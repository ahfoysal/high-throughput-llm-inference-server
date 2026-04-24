"""M6: LoRA adapter hot-swap.

Low-Rank Adaptation (LoRA) adds a trainable low-rank delta to a frozen
linear layer:

    y = x W^T + b        (original)
    y = x W^T + b + (x A^T) B^T * (alpha / r)      (with LoRA)

Here `A` has shape `(r, in_features)` and `B` has shape
`(out_features, r)`. Storing + applying only `A` and `B` is tiny
compared to the full weight, and we can keep several adapters in RAM
and swap per-request.

This module implements:

1. `LoRAConfig` — describes an adapter (target module-name substrings,
   rank, alpha, and the per-target `(A, B)` tensors).
2. `LoRALinear` — a drop-in wrapper around `nn.Linear` that adds the
   LoRA delta on the fly. When no adapter is active it is a pure
   no-op (we literally just call the underlying linear).
3. `LoRAManager` — loads adapters (either from a dict of tensors or
   from a peft-compatible `adapter_model.safetensors` / `.bin`), keeps
   them in memory, and provides `activate(name)` / `deactivate()` that
   swap the active weights across every `LoRALinear` in O(#targets).
4. `apply_lora_to_model(model, target_substrings)` — walks the model
   and monkey-patches matching `nn.Linear` modules into `LoRALinear`,
   then returns a `LoRAManager`.

Design notes:

* **Hot-swap is a pointer swap, not a copy.** Activating an adapter
  iterates over the registered `LoRALinear` modules and sets each one's
  `_active_A` / `_active_B` references to tensors already in memory.
  No matmul happens at swap time — the cost is O(num_layers) Python
  attribute assignments.
* **peft-compatible file format.** peft saves adapters as a state-dict
  with keys like
  `base_model.model.<layer>.lora_A.weight` /
  `base_model.model.<layer>.lora_B.weight`.
  `LoRAManager.load_peft_dir()` parses either `.safetensors` or
  `.bin` checkpoints with that layout (falling back to a plain dict
  mapping `target -> (A, B)` if the user has hand-built one). We don't
  pull in `peft` itself — the format is simple enough to parse directly.
* Adapters that don't mention a given target are harmless: the
  `LoRALinear` keeps `_active_A = None` for that slot, which short-
  circuits the delta.

CPU-only, pure PyTorch. Works with any HF causal LM.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import torch
from torch import nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class LoRAConfig:
    """Describes one adapter.

    `weights` maps a fully-qualified module name (the dotted path inside
    the model, e.g. `"transformer.h.0.attn.c_attn"`) to a tuple
    `(A, B)`:

        A.shape == (r, in_features)
        B.shape == (out_features, r)

    `alpha / r` is the scaling factor applied at forward time. `r` is
    inferred from the tensors themselves.
    """

    name: str
    weights: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = field(default_factory=dict)
    alpha: float = 16.0

    @property
    def rank(self) -> int:
        for A, _ in self.weights.values():
            return int(A.shape[0])
        return 0

    @property
    def scaling(self) -> float:
        r = self.rank
        return (self.alpha / r) if r > 0 else 0.0

    def num_params(self) -> int:
        return sum(A.numel() + B.numel() for A, B in self.weights.values())


# ---------------------------------------------------------------------------
# Wrapped linear
# ---------------------------------------------------------------------------


def _linear_shape(base: nn.Module) -> Tuple[int, int]:
    """Return (in_features, out_features) for nn.Linear OR GPT-2 Conv1D.

    HF's GPT-2 uses `transformers.pytorch_utils.Conv1D`, which is a
    linear-ish layer with `.weight` of shape `(in, out)` (transposed
    relative to nn.Linear) and attribute `.nf` for the output size.
    """
    if isinstance(base, nn.Linear):
        return base.in_features, base.out_features
    # Conv1D duck-type: weight is (in, out), has `.nf`
    w = getattr(base, "weight", None)
    if w is not None and w.ndim == 2 and hasattr(base, "nf"):
        return int(w.shape[0]), int(base.nf)
    raise TypeError(f"unsupported linear-like module: {type(base).__name__}")


class LoRALinear(nn.Module):
    """nn.Linear (or GPT-2 Conv1D) + optional LoRA delta.

    The base weight/bias are kept verbatim (and frozen — we never train
    here, this is an inference-only adapter). `_active_A` and
    `_active_B` are non-Parameter tensors that the manager swaps in and
    out at request boundaries.
    """

    def __init__(self, base: nn.Module, fq_name: str) -> None:
        super().__init__()
        self.fq_name = fq_name
        # Keep the base module as a child so its state_dict keys are
        # stable (matters for saving/reloading the host model).
        self.base = base
        in_f, out_f = _linear_shape(base)
        self.in_features = in_f
        self.out_features = out_f
        # Freeze — inference only.
        for p in self.base.parameters():
            p.requires_grad_(False)

        self._active_A: Optional[torch.Tensor] = None
        self._active_B: Optional[torch.Tensor] = None
        self._active_scaling: float = 1.0
        self._active_name: Optional[str] = None

    def set_adapter(
        self,
        A: Optional[torch.Tensor],
        B: Optional[torch.Tensor],
        scaling: float,
        name: Optional[str],
    ) -> None:
        self._active_A = A
        self._active_B = B
        self._active_scaling = scaling
        self._active_name = name

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        out = self.base(x)
        if self._active_A is None or self._active_B is None:
            return out
        # x: (..., in); A: (r, in); B: (out, r)
        lora = torch.nn.functional.linear(x, self._active_A)   # (..., r)
        lora = torch.nn.functional.linear(lora, self._active_B)  # (..., out)
        return out + lora * self._active_scaling


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------


DEFAULT_TARGETS: Tuple[str, ...] = (
    # GPT-2 family
    "c_attn", "c_proj", "c_fc",
    # Llama / newer decoders
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
)


class LoRAManager:
    """Holds registered `LoRALinear` modules + a dict of adapters."""

    def __init__(self, lora_modules: Mapping[str, LoRALinear]) -> None:
        self.lora_modules: Dict[str, LoRALinear] = dict(lora_modules)
        self.adapters: Dict[str, LoRAConfig] = {}
        self.active_name: Optional[str] = None

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------
    def register(self, cfg: LoRAConfig) -> None:
        """Add an adapter. Does NOT activate it."""
        if cfg.rank <= 0:
            raise ValueError(f"adapter {cfg.name!r} has empty / zero-rank weights")
        # Sanity-check shapes against target modules.
        matched = 0
        for tgt, (A, B) in cfg.weights.items():
            mod = self._match(tgt)
            if mod is None:
                logger.debug("adapter %s: target %s not found, skipping", cfg.name, tgt)
                continue
            in_features = mod.in_features
            out_features = mod.out_features
            if A.shape[1] != in_features or B.shape[0] != out_features:
                raise ValueError(
                    f"adapter {cfg.name!r} target {tgt}: A={tuple(A.shape)} "
                    f"B={tuple(B.shape)} incompatible with Linear"
                    f"({in_features}, {out_features})"
                )
            matched += 1
        if matched == 0:
            raise ValueError(
                f"adapter {cfg.name!r} matched zero targets — check target names"
            )
        self.adapters[cfg.name] = cfg

    def _match(self, target: str) -> Optional[LoRALinear]:
        """Find the LoRALinear whose FQ name endswith or contains `target`."""
        if target in self.lora_modules:
            return self.lora_modules[target]
        # Fall back to suffix match, e.g. "model.layers.0.self_attn.q_proj"
        # against a short "q_proj".
        for fq, mod in self.lora_modules.items():
            if fq.endswith(target) or target in fq:
                return mod
        return None

    # ------------------------------------------------------------------
    # Hot-swap
    # ------------------------------------------------------------------
    def activate(self, name: str) -> None:
        if name not in self.adapters:
            raise KeyError(f"unknown adapter {name!r}; have {list(self.adapters)}")
        cfg = self.adapters[name]
        scaling = cfg.scaling
        # Default every slot to off — then turn on matching targets. This
        # way swapping from adapter X to adapter Y correctly clears any
        # targets that X hit but Y does not.
        for mod in self.lora_modules.values():
            mod.set_adapter(None, None, 1.0, None)
        for tgt, (A, B) in cfg.weights.items():
            mod = self._match(tgt)
            if mod is None:
                continue
            mod.set_adapter(A, B, scaling, name)
        self.active_name = name

    def deactivate(self) -> None:
        for mod in self.lora_modules.values():
            mod.set_adapter(None, None, 1.0, None)
        self.active_name = None

    # ------------------------------------------------------------------
    # Loading — peft-compatible
    # ------------------------------------------------------------------
    def load_peft_dir(self, path: str, name: Optional[str] = None) -> LoRAConfig:
        """Load a peft adapter checkpoint from a directory.

        Expects either `adapter_model.safetensors` or `adapter_model.bin`
        inside `path`, plus optionally an `adapter_config.json` for
        `alpha`. Keys look like
        `base_model.model.<whatever>.lora_A.weight`.
        """
        name = name or os.path.basename(path.rstrip("/")) or "adapter"
        alpha = 16.0
        cfg_path = os.path.join(path, "adapter_config.json")
        if os.path.exists(cfg_path):
            with open(cfg_path) as f:
                meta = json.load(f)
            alpha = float(meta.get("lora_alpha", alpha))

        state: Dict[str, torch.Tensor]
        st_path = os.path.join(path, "adapter_model.safetensors")
        bin_path = os.path.join(path, "adapter_model.bin")
        if os.path.exists(st_path):
            try:
                from safetensors.torch import load_file
                state = load_file(st_path)
            except Exception:  # noqa: BLE001
                # safetensors not installed — fall back to bin below.
                if not os.path.exists(bin_path):
                    raise
                state = torch.load(bin_path, map_location="cpu")
        elif os.path.exists(bin_path):
            state = torch.load(bin_path, map_location="cpu")
        else:
            raise FileNotFoundError(f"no adapter_model.* in {path}")

        return self.register_from_state_dict(name, state, alpha=alpha)

    def register_from_state_dict(
        self,
        name: str,
        state: Mapping[str, torch.Tensor],
        alpha: float = 16.0,
    ) -> LoRAConfig:
        """Parse a peft-layout state dict and register it.

        Groups keys by the target module name (everything between
        `base_model.model.` and `.lora_{A,B}.weight`).
        """
        buckets: Dict[str, Dict[str, torch.Tensor]] = {}
        for k, v in state.items():
            if ".lora_A" in k:
                side = "A"
            elif ".lora_B" in k:
                side = "B"
            else:
                continue
            # Strip peft prefix + .lora_{A,B}.weight suffix.
            target = k
            for pref in ("base_model.model.", "base_model."):
                if target.startswith(pref):
                    target = target[len(pref):]
                    break
            target = target.split(".lora_")[0]
            buckets.setdefault(target, {})[side] = v

        weights: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        for tgt, ab in buckets.items():
            if "A" in ab and "B" in ab:
                weights[tgt] = (ab["A"].detach(), ab["B"].detach())
        if not weights:
            raise ValueError(
                f"adapter {name!r}: no lora_A / lora_B pairs in state dict"
            )
        cfg = LoRAConfig(name=name, weights=weights, alpha=alpha)
        self.register(cfg)
        return cfg

    # ------------------------------------------------------------------
    # Synthetic adapters — for tests / benchmarks without external files.
    # ------------------------------------------------------------------
    def make_synthetic(
        self,
        name: str,
        rank: int = 8,
        alpha: float = 16.0,
        std: float = 1e-3,
        seed: Optional[int] = None,
    ) -> LoRAConfig:
        """Build a random-init adapter sized to fit every registered target."""
        if seed is not None:
            g = torch.Generator().manual_seed(seed)
        else:
            g = None
        weights: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        for fq, mod in self.lora_modules.items():
            in_f = mod.in_features
            out_f = mod.out_features
            A = torch.empty(rank, in_f)
            B = torch.zeros(out_f, rank)
            if g is None:
                A.normal_(0.0, std)
            else:
                A.normal_(0.0, std, generator=g)
            # B starts at zero so the fresh adapter is a no-op — classic
            # LoRA init. That also lets tests assert "activating a
            # zero-B adapter doesn't change logits" as a sanity check.
            weights[fq] = (A, B)
        cfg = LoRAConfig(name=name, weights=weights, alpha=alpha)
        self.register(cfg)
        return cfg


# ---------------------------------------------------------------------------
# Model-wide wiring
# ---------------------------------------------------------------------------


def apply_lora_to_model(
    model: nn.Module,
    target_substrings: Iterable[str] = DEFAULT_TARGETS,
) -> LoRAManager:
    """Walk `model`, wrap matching `nn.Linear` children in `LoRALinear`.

    Any linear whose fully-qualified name contains any of
    `target_substrings` gets swapped in-place. Returns a `LoRAManager`
    with handles to every wrapped module.
    """
    targets = tuple(target_substrings)
    wrapped: Dict[str, LoRALinear] = {}

    def _is_linear_like(m: nn.Module) -> bool:
        if isinstance(m, nn.Linear):
            return True
        # HF GPT-2 Conv1D
        return (
            type(m).__name__ == "Conv1D"
            and hasattr(m, "weight") and hasattr(m, "nf")
        )

    # Collect first, mutate after — modifying a module dict while
    # iterating over `named_modules()` is unsafe.
    to_wrap: List[Tuple[nn.Module, str, nn.Module, str]] = []
    for fq_name, module in model.named_modules():
        for child_name, child in list(module.named_children()):
            if not _is_linear_like(child):
                continue
            full = f"{fq_name}.{child_name}" if fq_name else child_name
            if any(t in full for t in targets):
                to_wrap.append((module, child_name, child, full))

    for parent, child_name, linear, full in to_wrap:
        wrapper = LoRALinear(linear, full)
        setattr(parent, child_name, wrapper)
        wrapped[full] = wrapper

    logger.info("LoRA: wrapped %d linear modules", len(wrapped))
    return LoRAManager(wrapped)


# ---------------------------------------------------------------------------
# Self-test / mini-benchmark — `python -m app.lora`
# ---------------------------------------------------------------------------


def _main() -> None:  # pragma: no cover
    import time

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = os.environ.get("LORA_MODEL", "sshleifer/tiny-gpt2")
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).eval()

    mgr = apply_lora_to_model(model)

    # Adapter 1: zero-B (identity). Adapter 2: random B with tiny std.
    a1 = mgr.make_synthetic("identity", rank=8, alpha=16, std=1e-3, seed=0)
    # For a2, flip B to a nonzero random so it actually perturbs logits.
    a2 = mgr.make_synthetic("perturb", rank=8, alpha=16, std=1e-3, seed=1)
    for tgt, (A, B) in list(a2.weights.items()):
        B2 = torch.empty_like(B).normal_(0.0, 1e-3, generator=torch.Generator().manual_seed(42))
        a2.weights[tgt] = (A, B2)

    inp = tok("The quick brown fox", return_tensors="pt").input_ids

    with torch.no_grad():
        base_logits = model(inp).logits
        mgr.activate("identity")
        id_logits = model(inp).logits
        mgr.activate("perturb")
        pb_logits = model(inp).logits
        mgr.deactivate()
        base2_logits = model(inp).logits

    id_diff = (base_logits - id_logits).abs().max().item()
    pb_diff = (base_logits - pb_logits).abs().max().item()
    restore_diff = (base_logits - base2_logits).abs().max().item()
    print(f"identity adapter (B=0) max |Δlogits| = {id_diff:.3e}  (should be ~0)")
    print(f"perturb  adapter     max |Δlogits| = {pb_diff:.3e}  (should be >0)")
    print(f"deactivate restore   max |Δlogits| = {restore_diff:.3e} (should be ~0)")
    assert id_diff < 1e-5, "zero-B adapter changed logits!"
    assert restore_diff < 1e-5, "deactivate() did not restore baseline!"
    assert pb_diff > 0.0, "nonzero adapter had no effect!"

    # Hot-swap benchmark.
    N = 200
    t0 = time.time()
    for i in range(N):
        mgr.activate("identity" if i % 2 == 0 else "perturb")
    swap_s = (time.time() - t0) / N
    print(f"hot-swap latency: {swap_s * 1e6:.1f} µs/swap over {N} swaps "
          f"({len(mgr.lora_modules)} targets, rank={a1.rank})")


if __name__ == "__main__":  # pragma: no cover
    _main()
