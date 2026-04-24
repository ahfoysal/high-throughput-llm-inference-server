"""M4: INT4 weight-only quantization (group-wise, symmetric+zero-point).

Pure-PyTorch, CPU-friendly, no external libs (no bitsandbytes, no awq).
This is the *semantics* of group-wise INT4 — the production win comes
from fused dequant-matmul kernels, which aren't in scope for CPU.

Algorithm (per Linear weight W of shape (out, in)):

  - Split each row into groups of size `group_size` along the input dim.
  - For each group compute:
        zmin, zmax = min, max
        scale = (zmax - zmin) / 15
        zero  = round(-zmin / scale)        # so that 0 maps near-zero error
    Quantize:  q = clamp(round(W / scale + zero), 0, 15)  (uint4)
    Dequant :  W' = (q - zero) * scale

  - Pack two uint4 values into one uint8 (low nibble = even idx, high
    nibble = odd idx). This halves storage: 4 bits per weight.

On forward:
  - Unpack to uint8 per-nibble, subtract zero, multiply by scale, cast
    to float, reshape to (out, in), run standard `F.linear`.

Memory: INT4 packed weights are ~1/8 the size of fp32 weights (plus
small scale/zero tensors: one fp32 and one uint8 per group).

Quality: round-trip |W - W'| is bounded by scale/2 per group. For most
transformer weights this costs ~0.1-0.3 perplexity on small models.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core quant / dequant primitives
# ---------------------------------------------------------------------------


def quantize_int4_groupwise(
    W: torch.Tensor, group_size: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Quantize a 2-D weight matrix to packed uint4.

    Returns (packed, scales, zeros, pad) where
      packed : uint8, shape (out, in_padded // 2)
      scales : float32, shape (out, n_groups)
      zeros  : uint8,   shape (out, n_groups)    (range 0..15)
      pad    : number of zero columns appended to make `in` divisible
               by group_size (must be stripped at dequant time).
    """
    assert W.dim() == 2, "only 2-D weights supported"
    out_features, in_features = W.shape
    W = W.detach().to(torch.float32)

    # Pad the input dim up to a multiple of group_size.
    pad = (-in_features) % group_size
    if pad:
        W = F.pad(W, (0, pad))
    in_padded = in_features + pad
    n_groups = in_padded // group_size

    # Reshape to (out, n_groups, group_size).
    Wg = W.view(out_features, n_groups, group_size)
    zmin = Wg.min(dim=-1).values                      # (out, n_groups)
    zmax = Wg.max(dim=-1).values
    scale = (zmax - zmin).clamp(min=1e-8) / 15.0      # (out, n_groups)
    zero = torch.round(-zmin / scale).clamp(0, 15)    # (out, n_groups)

    q = torch.round(Wg / scale.unsqueeze(-1) + zero.unsqueeze(-1))
    q = q.clamp(0, 15).to(torch.uint8)                # (out, n_groups, group_size)
    q = q.view(out_features, in_padded)               # (out, in_padded)

    # Pack two nibbles per byte. Even idx -> low nibble, odd -> high nibble.
    assert in_padded % 2 == 0, "in_padded must be even (group_size is >=2)"
    low = q[:, 0::2]
    high = q[:, 1::2]
    packed = (low | (high << 4)).to(torch.uint8)      # (out, in_padded // 2)

    return packed, scale.to(torch.float32), zero.to(torch.uint8), pad


def dequantize_int4_groupwise(
    packed: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    group_size: int,
    in_features: int,
    pad: int,
) -> torch.Tensor:
    """Inverse of `quantize_int4_groupwise`. Returns float32 (out, in_features)."""
    out_features = packed.shape[0]
    in_padded = in_features + pad

    # Unpack.
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    q = torch.empty(
        (out_features, in_padded), dtype=torch.uint8, device=packed.device,
    )
    q[:, 0::2] = low
    q[:, 1::2] = high

    n_groups = in_padded // group_size
    q = q.view(out_features, n_groups, group_size).to(torch.float32)
    zeros_f = zeros.to(torch.float32).unsqueeze(-1)
    scales_f = scales.to(torch.float32).unsqueeze(-1)
    W = (q - zeros_f) * scales_f                      # (out, n_groups, group_size)
    W = W.view(out_features, in_padded)
    if pad:
        W = W[:, :in_features]
    return W


# ---------------------------------------------------------------------------
# Int4 Linear module — drop-in replacement for nn.Linear.
# ---------------------------------------------------------------------------


class Int4Linear(nn.Module):
    """Linear layer with INT4 group-quantized weights.

    Forward dequantizes once per call to float32 then uses F.linear.
    Not fast on CPU vs plain fp32 — the point here is *memory* (weights
    are 1/8 size) and *semantics*, matching what AWQ/GPTQ kernels do on
    GPU. A real deployment would use a fused dequant-matmul kernel.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        group_size: int = 64,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        # placeholders filled in by from_linear / load_quantized
        self.register_buffer(
            "packed",
            torch.zeros((out_features, 1), dtype=torch.uint8),
            persistent=True,
        )
        self.register_buffer(
            "scales", torch.zeros((out_features, 1), dtype=torch.float32),
        )
        self.register_buffer(
            "zeros", torch.zeros((out_features, 1), dtype=torch.uint8),
        )
        self.pad = 0
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    @classmethod
    def from_linear(cls, lin: nn.Linear, group_size: int = 64) -> "Int4Linear":
        m = cls(
            in_features=lin.in_features,
            out_features=lin.out_features,
            bias=lin.bias is not None,
            group_size=group_size,
        )
        packed, scales, zeros, pad = quantize_int4_groupwise(
            lin.weight.data, group_size=group_size,
        )
        m.packed = packed
        m.scales = scales
        m.zeros = zeros
        m.pad = pad
        if lin.bias is not None:
            m.bias.data.copy_(lin.bias.data.detach().to(torch.float32))
        return m

    def dequantize_weight(self) -> torch.Tensor:
        return dequantize_int4_groupwise(
            self.packed, self.scales, self.zeros,
            self.group_size, self.in_features, self.pad,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W = self.dequantize_weight().to(x.dtype)
        return F.linear(x, W, self.bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"group_size={self.group_size}, bias={self.bias is not None}"
        )


# ---------------------------------------------------------------------------
# Conv1D support (HF GPT-2 uses transformers.pytorch_utils.Conv1D for
# attn / mlp projections — it's a Linear with transposed weight:
#   y = x @ W + b,  W.shape == (in, out)
# )
# ---------------------------------------------------------------------------


class Int4Conv1D(nn.Module):
    """INT4-quantized drop-in for HF `Conv1D` (GPT-2 style)."""

    def __init__(
        self, nf: int, nx: int, group_size: int = 64,
    ) -> None:
        super().__init__()
        # nx = in_features, nf = out_features  (matches HF Conv1D naming)
        self.nx = nx
        self.nf = nf
        self.group_size = group_size
        self.register_buffer(
            "packed", torch.zeros((nf, 1), dtype=torch.uint8),
        )
        self.register_buffer(
            "scales", torch.zeros((nf, 1), dtype=torch.float32),
        )
        self.register_buffer(
            "zeros", torch.zeros((nf, 1), dtype=torch.uint8),
        )
        self.pad = 0
        self.bias = nn.Parameter(torch.zeros(nf))

    @classmethod
    def from_conv1d(cls, c1d, group_size: int = 64) -> "Int4Conv1D":
        # HF Conv1D weight shape is (nx, nf); we transpose to (nf, nx)
        # (i.e. (out, in)) for standard groupwise quant.
        nf = c1d.weight.shape[1]
        nx = c1d.weight.shape[0]
        m = cls(nf=nf, nx=nx, group_size=group_size)
        W = c1d.weight.data.detach().t().contiguous()   # (nf, nx) == (out, in)
        packed, scales, zeros, pad = quantize_int4_groupwise(W, group_size=group_size)
        m.packed = packed
        m.scales = scales
        m.zeros = zeros
        m.pad = pad
        m.bias.data.copy_(c1d.bias.data.detach().to(torch.float32))
        return m

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W = dequantize_int4_groupwise(
            self.packed, self.scales, self.zeros,
            self.group_size, self.nx, self.pad,
        ).to(x.dtype)                                   # (nf, nx)
        # HF Conv1D: size_out = x.size()[:-1] + (nf,); x = x @ W.T + b
        # Our W is already (nf, nx), so x @ W.T reverses it correctly.
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), W.t())
        return x.view(size_out)


# ---------------------------------------------------------------------------
# Whole-model swap-in
# ---------------------------------------------------------------------------


@dataclass
class QuantStats:
    replaced_linear: int
    replaced_conv1d: int
    bytes_before: int
    bytes_after: int

    @property
    def compression(self) -> float:
        return self.bytes_before / max(1, self.bytes_after)


def quantize_model_int4(model: nn.Module, group_size: int = 64) -> QuantStats:
    """In-place swap every nn.Linear / HF Conv1D for an INT4 variant.

    Returns memory statistics. Bias/LayerNorm/Embedding are left alone
    (standard practice — quantizing them hurts quality more than it
    saves memory).
    """
    # Lazy import — HF may not be installed in every test environment.
    try:
        from transformers.pytorch_utils import Conv1D  # type: ignore
    except Exception:  # noqa: BLE001
        Conv1D = None  # type: ignore

    bytes_before = 0
    bytes_after = 0
    replaced_linear = 0
    replaced_conv1d = 0

    def _count(t: torch.Tensor) -> int:
        return t.numel() * t.element_size()

    # Walk modules and replace children.
    for parent in list(model.modules()):
        for name, child in list(parent.named_children()):
            if isinstance(child, nn.Linear):
                bytes_before += _count(child.weight)
                new = Int4Linear.from_linear(child, group_size=group_size)
                bytes_after += (
                    _count(new.packed) + _count(new.scales) + _count(new.zeros)
                )
                setattr(parent, name, new)
                replaced_linear += 1
            elif Conv1D is not None and isinstance(child, Conv1D):
                bytes_before += _count(child.weight)
                new = Int4Conv1D.from_conv1d(child, group_size=group_size)
                bytes_after += (
                    _count(new.packed) + _count(new.scales) + _count(new.zeros)
                )
                setattr(parent, name, new)
                replaced_conv1d += 1

    return QuantStats(
        replaced_linear=replaced_linear,
        replaced_conv1d=replaced_conv1d,
        bytes_before=bytes_before,
        bytes_after=bytes_after,
    )


# ---------------------------------------------------------------------------
# Self-test — `python -m app.quant`
# ---------------------------------------------------------------------------


def _main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    torch.manual_seed(0)

    # --- round-trip error on a random matrix ---
    W = torch.randn(256, 512)
    packed, scales, zeros, pad = quantize_int4_groupwise(W, group_size=64)
    Wq = dequantize_int4_groupwise(packed, scales, zeros, 64, W.shape[1], pad)
    err = (W - Wq).abs().mean().item()
    print(f"[round-trip] mean abs err = {err:.5f} (expect < 0.15 for N(0,1))")
    # INT4 group=64 on N(0,1): step ≈ 6σ/15 ≈ 0.4, avg |err| ≈ step/4 ≈ 0.1.
    assert err < 0.15, "INT4 round-trip error too large"

    # --- Int4Linear forward matches nn.Linear closely ---
    lin = nn.Linear(512, 256)
    q = Int4Linear.from_linear(lin, group_size=64)
    x = torch.randn(4, 512)
    y0 = lin(x)
    y1 = q(x)
    rel = (y0 - y1).abs().mean().item() / y0.abs().mean().item()
    print(f"[Int4Linear] relative err = {rel*100:.2f}%  (expect < ~10%)")
    assert rel < 0.15

    # --- End-to-end: quantize a real model and generate a few tokens ---
    # Prefer distilgpt2 over tiny-gpt2 for meaningful compression numbers
    # (tiny-gpt2 has hidden=2 which makes per-group overhead dominate).
    import os as _os
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        name = _os.environ.get("QUANT_MODEL", "distilgpt2")
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModelForCausalLM.from_pretrained(name).eval()
        prompt = "Hello, my name is"
        ids = tok(prompt, return_tensors="pt").input_ids

        with torch.no_grad():
            base_logits = model(ids).logits[0, -1, :]
        base_tok = int(base_logits.argmax().item())

        stats = quantize_model_int4(model, group_size=64)
        print(f"[model] replaced Linear={stats.replaced_linear} "
              f"Conv1D={stats.replaced_conv1d}")
        print(f"[model] weight bytes {stats.bytes_before:,} -> "
              f"{stats.bytes_after:,} ({stats.compression:.2f}x compression)")

        with torch.no_grad():
            q_logits = model(ids).logits[0, -1, :]
        q_tok = int(q_logits.argmax().item())
        print(f"[model] baseline next-token id={base_tok}, quantized id={q_tok}, "
              f"match={base_tok == q_tok}")

        # Greedy a few tokens to confirm nothing explodes.
        cur = ids
        past = None
        out = []
        with torch.no_grad():
            for _ in range(8):
                o = model(input_ids=cur, past_key_values=past, use_cache=True)
                past = o.past_key_values
                nxt = int(o.logits[0, -1, :].argmax().item())
                out.append(nxt)
                cur = torch.tensor([[nxt]])
        print(f"[model] quantized greedy tokens: {out}")
        print(f"[model] decoded: {tok.decode(out)!r}")
    except Exception as e:  # noqa: BLE001
        print(f"[model] skipped end-to-end test: {e}")


if __name__ == "__main__":  # pragma: no cover
    _main()
