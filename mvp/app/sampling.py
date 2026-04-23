"""Token sampling strategies: greedy, temperature, top-p (nucleus), top-k.

All functions take a 1-D tensor of logits for the next token and return an int.
Later milestones (M2+) will vectorize these across a batch.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: Optional[int] = None,
) -> int:
    """Sample a single next-token id from a 1-D logits tensor.

    - temperature == 0 OR temperature very small -> greedy argmax
    - top_k: keep only the k highest-logit tokens before sampling
    - top_p: nucleus — keep smallest set whose cumulative prob >= top_p
    """
    if logits.dim() != 1:
        logits = logits.view(-1)

    # Greedy shortcut
    if temperature is None or temperature <= 1e-6:
        return int(torch.argmax(logits).item())

    logits = logits / float(temperature)

    # top-k filter
    if top_k is not None and top_k > 0 and top_k < logits.size(-1):
        kth_val = torch.topk(logits, top_k).values[-1]
        logits = torch.where(
            logits < kth_val, torch.full_like(logits, float("-inf")), logits
        )

    # top-p (nucleus) filter
    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cum_probs = torch.cumsum(sorted_probs, dim=-1)
        # tokens to remove: those where cumulative prob is already past top_p
        # keep at least one token — shift mask right by 1
        remove = cum_probs > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        # scatter back to original order
        logits = torch.full_like(logits, float("-inf"))
        logits.scatter_(0, sorted_idx, sorted_logits)

    probs = F.softmax(logits, dim=-1)
    # Guard against all -inf (shouldn't happen given the keep-at-least-one rule)
    if torch.isnan(probs).any() or probs.sum().item() == 0.0:
        return int(torch.argmax(logits).item())

    next_id = int(torch.multinomial(probs, num_samples=1).item())
    return next_id
