from __future__ import annotations

import torch
import torch.nn.functional as F


def label_smoothed_nll_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    epsilon: float = 0.0,
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    Compute label-smoothed cross-entropy loss for seq2seq.
    """
    if epsilon <= 0.0:
        return F.cross_entropy(logits, targets, ignore_index=ignore_index)

    n_classes = logits.size(-1)
    log_probs = F.log_softmax(logits, dim=-1)
    nll = -log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    smooth = -log_probs.mean(dim=-1)
    if ignore_index is not None:
        mask = targets.ne(ignore_index)
        nll = nll[mask]
        smooth = smooth[mask]
    loss = (1.0 - epsilon) * nll + epsilon * smooth
    return loss.mean()
