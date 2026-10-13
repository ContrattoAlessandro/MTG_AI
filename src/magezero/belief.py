"""A distribution over legal hand multisets, conditioned on the public hand size.

p(n | h) is proportional to exp(sum_i logits[i, n_i]) for lower_i <= n_i <= upper_i
and sum_i n_i == h. Dynamic programming marginalizes counts; it does not search
game states or simulate actions. Correlations beyond these constraints are not
represented by this factorization.
"""

from dataclasses import dataclass

import torch
from torch import Tensor

NEG = -1e9  # finite sentinel avoids undefined derivatives at unreachable DP cells


@dataclass
class HandBelief:
    """Per-card count marginals of the opponent's hand."""

    probabilities: Tensor  # [batch, card kinds, count classes]
    expected_counts: Tensor  # [batch, card kinds]
    log_partition: Tensor  # [batch]
    scores: Tensor  # [batch, card kinds, count classes], unnormalized log factors

    def nll(self, counts: Tensor) -> Tensor:
        """Joint negative log-likelihood of the true counts [batch, card kinds]."""
        selected = self.scores.gather(-1, counts.long().unsqueeze(-1)).squeeze(-1)
        return self.log_partition - selected.sum(-1)


def constrained_hand_belief(logits: Tensor, lower: Tensor, upper: Tensor, hand_size: Tensor) -> HandBelief:
    """Compute exact marginals of the count-factor model in log space.

    Arithmetic runs in float32 even under autocast. Zero-capacity padded cards
    have a point mass at zero. Inputs must contain a feasible set of bounds.
    """
    scores = logits.float()
    batch, kinds, classes = scores.shape
    if lower.shape != (batch, kinds) or upper.shape != lower.shape:
        raise ValueError("belief bounds must match the card candidate matrix")
    if hand_size.shape != (batch,):
        raise ValueError("hand_size must have one entry per observation")
    if ((lower < 0) | (upper < lower) | (lower >= classes)).any():
        raise ValueError("invalid belief count bounds")
    if ((hand_size < lower.sum(-1)) | (hand_size > upper.sum(-1)) | (hand_size >= classes) | (hand_size < 0)).any():
        raise ValueError("hand size cannot be represented by these bounds/count classes")
    horizon = int(hand_size.max().item())
    counts = torch.arange(classes, device=scores.device)
    support = (counts >= lower.unsqueeze(-1)) & (counts <= upper.unsqueeze(-1)) & (counts <= hand_size[:, None, None])
    scores = scores.masked_fill(~support, NEG)
    # Subtract a card-wise constant for numerical stability. This changes neither
    # the normalized joint distribution nor its negative log likelihood.
    scores = scores - scores.max(-1, keepdim=True).values
    base = scores.new_full((batch, horizon + 1), NEG)
    base[:, 0] = 0

    active_classes = min(classes, horizon + 1)
    totals = torch.arange(horizon + 1, device=scores.device)
    active_counts = torch.arange(active_classes, device=scores.device)
    predecessors = totals[:, None] - active_counts[None, :]
    valid_predecessors = predecessors >= 0

    def convolve(previous: Tensor, card: Tensor) -> Tensor:
        terms = previous[:, predecessors.clamp(min=0)] + card[:, None, :active_classes]
        terms = terms.masked_fill(~valid_predecessors[None], NEG)
        return torch.logsumexp(terms, -1)

    forward = [base]
    for i in range(kinds):
        forward.append(convolve(forward[-1], scores[:, i]))
    backward = [base]
    for i in range(kinds - 1, -1, -1):
        backward.append(convolve(backward[-1], scores[:, i]))
    backward.reverse()
    partition = forward[-1].gather(1, hand_size[:, None]).squeeze(1)
    marginal_rows = []
    remaining = hand_size[:, None, None] - active_counts[None, :, None] - totals[None, None, :]
    valid_remainder = (remaining >= 0) & (remaining <= horizon)
    for i in range(kinds):
        suffix = backward[i + 1][:, None, :].expand(-1, active_classes, -1)
        suffix = suffix.gather(-1, remaining.clamp(0, horizon))
        terms = (forward[i][:, None, :] + suffix).masked_fill(~valid_remainder, NEG)
        row = torch.logsumexp(terms, -1) + scores[:, i, :active_classes] - partition[:, None]
        row = torch.nn.functional.pad(row, (0, classes - active_classes), value=NEG)
        row = row.masked_fill(~support[:, i], NEG)
        marginal_rows.append(torch.softmax(row, -1))
    probabilities = torch.stack(marginal_rows, 1)
    expected = (probabilities * counts).sum(-1)
    return HandBelief(probabilities, expected, partition, scores)
