"""A vectorized without-replacement hand prior; no learned counts and no count DP.

Public lower bounds denote individually known copies in hand. Upper bounds
count all copies still possible in hand/library, including those known copies.
The unknown hand slots are modeled as a random subset of the remaining pool.
Opponent choices, mulligans and private selection can violate that assumption.
"""

import torch
from torch import Tensor

from .belief import HandBelief


def log_choose(n: Tensor, k: Tensor) -> Tensor:
    """log C(n, k) in float64; -inf where the binomial coefficient is zero."""
    n, k = n.double(), k.double()
    value = (torch.lgamma(n.clamp(min=0) + 1) - torch.lgamma(k.clamp(min=0) + 1)
             - torch.lgamma((n - k).clamp(min=0) + 1))
    return value.masked_fill((n < 0) | (k < 0) | (k > n), float("-inf"))


def _validate(lower, upper, hand_size, classes):
    if lower.ndim != 2 or upper.shape != lower.shape or hand_size.shape != lower.shape[:1]:
        raise ValueError("statistical bounds must be [batch, kinds] with one hand size per row")
    if any(t.dtype not in (torch.int32, torch.int64) for t in (lower, upper, hand_size)):
        raise ValueError("statistical counts must be integer tensors")
    if classes <= 0 or ((lower < 0) | (upper < lower)).any():
        raise ValueError("invalid statistical count bounds")
    if ((hand_size < lower.sum(-1)) | (hand_size > upper.sum(-1))
            | (hand_size < 0) | (hand_size >= classes)).any():
        raise ValueError("hand size cannot be represented by these bounds/count classes")


def statistical_hand_belief(
    lower: Tensor, upper: Tensor, hand_size: Tensor, classes: int, *, validate: bool = True
) -> HandBelief:
    """Exact per-card count marginals of the hypergeometric hand prior.

    validate=False skips the host-synchronizing checks for bounds that
    validate_observation already enforced (the model's forward pass).
    """
    if validate:
        _validate(lower, upper, hand_size, classes)
    remaining = upper - lower
    population = remaining.sum(-1)
    draws = hand_size - lower.sum(-1)
    counts = torch.arange(classes, device=lower.device)[None, None]
    selected = counts - lower.unsqueeze(-1)
    scores = log_choose(remaining.unsqueeze(-1), selected)
    partition = log_choose(population, draws)
    log_marginals = (scores + log_choose((population[:, None] - remaining).unsqueeze(-1),
                                       draws[:, None, None] - selected)
                     - partition[:, None, None])
    probabilities = torch.softmax(log_marginals, -1).float()
    expected = (probabilities * counts).sum(-1)
    return HandBelief(probabilities, expected, partition.float(), scores.float())


def role_statistics(lower: Tensor, upper: Tensor, hand_size: Tensor,
                    membership: Tensor, *, validate: bool = True) -> tuple[Tensor, Tensor]:
    """Exact role presence/expectation under the prior, including overlaps.

    membership is [batch, kinds, roles]. A physical copy is counted once per
    role even if it has several abilities/modes providing that capability.
    Roles are not exclusive and their presence probabilities need not sum to 1.
    """
    if membership.ndim != 3 or membership.shape[:2] != lower.shape:
        raise ValueError("role membership must match the deck candidates")
    if validate:  # the model checks its role matrix once, at construction
        if not torch.all((membership == 0) | (membership == 1)):
            raise ValueError("role membership must be binary, not soft clusters")
        _validate(lower, upper, hand_size, int(hand_size.max().item()) + 1)
    remaining = upper - lower
    population = remaining.sum(-1)
    draws = hand_size - lower.sum(-1)
    role_copies = (remaining.unsqueeze(-1) * membership).sum(1)
    known = (lower.unsqueeze(-1) * membership).sum(1)
    log_absent = log_choose(population[:, None] - role_copies, draws[:, None]) - log_choose(
        population, draws)[:, None]
    presence = -torch.expm1(log_absent.clamp(max=0))
    presence = torch.where(known > 0, torch.ones_like(presence), presence).float()
    expected = known + draws[:, None] * role_copies / population[:, None].clamp(min=1)
    return presence, expected.float()
