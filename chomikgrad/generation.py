from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple


@dataclass(frozen=True)
class GreedyVerification:
    """Backend-neutral decision for one greedy speculative decode block."""

    emitted_tokens: Tuple[int, ...]
    accepted_draft_tokens: int


def verify_greedy_candidates(
    draft_tokens: Sequence[int], target_tokens: Sequence[int]
) -> GreedyVerification:
    """Accept a draft prefix and use the target token at the first mismatch."""
    draft = tuple(int(token) for token in draft_tokens)
    target = tuple(int(token) for token in target_tokens)
    if not draft:
        raise ValueError("speculative verification requires at least one token")
    if len(draft) != len(target):
        raise ValueError("draft and target blocks must have equal lengths")
    for index, (candidate, verified) in enumerate(zip(draft, target)):
        if candidate != verified:
            return GreedyVerification(draft[:index] + (verified,), index)
    return GreedyVerification(draft, len(draft))
