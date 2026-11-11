"""Executable-state frontiers and recovery gaps at planning time."""

from __future__ import annotations

from typing import Iterable

from .state_catalog import CheckpointCandidate, is_compatible
from .workflow import PendingContinuation


def executable_frontier(
    continuation: PendingContinuation,
    selected: Iterable[CheckpointCandidate],
) -> int:
    """Return the deepest token position among compatible selected checkpoints, or zero."""

    compatible_positions = (
        checkpoint.token_pos
        for checkpoint in selected
        if is_compatible(checkpoint, continuation)
    )
    return max(compatible_positions, default=0)


def recovery_gap(
    continuation: PendingContinuation,
    selected: Iterable[CheckpointCandidate],
) -> int:
    """Return the recovery gap between the planning target and the executable frontier."""

    frontier = executable_frontier(continuation, selected)
    gap = continuation.planning_target - frontier
    if gap < 0:
        raise ValueError("recovery gap must not be negative")
    return gap
