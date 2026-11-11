"""Connect global optimization results to runtime state operations."""

from __future__ import annotations

from typing import Mapping, Protocol, Sequence

from .adapters.sglang import RuntimeCheckpointHandle
from .optimizer import AllocationResult, GlobalOptimizer
from .state_catalog import CheckpointCandidate, validate_unique_checkpoint_ids
from .workflow import PendingContinuation


class RuntimeAdapter(Protocol):
    """Describe the minimum runtime capabilities required by the controller to evict state."""

    def evict_mamba_only(
        self,
        handle: RuntimeCheckpointHandle,
    ) -> None:
        """Evict only the Mamba state referenced by the handle."""


class ReconcileExecutionError(RuntimeError):
    """Indicate that some evictions succeeded during reconciliation before a subsequent eviction failed."""

    def __init__(
        self,
        completed_evictions: tuple[str, ...],
        failed_checkpoint_id: str,
        cause: Exception,
    ) -> None:
        self.completed_evictions = completed_evictions
        self.failed_checkpoint_id = failed_checkpoint_id
        self.cause = cause
        completed = ", ".join(completed_evictions) or "无"
        super().__init__(
            "状态协调执行失败："
            f"已完成驱逐 {completed}；"
            f"失败检查点 {failed_checkpoint_id}；"
            f"原因：{cause}"
        )


class StateController:
    """Translate optimizer selections into deterministic keep and evict actions."""

    def __init__(
        self,
        optimizer: GlobalOptimizer,
        runtime_adapter: RuntimeAdapter,
    ) -> None:
        self._optimizer = optimizer
        self._runtime_adapter = runtime_adapter

    def reconcile(
        self,
        continuations: Sequence[PendingContinuation],
        candidates: Sequence[CheckpointCandidate],
        handles: Mapping[str, RuntimeCheckpointHandle],
        budget_bytes: int,
    ) -> AllocationResult:
        """Select states from the current snapshot and evict resident states that are not retained.

        The ``recurrent_resident`` value supplied on each call must reflect the
        current decision point; the controller does not retain residency state
        across rounds.
        """
        validate_unique_checkpoint_ids(candidates)
        self._validate_handles(handles)
        allocation = self._optimizer.select(
            continuations,
            candidates,
            budget_bytes,
        )
        selected_ids = {
            candidate.checkpoint_id for candidate in allocation.selected
        }
        eviction_candidates = tuple(
            sorted(
                (
                    candidate
                    for candidate in candidates
                    if candidate.recurrent_resident
                    and candidate.checkpoint_id not in selected_ids
                ),
                key=lambda candidate: candidate.checkpoint_id,
            )
        )

        missing_handle_ids = tuple(
            candidate.checkpoint_id
            for candidate in eviction_candidates
            if candidate.checkpoint_id not in handles
        )
        if missing_handle_ids:
            missing = ", ".join(missing_handle_ids)
            raise ValueError(f"待驱逐检查点缺少运行时句柄：{missing}")

        completed_evictions = []
        for candidate in eviction_candidates:
            try:
                self._runtime_adapter.evict_mamba_only(
                    handles[candidate.checkpoint_id]
                )
            except Exception as cause:
                raise ReconcileExecutionError(
                    completed_evictions=tuple(completed_evictions),
                    failed_checkpoint_id=candidate.checkpoint_id,
                    cause=cause,
                ) from cause
            completed_evictions.append(candidate.checkpoint_id)

        return allocation

    @staticmethod
    def reconcile_strict(epoch, plan, runtime):
        """Explicit strict execution entry point; do not invoke the legacy optimizer or implicitly reselect through the current adapter."""
        from .strict_control import StrictReconciler
        return StrictReconciler().reconcile_strict(epoch, plan, runtime)

    @staticmethod
    def _validate_handles(
        handles: Mapping[str, RuntimeCheckpointHandle],
    ) -> None:
        """Validate handle mapping consistency before policy computation or runtime mutation."""
        seen_handles: dict[str, RuntimeCheckpointHandle] = {}
        for mapping_key, handle in handles.items():
            if not isinstance(handle, RuntimeCheckpointHandle):
                raise TypeError(
                    f"运行时句柄 {mapping_key} 必须是 RuntimeCheckpointHandle"
                )
            if mapping_key != handle.checkpoint_id:
                raise ValueError(
                    "运行时句柄字典键与检查点标识不一致："
                    f"键 {mapping_key}，句柄 {handle.checkpoint_id}"
                )

            previous = seen_handles.get(handle.checkpoint_id)
            if previous is not None and previous != handle:
                raise ValueError(
                    "同一 checkpoint_id 对应多个冲突句柄："
                    f"{handle.checkpoint_id}"
                )
            seen_handles[handle.checkpoint_id] = handle
