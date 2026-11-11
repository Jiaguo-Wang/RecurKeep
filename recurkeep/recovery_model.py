"""Formal position-aware recovery cost model and the historical single-variable model."""

from __future__ import annotations

from bisect import bisect_left
import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Optional, Tuple, Union


# Historical profiles ship with the package so the release does not depend on
# the research repository layout.
_ARTIFACT_DIRECTORY = Path(__file__).resolve().parent / "profiles"
_DEFAULT_PROFILE_PATH = _ARTIFACT_DIRECTORY / "replay_cost.csv"
_DEFAULT_FIT_PATH = _ARTIFACT_DIRECTORY / "fit_metrics.json"


@dataclass(frozen=True)
class RecoveryModelMetadata:
    """Version, coefficients, units, and the validated range of the formal model."""

    name: str
    coefficient_a: float
    coefficient_b: float
    coefficient_c: float
    gap_unit: str
    target_unit: str
    output_unit: str
    calibration_artifact: str
    minimum_gap_tokens: int
    maximum_target_tokens: int


FORMAL_RECOVERY_MODEL_METADATA = RecoveryModelMetadata(
    name="position_aware_quadratic_v1",
    coefficient_a=37.828150,
    coefficient_b=0.345974143,
    coefficient_c=-0.156201917,
    gap_unit="tokens",
    target_unit="tokens",
    output_unit="ms",
    calibration_artifact="recovery_model_freeze_20260826_154235_266020",
    minimum_gap_tokens=0,
    maximum_target_tokens=131_072,
)


class RecoveryCostModel:
    """Estimate recovery latency for a given recovery interval and absolute target position."""

    metadata = FORMAL_RECOVERY_MODEL_METADATA

    def estimate(self, gap_tokens: int, target_tokens: int) -> float:
        """Return the position-aware incremental recovery latency in milliseconds."""
        self._validate_inputs(gap_tokens, target_tokens)
        if gap_tokens == 0:
            return 0.0

        gap_ki_tokens = gap_tokens / 1024.0
        target_ki_tokens = target_tokens / 1024.0
        metadata = self.metadata
        return (
            metadata.coefficient_a * gap_ki_tokens
            + metadata.coefficient_b * gap_ki_tokens * target_ki_tokens
            + metadata.coefficient_c * gap_ki_tokens * gap_ki_tokens
        )

    def cost(self, gap_tokens: int, target_tokens: int) -> float:
        """Return the position-aware recovery latency through an explicit cost interface."""
        return self.estimate(gap_tokens, target_tokens)

    def derivative_ms_per_ki_token(
        self,
        gap_tokens: int,
        target_tokens: int,
    ) -> float:
        """Return the first derivative with respect to the gap in Ki tokens at a fixed target."""
        self._validate_inputs(gap_tokens, target_tokens)
        gap_ki_tokens = gap_tokens / 1024.0
        target_ki_tokens = target_tokens / 1024.0
        metadata = self.metadata
        return (
            metadata.coefficient_a
            + metadata.coefficient_b * target_ki_tokens
            + 2.0 * metadata.coefficient_c * gap_ki_tokens
        )

    @classmethod
    def _validate_inputs(cls, gap_tokens: int, target_tokens: int) -> None:
        """Reject inputs outside the validated range or with invalid recovery semantics."""
        if gap_tokens < 0:
            raise ValueError("gap_tokens must be non-negative")
        if target_tokens < 0:
            raise ValueError("target_tokens must be non-negative")
        if gap_tokens > target_tokens:
            raise ValueError("gap_tokens must not exceed target_tokens")
        if target_tokens > cls.metadata.maximum_target_tokens:
            raise ValueError(
                "target_tokens is outside the validated range of the formal recovery model: "
                f"{target_tokens} > {cls.metadata.maximum_target_tokens}"
            )


class HistoricalRecoveryCostModel:
    """Reproduce the historical single-variable recovery cost from WP2 profile points."""

    def __init__(
        self,
        profile_path: Optional[Union[str, Path]] = None,
        fit_metrics_path: Optional[Union[str, Path]] = None,
    ) -> None:
        profile = Path(profile_path) if profile_path is not None else _DEFAULT_PROFILE_PATH
        fit_metrics = (
            Path(fit_metrics_path)
            if fit_metrics_path is not None
            else _DEFAULT_FIT_PATH
        )

        self._profile_points = self._load_profile(profile)
        self._tokens = tuple(point[0] for point in self._profile_points)
        self._fit_intercept_ms, self._fit_slope_ms_per_token = self._load_fit(
            fit_metrics
        )

    @staticmethod
    def _load_profile(path: Path) -> Tuple[Tuple[int, float], ...]:
        """Read total-latency medians and convert them to costs relative to the zero-replay baseline."""
        required_fields = {
            "expected_replay_tokens",
            "recovery_latency_median_ms",
        }
        total_latency_points = []

        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            missing_fields = required_fields.difference(reader.fieldnames or ())
            if missing_fields:
                missing = ", ".join(sorted(missing))
                raise ValueError(f"profile file is missing fields: {missing}")

            for row in reader:
                try:
                    replay_tokens = int(row["expected_replay_tokens"])
                    total_latency_ms = float(row["recovery_latency_median_ms"])
                except (TypeError, ValueError) as error:
                    raise ValueError(f"invalid profile row: {row}") from error

                if replay_tokens < 0:
                    raise ValueError(
                        f"replay tokens in the profile must be non-negative: {replay_tokens}"
                    )
                total_latency_points.append((replay_tokens, total_latency_ms))

        if not total_latency_points:
            raise ValueError("profile file contains no usable data points")

        total_latency_points.sort(key=lambda point: point[0])
        for previous, current in zip(
            total_latency_points, total_latency_points[1:]
        ):
            if previous[0] == current[0]:
                raise ValueError(f"profile contains duplicate replay tokens: {current[0]}")

        baseline_points = [point for point in total_latency_points if point[0] == 0]
        if not baseline_points:
            raise ValueError("profile is missing the zero-replay baseline")
        baseline_ms = baseline_points[0][1]

        recovery_points = tuple(
            (replay_tokens, 0.0 if replay_tokens == 0 else total_ms - baseline_ms)
            for replay_tokens, total_ms in total_latency_points
        )
        HistoricalRecoveryCostModel._validate_monotonic(recovery_points)
        return recovery_points

    @staticmethod
    def _validate_monotonic(points: Tuple[Tuple[int, float], ...]) -> None:
        """Check that adjacent recovery costs are non-decreasing."""
        for previous, current in zip(points, points[1:]):
            if current[1] < previous[1]:
                raise ValueError(
                    "recovery-cost profile is not non-decreasing: "
                    f"{previous[0]} tokens is {previous[1]:.6f} ms, "
                    f"{current[0]} tokens is {current[1]:.6f} ms"
                )

    @staticmethod
    def _load_fit(path: Path) -> Tuple[float, float]:
        """Read the linear fit over raw runs already produced by WP2."""
        with path.open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)

        raw_run_fit = metrics.get("raw_run_ols")
        if not isinstance(raw_run_fit, dict):
            raise ValueError("fit metrics are missing raw_run_ols")

        try:
            intercept_ms = float(raw_run_fit["intercept_ms"])
            slope_ms_per_token = float(raw_run_fit["slope_ms_per_token"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("fit metrics are missing a valid intercept or slope") from error

        if slope_ms_per_token < 0:
            raise ValueError(
                f"fit slope must be non-negative: {slope_ms_per_token}"
            )
        return intercept_ms, slope_ms_per_token

    def estimate(
        self,
        replay_tokens: int,
        target_tokens: int | None = None,
    ) -> float:
        """Return the historical single-variable recovery latency; the target may be ignored for legacy audits."""
        if replay_tokens < 0:
            raise ValueError("replay_tokens must be non-negative")
        if replay_tokens == 0:
            return 0.0

        maximum_profile_tokens = self._tokens[-1]
        if replay_tokens > maximum_profile_tokens:
            fitted_total_ms = (
                self._fit_intercept_ms
                + self._fit_slope_ms_per_token * replay_tokens
            )
            fitted_baseline_ms = self._fit_intercept_ms
            recovery_cost_ms = fitted_total_ms - fitted_baseline_ms
            if recovery_cost_ms < 0:
                raise ValueError(
                    f"linear extrapolation produced a negative recovery cost: {recovery_cost_ms} ms"
                )
            return recovery_cost_ms

        upper_index = bisect_left(self._tokens, replay_tokens)
        upper_tokens, upper_cost_ms = self._profile_points[upper_index]
        if upper_tokens == replay_tokens:
            return upper_cost_ms

        lower_tokens, lower_cost_ms = self._profile_points[upper_index - 1]
        position = (replay_tokens - lower_tokens) / (upper_tokens - lower_tokens)
        return lower_cost_ms + position * (upper_cost_ms - lower_cost_ms)
