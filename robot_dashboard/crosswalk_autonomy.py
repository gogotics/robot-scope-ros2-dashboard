"""Fail-closed crosswalk geometry and shadow-mode motion recommendations.

The geometry and state-machine design is derived from the MIT-licensed
``gogotics/go2-crosswalk-autonomy`` project.  This module deliberately has no
ROS publisher, control lease, or robot command dependency.
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Optional, Sequence

import numpy as np


class CrosswalkState(str, Enum):
    SEARCH = "search"
    APPROACH = "approach"
    ENTERING = "entering"
    ON_CROSSWALK = "on_crosswalk"
    HOLD = "hold"
    EXITED = "exited"


@dataclass(frozen=True)
class CrosswalkObservation:
    visible: bool
    confidence: float = 0.0
    area_ratio: float = 0.0
    bottom_ratio: float = 0.0
    near_width_ratio: float = 0.0
    lateral_error: float = 0.0
    heading_error: float = 0.0
    left_boundary_ratio: float = 0.0
    right_boundary_ratio: float = 0.0
    near_center_x_ratio: float = 0.5
    far_center_x_ratio: float = 0.5


@dataclass(frozen=True)
class NavigationCommand:
    state: CrosswalkState
    linear_x: float
    angular_z: float
    recommended_linear_x: float
    recommended_angular_z: float
    observation: CrosswalkObservation
    motion_authorized: bool
    reason: str


@dataclass
class CrosswalkPolicyConfig:
    confidence: float = 0.50
    visible_frames: int = 3
    recovery_frames: int = 3
    missing_frames: int = 8
    entry_bottom_ratio: float = 0.08
    entry_area_ratio: float = 0.025
    centered_width_ratio: float = 0.25
    centered_area_ratio: float = 0.07
    min_boundary_ratio: float = 0.06
    max_lateral_error: float = 0.70
    max_heading_error: float = 0.70
    stale_after_s: float = 0.30
    ema_alpha: float = 0.28
    approach_speed: float = 0.10
    crossing_speed: float = 0.12
    lateral_gain: float = 1.15
    heading_gain: float = 0.85
    max_angular_z: float = 0.35
    slow_turn_threshold: float = 0.25


def _row_geometry(binary: np.ndarray) -> list[tuple[int, float, float, float]]:
    rows: list[tuple[int, float, float, float]] = []
    for y in np.flatnonzero(binary.any(axis=1)):
        xs = np.flatnonzero(binary[y])
        if xs.size >= 2:
            left = float(xs[0])
            right = float(xs[-1])
            rows.append((int(y), (left + right) / 2.0, right - left, left))
    return rows


def _median_span(
    rows: Sequence[tuple[int, float, float, float]], start: int, stop: int
) -> tuple[float, float, float, float]:
    selected = rows[start:stop]
    if not selected:
        selected = rows[-1:]
    values = [
        float(np.median([row[index] for row in selected])) for index in range(4)
    ]
    return values[0], values[1], values[2], values[3]


def observation_from_mask(mask: np.ndarray, confidence: float) -> CrosswalkObservation:
    """Measure a crosswalk centerline and boundary clearance in image space."""

    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2 or min(binary.shape) < 2 or not binary.any():
        return CrosswalkObservation(False)
    if not math.isfinite(float(confidence)) or not 0.0 <= float(confidence) <= 1.0:
        return CrosswalkObservation(False)

    height, width = binary.shape
    rows = _row_geometry(binary)
    if len(rows) < 2:
        return CrosswalkObservation(False)

    count = len(rows)
    near_start = max(0, int(count * 0.75))
    far_start = max(0, int(count * 0.20))
    far_stop = max(far_start + 1, int(count * 0.45))
    near_y, near_center, near_width, near_left = _median_span(
        rows, near_start, count
    )
    far_y, far_center, _far_width, _far_left = _median_span(
        rows, far_start, far_stop
    )
    near_right = near_left + near_width
    image_center = width / 2.0
    lateral = (near_center - image_center) / image_center
    heading = math.atan2(near_center - far_center, max(1.0, near_y - far_y))

    return CrosswalkObservation(
        visible=True,
        confidence=float(confidence),
        area_ratio=float(binary.mean()),
        bottom_ratio=float(binary[int(height * 0.90) :].mean()),
        near_width_ratio=near_width / width,
        lateral_error=float(lateral),
        heading_error=float(heading),
        left_boundary_ratio=(image_center - near_left) / width,
        right_boundary_ratio=(near_right - image_center) / width,
        near_center_x_ratio=near_center / width,
        far_center_x_ratio=far_center / width,
    )


def select_crosswalk_mask(
    masks: Iterable[np.ndarray],
    confidences: Iterable[float],
    *,
    min_confidence: float = 0.0,
) -> tuple[Optional[np.ndarray], float, Optional[int]]:
    """Select one nearby crosswalk instance instead of merging every mask."""

    mask_items = [np.asarray(mask, dtype=bool) for mask in masks]
    confidence_items = [float(value) for value in confidences]
    if len(mask_items) != len(confidence_items):
        raise ValueError("mask and confidence counts differ")
    if not math.isfinite(float(min_confidence)) or not 0.0 <= min_confidence <= 1.0:
        raise ValueError("minimum confidence is outside [0, 1]")

    candidates: list[tuple[tuple[float, ...], int]] = []
    expected_shape: Optional[tuple[int, int]] = None
    for index, (mask, confidence) in enumerate(zip(mask_items, confidence_items)):
        if (
            mask.ndim != 2
            or not mask.any()
            or not math.isfinite(confidence)
            or not min_confidence <= confidence <= 1.0
        ):
            continue
        if expected_shape is None:
            expected_shape = mask.shape
        if mask.shape != expected_shape:
            continue
        height, width = mask.shape
        ys, xs = np.nonzero(mask)
        bottom = float(ys.max()) / max(1, height - 1)
        lower = ys >= np.quantile(ys, 0.75)
        center_distance = abs(float(np.median(xs[lower])) - width / 2.0) / width
        area = float(mask.mean())
        # Proximity dominates, then centrality, confidence, and visible area.
        score = (bottom, -center_distance, confidence, area, -float(index))
        candidates.append((score, index))

    if not candidates:
        return None, 0.0, None
    _score, selected_index = max(candidates)
    return (
        mask_items[selected_index],
        confidence_items[selected_index],
        selected_index,
    )


class CrosswalkNavigator:
    """Recommend bounded velocity while keeping emitted motion opt-in."""

    def __init__(self, config: Optional[CrosswalkPolicyConfig] = None):
        self.config = config or CrosswalkPolicyConfig()
        self.state = CrosswalkState.SEARCH
        self.visible_streak = 0
        self.missing_streak = 0
        self._resume_state = CrosswalkState.ENTERING
        self._lateral = 0.0
        self._heading = 0.0

    def reset(self) -> None:
        self.__init__(self.config)

    def _validity_reason(
        self,
        observation: CrosswalkObservation,
        *,
        captured_at_s: Optional[float],
        now_s: float,
    ) -> str:
        cfg = self.config
        if not observation.visible:
            return "crosswalk not visible"
        if observation.confidence < cfg.confidence:
            return "crosswalk confidence below threshold"
        values = (
            observation.area_ratio,
            observation.bottom_ratio,
            observation.near_width_ratio,
            observation.lateral_error,
            observation.heading_error,
            observation.left_boundary_ratio,
            observation.right_boundary_ratio,
        )
        if any(not math.isfinite(value) for value in values):
            return "crosswalk geometry is non-finite"
        if captured_at_s is not None and (
            captured_at_s > now_s or now_s - captured_at_s > cfg.stale_after_s
        ):
            return "camera observation is stale"
        if abs(observation.lateral_error) > cfg.max_lateral_error:
            return "crosswalk center is outside steering range"
        if abs(observation.heading_error) > cfg.max_heading_error:
            return "crosswalk heading is outside steering range"
        if (
            observation.left_boundary_ratio < cfg.min_boundary_ratio
            or observation.right_boundary_ratio < cfg.min_boundary_ratio
        ):
            return "safe crosswalk corridor is not visible"
        return ""

    def update(
        self,
        observation: CrosswalkObservation,
        *,
        captured_at_s: Optional[float] = None,
        now_s: Optional[float] = None,
        motion_authorized: bool = False,
        exit_confirmed: bool = False,
    ) -> NavigationCommand:
        if not isinstance(motion_authorized, bool):
            raise ValueError("motion authorization must be boolean")
        try:
            current = time.monotonic() if now_s is None else float(now_s)
            captured = None if captured_at_s is None else float(captured_at_s)
        except (TypeError, ValueError):
            current = math.nan
            captured = math.nan
        invalid_reason = self._validity_reason(
            observation, captured_at_s=captured, now_s=current
        )
        if not math.isfinite(current):
            invalid_reason = "evaluation timestamp is invalid"
        elif captured is not None and not math.isfinite(captured):
            invalid_reason = "camera timestamp is invalid"
        elif motion_authorized and captured is None:
            invalid_reason = "camera timestamp is unavailable"
        valid = not invalid_reason
        cfg = self.config

        if valid:
            self.visible_streak += 1
            self.missing_streak = 0
            alpha = cfg.ema_alpha
            self._lateral = alpha * observation.lateral_error + (1 - alpha) * self._lateral
            self._heading = alpha * observation.heading_error + (1 - alpha) * self._heading
        else:
            self.visible_streak = 0
            self.missing_streak += 1

        if exit_confirmed and self.state in {
            CrosswalkState.ENTERING,
            CrosswalkState.ON_CROSSWALK,
            CrosswalkState.HOLD,
        }:
            self.state = CrosswalkState.EXITED
        elif self.state == CrosswalkState.SEARCH:
            if self.visible_streak >= cfg.visible_frames:
                self.state = CrosswalkState.APPROACH
        elif self.state == CrosswalkState.APPROACH:
            if not valid and self.missing_streak >= cfg.missing_frames:
                self.state = CrosswalkState.SEARCH
            elif (
                valid
                and observation.bottom_ratio >= cfg.entry_bottom_ratio
                and observation.area_ratio >= cfg.entry_area_ratio
            ):
                self.state = CrosswalkState.ENTERING
        elif self.state == CrosswalkState.ENTERING:
            if not valid:
                self._resume_state = CrosswalkState.ENTERING
                self.state = CrosswalkState.HOLD
            elif (
                observation.near_width_ratio >= cfg.centered_width_ratio
                and observation.area_ratio >= cfg.centered_area_ratio
            ):
                self.state = CrosswalkState.ON_CROSSWALK
        elif self.state == CrosswalkState.ON_CROSSWALK:
            if not valid:
                self._resume_state = CrosswalkState.ON_CROSSWALK
                self.state = CrosswalkState.HOLD
        elif self.state == CrosswalkState.HOLD:
            if valid and self.visible_streak >= cfg.recovery_frames:
                self.state = self._resume_state

        recommended_x = 0.0
        recommended_z = 0.0
        reason = invalid_reason or "waiting for stable crosswalk"
        if valid and self.state in {
            CrosswalkState.APPROACH,
            CrosswalkState.ENTERING,
            CrosswalkState.ON_CROSSWALK,
        }:
            recommended_x = (
                cfg.crossing_speed
                if self.state == CrosswalkState.ON_CROSSWALK
                else cfg.approach_speed
            )
            yaw = cfg.lateral_gain * self._lateral + cfg.heading_gain * self._heading
            recommended_z = -max(-cfg.max_angular_z, min(cfg.max_angular_z, yaw))
            if abs(recommended_z) > cfg.slow_turn_threshold:
                recommended_x *= 0.5
            reason = "tracking crosswalk centerline"
        elif self.state == CrosswalkState.EXITED:
            reason = "crosswalk exit explicitly confirmed"

        emitted_x = recommended_x if motion_authorized else 0.0
        emitted_z = recommended_z if motion_authorized else 0.0
        if recommended_x and not motion_authorized:
            reason = "shadow mode: motion recommendation only"
        return NavigationCommand(
            state=self.state,
            linear_x=emitted_x,
            angular_z=emitted_z,
            recommended_linear_x=recommended_x,
            recommended_angular_z=recommended_z,
            observation=observation,
            motion_authorized=bool(motion_authorized),
            reason=reason,
        )


_CROSSWALK_ID = re.compile(r"^[A-Z][A-Z0-9_]{0,63}$")


def route_perception_snapshot(
    observation: CrosswalkObservation,
    *,
    sequence: int,
    observed_at_ns: int,
    crosswalk_width_m: float,
    crosswalk_id: str = "CAMERA_CROSSWALK",
    minimum_confidence: float = 0.50,
) -> dict[str, object]:
    """Project image geometry into the Route Planner's fixed v1 envelope.

    The metric scale assumes the detected near span represents the configured
    physical crosswalk width.  It is intended for offline calibration and must
    be replaced by a camera-to-ground transform before live control.
    """

    width_m = float(crosswalk_width_m)
    if not math.isfinite(width_m) or not 0.2 <= width_m <= 30.0:
        raise ValueError("crosswalk width must be between 0.2 and 30 metres")
    if not _CROSSWALK_ID.fullmatch(crosswalk_id):
        raise ValueError("crosswalk ID is invalid")
    threshold = float(minimum_confidence)
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("minimum confidence is outside [0, 1]")
    if (
        isinstance(sequence, bool)
        or not isinstance(sequence, int)
        or not 0 <= sequence <= (1 << 64) - 1
    ):
        raise ValueError("sequence is invalid")
    if (
        isinstance(observed_at_ns, bool)
        or not isinstance(observed_at_ns, int)
        or not 0 <= observed_at_ns <= (1 << 64) - 1
    ):
        raise ValueError("observation timestamp is invalid")

    observation_values = (
        observation.confidence,
        observation.near_width_ratio,
        observation.lateral_error,
        observation.heading_error,
        observation.left_boundary_ratio,
        observation.right_boundary_ratio,
    )
    finite = all(math.isfinite(value) for value in observation_values)

    scale = (
        width_m / observation.near_width_ratio
        if finite and observation.visible and observation.near_width_ratio > 0.0
        else 0.0
    )
    left_distance = (
        max(0.0, observation.left_boundary_ratio * scale) if finite else 0.0
    )
    right_distance = (
        max(0.0, observation.right_boundary_ratio * scale) if finite else 0.0
    )
    visible = bool(
        observation.visible
        and finite
        and observation.confidence >= threshold
        and observation.left_boundary_ratio >= 0.0
        and observation.right_boundary_ratio >= 0.0
    )
    confidence = (
        min(1.0, max(0.0, float(observation.confidence))) if finite else 0.0
    )
    lateral_offset_m = (
        -observation.lateral_error * width_m / 2.0 if finite else 0.0
    )
    heading_error_rad = observation.heading_error if finite else 0.0
    return {
        "schema_version": 1,
        "source": "crosswalk-segmentation-shadow",
        "frame_id": "base_link",
        "observed_at_ns": observed_at_ns,
        "sequence": sequence,
        "state": "READY" if visible else "UNKNOWN",
        "confidence": confidence,
        "traffic": [],
        "crosswalks": [
            {
                "crosswalk_id": crosswalk_id,
                "visible": visible,
                "lateral_offset_m": lateral_offset_m,
                "heading_error_rad": heading_error_rad,
                "left_boundary_distance_m": left_distance,
                "right_boundary_distance_m": right_distance,
                "confidence": confidence,
            }
        ],
        "people": [],
        "aruco": [],
        "underpass_blocked": None,
    }


__all__ = [
    "CrosswalkNavigator",
    "CrosswalkObservation",
    "CrosswalkPolicyConfig",
    "CrosswalkState",
    "NavigationCommand",
    "observation_from_mask",
    "route_perception_snapshot",
    "select_crosswalk_mask",
]
