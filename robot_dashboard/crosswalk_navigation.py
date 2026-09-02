"""Fail-closed crosswalk navigation policy driven by segmentation masks."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Optional

import numpy as np


class CrosswalkState(str, Enum):
    SEARCH = "search"
    APPROACH = "approach"
    ENTERING = "entering"
    ON_CROSSWALK = "on_crosswalk"
    EXITING = "exiting"
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
    polygon: Optional[np.ndarray] = None


@dataclass(frozen=True)
class NavigationCommand:
    state: CrosswalkState
    linear_x: float
    angular_z: float
    observation: CrosswalkObservation
    reason: str


@dataclass
class CrosswalkPolicyConfig:
    confidence: float = 0.50
    visible_frames: int = 3
    missing_frames: int = 8
    entry_bottom_ratio: float = 0.08
    entry_area_ratio: float = 0.025
    centered_width_ratio: float = 0.25
    centered_area_ratio: float = 0.07
    ema_alpha: float = 0.28
    approach_speed: float = 0.10
    crossing_speed: float = 0.12
    exit_speed: float = 0.06
    lateral_gain: float = 1.15
    heading_gain: float = 0.85
    max_angular_z: float = 0.35


def observation_from_mask(mask: np.ndarray, confidence: float) -> CrosswalkObservation:
    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2 or not binary.any():
        return CrosswalkObservation(False)
    h, w = binary.shape
    ys, xs = np.nonzero(binary)
    area_ratio = float(binary.mean())
    bottom_ratio = float(binary[int(h * 0.90):].mean())

    def span(y_fraction: float):
        target = int(h * y_fraction)
        for delta in range(0, max(5, int(h * 0.04))):
            for y in (target + delta, target - delta):
                if 0 <= y < h:
                    row = np.flatnonzero(binary[y])
                    if row.size >= 2:
                        return float((row[0] + row[-1]) / 2), float(row[-1] - row[0]), y
        return None

    near = span(0.85)
    far = span(0.55)
    if near is None:
        y = int(ys.max())
        row = np.flatnonzero(binary[y])
        near = (float((row[0] + row[-1]) / 2), float(row[-1] - row[0]), y)
    if far is None:
        y = int(np.quantile(ys, 0.20))
        row = np.flatnonzero(binary[y])
        if row.size >= 2:
            far = (float((row[0] + row[-1]) / 2), float(row[-1] - row[0]), y)
        else:
            far = near

    lateral = (near[0] - w / 2) / (w / 2)
    dy = max(1.0, near[2] - far[2])
    heading = math.atan2(near[0] - far[0], dy)
    return CrosswalkObservation(
        True,
        float(confidence),
        area_ratio,
        bottom_ratio,
        near[1] / w,
        float(lateral),
        float(heading),
    )


def combine_masks(masks: Iterable[np.ndarray]) -> Optional[np.ndarray]:
    items = [np.asarray(mask, dtype=bool) for mask in masks]
    if not items:
        return None
    shape = items[0].shape
    valid = [item for item in items if item.shape == shape]
    return np.logical_or.reduce(valid) if valid else None


class CrosswalkNavigator:
    def __init__(self, config: Optional[CrosswalkPolicyConfig] = None):
        self.config = config or CrosswalkPolicyConfig()
        self.state = CrosswalkState.SEARCH
        self.visible_streak = 0
        self.missing_streak = 0
        self._lateral = 0.0
        self._heading = 0.0

    def reset(self):
        self.__init__(self.config)

    def update(self, observation: CrosswalkObservation) -> NavigationCommand:
        cfg = self.config
        valid = observation.visible and observation.confidence >= cfg.confidence
        if valid:
            self.visible_streak += 1
            self.missing_streak = 0
            a = cfg.ema_alpha
            self._lateral = a * observation.lateral_error + (1 - a) * self._lateral
            self._heading = a * observation.heading_error + (1 - a) * self._heading
        else:
            self.visible_streak = 0
            self.missing_streak += 1

        if self.state in (CrosswalkState.SEARCH, CrosswalkState.EXITED):
            if self.visible_streak >= cfg.visible_frames:
                self.state = CrosswalkState.APPROACH
        elif self.state == CrosswalkState.APPROACH:
            if not valid and self.missing_streak >= cfg.missing_frames:
                self.state = CrosswalkState.SEARCH
            elif valid and observation.bottom_ratio >= cfg.entry_bottom_ratio and observation.area_ratio >= cfg.entry_area_ratio:
                self.state = CrosswalkState.ENTERING
        elif self.state == CrosswalkState.ENTERING:
            if valid and observation.near_width_ratio >= cfg.centered_width_ratio and observation.area_ratio >= cfg.centered_area_ratio:
                self.state = CrosswalkState.ON_CROSSWALK
            elif not valid and self.missing_streak >= cfg.missing_frames:
                self.state = CrosswalkState.SEARCH
        elif self.state == CrosswalkState.ON_CROSSWALK:
            if not valid and self.missing_streak >= 3:
                self.state = CrosswalkState.EXITING
        elif self.state == CrosswalkState.EXITING:
            if valid:
                self.state = CrosswalkState.ON_CROSSWALK
            elif self.missing_streak >= cfg.missing_frames:
                self.state = CrosswalkState.EXITED

        if self.state in (CrosswalkState.SEARCH, CrosswalkState.EXITED) or not valid and self.state != CrosswalkState.EXITING:
            return NavigationCommand(self.state, 0.0, 0.0, observation, "fail-closed: no stable crosswalk")

        speed = {
            CrosswalkState.APPROACH: cfg.approach_speed,
            CrosswalkState.ENTERING: cfg.approach_speed * 0.8,
            CrosswalkState.ON_CROSSWALK: cfg.crossing_speed,
            CrosswalkState.EXITING: cfg.exit_speed,
        }.get(self.state, 0.0)
        yaw = cfg.lateral_gain * self._lateral + cfg.heading_gain * self._heading
        yaw = max(-cfg.max_angular_z, min(cfg.max_angular_z, yaw))
        # Never extend motion open-loop after perception disappears.
        if not valid:
            speed = 0.0
            yaw = 0.0
        elif abs(yaw) > 0.25:
            speed *= 0.5
        return NavigationCommand(self.state, speed, -yaw, observation, "tracking crosswalk centerline")
