"""Detect a red/black taped parking bay without relying on fiducial IDs."""
from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np

from robot_dashboard.aruco_parking import diagonal_intersection


@dataclass(frozen=True)
class TapeParkingDetection:
    """Image-space parking-bay observation.

    Corners are ordered near-left, near-right, far-left, far-right.  "Near"
    means the edge with the larger average image y coordinate, which is the
    expected geometry for a forward/downward-facing robot camera.
    """

    corners: np.ndarray
    center: tuple[float, float]
    forward: tuple[float, float]
    area_ratio: float
    red_coverage: float
    black_support: float
    confidence: float


@dataclass(frozen=True)
class TapeParkingTrack:
    """One temporal tracker update."""

    state: str
    detection: TapeParkingDetection | None
    tracked_frames: int
    inlier_ratio: float


def red_tape_mask(frame: np.ndarray) -> np.ndarray:
    """Return a cleaned binary mask for both lobes of red in HSV space."""
    if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("frame must be a BGR image")
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    low_red = cv2.inRange(hsv, (0, 75, 45), (14, 255, 255))
    high_red = cv2.inRange(hsv, (165, 75, 45), (179, 255, 255))
    mask = cv2.bitwise_or(low_red, high_red)
    scale = max(3, int(round(min(frame.shape[:2]) * 0.009)))
    if scale % 2 == 0:
        scale += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (scale, scale))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)


def _order_corners(points: np.ndarray) -> np.ndarray:
    """Order a convex quadrilateral as near-left/right then far-left/right."""
    points = np.asarray(points, dtype=np.float32).reshape(4, 2)
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    cycle = points[np.argsort(angles)]
    edges = [
        (cycle[index], cycle[(index + 1) % 4])
        for index in range(4)
    ]
    near_a, near_b = max(edges, key=lambda edge: float(edge[0][1] + edge[1][1]))
    far_points = [point for point in cycle if not any(np.allclose(point, near) for near in (near_a, near_b))]
    near_left, near_right = sorted((near_a, near_b), key=lambda point: float(point[0]))
    far_left, far_right = sorted(far_points, key=lambda point: float(point[0]))
    return np.asarray((near_left, near_right, far_left, far_right), dtype=np.float32)


def _quad_from_contour(contour: np.ndarray) -> np.ndarray | None:
    hull = cv2.convexHull(contour)
    perimeter = cv2.arcLength(hull, True)
    if perimeter <= 0:
        return None
    for epsilon in (0.015, 0.02, 0.025, 0.03, 0.04, 0.05):
        polygon = cv2.approxPolyDP(hull, epsilon * perimeter, True)
        if len(polygon) == 4 and cv2.isContourConvex(polygon):
            return polygon.reshape(4, 2).astype(np.float32)
    return None


def _edge_support(mask: np.ndarray, corners: np.ndarray, thickness: int) -> float:
    band = np.zeros_like(mask)
    polygon = np.rint(corners).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(band, [polygon], True, 255, thickness, cv2.LINE_AA)
    selected = band > 0
    return float(np.count_nonzero((mask > 0) & selected) / max(1, np.count_nonzero(selected)))


def _detection_from_corners(
    frame: np.ndarray,
    corners: np.ndarray,
    confidence: float,
) -> TapeParkingDetection | None:
    """Rebuild image geometry for a quadrilateral propagated by tracking."""
    height, width = frame.shape[:2]
    ordered = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    polygon = ordered[[0, 1, 3, 2]]
    if not np.isfinite(ordered).all():
        return None
    if not cv2.isContourConvex(np.rint(polygon).astype(np.int32).reshape(-1, 1, 2)):
        return None
    area_ratio = abs(float(cv2.contourArea(polygon))) / float(height * width)
    if not 0.015 <= area_ratio <= 3.0:
        return None
    padding_x, padding_y = width * 0.75, height * 0.75
    if (
        np.any(ordered[:, 0] < -padding_x)
        or np.any(ordered[:, 0] > width + padding_x)
        or np.any(ordered[:, 1] < -padding_y)
        or np.any(ordered[:, 1] > height + padding_y)
    ):
        return None
    near_left, near_right, far_left, far_right = ordered
    try:
        center = diagonal_intersection(near_left, far_right, near_right, far_left)
    except ValueError:
        return None
    near_midpoint = (near_left + near_right) / 2.0
    far_midpoint = (far_left + far_right) / 2.0
    forward = far_midpoint - near_midpoint
    length = float(np.linalg.norm(forward))
    if length < max(8.0, min(height, width) * 0.025):
        return None
    forward /= length

    red = red_tape_mask(frame)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dark = cv2.inRange(hsv, (0, 0, 0), (179, 255, 75))
    dark[red > 0] = 0
    thickness = max(5, int(round(min(height, width) * 0.025)))
    return TapeParkingDetection(
        corners=ordered,
        center=(float(center[0]), float(center[1])),
        forward=(float(forward[0]), float(forward[1])),
        area_ratio=area_ratio,
        red_coverage=_edge_support(red, polygon, thickness),
        black_support=_edge_support(dark, polygon, thickness * 2),
        confidence=max(0.0, min(1.0, float(confidence))),
    )


def detect_tape_parking_bay(
    frame: np.ndarray,
    *,
    min_area_ratio: float = 0.08,
    min_confidence: float = 0.42,
    min_black_support: float = 0.10,
) -> TapeParkingDetection | None:
    """Detect the strongest red-outlined parking rectangle in ``frame``.

    The detector is intentionally observation-only.  It returns image-space
    geometry and never publishes motion commands.
    """
    if not 0.0 < min_area_ratio < 1.0:
        raise ValueError("min_area_ratio must be between zero and one")
    if not 0.0 <= min_confidence <= 1.0:
        raise ValueError("min_confidence must be between zero and one")
    if not 0.0 <= min_black_support <= 1.0:
        raise ValueError("min_black_support must be between zero and one")

    height, width = frame.shape[:2]
    image_area = float(width * height)
    red = red_tape_mask(frame)
    contours, _ = cv2.findContours(red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[TapeParkingDetection] = []
    thickness = max(5, int(round(min(height, width) * 0.025)))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dark = cv2.inRange(hsv, (0, 0, 0), (179, 255, 75))
    dark[red > 0] = 0

    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:12]:
        quad = _quad_from_contour(contour)
        if quad is None:
            continue
        area_ratio = abs(float(cv2.contourArea(quad))) / image_area
        if area_ratio < min_area_ratio:
            continue
        ordered = _order_corners(quad)
        near_left, near_right, far_left, far_right = ordered
        try:
            center = diagonal_intersection(near_left, far_right, near_right, far_left)
        except ValueError:
            continue
        near_midpoint = (near_left + near_right) / 2.0
        far_midpoint = (far_left + far_right) / 2.0
        forward = far_midpoint - near_midpoint
        length = float(np.linalg.norm(forward))
        if length < max(10.0, min(height, width) * 0.04):
            continue
        forward /= length

        red_coverage = _edge_support(red, ordered[[0, 1, 3, 2]], thickness)
        black_support = _edge_support(dark, ordered[[0, 1, 3, 2]], thickness * 2)
        if black_support < min_black_support:
            continue
        area_score = min(1.0, area_ratio / 0.30)
        confidence = max(0.0, min(1.0, 0.55 * red_coverage + 0.25 * black_support + 0.20 * area_score))
        candidates.append(
            TapeParkingDetection(
                corners=ordered,
                center=(float(center[0]), float(center[1])),
                forward=(float(forward[0]), float(forward[1])),
                area_ratio=area_ratio,
                red_coverage=red_coverage,
                black_support=black_support,
                confidence=confidence,
            )
        )

    if not candidates:
        return None
    result = max(candidates, key=lambda candidate: candidate.confidence)
    return result if result.confidence >= min_confidence and math.isfinite(result.confidence) else None


def draw_tape_parking_detection(frame: np.ndarray, detection: TapeParkingDetection) -> np.ndarray:
    """Draw the detected bay, center, and forward direction in place."""
    near_left, near_right, far_left, far_right = detection.corners
    polygon = np.rint((near_left, near_right, far_right, far_left)).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(frame, [polygon], True, (0, 255, 0), 4, cv2.LINE_AA)
    center = np.asarray(detection.center)
    forward = np.asarray(detection.forward)
    center_pixel = tuple(np.rint(center).astype(int))
    cv2.drawMarker(frame, center_pixel, (0, 255, 255), cv2.MARKER_TILTED_CROSS, 30, 4)
    cv2.arrowedLine(
        frame,
        center_pixel,
        tuple(np.rint(center + forward * 80).astype(int)),
        (0, 255, 255),
        4,
        cv2.LINE_AA,
        tipLength=0.25,
    )
    return frame


class TapeParkingTracker:
    """Track an acquired bay through short blur, clipping, and missed detections.

    A full red/black quadrilateral must be detected before tracking starts.
    Afterwards, KLT optical flow and a RANSAC homography propagate the bay for
    a bounded number of frames.  The bound prevents a stale image estimate from
    authorizing indefinite motion; longer gaps must be handled with a separately
    validated odometry-frame target or by stopping.
    """

    def __init__(
        self,
        *,
        detect_every: int = 1,
        max_track_frames: int = 18,
        min_track_points: int = 8,
        min_inlier_ratio: float = 0.55,
        min_area_ratio: float = 0.08,
        min_confidence: float = 0.42,
        min_black_support: float = 0.10,
    ) -> None:
        if detect_every < 1:
            raise ValueError("detect_every must be positive")
        if max_track_frames < 0:
            raise ValueError("max_track_frames must not be negative")
        if min_track_points < 4:
            raise ValueError("min_track_points must be at least four")
        if not 0.0 <= min_inlier_ratio <= 1.0:
            raise ValueError("min_inlier_ratio must be between zero and one")
        self.detect_every = detect_every
        self.max_track_frames = max_track_frames
        self.min_track_points = min_track_points
        self.min_inlier_ratio = min_inlier_ratio
        self.detector_options = {
            "min_area_ratio": min_area_ratio,
            "min_confidence": min_confidence,
            "min_black_support": min_black_support,
        }
        self.reset()

    def reset(self) -> None:
        self._frame_index = 0
        self._frames_since_detection = 0
        self._gray: np.ndarray | None = None
        self._detection: TapeParkingDetection | None = None

    def _features(self, gray: np.ndarray, detection: TapeParkingDetection) -> np.ndarray | None:
        mask = np.zeros_like(gray)
        polygon = np.rint(detection.corners[[0, 1, 3, 2]]).astype(np.int32).reshape(-1, 1, 2)
        thickness = max(24, int(round(min(gray.shape[:2]) * 0.10)))
        cv2.polylines(mask, [polygon], True, 255, thickness, cv2.LINE_AA)
        return cv2.goodFeaturesToTrack(
            gray,
            maxCorners=160,
            qualityLevel=0.008,
            minDistance=5,
            mask=mask,
            blockSize=5,
        )

    def _propagate(self, frame: np.ndarray, gray: np.ndarray) -> tuple[TapeParkingDetection, float] | None:
        if self._gray is None or self._detection is None:
            return None
        previous_points = self._features(self._gray, self._detection)
        if previous_points is None or len(previous_points) < self.min_track_points:
            return None
        current_points, status, _ = cv2.calcOpticalFlowPyrLK(
            self._gray,
            gray,
            previous_points,
            None,
            winSize=(31, 31),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        if current_points is None or status is None:
            return None
        backward_points, backward_status, _ = cv2.calcOpticalFlowPyrLK(
            gray,
            self._gray,
            current_points,
            None,
            winSize=(31, 31),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        if backward_points is None or backward_status is None:
            return None
        forward_backward_error = np.linalg.norm(previous_points - backward_points, axis=2).reshape(-1)
        valid = (status.reshape(-1) == 1) & (backward_status.reshape(-1) == 1) & (forward_backward_error <= 2.0)
        source = previous_points.reshape(-1, 2)[valid]
        destination = current_points.reshape(-1, 2)[valid]
        if len(source) < self.min_track_points:
            return None
        homography, inliers = cv2.findHomography(source, destination, cv2.RANSAC, 3.0)
        if homography is None or inliers is None:
            return None
        inlier_ratio = float(np.count_nonzero(inliers) / len(inliers))
        if inlier_ratio < self.min_inlier_ratio:
            return None
        transformed = cv2.perspectiveTransform(
            self._detection.corners.reshape(1, 4, 2),
            homography,
        ).reshape(4, 2)
        previous_area = max(self._detection.area_ratio, 1e-6)
        raw_area = abs(float(cv2.contourArea(transformed[[0, 1, 3, 2]]))) / float(frame.shape[0] * frame.shape[1])
        scale = raw_area / previous_area
        if not 0.55 <= scale <= 1.80:
            return None
        previous_center = np.asarray(self._detection.center)
        candidate_confidence = self._detection.confidence * 0.94 * (0.65 + 0.35 * inlier_ratio)
        detection = _detection_from_corners(frame, transformed, candidate_confidence)
        if detection is None:
            return None
        shift = float(np.linalg.norm(np.asarray(detection.center) - previous_center))
        if shift > math.hypot(frame.shape[1], frame.shape[0]) * 0.20:
            return None
        return detection, inlier_ratio

    def update(self, frame: np.ndarray) -> TapeParkingTrack:
        if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame must be a BGR image")
        self._frame_index += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        should_detect = self._detection is None or self._frame_index % self.detect_every == 0
        fresh = detect_tape_parking_bay(frame, **self.detector_options) if should_detect else None
        if fresh is not None:
            self._gray = gray
            self._detection = fresh
            self._frames_since_detection = 0
            return TapeParkingTrack("DETECTED", fresh, 0, 1.0)

        if self._detection is None or self._frames_since_detection >= self.max_track_frames:
            self._gray = gray
            self._detection = None
            self._frames_since_detection = 0
            return TapeParkingTrack("LOST", None, 0, 0.0)

        propagated = self._propagate(frame, gray)
        if propagated is None:
            self._gray = gray
            self._detection = None
            self._frames_since_detection = 0
            return TapeParkingTrack("LOST", None, 0, 0.0)
        detection, inlier_ratio = propagated
        self._gray = gray
        self._detection = detection
        self._frames_since_detection += 1
        return TapeParkingTrack("TRACKED", detection, self._frames_since_detection, inlier_ratio)
