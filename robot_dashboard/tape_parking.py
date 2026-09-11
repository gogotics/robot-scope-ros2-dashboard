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
