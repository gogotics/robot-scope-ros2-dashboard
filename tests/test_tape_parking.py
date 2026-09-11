import unittest

import cv2
import numpy as np

from robot_dashboard.tape_parking import TapeParkingTracker, detect_tape_parking_bay


class TapeParkingTests(unittest.TestCase):
    @staticmethod
    def parking_image():
        image = np.full((600, 900, 3), 190, np.uint8)
        for x in range(0, 900, 70):
            cv2.line(image, (x, 0), (x, 599), (140, 140, 140), 2)
        for y in range(0, 600, 70):
            cv2.line(image, (0, y), (899, y), (140, 140, 140), 2)
        corners = np.array([[110, 500], [800, 480], [300, 100], [650, 120]], np.int32)
        polygon = corners[[0, 1, 3, 2]].reshape(-1, 1, 2)
        cv2.polylines(image, [polygon], True, (15, 15, 15), 42, cv2.LINE_AA)
        cv2.polylines(image, [polygon], True, (0, 0, 210), 18, cv2.LINE_AA)
        return image

    def test_detects_perspective_red_black_bay(self):
        image = self.parking_image()

        detection = detect_tape_parking_bay(image)

        self.assertIsNotNone(detection)
        self.assertGreater(detection.confidence, 0.42)
        # Perspective diagonals intersect closer to the narrower far edge.
        self.assertAlmostEqual(detection.center[0], 485, delta=20)
        self.assertAlmostEqual(detection.center[1], 231, delta=20)
        self.assertLess(detection.forward[1], 0)

    def test_tracks_acquired_bay_when_fresh_detection_is_skipped(self):
        first = self.parking_image()
        transform = np.float32([[1, 0, 9], [0, 1, -6]])
        second = cv2.warpAffine(first, transform, (first.shape[1], first.shape[0]))
        tracker = TapeParkingTracker(detect_every=100, max_track_frames=5)

        acquired = tracker.update(first)
        tracked = tracker.update(second)

        self.assertEqual(acquired.state, "DETECTED")
        self.assertEqual(tracked.state, "TRACKED")
        self.assertIsNotNone(tracked.detection)
        self.assertGreaterEqual(tracked.inlier_ratio, 0.55)
        self.assertAlmostEqual(tracked.detection.center[0] - acquired.detection.center[0], 9, delta=3)
        self.assertAlmostEqual(tracked.detection.center[1] - acquired.detection.center[1], -6, delta=3)

    def test_tracking_expires_instead_of_reusing_stale_target_forever(self):
        frame = self.parking_image()
        tracker = TapeParkingTracker(detect_every=100, max_track_frames=2)
        self.assertEqual(tracker.update(frame).state, "DETECTED")
        self.assertEqual(tracker.update(frame).state, "TRACKED")
        self.assertEqual(tracker.update(frame).state, "TRACKED")
        self.assertEqual(tracker.update(frame).state, "LOST")

    def test_rejects_red_object_without_parking_area(self):
        image = np.full((480, 640, 3), 180, np.uint8)
        cv2.rectangle(image, (30, 30), (120, 90), (0, 0, 220), -1)
        self.assertIsNone(detect_tape_parking_bay(image))

    def test_rejects_large_red_quad_without_black_boundary(self):
        image = np.full((480, 640, 3), 180, np.uint8)
        polygon = np.array([[60, 400], [580, 390], [470, 80], [170, 70]], np.int32)
        cv2.polylines(image, [polygon.reshape(-1, 1, 2)], True, (0, 0, 220), 24)
        self.assertIsNone(detect_tape_parking_bay(image))

    def test_rejects_invalid_thresholds(self):
        image = np.zeros((100, 100, 3), np.uint8)
        with self.assertRaises(ValueError):
            detect_tape_parking_bay(image, min_area_ratio=0)
        with self.assertRaises(ValueError):
            detect_tape_parking_bay(image, min_confidence=2)
        with self.assertRaises(ValueError):
            detect_tape_parking_bay(image, min_black_support=-1)


if __name__ == "__main__":
    unittest.main()
