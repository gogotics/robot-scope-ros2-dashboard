import unittest

import numpy as np

from robot_dashboard.crosswalk_autonomy import (
    CrosswalkNavigator,
    CrosswalkObservation,
    CrosswalkState,
    observation_from_mask,
    route_perception_snapshot,
    select_crosswalk_mask,
)
from robot_dashboard.route_planner.perception import normalize_perception_snapshot


def trapezoid(*, offset: int = 0) -> np.ndarray:
    mask = np.zeros((120, 240), dtype=bool)
    for y in range(30, 120):
        center = 120 + offset
        half_width = 20 + (y - 30) // 2
        mask[y, center - half_width : center + half_width] = True
    return mask


class CrosswalkGeometryTests(unittest.TestCase):
    def test_centered_trapezoid_has_two_visible_boundaries(self):
        observation = observation_from_mask(trapezoid(), 0.9)

        self.assertTrue(observation.visible)
        self.assertAlmostEqual(observation.lateral_error, 0.0, places=2)
        self.assertGreater(observation.near_width_ratio, 0.35)
        self.assertGreater(observation.left_boundary_ratio, 0.15)
        self.assertGreater(observation.right_boundary_ratio, 0.15)

    def test_selector_keeps_one_near_instance_instead_of_merging(self):
        far = np.zeros((120, 240), dtype=bool)
        far[20:60, 20:100] = True
        near = np.zeros((120, 240), dtype=bool)
        near[50:118, 130:220] = True

        selected, confidence, index = select_crosswalk_mask(
            [far, near], [0.99, 0.75], min_confidence=0.5
        )

        self.assertEqual(index, 1)
        self.assertEqual(confidence, 0.75)
        self.assertTrue(np.array_equal(selected, near))

    def test_metric_projection_matches_route_planner_contract(self):
        observation = observation_from_mask(trapezoid(offset=8), 0.91)
        snapshot = route_perception_snapshot(
            observation,
            sequence=7,
            observed_at_ns=5_000_000_000,
            crosswalk_width_m=3.0,
        )

        normalized = normalize_perception_snapshot(snapshot, now_ns=5_100_000_000)

        self.assertTrue(normalized["fresh"])
        self.assertEqual(normalized["crosswalks"][0]["crosswalk_id"], "CAMERA_CROSSWALK")
        self.assertGreater(normalized["crosswalks"][0]["left_boundary_distance_m"], 0.0)
        self.assertGreater(normalized["crosswalks"][0]["right_boundary_distance_m"], 0.0)

    def test_nonfinite_projection_fails_closed_but_remains_valid_json_shape(self):
        observation = CrosswalkObservation(
            True, 0.9, 0.1, 0.1, 0.4, float("nan"), 0.0, 0.2, 0.2
        )
        snapshot = route_perception_snapshot(
            observation,
            sequence=8,
            observed_at_ns=5_000_000_000,
            crosswalk_width_m=3.0,
        )

        normalized = normalize_perception_snapshot(snapshot, now_ns=5_100_000_000)

        self.assertEqual(normalized["state"], "UNKNOWN")
        self.assertFalse(normalized["crosswalks"][0]["visible"])
        self.assertEqual(normalized["crosswalks"][0]["lateral_offset_m"], 0.0)


class CrosswalkNavigatorTests(unittest.TestCase):
    def _enter_crosswalk(self, navigator: CrosswalkNavigator) -> CrosswalkObservation:
        far = CrosswalkObservation(
            True, 0.9, 0.04, 0.01, 0.20, 0.0, 0.0, 0.15, 0.15
        )
        entry = CrosswalkObservation(
            True, 0.9, 0.10, 0.10, 0.40, 0.0, 0.0, 0.20, 0.20
        )
        for _ in range(3):
            navigator.update(far, captured_at_s=1.0, now_s=1.0)
        navigator.update(entry, captured_at_s=1.0, now_s=1.0)
        command = navigator.update(entry, captured_at_s=1.0, now_s=1.0)
        self.assertEqual(command.state, CrosswalkState.ON_CROSSWALK)
        return entry

    def test_shadow_mode_reports_recommendation_but_emits_zero(self):
        navigator = CrosswalkNavigator()
        entry = self._enter_crosswalk(navigator)

        command = navigator.update(entry, captured_at_s=1.0, now_s=1.0)

        self.assertGreater(command.recommended_linear_x, 0.0)
        self.assertEqual(command.linear_x, 0.0)
        self.assertEqual(command.angular_z, 0.0)
        self.assertFalse(command.motion_authorized)

    def test_stale_frame_and_missing_boundary_stop_immediately(self):
        navigator = CrosswalkNavigator()
        self._enter_crosswalk(navigator)
        centered = CrosswalkObservation(
            True, 0.9, 0.10, 0.10, 0.40, 0.0, 0.0, 0.20, 0.20
        )

        stale = navigator.update(
            centered,
            captured_at_s=1.0,
            now_s=1.31,
            motion_authorized=True,
        )

        self.assertEqual(stale.state, CrosswalkState.HOLD)
        self.assertEqual(stale.linear_x, 0.0)
        self.assertIn("stale", stale.reason)

    def test_mask_loss_is_not_mistaken_for_completed_crossing(self):
        navigator = CrosswalkNavigator()
        self._enter_crosswalk(navigator)

        for _ in range(20):
            command = navigator.update(
                CrosswalkObservation(False),
                captured_at_s=2.0,
                now_s=2.0,
                motion_authorized=True,
            )

        self.assertEqual(command.state, CrosswalkState.HOLD)
        self.assertEqual(command.linear_x, 0.0)
        exited = navigator.update(
            CrosswalkObservation(False),
            captured_at_s=2.0,
            now_s=2.0,
            exit_confirmed=True,
            motion_authorized=True,
        )
        self.assertEqual(exited.state, CrosswalkState.EXITED)
        self.assertEqual(exited.linear_x, 0.0)

    def test_boundary_guard_blocks_authorized_motion(self):
        navigator = CrosswalkNavigator()
        unsafe = CrosswalkObservation(
            True, 0.95, 0.1, 0.1, 0.35, 0.4, 0.0, -0.02, 0.30
        )

        for _ in range(5):
            command = navigator.update(
                unsafe,
                captured_at_s=1.0,
                now_s=1.0,
                motion_authorized=True,
            )

        self.assertEqual(command.linear_x, 0.0)
        self.assertEqual(command.angular_z, 0.0)
        self.assertIn("corridor", command.reason)

    def test_authorized_motion_requires_capture_timestamp(self):
        navigator = CrosswalkNavigator()
        centered = CrosswalkObservation(
            True, 0.9, 0.1, 0.1, 0.4, 0.0, 0.0, 0.2, 0.2
        )

        for _ in range(5):
            command = navigator.update(centered, motion_authorized=True)

        self.assertEqual(command.linear_x, 0.0)
        self.assertIn("timestamp", command.reason)


if __name__ == "__main__":
    unittest.main()
