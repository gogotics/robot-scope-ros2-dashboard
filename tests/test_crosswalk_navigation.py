import unittest

import numpy as np

from robot_dashboard.crosswalk_navigation import (
    CrosswalkNavigator,
    CrosswalkObservation,
    CrosswalkState,
    combine_masks,
    observation_from_mask,
)


class CrosswalkNavigationTests(unittest.TestCase):
    def test_geometry_center_and_direction(self):
        mask = np.zeros((100, 200), dtype=bool)
        for y in range(40, 100):
            center = 80 + (y - 40) // 4
            half = 15 + (y - 40) // 3
            mask[y, center-half:center+half] = True
        obs = observation_from_mask(mask, 0.9)
        self.assertTrue(obs.visible)
        self.assertGreater(obs.heading_error, 0)
        self.assertGreater(obs.near_width_ratio, 0.2)

    def test_state_machine_and_fail_closed_exit(self):
        nav = CrosswalkNavigator()
        far = CrosswalkObservation(True, 0.9, 0.04, 0.01, 0.15, 0.0, 0.0)
        entry = CrosswalkObservation(True, 0.9, 0.10, 0.10, 0.35, 0.0, 0.0)
        for _ in range(3): command = nav.update(far)
        self.assertEqual(command.state, CrosswalkState.APPROACH)
        command = nav.update(entry)
        self.assertEqual(command.state, CrosswalkState.ENTERING)
        command = nav.update(entry)
        self.assertEqual(command.state, CrosswalkState.ON_CROSSWALK)
        missing = CrosswalkObservation(False)
        command = nav.update(missing)
        self.assertEqual(command.linear_x, 0.0)
        self.assertEqual(command.angular_z, 0.0)
        for _ in range(7): command = nav.update(missing)
        self.assertEqual(command.state, CrosswalkState.EXITED)
        self.assertEqual(command.linear_x, 0.0)
        self.assertEqual(command.angular_z, 0.0)

    def test_low_confidence_never_moves(self):
        nav = CrosswalkNavigator()
        low = CrosswalkObservation(True, 0.49, 0.2, 0.2, 0.5, 0.2, 0.1)
        for _ in range(20): command = nav.update(low)
        self.assertEqual(command.linear_x, 0.0)

    def test_combine_masks(self):
        a = np.zeros((4, 4), dtype=bool); a[0, 0] = True
        b = np.zeros((4, 4), dtype=bool); b[3, 3] = True
        merged = combine_masks([a, b])
        self.assertEqual(int(merged.sum()), 2)


if __name__ == "__main__":
    unittest.main()
