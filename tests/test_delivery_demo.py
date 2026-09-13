import unittest

from scripts import run_delivery_demo as delivery


MAP = {
    "id": "a" * 24,
    "revision": "b" * 64,
    "name": "Llllc",
    "format": "map-server-pgm",
}
POINTS = tuple(
    {
        "id": str(index) * 24,
        "name": name,
        "type": "POI",
        "pose": {"x": float(index), "y": float(index + 1), "yaw": 0.0},
    }
    for index, name in enumerate(("출발지", "픽업지", "배송지", "복귀지"), start=1)
)
ANNOTATIONS = {
    "annotation_revision": "c" * 64,
    "points": list(POINTS),
}


class FakeApi:
    def __init__(self, *, wrong_map=False, fail_goal_number=None):
        self.pipeline_state = "running" if wrong_map else "idle"
        self.map = {**MAP, "id": "f" * 24} if wrong_map else None
        self.localized = False
        self.goal = {"state": "idle", "goal_id": None}
        self.goal_number = 0
        self.fail_goal_number = fail_goal_number
        self.posts = []

    def get(self, path):
        if path == "/api/v1/navigation/parameters":
            return {"revision": "d" * 64}
        if path != "/api/v1/navigation":
            raise AssertionError(path)
        if self.goal.get("state") == "active":
            self.goal = {
                **self.goal,
                "state": (
                    "failed"
                    if self.goal_number == self.fail_goal_number
                    else "succeeded"
                ),
                "error": "blocked" if self.goal_number == self.fail_goal_number else None,
            }
        return {
            "pipeline": {"state": self.pipeline_state},
            "map": self.map,
            "localization": {
                "state": "localized" if self.localized else "uninitialized"
            },
            "goal": dict(self.goal),
            "safety": {
                "can_set_initial_pose": self.pipeline_state == "running",
                "can_send_goal": self.pipeline_state == "running" and self.localized,
                "blockers": [],
            },
        }

    def post(self, path, payload):
        self.posts.append((path, dict(payload)))
        if path == "/api/v1/navigation/start":
            self.pipeline_state = "running"
            self.map = dict(MAP)
            return {"accepted": True}
        if path == "/api/v1/navigation/initial-pose":
            self.localized = True
            return {"accepted": True}
        if path == "/api/v1/navigation/goal/annotation":
            self.goal_number += 1
            self.goal = {
                "state": "active",
                "goal_id": f"goal-{self.goal_number:016d}",
                "error": None,
            }
            return {"navigation": {"goal": dict(self.goal)}}
        if path == "/api/v1/navigation/cancel":
            self.goal = {**self.goal, "state": "canceled"}
            return {"navigation": {"goal": dict(self.goal)}}
        raise AssertionError(path)


class DeliveryResolutionTests(unittest.TestCase):
    def test_map_can_be_selected_by_case_insensitive_name_or_exact_id(self):
        catalog = {"maps": [MAP]}
        self.assertIs(delivery.resolve_map(catalog, "llllc"), MAP)
        self.assertIs(delivery.resolve_map(catalog, "a" * 24), MAP)
        with self.assertRaisesRegex(delivery.DeliveryDemoError, "not found"):
            delivery.resolve_map(catalog, "missing")

    def test_four_named_points_are_resolved_in_requested_order(self):
        resolved = delivery.resolve_points(
            ANNOTATIONS,
            ("출발지", "픽업지", "배송지", "복귀지"),
        )
        self.assertEqual([item["id"] for item in resolved], [item["id"] for item in POINTS])

        duplicate = {**ANNOTATIONS, "points": [*POINTS, dict(POINTS[1])]}
        with self.assertRaisesRegex(delivery.DeliveryDemoError, "ambiguous"):
            delivery.resolve_points(
                duplicate,
                ("출발지", "픽업지", "배송지", "복귀지"),
            )


class DeliveryMissionTests(unittest.TestCase):
    def mission(self, api, *, sleeps=None):
        return delivery.DeliveryMission(
            api,
            map_record=MAP,
            annotations=ANNOTATIONS,
            points=POINTS,
            delivery_dwell_s=5.0,
            poll_interval_s=0.05,
            timeouts=delivery.MissionTimeouts(1.0, 1.0, 1.0),
            sleep=(sleeps.append if sleeps is not None else lambda _: None),
            output=lambda _: None,
        )

    def test_starts_localizes_visits_three_goals_and_waits_at_delivery(self):
        api = FakeApi()
        sleeps = []
        self.mission(api, sleeps=sleeps).run()

        self.assertEqual(
            [path for path, _ in api.posts],
            [
                "/api/v1/navigation/start",
                "/api/v1/navigation/initial-pose",
                "/api/v1/navigation/goal/annotation",
                "/api/v1/navigation/goal/annotation",
                "/api/v1/navigation/goal/annotation",
            ],
        )
        sent_ids = [
            payload["annotation_id"]
            for path, payload in api.posts
            if path.endswith("/goal/annotation")
        ]
        self.assertEqual(sent_ids, [POINTS[1]["id"], POINTS[2]["id"], POINTS[3]["id"]])
        self.assertEqual(sleeps, [5.0])

    def test_failed_delivery_goal_aborts_before_return(self):
        api = FakeApi(fail_goal_number=2)
        with self.assertRaisesRegex(delivery.DeliveryDemoError, "배송지 navigation failed"):
            self.mission(api).run()
        goal_posts = [path for path, _ in api.posts if path.endswith("/goal/annotation")]
        self.assertEqual(len(goal_posts), 2)

    def test_running_different_map_is_rejected(self):
        api = FakeApi(wrong_map=True)
        with self.assertRaisesRegex(delivery.DeliveryDemoError, "different map revision"):
            self.mission(api).run()
        self.assertEqual(api.posts, [])


if __name__ == "__main__":
    unittest.main()
