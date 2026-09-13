#!/usr/bin/env python3
"""Run a small delivery loop through Robot Scope's guarded navigation API.

The operator first stores four point annotations on the selected map.  This
client then starts Nav2 when necessary, uses the start point as the initial
pose, and visits pickup, delivery, and return in order.  Motion still passes
through Robot Scope's normal map-revision, localization, lease, and watchdog
checks; this script does not publish ROS velocity commands.
"""

from __future__ import annotations

import argparse
import json
import math
import signal
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


TERMINAL_GOAL_STATES = frozenset({"succeeded", "failed", "canceled"})
ACTIVE_GOAL_STATES = frozenset({"pending", "active", "canceling"})


class DeliveryDemoError(RuntimeError):
    """Expected setup, API, or navigation failure."""


class _NoRedirect(HTTPRedirectHandler):
    """urllib redirect handler that keeps requests on the configured origin."""

    def http_error_301(
        self,
        request: Any,
        response: Any,
        code: int,
        message: str,
        headers: Any,
    ) -> Any:
        del request, response, code, message, headers
        raise DeliveryDemoError("Robot Scope API redirects are not allowed")

    http_error_302 = http_error_301
    http_error_303 = http_error_301
    http_error_307 = http_error_301
    http_error_308 = http_error_301


class RobotScopeApi:
    """Small same-origin JSON client for one Robot Scope instance."""

    def __init__(self, base_url: str, *, request_timeout_s: float = 5.0) -> None:
        parsed = urlsplit(str(base_url).strip())
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.netloc
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise DeliveryDemoError("--base-url must be an HTTP(S) origin")
        try:
            parsed.port
        except ValueError as exc:
            raise DeliveryDemoError("--base-url contains an invalid port") from exc
        path = parsed.path.rstrip("/")
        if path:
            raise DeliveryDemoError("--base-url must not contain a path")
        if not math.isfinite(request_timeout_s) or not 0.1 <= request_timeout_s <= 60.0:
            raise DeliveryDemoError("request timeout must be from 0.1 to 60 seconds")
        self.base_url = f"{parsed.scheme}://{parsed.netloc}"
        self.origin = self.base_url
        self.request_timeout_s = request_timeout_s
        self._opener = build_opener(_NoRedirect())

    def get(self, path: str) -> dict[str, Any]:
        return self._request("GET", path)

    def post(self, path: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        return self._request("POST", path, payload)

    def _request(
        self,
        method: str,
        path: str,
        payload: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not path.startswith("/") or path.startswith("//"):
            raise DeliveryDemoError("API path is invalid")
        body = None
        headers = {"Accept": "application/json", "Origin": self.origin}
        if payload is not None:
            body = json.dumps(dict(payload), separators=(",", ":")).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = Request(
            f"{self.base_url}{path}",
            data=body,
            headers=headers,
            method=method,
        )
        try:
            with self._opener.open(request, timeout=self.request_timeout_s) as response:
                raw = response.read(2_000_001)
        except HTTPError as exc:
            detail = ""
            try:
                error_body = exc.read(65_537)
                decoded = json.loads(error_body.decode("utf-8"))
                if isinstance(decoded, Mapping):
                    detail = str(decoded.get("detail", ""))[:300]
            except (UnicodeDecodeError, json.JSONDecodeError, OSError):
                detail = ""
            suffix = f": {detail}" if detail else ""
            raise DeliveryDemoError(f"Robot Scope API returned HTTP {exc.code}{suffix}") from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise DeliveryDemoError(f"Robot Scope API is unavailable: {exc}") from exc
        if len(raw) > 2_000_000:
            raise DeliveryDemoError("Robot Scope API response is unexpectedly large")
        try:
            decoded = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DeliveryDemoError("Robot Scope API returned invalid JSON") from exc
        if not isinstance(decoded, dict):
            raise DeliveryDemoError("Robot Scope API returned an invalid document")
        return decoded


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DeliveryDemoError(f"Robot Scope {label} is unavailable")
    return value


def resolve_map(catalog: Mapping[str, Any], selector: str) -> Mapping[str, Any]:
    maps = catalog.get("maps")
    if not isinstance(maps, list):
        raise DeliveryDemoError("saved-map catalog is unavailable")
    normalized = str(selector).strip()
    exact_ids = [item for item in maps if isinstance(item, Mapping) and item.get("id") == normalized]
    named = [
        item
        for item in maps
        if isinstance(item, Mapping)
        and str(item.get("name", "")).casefold() == normalized.casefold()
    ]
    matches = exact_ids or named
    if len(matches) != 1:
        reason = "not found" if not matches else "ambiguous"
        raise DeliveryDemoError(f"saved map {selector!r} is {reason}")
    selected = matches[0]
    if selected.get("format") != "map-server-pgm":
        raise DeliveryDemoError("delivery demo requires a map-server PGM/YAML map")
    if not isinstance(selected.get("id"), str) or not isinstance(selected.get("revision"), str):
        raise DeliveryDemoError("saved map identity is invalid")
    return selected


def resolve_points(
    annotations: Mapping[str, Any],
    names: Sequence[str],
) -> tuple[Mapping[str, Any], ...]:
    points = annotations.get("points")
    if not isinstance(points, list):
        raise DeliveryDemoError("map point annotations are unavailable")
    resolved: list[Mapping[str, Any]] = []
    for requested in names:
        key = requested.strip().casefold()
        matches = [
            item
            for item in points
            if isinstance(item, Mapping)
            and str(item.get("name", "")).strip().casefold() == key
        ]
        if len(matches) != 1:
            reason = "not found" if not matches else "ambiguous"
            raise DeliveryDemoError(f"point annotation {requested!r} is {reason}")
        point = matches[0]
        pose = point.get("pose")
        if not isinstance(point.get("id"), str) or not isinstance(pose, Mapping):
            raise DeliveryDemoError(f"point annotation {requested!r} is invalid")
        try:
            coordinates = tuple(float(pose[key]) for key in ("x", "y", "yaw"))
        except (KeyError, TypeError, ValueError) as exc:
            raise DeliveryDemoError(f"point annotation {requested!r} pose is invalid") from exc
        if not all(math.isfinite(value) for value in coordinates):
            raise DeliveryDemoError(f"point annotation {requested!r} pose is invalid")
        resolved.append(point)
    return tuple(resolved)


@dataclass(frozen=True)
class MissionTimeouts:
    startup_s: float = 180.0
    localization_s: float = 30.0
    goal_s: float = 300.0

    def __post_init__(self) -> None:
        for value in (self.startup_s, self.localization_s, self.goal_s):
            if not math.isfinite(value) or not 1.0 <= value <= 3_600.0:
                raise DeliveryDemoError("mission timeouts must be from 1 to 3600 seconds")


class DeliveryMission:
    """Orchestrate one revision-pinned delivery loop over HTTP."""

    def __init__(
        self,
        api: Any,
        *,
        map_record: Mapping[str, Any],
        annotations: Mapping[str, Any],
        points: Sequence[Mapping[str, Any]],
        delivery_dwell_s: float,
        poll_interval_s: float = 0.5,
        timeouts: MissionTimeouts = MissionTimeouts(),
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
        output: Callable[[str], None] = print,
    ) -> None:
        if len(points) != 4:
            raise DeliveryDemoError("exactly four mission points are required")
        if not math.isfinite(delivery_dwell_s) or not 0.0 <= delivery_dwell_s <= 300.0:
            raise DeliveryDemoError("delivery dwell must be from 0 to 300 seconds")
        if not math.isfinite(poll_interval_s) or not 0.05 <= poll_interval_s <= 10.0:
            raise DeliveryDemoError("poll interval must be from 0.05 to 10 seconds")
        self.api = api
        self.map = map_record
        self.annotations = annotations
        self.points = tuple(points)
        self.delivery_dwell_s = delivery_dwell_s
        self.poll_interval_s = poll_interval_s
        self.timeouts = timeouts
        self.sleep = sleep
        self.monotonic = monotonic
        self.output = output
        self.active_goal_id: str | None = None

    @property
    def map_id(self) -> str:
        return str(self.map["id"])

    @property
    def map_revision(self) -> str:
        return str(self.map["revision"])

    @property
    def annotation_revision(self) -> str:
        revision = self.annotations.get("annotation_revision")
        if not isinstance(revision, str):
            raise DeliveryDemoError("annotation revision is invalid")
        return revision

    def run(self) -> None:
        start, pickup, delivery, return_point = self.points
        self._ensure_navigation()
        self._set_initial_pose(start)
        self._visit("픽업지", pickup)
        self._visit("배송지", delivery)
        if self.delivery_dwell_s:
            self.output(f"[delivery] 배송 완료 대기 {self.delivery_dwell_s:g}초")
            self.sleep(self.delivery_dwell_s)
        self._visit("복귀지", return_point)
        self.output("[delivery] 미션 완료: 복귀지 도착")

    def cancel_active_goal(self) -> None:
        identifier = self.active_goal_id
        if not identifier:
            return
        try:
            self.api.post("/api/v1/navigation/cancel", {"goal_id": identifier})
        except DeliveryDemoError as exc:
            self.output(f"[delivery] 목표 취소 확인 실패: {exc}")
        finally:
            self.active_goal_id = None

    def _navigation(self) -> Mapping[str, Any]:
        return self.api.get("/api/v1/navigation")

    def _wait_for(
        self,
        label: str,
        timeout_s: float,
        predicate: Callable[[Mapping[str, Any]], bool],
    ) -> Mapping[str, Any]:
        deadline = self.monotonic() + timeout_s
        last: Mapping[str, Any] = {}
        while self.monotonic() < deadline:
            last = self._navigation()
            if predicate(last):
                return last
            pipeline = _mapping(last.get("pipeline", {}), "navigation pipeline")
            if pipeline.get("state") == "failed":
                detail = str(pipeline.get("error") or "unknown error")[:300]
                raise DeliveryDemoError(f"navigation pipeline failed: {detail}")
            self.sleep(self.poll_interval_s)
        blockers = _mapping(last.get("safety", {}), "navigation safety").get("blockers", [])
        raise DeliveryDemoError(f"timed out waiting for {label}; blockers={blockers}")

    def _ensure_navigation(self) -> None:
        status = self._navigation()
        pipeline = _mapping(status.get("pipeline"), "navigation pipeline")
        state = str(pipeline.get("state", ""))
        if state in {"running", "starting"}:
            active_map = status.get("map")
            if state == "running" and (
                not isinstance(active_map, Mapping)
                or active_map.get("id") != self.map_id
                or active_map.get("revision") != self.map_revision
            ):
                raise DeliveryDemoError("running Nav2 session uses a different map revision")
            self.output("[delivery] 기존 Nav2 시작 작업을 사용합니다")
        elif state in {"idle", "failed"}:
            parameters = self.api.get("/api/v1/navigation/parameters")
            revision = parameters.get("revision")
            if not isinstance(revision, str):
                raise DeliveryDemoError("navigation parameter revision is invalid")
            self.api.post(
                "/api/v1/navigation/start",
                {
                    "map_id": self.map_id,
                    "map_revision": self.map_revision,
                    "parameters_revision": revision,
                },
            )
            self.output("[delivery] Nav2 시작 요청을 보냈습니다")
        else:
            raise DeliveryDemoError(f"navigation pipeline is busy: {state or 'unknown'}")

        def ready(candidate: Mapping[str, Any]) -> bool:
            pipeline_view = candidate.get("pipeline")
            map_view = candidate.get("map")
            safety = candidate.get("safety")
            return bool(
                isinstance(pipeline_view, Mapping)
                and pipeline_view.get("state") == "running"
                and isinstance(map_view, Mapping)
                and map_view.get("id") == self.map_id
                and map_view.get("revision") == self.map_revision
                and isinstance(safety, Mapping)
                and safety.get("can_set_initial_pose") is True
            )

        self._wait_for("Nav2 startup", self.timeouts.startup_s, ready)

    def _set_initial_pose(self, point: Mapping[str, Any]) -> None:
        pose = _mapping(point.get("pose"), "start pose")
        self.api.post(
            "/api/v1/navigation/initial-pose",
            {
                "map_id": self.map_id,
                "map_revision": self.map_revision,
                "pose": {key: float(pose[key]) for key in ("x", "y", "yaw")},
            },
        )
        self.output(f"[delivery] 출발지 초기 위치 설정: {point.get('name')}")

        def localized(candidate: Mapping[str, Any]) -> bool:
            localization = candidate.get("localization")
            safety = candidate.get("safety")
            return bool(
                isinstance(localization, Mapping)
                and localization.get("state") == "localized"
                and isinstance(safety, Mapping)
                and safety.get("can_send_goal") is True
            )

        self._wait_for("localization", self.timeouts.localization_s, localized)

    def _visit(self, label: str, point: Mapping[str, Any]) -> None:
        response = self.api.post(
            "/api/v1/navigation/goal/annotation",
            {
                "map_id": self.map_id,
                "map_revision": self.map_revision,
                "annotation_revision": self.annotation_revision,
                "annotation_id": str(point["id"]),
                "confirmed": True,
            },
        )
        navigation = _mapping(response.get("navigation"), "navigation response")
        goal = _mapping(navigation.get("goal"), "navigation goal")
        identifier = goal.get("goal_id")
        if not isinstance(identifier, str) or not identifier:
            raise DeliveryDemoError("navigation goal id is unavailable")
        self.active_goal_id = identifier
        self.output(f"[delivery] {label} 이동 시작: {point.get('name')}")
        deadline = self.monotonic() + self.timeouts.goal_s
        while self.monotonic() < deadline:
            status = self._navigation()
            current = _mapping(status.get("goal"), "navigation goal")
            if current.get("goal_id") != identifier:
                raise DeliveryDemoError("navigation goal changed outside the delivery mission")
            state = str(current.get("state", ""))
            if state in TERMINAL_GOAL_STATES:
                self.active_goal_id = None
                if state != "succeeded":
                    detail = str(current.get("error") or state)[:300]
                    raise DeliveryDemoError(f"{label} navigation {state}: {detail}")
                self.output(f"[delivery] {label} 도착")
                return
            if state not in ACTIVE_GOAL_STATES:
                raise DeliveryDemoError(f"{label} navigation entered invalid state: {state}")
            self.sleep(self.poll_interval_s)
        self.cancel_active_goal()
        raise DeliveryDemoError(f"timed out navigating to {label}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8088")
    parser.add_argument("--map", required=True, help="exact saved-map name or opaque ID")
    parser.add_argument("--start", default="출발지", help="start point annotation name")
    parser.add_argument("--pickup", default="픽업지", help="pickup point annotation name")
    parser.add_argument("--delivery", default="배송지", help="delivery point annotation name")
    parser.add_argument("--return-point", default="복귀지", help="return point annotation name")
    parser.add_argument("--delivery-dwell", type=float, default=5.0, metavar="SECONDS")
    parser.add_argument("--startup-timeout", type=float, default=180.0, metavar="SECONDS")
    parser.add_argument("--localization-timeout", type=float, default=30.0, metavar="SECONDS")
    parser.add_argument("--goal-timeout", type=float, default=300.0, metavar="SECONDS")
    parser.add_argument("--poll-interval", type=float, default=0.5, metavar="SECONDS")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    mission: DeliveryMission | None = None

    def handle_signal(signum: int, frame: Any) -> None:
        del signum, frame
        raise KeyboardInterrupt

    previous_handlers = {
        signal.SIGINT: signal.signal(signal.SIGINT, handle_signal),
        signal.SIGTERM: signal.signal(signal.SIGTERM, handle_signal),
    }
    try:
        api = RobotScopeApi(args.base_url)
        selected_map = resolve_map(api.get("/api/v1/saved-maps"), args.map)
        annotations = api.get(
            f"/api/v1/saved-maps/{selected_map['id']}/annotations"
        )
        points = resolve_points(
            annotations,
            (args.start, args.pickup, args.delivery, args.return_point),
        )
        mission = DeliveryMission(
            api,
            map_record=selected_map,
            annotations=annotations,
            points=points,
            delivery_dwell_s=args.delivery_dwell,
            poll_interval_s=args.poll_interval,
            timeouts=MissionTimeouts(
                startup_s=args.startup_timeout,
                localization_s=args.localization_timeout,
                goal_s=args.goal_timeout,
            ),
        )
        mission.run()
        return 0
    except KeyboardInterrupt:
        if mission is not None:
            mission.cancel_active_goal()
        print("[delivery] 실패: mission interrupted", file=sys.stderr)
        return 130
    except DeliveryDemoError as exc:
        if mission is not None:
            mission.cancel_active_goal()
        print(f"[delivery] 실패: {exc}", file=sys.stderr)
        return 1
    finally:
        for key, handler in previous_handlers.items():
            signal.signal(key, handler)


if __name__ == "__main__":
    raise SystemExit(main())
