#!/usr/bin/env python3
"""Detect a red/black taped parking bay in an image or camera stream."""
from __future__ import annotations

import argparse
import platform
from pathlib import Path
import time

import cv2

from robot_dashboard.tape_parking import (
    TapeParkingTracker,
    detect_tape_parking_bay,
    draw_tape_parking_detection,
)


def _camera_source(value: str):
    return int(value) if value.isdigit() else value


def _camera_backend(source) -> int:
    if not isinstance(source, int) and not str(source).startswith("/dev/video"):
        return cv2.CAP_ANY
    if platform.system() == "Darwin":
        return cv2.CAP_AVFOUNDATION
    if platform.system() == "Linux":
        return cv2.CAP_V4L2
    return cv2.CAP_ANY


def _describe(detection) -> str:
    if detection is None:
        return "PARKING_NOT_FOUND"
    x, y = detection.center
    dx, dy = detection.forward
    return (
        f"PARKING_FOUND center=({x:.1f},{y:.1f})px "
        f"forward=({dx:+.3f},{dy:+.3f}) confidence={detection.confidence:.3f} "
        f"red={detection.red_coverage:.3f} black={detection.black_support:.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path)
    source.add_argument("--device")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--seconds", type=float, default=300.0)
    parser.add_argument("--min-area-ratio", type=float, default=0.08)
    parser.add_argument("--min-confidence", type=float, default=0.42)
    parser.add_argument("--min-black-support", type=float, default=0.10)
    parser.add_argument("--detect-every", type=int, default=1)
    parser.add_argument("--max-track-frames", type=int, default=18)
    parser.add_argument("--display", action="store_true")
    args = parser.parse_args()

    if args.image:
        frame = cv2.imread(str(args.image))
        if frame is None:
            parser.error(f"cannot read image: {args.image}")
        detection = detect_tape_parking_bay(
            frame,
            min_area_ratio=args.min_area_ratio,
            min_confidence=args.min_confidence,
            min_black_support=args.min_black_support,
        )
        print(_describe(detection))
        if args.display:
            if detection:
                draw_tape_parking_detection(frame, detection)
            cv2.imshow("Tape parking bay", frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        raise SystemExit(0 if detection else 2)

    camera_source = _camera_source(args.device)
    capture = cv2.VideoCapture(camera_source, _camera_backend(camera_source))
    if not capture.isOpened():
        parser.error(f"cannot open camera: {args.device}")
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    capture.set(cv2.CAP_PROP_FPS, args.fps)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    tracker = TapeParkingTracker(
        detect_every=args.detect_every,
        max_track_frames=args.max_track_frames,
        min_area_ratio=args.min_area_ratio,
        min_confidence=args.min_confidence,
        min_black_support=args.min_black_support,
    )
    started = time.monotonic()
    stable_frames = 0
    last_state = None
    try:
        while time.monotonic() - started < args.seconds:
            ok, frame = capture.read()
            if not ok:
                continue
            observation = tracker.update(frame)
            detection = observation.detection
            stable_frames = stable_frames + 1 if detection else 0
            stable = detection is not None and stable_frames >= 5
            state = observation.state if stable else "ACQUIRING" if detection else "LOST"
            if state != last_state:
                print(
                    f"state={state} tracked_frames={observation.tracked_frames} "
                    f"inliers={observation.inlier_ratio:.3f} {_describe(detection)}",
                    flush=True,
                )
                last_state = state
            if detection:
                draw_tape_parking_detection(frame, detection)
            color = (0, 255, 0) if state == "DETECTED" else (0, 255, 255) if detection else (0, 0, 255)
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 46), (0, 0, 0), -1)
            cv2.putText(frame, state, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            if args.display:
                cv2.imshow("Tape parking bay", frame)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
    finally:
        capture.release()
        if args.display:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
