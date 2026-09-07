#!/usr/bin/env python3
"""Replay crosswalk segmentation in shadow mode; never sends robot commands."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path

import numpy as np

from robot_dashboard.crosswalk_autonomy import (
    CrosswalkNavigator,
    CrosswalkObservation,
    CrosswalkPolicyConfig,
    observation_from_mask,
    route_perception_snapshot,
    select_crosswalk_mask,
)

DEFAULT_YOLOE_MODEL = "yoloe-26s-seg.pt"
DEFAULT_CLASS_NAMES = ("crosswalk", "zebra crossing")
DOWNLOADABLE_YOLOE_MODELS = frozenset(
    f"yoloe-26{size}-seg.pt" for size in ("n", "s", "m", "l", "x")
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate crosswalk center tracking on a recorded image or video"
    )
    parser.add_argument("video", type=Path, help="recorded image or video input")
    parser.add_argument(
        "--model",
        default=DEFAULT_YOLOE_MODEL,
        help=(
            "local segmentation weights or an official YOLOE-26 model name "
            f"(default: {DEFAULT_YOLOE_MODEL})"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--confidence", type=float, default=0.20)
    parser.add_argument(
        "--class-name",
        dest="class_names",
        action="append",
        help=(
            "accepted class or YOLOE text prompt; repeat for aliases "
            "(default: crosswalk + zebra crossing)"
        ),
    )
    parser.add_argument("--image-size", type=int, default=768)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--crosswalk-width-m", type=float, default=3.0)
    return parser


def _class_name(names: object, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, ""))
    if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        return str(names[class_id])
    return ""


def _normalized_class_names(values: list[str] | None) -> tuple[str, ...]:
    source = values if values is not None else list(DEFAULT_CLASS_NAMES)
    normalized = tuple(dict.fromkeys(value.strip().lower() for value in source if value.strip()))
    if not normalized:
        raise ValueError("at least one non-empty --class-name is required")
    return normalized


def _resolve_model(model: str) -> tuple[str, bool]:
    """Return (model reference, uses YOLOE text prompts)."""
    candidate = Path(model).expanduser()
    if candidate.is_file():
        return str(candidate), candidate.name.lower().startswith("yoloe-")
    if model.lower() in DOWNLOADABLE_YOLOE_MODELS and Path(model).name == model:
        return model.lower(), True
    raise ValueError(
        f"model not found: {model}. Use a local .pt file or one of: "
        + ", ".join(sorted(DOWNLOADABLE_YOLOE_MODELS))
    )


def main() -> int:
    args = _parser().parse_args()
    try:
        import cv2  # type: ignore[import-not-found]
        from ultralytics import YOLO, YOLOE  # type: ignore[import-not-found]
    except ImportError as exc:
        raise SystemExit(
            f"vision dependency import failed: {exc}. "
            "Install with: pip install -r requirements-crosswalk.txt"
        ) from exc

    if not args.video.is_file():
        raise SystemExit(f"video not found: {args.video}")
    try:
        class_names = _normalized_class_names(args.class_names)
        model_reference, uses_yoloe = _resolve_model(args.model)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output.with_suffix(".csv")
    jsonl_path = output.with_suffix(".perception.jsonl")
    summary_path = output.with_suffix(".summary.json")

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise SystemExit(f"cannot open video: {args.video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height)
    )
    if not writer.isOpened():
        capture.release()
        raise SystemExit(f"cannot create output video: {output}")

    if uses_yoloe:
        model = YOLOE(model_reference)
        model.set_classes(list(class_names))
    else:
        model = YOLO(model_reference)
    navigator = CrosswalkNavigator(
        CrosswalkPolicyConfig(confidence=args.confidence)
    )
    frame_index = 0
    detected_frames = 0
    hold_frames = 0
    inference_ms: list[float] = []
    absolute_lateral: list[float] = []
    fieldnames = [
        "frame",
        "state",
        "confidence",
        "area_ratio",
        "bottom_ratio",
        "near_width_ratio",
        "lateral_error",
        "heading_error",
        "left_boundary_ratio",
        "right_boundary_ratio",
        "recommended_linear_x",
        "recommended_angular_z",
        "emitted_linear_x",
        "emitted_angular_z",
        "inference_ms",
        "reason",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as csv_stream, jsonl_path.open(
        "w", encoding="utf-8"
    ) as jsonl_stream:
        table = csv.DictWriter(csv_stream, fieldnames=fieldnames)
        table.writeheader()
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            inference_started = time.perf_counter()
            result = model.predict(
                frame,
                imgsz=args.image_size,
                conf=args.confidence,
                device=args.device,
                verbose=False,
            )[0]
            elapsed_ms = (time.perf_counter() - inference_started) * 1000.0
            inference_ms.append(elapsed_ms)

            masks: list[np.ndarray] = []
            confidences: list[float] = []
            if result.masks is not None and result.boxes is not None:
                raw_masks = result.masks.data.cpu().numpy()
                raw_confidences = result.boxes.conf.cpu().numpy().tolist()
                raw_classes = result.boxes.cls.cpu().numpy().astype(int).tolist()
                for raw_mask, confidence, class_id in zip(
                    raw_masks, raw_confidences, raw_classes
                ):
                    name = _class_name(result.names, class_id).strip().lower()
                    if name not in class_names:
                        continue
                    resized = cv2.resize(
                        raw_mask.astype(np.uint8),
                        (width, height),
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(bool)
                    masks.append(resized)
                    confidences.append(float(confidence))

            selected, confidence, _selected_index = select_crosswalk_mask(
                masks, confidences, min_confidence=args.confidence
            )
            observation = (
                observation_from_mask(selected, confidence)
                if selected is not None
                else CrosswalkObservation(False)
            )
            frame_time_s = frame_index / fps
            command = navigator.update(
                observation,
                captured_at_s=frame_time_s,
                now_s=frame_time_s,
                motion_authorized=False,
            )
            observed_at_ns = time.time_ns()
            snapshot = route_perception_snapshot(
                observation,
                sequence=frame_index,
                observed_at_ns=observed_at_ns,
                crosswalk_width_m=args.crosswalk_width_m,
                minimum_confidence=args.confidence,
            )
            jsonl_stream.write(json.dumps(snapshot, separators=(",", ":")) + "\n")

            if observation.visible:
                detected_frames += 1
                absolute_lateral.append(abs(observation.lateral_error))
            if command.state.value == "hold":
                hold_frames += 1

            drawn = frame.copy()
            if selected is not None:
                overlay = drawn.copy()
                overlay[selected] = (255, 80, 30)
                drawn = cv2.addWeighted(overlay, 0.35, drawn, 0.65, 0)
                near = (
                    int(observation.near_center_x_ratio * width), int(height * 0.85)
                )
                far = (
                    int(observation.far_center_x_ratio * width), int(height * 0.55)
                )
                cv2.line(drawn, near, far, (0, 255, 255), 4)
            label = (
                f"SHADOW {command.state.value} conf={observation.confidence:.2f} "
                f"lat={observation.lateral_error:+.3f} head={observation.heading_error:+.3f} "
                f"suggest=({command.recommended_linear_x:.2f},"
                f"{command.recommended_angular_z:+.2f})"
            )
            cv2.rectangle(drawn, (0, 0), (width, 55), (0, 0, 0), -1)
            cv2.putText(
                drawn,
                label,
                (12, 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )
            writer.write(drawn)
            table.writerow(
                {
                    "frame": frame_index,
                    "state": command.state.value,
                    "confidence": observation.confidence,
                    "area_ratio": observation.area_ratio,
                    "bottom_ratio": observation.bottom_ratio,
                    "near_width_ratio": observation.near_width_ratio,
                    "lateral_error": observation.lateral_error,
                    "heading_error": observation.heading_error,
                    "left_boundary_ratio": observation.left_boundary_ratio,
                    "right_boundary_ratio": observation.right_boundary_ratio,
                    "recommended_linear_x": command.recommended_linear_x,
                    "recommended_angular_z": command.recommended_angular_z,
                    "emitted_linear_x": command.linear_x,
                    "emitted_angular_z": command.angular_z,
                    "inference_ms": round(elapsed_ms, 3),
                    "reason": command.reason,
                }
            )
            frame_index += 1

    capture.release()
    writer.release()
    sorted_latency = sorted(inference_ms)
    p95_index = max(0, min(len(sorted_latency) - 1, int(len(sorted_latency) * 0.95)))
    summary = {
        "mode": "SHADOW",
        "robot_commands_sent": 0,
        "model": model_reference,
        "class_names": list(class_names),
        "frames": frame_index,
        "detected_frame_ratio": detected_frames / frame_index if frame_index else 0.0,
        "hold_frame_ratio": hold_frames / frame_index if frame_index else 0.0,
        "mean_abs_lateral_error": statistics.fmean(absolute_lateral)
        if absolute_lateral
        else None,
        "mean_inference_ms": statistics.fmean(inference_ms) if inference_ms else None,
        "p95_inference_ms": sorted_latency[p95_index] if sorted_latency else None,
        "video": str(output),
        "csv": str(csv_path),
        "perception_jsonl": str(jsonl_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
