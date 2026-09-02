#!/usr/bin/env python3
"""Replay the crosswalk navigator on video; never sends robot commands."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from robot_dashboard.crosswalk_navigation import (
    CrosswalkNavigator,
    CrosswalkObservation,
    combine_masks,
    observation_from_mask,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("--model", type=Path, default=Path("/home/chosun/crosswalk_runs/yolo11m_768/weights/best.pt"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--confidence", type=float, default=0.50)
    args = parser.parse_args()

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))
    model = YOLO(str(args.model))
    navigator = CrosswalkNavigator()
    csv_path = args.output.with_suffix(".csv")
    rows = []
    frame_id = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        result = model.predict(frame, imgsz=768, conf=args.confidence, device=0, verbose=False)[0]
        masks, confidences = [], []
        if result.masks is not None:
            raw = result.masks.data.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy().tolist()
            masks = [cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool) for m in raw]
        combined = combine_masks(masks)
        observation = observation_from_mask(combined, max(confidences)) if combined is not None else CrosswalkObservation(False)
        command = navigator.update(observation)
        drawn = frame.copy()
        if combined is not None:
            overlay = drawn.copy(); overlay[combined] = (255, 80, 30)
            drawn = cv2.addWeighted(overlay, 0.35, drawn, 0.65, 0)
        center = (w // 2, int(h * 0.85))
        target = (int(w / 2 + observation.lateral_error * w / 2), int(h * 0.85))
        cv2.line(drawn, center, target, (0, 255, 255), 4)
        text = f"{command.state.value} conf={observation.confidence:.2f} lat={observation.lateral_error:+.3f} head={observation.heading_error:+.3f} vx={command.linear_x:.2f} wz={command.angular_z:+.2f}"
        cv2.rectangle(drawn, (0, 0), (w, 52), (0, 0, 0), -1)
        cv2.putText(drawn, text, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)
        writer.write(drawn)
        rows.append([frame_id, command.state.value, observation.confidence, observation.area_ratio, observation.bottom_ratio, observation.near_width_ratio, observation.lateral_error, observation.heading_error, command.linear_x, command.angular_z])
        frame_id += 1
    cap.release(); writer.release()
    with csv_path.open("w", newline="") as f:
        out = csv.writer(f); out.writerow(["frame","state","confidence","area_ratio","bottom_ratio","near_width_ratio","lateral_error","heading_error","linear_x","angular_z"]); out.writerows(rows)
    print(f"frames={frame_id} video={args.output} decisions={csv_path}")


if __name__ == "__main__":
    main()
