"""Realtime webcam/video inference for the 3-class model.

Keeps usage close to typical Ultralytics scripts with a simple CLI.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Realtime controlled_fire/fire/smoke inference")
    parser.add_argument(
        "--weights",
        type=Path,
        required=True,
        help="Path to best.pt from training.",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Webcam index (0), video path, stream URL, or folder.",
    )
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="0")
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    if not args.weights.exists():
        raise FileNotFoundError(f"Weights not found: {args.weights}")

    model = YOLO(str(args.weights))
    model.predict(
        source=args.source,
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
        show=True,
        save=False,
        stream=False,
    )


if __name__ == "__main__":
    run(parse_args())
