"""Run inference with the trained 3-class fire model."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict using a trained controlled_fire/fire/smoke model")
    parser.add_argument(
        "--weights",
        type=Path,
        required=True,
        help="Path to trained weights, e.g. runs/fire3class/yolo11n_transfer/weights/best.pt",
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Image/video/folder/webcam source.",
    )
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument(
        "--save-txt",
        action="store_true",
        help="Save YOLO txt predictions alongside images.",
    )
    return parser.parse_args()


def predict(args: argparse.Namespace) -> None:
    if not args.weights.exists():
        raise FileNotFoundError(f"Weights not found: {args.weights}")

    model = YOLO(str(args.weights))
    model.predict(
        source=args.source,
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
        save=True,
        save_txt=args.save_txt,
        show=False,
    )


if __name__ == "__main__":
    predict(parse_args())
