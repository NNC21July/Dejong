"""Fine-tune a YOLO11 model for 3-class controlled_fire/fire/smoke detection."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


DEFAULT_MODEL = "yolo11n.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a 3-class fire detector (controlled_fire, fire, smoke) "
            "from a YOLO11 checkpoint."
        )
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("model_training/configs/dataset_3class.yaml"),
        help="Path to YOLO dataset YAML.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="Base YOLO checkpoint (e.g., yolo11n.pt, yolo11s.pt).",
    )
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument(
        "--patience",
        type=int,
        default=25,
        help="Early stopping patience.",
    )
    parser.add_argument(
        "--freeze",
        type=int,
        default=10,
        help=(
            "Freeze first N layers during initial learning. "
            "Useful for small datasets."
        ),
    )
    parser.add_argument(
        "--project",
        type=str,
        default="runs/fire3class",
        help="Parent output folder for training runs.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="yolo11n_transfer",
        help="Run name within the project directory.",
    )
    parser.add_argument("--device", type=str, default="0", help="Device id or cpu.")
    return parser.parse_args()


def train(args: argparse.Namespace) -> None:
    if not args.data.exists():
        raise FileNotFoundError(
            f"Dataset config does not exist: {args.data}. "
            "Edit model_training/configs/dataset_3class.yaml first."
        )

    model = YOLO(args.model)

    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        project=args.project,
        name=args.name,
        device=args.device,
        optimizer="AdamW",
        lr0=5e-4,
        lrf=0.01,
        weight_decay=5e-4,
        warmup_epochs=3,
        close_mosaic=8,
        freeze=args.freeze,
        hsv_h=0.02,
        hsv_s=0.6,
        hsv_v=0.4,
        translate=0.08,
        scale=0.35,
        fliplr=0.5,
        degrees=7.0,
        mixup=0.1,
        copy_paste=0.0,
    )


if __name__ == "__main__":
    train(parse_args())
