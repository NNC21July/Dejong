# 3-Class Fire/Smoke Detection (YOLO11)

This project adapts the original fire/smoke detector to **3 classes**:

- `controlled_fire`
- `fire`
- `smoke`

The defaults are tuned for small datasets (~500 augmented images) using transfer learning.

## 1) Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2) What to do with your current `.pt` model (important)

If you already have a model you want to improve (for example from the original repo), do this:

1. Download/copy that model file to your machine (example: `/home/you/models/current_fire_model.pt`).
2. Keep that file path ready.
3. Pass that path to `--model` during training.

### Example

```bash
python model_training/src/train_three_class.py \
  --data /absolute/path/to/your/data.yaml \
  --model /home/you/models/current_fire_model.pt \
  --epochs 120 \
  --batch 8 \
  --imgsz 640
```

That command means: **start from your existing `.pt` weights** and fine-tune to the new 3-class dataset.

> If you do not have a custom `.pt` yet, use `--model yolo11n.pt` and Ultralytics will download it automatically.

## 3) Prepare dataset

You can use either of these dataset styles:

### Option A: This repo template style

```text
dataset_root/
  images/
    train/
    val/
    test/      # optional
  labels/
    train/
    val/
    test/      # optional
```

Then edit `model_training/configs/dataset_3class.yaml`:

- set `path:` to your `dataset_root`
- keep classes in this exact order:
  - `0 = controlled_fire`
  - `1 = fire`
  - `2 = smoke`

### Option B: Roboflow exported YAML

If your `data.yaml` already has paths like `../train/images`, `../valid/images`, etc., you can use it directly with `--data /path/to/data.yaml`.

## 4) Validate dataset (recommended)

```bash
python model_training/src/check_dataset.py --dataset-root /absolute/path/to/dataset_root
```

If your dataset uses a different folder structure (e.g., Roboflow with `train/valid/test`), you can skip this script and rely on your `data.yaml` paths directly.

## 5) Train (3-class fine-tuning)

### Using your existing `.pt` (recommended for your case)

```bash
python model_training/src/train_three_class.py \
  --data /absolute/path/to/your/data.yaml \
  --model /absolute/path/to/your/current_fire_model.pt
```

### Using default base model

```bash
python model_training/src/train_three_class.py \
  --data model_training/configs/dataset_3class.yaml \
  --model yolo11n.pt
```

Outputs are saved to:

```text
runs/fire3class/yolo11n_transfer/
```

Best model path:

```text
runs/fire3class/yolo11n_transfer/weights/best.pt
```

## 6) Use the tuned model later (inference only)

Once training is done, copy/use `best.pt` anywhere you like.

### Batch/file inference (save predictions)

```bash
python model_training/src/predict_three_class.py \
  --weights runs/fire3class/yolo11n_transfer/weights/best.pt \
  --source /path/to/image_or_video_or_folder \
  --conf 0.25
```

### Realtime mode (webcam/video/stream)

```bash
python model_training/src/realtime_infer.py \
  --weights runs/fire3class/yolo11n_transfer/weights/best.pt \
  --source 0 \
  --conf 0.25
```

- `--source 0` = default webcam.
- You can pass a video file path or RTSP/HTTP stream URL instead.

### Ultralytics CLI (alternative)

```bash
yolo predict \
  model=runs/fire3class/yolo11n_transfer/weights/best.pt \
  source=/path/to/image_or_video_or_folder \
  conf=0.25
```

## 7) Quick beginner checklist

- [ ] I installed dependencies.
- [ ] I have my dataset YAML path.
- [ ] My class order is `controlled_fire`, `fire`, `smoke`.
- [ ] I downloaded my current `.pt` model and know its full path.
- [ ] I ran training with `--model /path/to/current_model.pt`.
- [ ] I got `best.pt` and can run `predict` or `realtime_infer`.

## Notes for small datasets

- Keep validation clean and representative.
- If overfitting appears, increase `--freeze` (e.g., 15), reduce epochs, or use a smaller base checkpoint.
- Improve class balance, especially for underrepresented classes (especially `controlled_fire`).
