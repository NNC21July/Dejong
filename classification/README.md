# Classification Module (Single-Window Analysis)

This module runs **one analysis window per command execution**.

- You choose a start time and duration.
- The script samples frames in that window.
- It outputs exactly **one JSON** and **one JPG** for that run.

## 1) Install

From repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

To use a specific Python version (3.10-3.12), create the venv with that interpreter:

- macOS/Linux (example 3.11):
  ```bash
  python3.11 -m venv .venv
  ```
- Windows PowerShell (example 3.11):
  ```powershell
  py -3.11 -m venv .venv
  ```

## 2) OpenAI key setup (optional)

OpenAI is optional. Without a key, the script runs local-only.

```bash
cp .env.example .env
```

Then edit `.env`:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

### Manual env var (optional alternative)

- macOS/Linux:
  ```bash
  export OPENAI_API_KEY="..."
  ```
- PowerShell:
  ```powershell
  $env:OPENAI_API_KEY="..."
  ```

## 3) Run command (core usage)

```bash
python classification/analyze_video.py \
  --video /path/to/video.mp4 \
  --weights classification_model.pt \
  --start-seconds 0 \
  --analyze-seconds 10 \
  --sample-fps 2 \
  --conf 0.25 \
  --results-dir results \
  --run-label incident_cam01_0000_0010 \
  --camera-id cam_01 \
  --location-type warehouse \
  --device cpu
```

PowerShell one-line example:

```powershell
python classification/analyze_video.py --video <path_to_video.mp4> --weights classification_model.pt --start-seconds 0 --analyze-seconds 10 --sample-fps 2 --conf 0.25 --results-dir results --run-label incident_cam01_0000_0010 --camera-id cam_01 --location-type warehouse --device cpu
```

## 4) Flags reference

- `--video` (required): input video file.
- `--weights`: model path (default `classification_model.pt`).
- `--start-seconds`: analysis window start offset in video.
- `--analyze-seconds`: analysis window duration.
- `--sample-fps`: frame sampling rate in the selected window.
- `--conf`: YOLO confidence threshold.
- `--results-dir`: parent output folder.
- `--run-label`: output subfolder name. If omitted, auto-generated.
- `--camera-id`, `--location-type`: metadata forwarded into output/OpenAI payload.
- `--device`: inference device (`cpu` recommended for cross-device compatibility).

## 5) Output layout (one run)

For each execution, files are written to:

`<results-dir>/<run-label>/`

with:

- `metrics.json`
- `top_fire.jpg` (fallback: first sampled frame when no fire box is detected)

No timeline file and no multi-interval files are produced.

## 6) What is inside `metrics.json`

Top-level keys:

- `analysis` (window label/start/end/duration/output path)
- `input` (video/model/runtime mode metadata)
- `sampling` (sample fps, sampled frame count, conf threshold)
- `summary` (aggregate confidences + risk variables)
- `openai` (if used/skipped, reason, context score/scenario/confidence/rationale)
- `decision` (`local_score`, `final_score`, `decision_confidence`, `scenario_rank`)
- `artifacts` (json/jpg output paths)

## 7) OpenAI usage rules

OpenAI is called only when:

1. uncertainty conditions are met, **or**
2. local-only rank is `Emergency` (forced context verification).

If key/client is unavailable, OpenAI is skipped and recorded in `openai.note`.

## 8) Scoring and scenario levels

Configured in `classification/configs/scoring.yaml`:

- local score weights (`local_score_weights`)
- uncertainty triggers (`uncertainty`)
- scenario thresholds/hysteresis (`scenario_thresholds`)
- context blend behavior (`context_weighting`)

Current level behavior:

- `No Fire Risk`: very low scores.
- `Elevated Risk`: low-risk smoke-heavy, minimal visible fire.
- `Hazard`: visible fire / meaningful risk.
- `Emergency`: higher-risk uncontrolled growth.

## 9) Troubleshooting

If you see missing module errors:

```bash
pip install -r requirements.txt
```

If OpenAI is not called when expected, check in `metrics.json`:

- `openai.eligible`
- `openai.trigger_reason`
- `openai.note`
