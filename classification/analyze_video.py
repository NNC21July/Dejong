"""Single-window video fire analysis with optional OpenAI context tie-breaker.

One execution analyzes one requested time window and writes:
- <results-dir>/<run-label>/metrics.json
- <results-dir>/<run-label>/top_fire.jpg
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

import cv2
import numpy as np
import yaml

try:
    from dotenv import load_dotenv
except ImportError:  # optional in local-only runtime
    def load_dotenv(*_args, **_kwargs):
        return False

from ultralytics import YOLO



def _bootstrap_import_paths() -> None:
    """Make direct script execution robust on Windows/Linux shells."""
    import sys

    this_dir = Path(__file__).resolve().parent
    repo_root = this_dir.parent

    for candidate in (repo_root, this_dir):
        candidate_str = str(candidate)
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)


_bootstrap_import_paths()

try:
    from classification.src.openai_reasoner.client import get_openai_client
    from classification.src.openai_reasoner.reasoner import reason_with_openai
    from classification.src.scoring import (
        LocalScoreWeights,
        ScenarioThresholds,
        UncertaintyThresholds,
        assign_scenario_rank,
        compute_decision_confidence,
        compute_local_score,
        is_uncertain,
    )
except ModuleNotFoundError:
    try:
        from src.openai_reasoner.client import get_openai_client
        from src.openai_reasoner.reasoner import reason_with_openai
        from src.scoring import (
            LocalScoreWeights,
            ScenarioThresholds,
            UncertaintyThresholds,
            assign_scenario_rank,
            compute_decision_confidence,
            compute_local_score,
            is_uncertain,
        )
    except ModuleNotFoundError:
        from openai_reasoner.client import get_openai_client
        from openai_reasoner.reasoner import reason_with_openai
        from scoring import (
            LocalScoreWeights,
            ScenarioThresholds,
            UncertaintyThresholds,
            assign_scenario_rank,
            compute_decision_confidence,
            compute_local_score,
            is_uncertain,
        )

CLASS_NAMES = {0: "controlled_fire", 1: "fire", 2: "smoke"}


@dataclass
class AggregateStats:
    counts: dict[str, int] = field(default_factory=lambda: {"controlled_fire": 0, "fire": 0, "smoke": 0})
    sum_conf: dict[str, float] = field(default_factory=lambda: {"controlled_fire": 0.0, "fire": 0.0, "smoke": 0.0})
    max_conf: dict[str, float] = field(default_factory=lambda: {"controlled_fire": 0.0, "fire": 0.0, "smoke": 0.0})
    fire_conf_series: list[float] = field(default_factory=list)
    fire_area_series: list[float] = field(default_factory=list)
    num_detections_total: int = 0
    sampled_frames: int = 0


@dataclass
class FrameSignal:
    class_name: str
    confidence: float
    bbox_area_ratio: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze one requested video window with classification_model.pt")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--weights", type=Path, default=Path("classification_model.pt"))
    parser.add_argument("--start-seconds", type=float, default=0.0, help="Start time offset (seconds) from beginning of video")
    parser.add_argument("--analyze-seconds", type=float, default=10.0, help="Window duration to analyze (seconds)")
    parser.add_argument("--sample-fps", type=float, default=2.0)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--run-label", type=str, default="", help="Optional output subfolder name under --results-dir")
    parser.add_argument("--scoring-config", type=Path, default=Path("classification/configs/scoring.yaml"))
    parser.add_argument("--camera-id", type=str, default="unknown_camera")
    parser.add_argument("--location-type", type=str, default="unknown_location")
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _safe_mean(total: float, count: int) -> float:
    return total / count if count > 0 else 0.0


def _area_ratio(xyxy: list[float], frame_w: int, frame_h: int) -> float:
    x1, y1, x2, y2 = xyxy
    box_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return box_area / float(max(1, frame_w * frame_h))


def _fire_frame_score(frame_signals: list[FrameSignal]) -> float:
    fire = [d for d in frame_signals if d.class_name == "fire"]
    controlled = [d for d in frame_signals if d.class_name == "controlled_fire"]
    if not fire:
        return 0.0
    fire_strength = max(d.confidence * (0.7 + 0.3 * _clamp01(d.bbox_area_ratio * 5.0)) for d in fire)
    controlled_penalty = max((d.confidence for d in controlled), default=0.0) * 0.20
    return _clamp01(fire_strength - controlled_penalty)


def _update_stats(stats: AggregateStats, frame_signals: list[FrameSignal]) -> None:
    stats.sampled_frames += 1
    for d in frame_signals:
        stats.num_detections_total += 1
        stats.counts[d.class_name] += 1
        stats.sum_conf[d.class_name] += d.confidence
        stats.max_conf[d.class_name] = max(stats.max_conf[d.class_name], d.confidence)
        if d.class_name == "fire":
            stats.fire_conf_series.append(d.confidence)
            stats.fire_area_series.append(d.bbox_area_ratio)


def _compute_behavior(stats: AggregateStats) -> tuple[float, float]:
    fire_flicker_score = 0.0
    if len(stats.fire_conf_series) > 1:
        deltas = [
            abs(stats.fire_conf_series[i] - stats.fire_conf_series[i - 1])
            for i in range(1, len(stats.fire_conf_series))
        ]
        fire_flicker_score = mean(deltas)

    fire_spread_score = 0.0
    if len(stats.fire_area_series) > 1:
        fire_spread_score = max(stats.fire_area_series) - min(stats.fire_area_series)

    return fire_flicker_score, fire_spread_score


def _compute_aggregate_confidence(
    mean_controlled: float,
    mean_fire: float,
    mean_smoke: float,
    fire_spread_score: float,
    fire_flicker_score: float,
) -> dict[str, float]:
    spread_n = _clamp01(fire_spread_score * 3.0)
    flicker_n = _clamp01(fire_flicker_score * 4.0)

    controlled_raw = (
        0.45 * mean_controlled
        + 0.15 * (1.0 - spread_n)
        + 0.10 * (1.0 - flicker_n)
        + 0.30 * (1.0 - mean_fire)
    )
    controlled_raw *= 1.0 - 0.35 * mean_smoke

    fire_raw = 0.60 * mean_fire + 0.20 * spread_n + 0.15 * flicker_n + 0.05 * mean_smoke
    fire_raw *= 1.0 - 0.15 * mean_controlled

    smoke_raw = 0.75 * mean_smoke + 0.20 * mean_fire + 0.05 * flicker_n

    total = controlled_raw + fire_raw + smoke_raw
    if total <= 1e-12:
        return {"controlled_fire": 0.0, "fire": 0.0, "smoke": 0.0}

    return {
        "controlled_fire": _clamp01(controlled_raw / total),
        "fire": _clamp01(fire_raw / total),
        "smoke": _clamp01(smoke_raw / total),
    }


def _compute_risk_numbers(aggregate: dict[str, float], fire_spread_score: float, fire_flicker_score: float) -> dict[str, float]:
    fire = aggregate["fire"]
    controlled = aggregate["controlled_fire"]
    smoke = aggregate["smoke"]
    spread_n = _clamp01(fire_spread_score * 3.0)
    flicker_n = _clamp01(fire_flicker_score * 4.0)

    fire_vs_controlled_gap = fire - controlled
    fire_to_controlled_ratio = fire / max(controlled, 1e-6)
    dangerous_fire_index = _clamp01(0.55 * fire + 0.20 * smoke + 0.15 * spread_n + 0.10 * flicker_n)

    return {
        "dangerous_fire_index": dangerous_fire_index,
        "fire_vs_controlled_gap": fire_vs_controlled_gap,
        "fire_to_controlled_ratio": fire_to_controlled_ratio,
        "spread_normalized": spread_n,
        "flicker_normalized": flicker_n,
    }


def _summarize_stats(stats: AggregateStats) -> dict:
    mean_controlled = _safe_mean(stats.sum_conf["controlled_fire"], stats.counts["controlled_fire"])
    mean_fire = _safe_mean(stats.sum_conf["fire"], stats.counts["fire"])
    mean_smoke = _safe_mean(stats.sum_conf["smoke"], stats.counts["smoke"])

    fire_flicker_score, fire_spread_score = _compute_behavior(stats)
    aggregate_relative_confidence = _compute_aggregate_confidence(
        mean_controlled=mean_controlled,
        mean_fire=mean_fire,
        mean_smoke=mean_smoke,
        fire_spread_score=fire_spread_score,
        fire_flicker_score=fire_flicker_score,
    )
    risk_numbers = _compute_risk_numbers(
        aggregate=aggregate_relative_confidence,
        fire_spread_score=fire_spread_score,
        fire_flicker_score=fire_flicker_score,
    )

    return {
        "num_detections_total": stats.num_detections_total,
        "sampled_frames": stats.sampled_frames,
        "counts": stats.counts,
        "max_confidence": stats.max_conf,
        "mean_confidence": {
            "controlled_fire": mean_controlled,
            "fire": mean_fire,
            "smoke": mean_smoke,
        },
        "video_behavior_signals": {
            "fire_flicker_score": fire_flicker_score,
            "fire_spread_score": fire_spread_score,
        },
        "aggregate_relative_confidence": aggregate_relative_confidence,
        "risk_numbers": risk_numbers,
    }


def _load_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Scoring config not found: {path}")
    return yaml.safe_load(path.read_text())


def _as_uncertainty(cfg: dict) -> UncertaintyThresholds:
    return UncertaintyThresholds(**cfg["uncertainty"])


def _as_local_weights(cfg: dict) -> LocalScoreWeights:
    return LocalScoreWeights(**cfg["local_score_weights"])


def _as_scenario_thresholds(cfg: dict) -> ScenarioThresholds:
    return ScenarioThresholds(**cfg["scenario_thresholds"])


def _build_run_label(args: argparse.Namespace, start_s: float, end_s: float) -> str:
    if args.run_label:
        return args.run_label
    return f"analysis_{int(start_s):06d}s_{int(end_s):06d}s"


def analyze_video(args: argparse.Namespace) -> dict:
    load_dotenv()

    if not args.video.exists():
        raise FileNotFoundError(f"Video not found: {args.video}")
    if not args.weights.exists():
        raise FileNotFoundError(f"Weights not found: {args.weights}")

    cfg = _load_config(args.scoring_config)
    uncertainty_cfg = _as_uncertainty(cfg)
    local_weights = _as_local_weights(cfg)
    scenario_thresholds = _as_scenario_thresholds(cfg)

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1)
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_s = (total_frames / fps) if fps > 0 else 0.0

    start_s = max(0.0, args.start_seconds)
    if start_s >= duration_s and duration_s > 0:
        raise ValueError(f"--start-seconds ({start_s}) must be smaller than video duration ({duration_s:.2f}s)")
    end_s = min(duration_s, start_s + max(0.1, args.analyze_seconds)) if duration_s > 0 else start_s + max(0.1, args.analyze_seconds)

    model = YOLO(str(args.weights))
    sample_stride = max(1, int(round(fps / max(args.sample_fps, 0.01))))

    stats = AggregateStats()
    top_fire_frame_score = -1.0
    top_fire_frame_timestamp = None
    top_fire_frame = None
    first_sample_frame = None
    first_sample_timestamp = None

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        t = frame_idx / fps if fps > 0 else 0.0
        in_window = start_s <= t < end_s
        should_sample = in_window and (frame_idx % sample_stride == 0)

        if should_sample:
            results = model.predict(source=frame, conf=args.conf, verbose=False, device=args.device)
            if first_sample_frame is None:
                first_sample_frame = frame.copy()
                first_sample_timestamp = t

            frame_signals: list[FrameSignal] = []
            if results:
                r = results[0]
                if r.boxes is not None:
                    boxes = r.boxes
                    for i in range(len(boxes)):
                        cls_id = int(boxes.cls[i].item())
                        conf = float(boxes.conf[i].item())
                        xyxy = [float(v) for v in boxes.xyxy[i].tolist()]
                        class_name = CLASS_NAMES.get(cls_id, str(cls_id))
                        frame_signals.append(
                            FrameSignal(
                                class_name=class_name,
                                confidence=conf,
                                bbox_area_ratio=_area_ratio(xyxy, frame_w, frame_h),
                            )
                        )

            _update_stats(stats, frame_signals)
            frame_fire_score = _fire_frame_score(frame_signals)
            if frame_fire_score > top_fire_frame_score:
                top_fire_frame_score = frame_fire_score
                top_fire_frame_timestamp = t
                top_fire_frame = frame.copy()

        frame_idx += 1

    cap.release()

    args.results_dir.mkdir(parents=True, exist_ok=True)
    run_label = _build_run_label(args, start_s, end_s)
    run_dir = args.results_dir / run_label
    run_dir.mkdir(parents=True, exist_ok=True)

    top_fire_frame_path = run_dir / "top_fire.jpg"
    frame_to_write = top_fire_frame if top_fire_frame is not None else first_sample_frame
    if frame_to_write is None:
        # Always emit a JPG artifact even when no frame is sampled in the selected window.
        frame_to_write = np.zeros((max(frame_h, 1), max(frame_w, 1), 3), dtype=np.uint8)
    wrote_jpg = cv2.imwrite(str(top_fire_frame_path), frame_to_write)
    top_fire_frame_path_str = str(top_fire_frame_path) if wrote_jpg else None

    summary = _summarize_stats(stats)
    summary["top_fire_frame"] = {
        "path": top_fire_frame_path_str,
        "timestamp_s": top_fire_frame_timestamp if top_fire_frame_timestamp is not None else first_sample_timestamp,
        "fire_frame_score": max(0.0, top_fire_frame_score),
    }

    agg = summary["aggregate_relative_confidence"]
    risk = summary["risk_numbers"]

    local_score = compute_local_score(agg, risk, local_weights)
    uncertain = is_uncertain(summary, uncertainty_cfg)
    local_pre_openai_rank = assign_scenario_rank(local_score, None, scenario_thresholds, agg)
    emergency_needs_verification = local_pre_openai_rank == "Emergency"
    should_call_openai = uncertain or emergency_needs_verification

    openai_client = get_openai_client()
    openai_enabled = openai_client is not None
    openai_unavailable_reason = "missing_api_key_or_client_init_failure" if openai_client is None else ""
    runtime_mode = "demo_local" if not openai_enabled else "openai_enabled"

    openai_payload = {
        "used": False,
        "eligible": should_call_openai,
        "trigger_reason": "emergency_verification" if emergency_needs_verification else ("uncertainty" if uncertain else "none"),
        "context_score": 0.0,
        "scenario": None,
        "confidence": 0.0,
        "rationale": [],
        "note": "",
    }

    if should_call_openai and openai_enabled and top_fire_frame_path_str is not None:
        model_name = cfg["openai"]["model"]
        if local_score >= float(cfg["openai"].get("high_risk_switch_threshold", 1.0)):
            model_name = cfg["openai"].get("high_risk_model", model_name)

        metadata = {
            "analysis_label": run_label,
            "camera_id": args.camera_id,
            "location_type": args.location_type,
            "start_s": start_s,
            "end_s": end_s,
            "openai_trigger_reason": openai_payload["trigger_reason"],
        }
        try:
            openai_result = reason_with_openai(
                client=openai_client,
                model=model_name,
                image_path=Path(top_fire_frame_path_str),
                metadata=metadata,
                metrics={"aggregate_relative_confidence": agg, "risk_numbers": risk, "local_pre_openai_rank": local_pre_openai_rank},
            )
            openai_payload = {"used": True, **openai_result}
        except Exception as exc:
            # Keep pipeline alive if OpenAI call fails (e.g., bad key/network/rate limit).
            openai_payload["used"] = False
            openai_payload["note"] = f"OpenAI call failed; fallback to local-only scoring ({exc.__class__.__name__})"
    elif should_call_openai and not openai_enabled:
        if emergency_needs_verification:
            openai_payload["note"] = f"local Emergency flagged; OpenAI verification skipped ({openai_unavailable_reason})"
        else:
            openai_payload["note"] = f"uncertain interval; OpenAI skipped ({openai_unavailable_reason})"
    elif should_call_openai and top_fire_frame_path_str is None:
        openai_payload["note"] = "OpenAI eligible but skipped because no sampled frame was available"
    else:
        openai_payload["note"] = "analysis window not uncertain and not local-Emergency; OpenAI skipped"

    context_cfg = cfg["context_weighting"]
    w_context = float(context_cfg["w_context"])
    scale_by_openai_confidence = bool(context_cfg.get("scale_by_openai_confidence", True))

    if openai_payload["used"]:
        context_weight = w_context
        if scale_by_openai_confidence:
            confidence = float(openai_payload.get("confidence", 0.0))
            context_weight *= max(confidence, float(context_cfg.get("min_openai_confidence_floor", 0.75)))
        context_weight = max(context_weight, float(context_cfg.get("min_context_weight", 0.70)))
        context_weight = min(context_weight, 0.95)
        final_score = (1.0 - context_weight) * local_score + context_weight * float(openai_payload["context_score"])
        if openai_payload.get("scenario") == "Emergency":
            final_score = max(final_score, float(openai_payload.get("context_score", 0.0)))
    else:
        final_score = local_score

    decision_confidence = compute_decision_confidence(
        aggregate_relative_confidence=agg,
        risk_numbers=risk,
        openai_output_optional=openai_payload if openai_payload["used"] else None,
    )
    scenario_rank = assign_scenario_rank(final_score, None, scenario_thresholds, agg)

    decision = {
        "local_score": _clamp01(local_score),
        "final_score": _clamp01(final_score),
        "decision_confidence": decision_confidence,
        "scenario_rank": scenario_rank,
    }

    metrics = {
        "analysis": {
            "label": run_label,
            "start_s": start_s,
            "end_s": end_s,
            "duration_s": max(0.0, end_s - start_s),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "output_dir": str(run_dir),
        },
        "input": {
            "video": str(args.video),
            "weights": str(args.weights),
            "video_duration_seconds": duration_s,
            "fps": fps,
            "frame_size": {"width": frame_w, "height": frame_h},
            "camera_id": args.camera_id,
            "location_type": args.location_type,
            "runtime_mode": runtime_mode,
            "openai_enabled": openai_enabled,
            "openai_unavailable_reason": openai_unavailable_reason or None,
        },
        "sampling": {
            "sample_fps": args.sample_fps,
            "sampled_frames": stats.sampled_frames,
            "confidence_threshold": args.conf,
        },
        "summary": summary,
        "openai": openai_payload,
        "decision": decision,
        "artifacts": {
            "metrics_json": str(run_dir / "metrics.json"),
            "top_fire_frame_jpg": top_fire_frame_path_str,
        },
    }

    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    return metrics


def main() -> None:
    args = parse_args()
    metrics = analyze_video(args)
    print("Analysis complete")
    print(f"Runtime mode: {metrics['input']['runtime_mode']}")
    print(f"JSON: {metrics['artifacts']['metrics_json']}")
    print(f"JPG: {metrics['artifacts']['top_fire_frame_jpg']}")
    print(
        "decision: "
        f"openai_used={metrics['openai']['used']}, "
        f"scenario={metrics['decision']['scenario_rank']}, "
        f"final_score={metrics['decision']['final_score']:.3f}, "
        f"decision_confidence={metrics['decision']['decision_confidence']:.3f}"
    )


if __name__ == "__main__":
    main()
