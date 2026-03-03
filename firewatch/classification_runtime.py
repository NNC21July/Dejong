import json
import subprocess
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any

import cv2
from django.utils import timezone

from .models import Event

BASE_DIR = Path(__file__).resolve().parent.parent


class ClassificationRunError(RuntimeError):
    pass


def _resolve_video_path(explicit_path: str | None = None) -> Path:
    candidates: list[Path] = []
    if explicit_path:
        candidates.append(Path(explicit_path))
    candidates.extend([BASE_DIR / "video.mp4", BASE_DIR / "video.MP4", BASE_DIR / "video.Mp4"])

    for p in candidates:
        if p.exists():
            return p
    raise ClassificationRunError("video.mp4 not found in workspace root. Add the footage file first.")


def _append_notification(event: Event, payload: dict[str, Any]) -> None:
    logs = list(event.authority_notifications_json or [])
    logs.append(payload)
    event.authority_notifications_json = logs


def get_video_duration_seconds(video_path: str | None = None) -> float:
    video = _resolve_video_path(video_path)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return 0.0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    cap.release()
    if fps <= 0:
        return 0.0
    return float(frame_count / fps)


def run_classification_window(
    event: Event,
    start_seconds: float,
    analyze_seconds: float = 10.0,
    sample_fps: float = 2.0,
    conf: float = 0.25,
    location_type: str | None = None,
    video_path: str | None = None,
    camera_id: str = "video_feed",
    device: str = "cpu",
) -> dict[str, Any]:
    video = _resolve_video_path(video_path)
    weights = BASE_DIR / "classification_model.pt"
    script = BASE_DIR / "classification" / "analyze_video.py"
    results_dir = BASE_DIR / "results"

    if not script.exists():
        raise ClassificationRunError("classification/analyze_video.py not found.")
    if not weights.exists():
        raise ClassificationRunError("classification_model.pt not found in workspace root.")

    end_seconds = start_seconds + analyze_seconds
    run_label = f"{event.event_id}_{int(start_seconds):06d}_{int(end_seconds):06d}_{timezone.now().strftime('%H%M%S')}"

    cmd = [
        sys.executable,
        str(script),
        "--video",
        str(video),
        "--weights",
        str(weights),
        "--start-seconds",
        str(start_seconds),
        "--analyze-seconds",
        str(analyze_seconds),
        "--sample-fps",
        str(sample_fps),
        "--conf",
        str(conf),
        "--results-dir",
        str(results_dir),
        "--run-label",
        run_label,
        "--camera-id",
        camera_id or "video_feed",
        "--location-type",
        location_type or event.zone.location_type or "unknown_location",
        "--device",
        device or "cpu",
    ]

    proc = subprocess.run(cmd, cwd=str(BASE_DIR), capture_output=True, text=True)

    metrics_path = results_dir / run_label / "metrics.json"
    if proc.returncode != 0 or not metrics_path.exists():
        raise ClassificationRunError(
            "Classification failed. "
            f"return_code={proc.returncode}; stderr={proc.stderr[-500:]}; stdout={proc.stdout[-500:]}"
        )

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    decision = metrics.get("decision", {})
    summary = metrics.get("summary", {})
    top_fire = summary.get("top_fire_frame", {})
    scenario_rank = decision.get("scenario_rank", "No Fire Risk")
    final_score = float(decision.get("final_score", 0.0))
    decision_conf = float(decision.get("decision_confidence", 0.0))

    run_record = {
        "run_label": run_label,
        "start_seconds": start_seconds,
        "end_seconds": end_seconds,
        "camera_id": camera_id or "video_feed",
        "video": str(video),
        "scenario_rank": scenario_rank,
        "final_score": final_score,
        "decision_confidence": decision_conf,
        "top_fire_frame_path": top_fire.get("path"),
        "top_fire_timestamp_s": top_fire.get("timestamp_s"),
        "fire_frame_score": top_fire.get("fire_frame_score"),
        "metrics_json": str(metrics_path),
        "created_at": timezone.now().isoformat(),
    }

    runs = list(event.classification_runs_json or [])
    previous_scenario = str(runs[-1].get("scenario_rank")) if runs else None
    runs.append(run_record)
    event.classification_runs_json = runs

    # Keep existing score math untouched; these fields are display mapping from classifier output.
    mapped = {
        "No Fire Risk": "no_fire_risk",
        "Elevated Risk": "elevated_risk",
        "Hazard": "hazard",
        "Emergency": "emergency",
    }
    actions = {
        "no_fire_risk": "log_watch",
        "elevated_risk": "notify_tier1_tier2_live_feed",
        "hazard": "notify_tier1_tier2_live_feed",
        "emergency": "dispatch_and_notify_authorities",
    }
    decision_map = {
        "No Fire Risk": "monitor",
        "Elevated Risk": "request_human_verification",
        "Hazard": "request_human_verification",
        "Emergency": "dispatch",
    }

    event.risk_level = mapped.get(scenario_rank, event.risk_level)
    event.risk_action = actions.get(event.risk_level, event.risk_action)
    event.final_risk_score = final_score
    event.decision = decision_map.get(scenario_rank, event.decision)

    now = timezone.now()
    if scenario_rank in {"Elevated Risk", "Hazard"} and event.first_hazard_detected_at is None:
        event.first_hazard_detected_at = now
        first_type = "elevated_first_detection" if scenario_rank == "Elevated Risk" else "hazard_first_detection"
        first_style = "elevated" if scenario_rank == "Elevated Risk" else "hazard"
        _append_notification(
            event,
            {
                "type": first_type,
                "level": scenario_rank,
                "distinct_style": first_style,
                "created_at": now.isoformat(),
                "details": run_record,
            },
        )

    if scenario_rank == "Emergency" and event.first_emergency_detected_at is None:
        event.first_emergency_detected_at = now
        event.emergency_call_pending = True
        event.emergency_call_deadline = now + timedelta(seconds=20)
        event.emergency_call_status = "awaiting_user_escalation"
        _append_notification(
            event,
            {
                "type": "emergency_first_detection",
                "level": "Emergency",
                "distinct_style": "emergency",
                "created_at": now.isoformat(),
                "call_deadline": event.emergency_call_deadline.isoformat(),
                "details": run_record,
            },
        )
        _append_notification(
            event,
            {
                "type": "emergency_evacuation_announcement",
                "level": "Emergency",
                "distinct_style": "emergency",
                "created_at": now.isoformat(),
                "message": f"Announcing evacuation to people in zone {event.zone.code}. Please proceed to nearest safe exit immediately.",
                "status": "in_progress",
                "details": run_record,
            },
        )
        _append_notification(
            event,
            {
                "type": "emergency_evacuation_announcement_completed",
                "level": "Emergency",
                "distinct_style": "emergency",
                "created_at": now.isoformat(),
                "message": f"Evacuation announcement delivered to occupants in zone {event.zone.code}.",
                "status": "completed",
                "details": run_record,
            },
        )

    if previous_scenario and previous_scenario != scenario_rank:
        change_style = "emergency"
        if scenario_rank == "Hazard":
            change_style = "hazard"
        elif scenario_rank == "Elevated Risk":
            change_style = "elevated"
        _append_notification(
            event,
            {
                "type": "scenario_status_change",
                "level": scenario_rank,
                "distinct_style": change_style,
                "created_at": now.isoformat(),
                "from_scenario": previous_scenario,
                "to_scenario": scenario_rank,
                "details": run_record,
            },
        )

    outputs = dict(event.stage_outputs_json or {})
    outputs["classification_last_run"] = run_record
    outputs.setdefault("classification_runs", [])
    outputs["classification_runs"].append(run_record)
    event.stage_outputs_json = outputs

    event.save(
        update_fields=[
            "classification_runs_json",
            "risk_level",
            "risk_action",
            "final_risk_score",
            "decision",
            "first_hazard_detected_at",
            "first_emergency_detected_at",
            "emergency_call_pending",
            "emergency_call_deadline",
            "emergency_call_status",
            "authority_notifications_json",
            "stage_outputs_json",
            "updated_at",
        ]
    )

    return {
        "ok": True,
        "event_id": event.event_id,
        "run": run_record,
        "stdout": proc.stdout[-2000:],
        "stderr": proc.stderr[-2000:],
    }


def resolve_emergency_decision(event: Event, action: str) -> dict[str, Any]:
    now = timezone.now()
    logs = list(event.authority_notifications_json or [])

    if action == "call_now":
        event.emergency_call_pending = False
        event.emergency_call_status = "manual_escalation_requested_telegram_pending"
        logs.append(
            {
                "type": "emergency_escalation_manual_request",
                "created_at": now.isoformat(),
                "note": "Telegram forward module pending integration.",
            }
        )
    elif action == "cancel":
        event.emergency_call_pending = False
        event.emergency_call_status = "user_cancelled"
        logs.append({"type": "emergency_escalation_cancelled", "created_at": now.isoformat()})
    elif action == "auto_timeout":
        event.emergency_call_pending = False
        event.emergency_call_status = "auto_escalation_triggered_telegram_pending"
        logs.append(
            {
                "type": "emergency_escalation_auto_timeout",
                "created_at": now.isoformat(),
                "note": "Auto-escalation placeholder only. Telegram forward module pending import.",
            }
        )
    else:
        raise ValueError("Invalid emergency action")

    event.authority_notifications_json = logs
    event.save(update_fields=["emergency_call_pending", "emergency_call_status", "authority_notifications_json", "updated_at"])
    return {
        "ok": True,
        "event_id": event.event_id,
        "action": action,
        "emergency_call_status": event.emergency_call_status,
    }
