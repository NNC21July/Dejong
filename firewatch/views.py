import json
import random
import re
import threading
import time
import urllib.error
import urllib.request
from datetime import timedelta
from io import BytesIO
from urllib.parse import quote, urlencode
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
from django.contrib import messages
from django.conf import settings
from django.db import transaction
from django.http import FileResponse, Http404, HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_http_methods

from .classification_runtime import (
    ClassificationRunError,
    get_video_duration_seconds,
    resolve_emergency_decision,
    run_classification_window,
)
from .models import Camera, Building, Event, EventAction, FrameDetection, RiskConfig, Zone
from .pathfinding import astar_path, compute_routes_for_event, get_or_create_layout, normalize_layout
from .services import (
    aggregate_temporal_evidence,
    apply_advisory_reasoner,
    build_context_package,
    build_response_packet,
    map_risk_level,
    parse_iso_dt,
    score_event,
    stage3_ingest_frame_detection,
)


def _json_body(request: HttpRequest) -> dict:
    try:
        return json.loads(request.body.decode("utf-8"))
    except Exception:
        return {}


BASE_DIR = Path(__file__).resolve().parent.parent
MONITOR_LOCK = threading.Lock()
EVENT_MONITORS: dict[str, dict] = {}
CLIP_LOCK = threading.Lock()


def _monitor_loop(event_id: str) -> None:
    while True:
        with MONITOR_LOCK:
            state = EVENT_MONITORS.get(event_id)
            if not state or not state.get("enabled"):
                break
            next_start = float(state.get("next_start_seconds", 0.0))
            interval = float(state.get("interval_seconds", 5.0))
            camera_id = str(state.get("camera_id", "video_feed"))
            video_duration = float(state.get("video_duration", 0.0))

        try:
            event = Event.objects.select_related("zone").get(event_id=event_id)
            run_classification_window(
                event=event,
                start_seconds=next_start,
                analyze_seconds=interval,
                sample_fps=2.0,
                conf=0.25,
                location_type=event.zone.location_type,
                camera_id=camera_id,
                device="cpu",
            )
            new_next = next_start + interval
            reached_end = video_duration > 0 and new_next >= video_duration
            with MONITOR_LOCK:
                if event_id in EVENT_MONITORS:
                    EVENT_MONITORS[event_id]["last_tick"] = timezone.now().isoformat()
                    EVENT_MONITORS[event_id]["last_error"] = ""
                    if reached_end:
                        EVENT_MONITORS[event_id]["enabled"] = False
                    else:
                        EVENT_MONITORS[event_id]["next_start_seconds"] = new_next
            if reached_end:
                break
        except Exception as exc:
            with MONITOR_LOCK:
                if event_id in EVENT_MONITORS:
                    EVENT_MONITORS[event_id]["last_error"] = str(exc)
            time.sleep(2.0)
            continue

        # Keep processing interval-by-interval in background.
        time.sleep(0.6)


def _save_stage_output(event: Event, stage_key: str, payload: dict) -> None:
    outputs = dict(event.stage_outputs_json or {})
    outputs[stage_key] = payload
    event.stage_outputs_json = outputs
    event.save(update_fields=["stage_outputs_json", "updated_at"])


def _build_flow_validation(event: Event) -> dict:
    outputs = event.stage_outputs_json or {}
    required = [
        "stage1_event_trigger",
        "stage2_camera_selection",
        "stage3_yolo_detection",
        "stage4_temporal_aggregation",
        "stage5_context_package",
        "stage7_risk_score",
        "stage8_response_packet",
    ]
    stages = {}
    for key in required:
        stages[key] = {"ok": key in outputs, "has_output": key in outputs}

    if "stage3_yolo_detection" in outputs:
        ingested = int(outputs["stage3_yolo_detection"].get("ingested_frames", 0))
        stages["stage3_yolo_detection"]["ok"] = ingested > 0
        stages["stage3_yolo_detection"]["ingested_frames"] = ingested

    all_ok = all(v["ok"] for v in stages.values())
    summary = {"all_required_stages_ok": all_ok, "stages": stages}
    event.flow_validation_json = summary
    event.save(update_fields=["flow_validation_json", "updated_at"])
    return summary


def _latest_classification(event: Event) -> dict:
    latest = event.frame_detections.select_related("camera").order_by("-timestamp", "-frame_index").first()
    if not latest:
        return {"available": False}

    best = None
    for det in latest.detections_json:
        conf = float(det.get("confidence", 0.0))
        if not best or conf > float(best.get("confidence", 0.0)):
            best = det

    class_map = {0: "fire", 1: "smoke", 2: "controlled_flame"}
    class_name = "none"
    confidence = 0.0
    bbox = []
    if best:
        class_name = best.get("class_name") or class_map.get(int(best.get("class_id", -1)), "unknown")
        confidence = float(best.get("confidence", 0.0))
        bbox = best.get("bbox_xyxy", [])

    cam = latest.camera
    return {
        "available": True,
        "camera_id": cam.camera_id,
        "camera_rtsp_url": cam.rtsp_url,
        "camera_default_fps": cam.default_fps,
        "camera_active": cam.active,
        "zone": cam.zone.code,
        "frame_index": latest.frame_index,
        "timestamp": latest.timestamp.isoformat(),
        "top_classification": class_name,
        "confidence": round(confidence, 4),
        "bbox_xyxy": bbox,
        "detections_count": len(latest.detections_json or []),
    }


def _risk_presentation(event: Event) -> dict:
    level = event.risk_level or map_risk_level(event.final_risk_score)[0]
    action = event.risk_action or map_risk_level(event.final_risk_score)[1]
    label_map = {
        "no_fire_risk": "No Fire Risk",
        "elevated_risk": "Elevated Risk",
        "hazard": "Hazard",
        "emergency": "Emergency",
    }
    action_map = {
        "no_fire_risk": "Log + Watch",
        "elevated_risk": "Live Feed + Tier 1 + Tier 2 Notify",
        "hazard": "Live Feed + Tier 1 + Tier 2 Notify",
        "emergency": "Escalate To Authorities",
    }
    return {
        "risk_level": level,
        "risk_label": label_map[level],
        "risk_action": action,
        "risk_action_label": action_map[level],
        "confidence_percent": round(event.final_risk_score * 100, 1),
        "matches_decision": (
            (level == "emergency" and event.decision == "dispatch")
            or (level in {"hazard", "elevated_risk"} and event.decision == "request_human_verification")
            or (level == "no_fire_risk" and event.decision == "monitor")
        ),
    }


def _event_alert_camera(event: Event) -> str:
    latest = event.frame_detections.select_related("camera").order_by("-timestamp", "-frame_index").first()
    if latest:
        return latest.camera.camera_id

    runs = event.classification_runs_json or []
    if runs:
        return str(runs[-1].get("camera_id", "video_feed"))

    response_packet = event.response_packet_json or {}
    cam = ((response_packet.get("response_packet") or {}).get("camera_id")) if isinstance(response_packet, dict) else ""
    if cam:
        return str(cam)

    fallback = event.zone.cameras.filter(active=True).first()
    return fallback.camera_id if fallback else ""


def _event_footage_focus(event: Event) -> dict:
    runs = event.classification_runs_json or []
    if runs:
        last = runs[-1]
        return {
            "detection_start_s": float(last.get("start_seconds") or 0.0),
            "focus_time_s": float(last.get("top_fire_timestamp_s") or 0.0),
            "confidence": float(last.get("decision_confidence") or 0.0),
            "score": float(last.get("final_score") or 0.0),
            "scenario_rank": str(last.get("scenario_rank") or ""),
            "camera_id": str(last.get("camera_id") or _event_alert_camera(event) or "video_feed"),
            "frame_hint": "classification_top_frame",
        }
    latest = _latest_classification(event)
    if latest.get("available"):
        return {
            "detection_start_s": 0.0,
            "focus_time_s": 0.0,
            "confidence": float(latest.get("confidence") or 0.0),
            "score": float(event.final_risk_score),
            "scenario_rank": _risk_presentation(event)["risk_label"],
            "camera_id": str(latest.get("camera_id") or _event_alert_camera(event) or "video_feed"),
            "frame_hint": int(latest.get("frame_index") or 0),
        }
    return {
        "detection_start_s": 0.0,
        "focus_time_s": 0.0,
        "confidence": float(event.final_risk_score),
        "score": float(event.final_risk_score),
        "scenario_rank": _risk_presentation(event)["risk_label"],
        "camera_id": _event_alert_camera(event) or "video_feed",
        "frame_hint": "n/a",
    }


def _time_since_alarm_text(event: Event) -> str:
    alarm_at = event.first_hazard_detected_at or event.first_emergency_detected_at or event.trigger_time
    if not alarm_at:
        return "N/A"
    delta = timezone.now() - alarm_at
    total = int(max(delta.total_seconds(), 0))
    mins, secs = divmod(total, 60)
    hrs, mins = divmod(mins, 60)
    if hrs > 0:
        return f"{hrs}h {mins}m {secs}s"
    if mins > 0:
        return f"{mins}m {secs}s"
    return f"{secs}s"


def _alarm_started_at(event: Event):
    return event.first_hazard_detected_at or event.first_emergency_detected_at or event.trigger_time


def _latest_emergency_notification(event: Event) -> dict:
    logs = list(event.authority_notifications_json or [])
    for log in reversed(logs):
        if log.get("type") == "emergency_first_detection":
            return log
    return {}


def _latest_evacuation_announcement(event: Event) -> dict:
    logs = list(event.authority_notifications_json or [])
    for log in reversed(logs):
        if log.get("type") in {"emergency_evacuation_announcement_completed", "emergency_evacuation_announcement"}:
            return log
    return {}


def _build_whatsapp_payload(event: Event, phone_number: str) -> dict:
    classification = _latest_classification(event)
    risk = _risk_presentation(event)
    text = (
        f"FIREWATCH ALERT\n"
        f"Event: {event.event_id}\n"
        f"Risk: {risk['risk_label']} ({risk['confidence_percent']}%)\n"
        f"Decision: {event.decision}\n"
        f"Zone: {event.zone.code}\n"
        f"Camera: {classification.get('camera_id', 'N/A')}\n"
        f"Frame: {classification.get('frame_index', 'N/A')} at {classification.get('timestamp', 'N/A')}\n"
        f"Please verify and respond immediately."
    )
    return {
        "target_phone": phone_number,
        "message_text": text,
        "whatsapp_url": f"https://wa.me/{phone_number}?text={quote(text)}",
        "sms_formal_reference": "70995",
        "created_at": timezone.now().isoformat(),
    }


def _source_video_path() -> Path:
    candidates = [BASE_DIR / "video.mp4", BASE_DIR / "video.MP4", BASE_DIR / "video.Mp4"]
    video = next((p for p in candidates if p.exists()), None)
    if not video:
        raise Http404("video.mp4 not found in workspace root")
    return video


def _fourcc_to_text(value: int) -> str:
    return "".join(chr((value >> (8 * i)) & 0xFF) for i in range(4)).strip().lower()


def _clip_is_readable(path: Path) -> bool:
    if not path.exists() or path.stat().st_size <= 0:
        return False
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        return False
    ok, _ = cap.read()
    cap.release()
    return bool(ok)


def _build_5s_clip_for_event(event: Event, start_seconds: float, duration_seconds: float = 5.0) -> Path:
    source_video = _source_video_path()
    clips_dir = BASE_DIR / "runtime_clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    safe_event_id = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in event.event_id)
    stamp = int(source_video.stat().st_mtime)
    requested_start_seconds = max(0.0, float(start_seconds))
    duration_ms = int(max(1.0, duration_seconds) * 1000.0)
    codec_tag = "avc1"
    with CLIP_LOCK:
        cap = cv2.VideoCapture(str(source_video))
        if not cap.isOpened():
            raise Http404("Unable to open source video")

        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0:
            fps = 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720

        frame_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_total = (frame_count / fps) if (fps > 0 and frame_count > 0) else 0.0
        max_start = max(0.0, duration_total - duration_seconds) if duration_total > 0 else requested_start_seconds
        clamped_start = min(requested_start_seconds, max_start)
        start_ms = int(clamped_start * 1000.0)

        clip_path = clips_dir / f"{safe_event_id}_{start_ms}_{duration_ms}_{codec_tag}_{stamp}.mp4"
        if _clip_is_readable(clip_path):
            cap.release()
            return clip_path

        # Load frame window first so we can retry with different codecs deterministically.
        cap.set(cv2.CAP_PROP_POS_MSEC, float(start_ms))
        max_frames = max(1, int(round(duration_seconds * fps)))
        frames: list = []
        while len(frames) < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
        cap.release()

        # Retry frame extraction once from start of video if chosen interval had no frames.
        if not frames:
            cap_retry = cv2.VideoCapture(str(source_video))
            cap_retry.set(cv2.CAP_PROP_POS_MSEC, 0.0)
            while len(frames) < max_frames:
                ok, frame = cap_retry.read()
                if not ok:
                    break
                frames.append(frame)
            cap_retry.release()

        if not frames:
            raise Http404("No frames available for 5-second clip")

        for candidate in ("avc1", "H264", "X264", "mp4v"):
            temp_path = clips_dir / f"{clip_path.stem}_{candidate}.tmp.mp4"
            writer = cv2.VideoWriter(
                str(temp_path),
                cv2.VideoWriter_fourcc(*candidate),
                fps,
                (width, height),
            )
            if not writer.isOpened():
                writer.release()
                if temp_path.exists():
                    temp_path.unlink()
                continue

            for frame in frames:
                writer.write(frame)
            writer.release()

            if not _clip_is_readable(temp_path):
                if temp_path.exists():
                    temp_path.unlink()
                continue

            # Validate codec family if possible (prefer h264/avc1).
            probe = cv2.VideoCapture(str(temp_path))
            clip_fourcc = _fourcc_to_text(int(probe.get(cv2.CAP_PROP_FOURCC)))
            probe.release()
            is_h264_family = clip_fourcc in {"h264", "avc1"}
            if candidate != "mp4v" and not is_h264_family:
                if temp_path.exists():
                    temp_path.unlink()
                continue

            if clip_path.exists():
                clip_path.unlink()
            temp_path.replace(clip_path)
            return clip_path

        raise Http404("Unable to generate browser-compatible 5-second clip")
    return clip_path


def _render_route_map_image(event: Event, route_data: dict) -> bytes:
    layout = (route_data or {}).get("layout") or {}
    cells = layout.get("cells") or []
    rows = int(layout.get("rows") or len(cells) or 20)
    cols = int(layout.get("cols") or (len(cells[0]) if cells else 20) or 20)
    if not isinstance(cells, list) or not cells:
        layout_obj = get_or_create_layout(event.zone)
        cells = normalize_layout(layout_obj.rows, layout_obj.cols, layout_obj.cells_json)
        rows = layout_obj.rows
        cols = layout_obj.cols

    cell_px = 28
    margin = 20
    img_h = rows * cell_px + margin * 2
    img_w = cols * cell_px + margin * 2
    img = np.full((img_h, img_w, 3), 255, dtype=np.uint8)

    colors = {
        "empty": (245, 245, 245),
        "wall": (55, 55, 55),
        "stairs": (209, 231, 255),
        "entrance": (179, 255, 179),
        "fire": (153, 153, 255),
    }

    for r in range(rows):
        row = cells[r] if r < len(cells) and isinstance(cells[r], list) else []
        for c in range(cols):
            t = row[c] if c < len(row) else "empty"
            tl = (margin + c * cell_px, margin + r * cell_px)
            br = (tl[0] + cell_px - 1, tl[1] + cell_px - 1)
            cv2.rectangle(img, tl, br, colors.get(t, colors["empty"]), -1)
            cv2.rectangle(img, tl, br, (215, 215, 215), 1)

    primary = ((route_data or {}).get("primary_route") or {}).get("path") or []
    points = []
    for p in primary:
        if isinstance(p, list) and len(p) == 2:
            rr, cc = int(p[0]), int(p[1])
            points.append((margin + cc * cell_px + cell_px // 2, margin + rr * cell_px + cell_px // 2))

    if len(points) >= 2:
        cv2.polylines(img, [np.array(points, dtype=np.int32)], False, (0, 0, 255), thickness=4)
    for pt in points:
        cv2.circle(img, pt, 4, (0, 0, 200), -1)

    entrance = (route_data or {}).get("entrance_cell")
    target = (route_data or {}).get("target_cell")
    if isinstance(entrance, list) and len(entrance) == 2:
        ept = (margin + int(entrance[1]) * cell_px + cell_px // 2, margin + int(entrance[0]) * cell_px + cell_px // 2)
        cv2.circle(img, ept, 8, (0, 180, 0), -1)
        cv2.putText(img, "ENTRANCE", (ept[0] + 10, ept[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 100, 0), 1)

    if isinstance(target, list) and len(target) == 2:
        tpt = (margin + int(target[1]) * cell_px + cell_px // 2, margin + int(target[0]) * cell_px + cell_px // 2)
        cv2.circle(img, tpt, 8, (0, 0, 255), -1)
        cv2.putText(img, "FIRE", (tpt[0] + 10, tpt[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 140), 1)

    cv2.putText(
        img,
        f"Event {event.event_id} | Zone {event.zone.code} | A* Route",
        (margin, max(16, margin - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (40, 40, 40),
        1,
    )
    ok, encoded = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("Failed to render route map image")
    return encoded.tobytes()


def _telegram_post(method: str, fields: dict, files: dict[str, tuple[str, bytes, str]] | None = None) -> dict:
    token = str(getattr(settings, "TELEGRAM_BOT_TOKEN", "") or "").strip()
    if not token:
        return {"ok": False, "error": "TELEGRAM_BOT_TOKEN not configured"}
    url = f"https://api.telegram.org/bot{token}/{method}"

    if files:
        boundary = f"----firewatch{uuid4().hex}"
        body = BytesIO()
        for k, v in fields.items():
            body.write(f"--{boundary}\r\n".encode())
            body.write(f'Content-Disposition: form-data; name="{k}"\r\n\r\n'.encode())
            body.write(str(v).encode())
            body.write(b"\r\n")
        for field, (name, data, mime) in files.items():
            body.write(f"--{boundary}\r\n".encode())
            body.write(
                f'Content-Disposition: form-data; name="{field}"; filename="{name}"\r\n'.encode()
            )
            body.write(f"Content-Type: {mime}\r\n\r\n".encode())
            body.write(data)
            body.write(b"\r\n")
        body.write(f"--{boundary}--\r\n".encode())
        req = urllib.request.Request(url, data=body.getvalue(), method="POST")
        req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    else:
        encoded = urlencode({k: str(v) for k, v in fields.items()}).encode("utf-8")
        req = urllib.request.Request(url, data=encoded, method="POST")
        req.add_header("Content-Type", "application/x-www-form-urlencoded")

    try:
        with urllib.request.urlopen(req, timeout=12) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        return {"ok": False, "error": str(exc)}


def _resolve_telegram_chat_id(telegram_username: str = "") -> str:
    fallback_chat_id = str(getattr(settings, "TELEGRAM_CHAT_ID", "") or "").strip()
    configured_username = str(getattr(settings, "TELEGRAM_USERNAME", "") or "").strip()
    username = str(telegram_username or configured_username).strip().lstrip("@").lower()
    if not username:
        return fallback_chat_id

    updates = _telegram_post("getUpdates", {})
    if not bool(updates.get("ok")):
        return fallback_chat_id

    for item in updates.get("result", []):
        message = item.get("message") or {}
        chat = message.get("chat") or {}
        if str(chat.get("username") or "").lower() == username and chat.get("id") is not None:
            return str(chat.get("id"))

    return fallback_chat_id


def _route_data_for_notification(event: Event) -> dict:
    default_route = compute_routes_for_event(event)
    sim = (event.stage_outputs_json or {}).get("sim_fire_seed") or {}
    fire_cell = sim.get("fire_cell")
    if not (isinstance(fire_cell, list) and len(fire_cell) == 2):
        return default_route

    layout = get_or_create_layout(event.zone)
    cells = normalize_layout(layout.rows, layout.cols, layout.cells_json)
    try:
        target = (int(fire_cell[0]), int(fire_cell[1]))
    except Exception:
        return default_route

    entrances = []
    for r in range(layout.rows):
        for c in range(layout.cols):
            if cells[r][c] == "entrance":
                entrances.append((r, c))
    if not entrances:
        entrances = [(0, 0)]

    best = None
    start = None
    for ent in entrances:
        route = astar_path(layout.rows, layout.cols, cells, ent, target)
        if not route:
            continue
        if best is None or float(route.get("cost", 1e9)) < float(best.get("cost", 1e9)):
            best = route
            start = ent

    if not best:
        default_route["target_cell"] = [target[0], target[1]]
        return default_route

    return {
        "event_id": event.event_id,
        "zone": event.zone.code,
        "target_type": "sim_fire_seed",
        "target_camera_id": None,
        "target_cell": [target[0], target[1]],
        "entrance_cell": [start[0], start[1]] if start else None,
        "primary_route": best,
        "alternative_routes": [],
        "blocked_cells": [],
        "layout": {
            "rows": layout.rows,
            "cols": layout.cols,
            "cells": cells,
            "camera_points": layout.camera_points_json,
        },
    }


def _send_telegram_authority_notification(event: Event, mode: str) -> dict:
    configured_username = str(getattr(settings, "TELEGRAM_USERNAME", "") or "").strip()
    chat_id = _resolve_telegram_chat_id(configured_username)
    if not chat_id:
        return {"ok": False, "error": "TELEGRAM_CHAT_ID not configured"}

    risk = _risk_presentation(event)
    cls = _latest_classification(event)
    focus = _event_footage_focus(event)
    route_data = _route_data_for_notification(event)
    if "error" in route_data:
        route_summary = f"Route error: {route_data['error']}"
    else:
        pr = route_data.get("primary_route") or {}
        route_summary = (
            f"Entrance {route_data.get('entrance_cell')} -> Fire {route_data.get('target_cell')} | "
            f"cost={pr.get('cost')} | steps={len(pr.get('path') or [])}"
        )

    escalation_origin = "manual" if mode == "manual" else "system escalated"

    text = (
        "🔥 FIREWATCH ESCALATION\n"
        f"Risk: {risk['risk_label']}\n"
        f"Decision: {escalation_origin}\n"
        f"Zone: {event.zone.code}\n"
        f"Camera ID: {focus.get('camera_id') or cls.get('camera_id') or 'N/A'}"
    )

    results = {"message": _telegram_post("sendMessage", {"chat_id": chat_id, "text": text})}

    try:
        start_at = max(0.0, float(focus.get("detection_start_s", 0.0)))
        clip_path = _build_5s_clip_for_event(event, start_at, duration_seconds=5.0)
        video_bytes = clip_path.read_bytes()
        results["video"] = _telegram_post(
            "sendVideo",
            {"chat_id": chat_id, "caption": f"Event {event.event_id} clip ({mode})"},
            files={"video": (clip_path.name, video_bytes, "video/mp4")},
        )
    except Exception as exc:
        results["video"] = {"ok": False, "error": str(exc)}

    try:
        map_png = _render_route_map_image(event, route_data)
        results["map"] = _telegram_post(
            "sendPhoto",
            {"chat_id": chat_id, "caption": f"A* path to fire for {event.event_id}"},
            files={"photo": (f"{event.event_id}_astar_map.png", map_png, "image/png")},
        )
    except Exception as exc:
        results["map"] = {"ok": False, "error": str(exc)}

    ok = all(bool((results.get(k) or {}).get("ok")) for k in ["message", "video", "map"])
    return {"ok": ok, "results": results, "route_data": route_data}


def _dashboard_live_payload() -> dict:
    zones = Zone.objects.select_related("building").all().order_by("building__name", "code")
    zone_overviews = []
    active_notifications = []
    seen_notification_keys: set[tuple[str, str, str, str]] = set()

    def add_notification(item: dict) -> None:
        key = (
            str(item.get("event_id", "")),
            str(item.get("risk_label", "")),
            str(item.get("zone", "")),
            str(item.get("camera_id", "")),
        )
        if key in seen_notification_keys:
            return
        seen_notification_keys.add(key)
        active_notifications.append(item)
    danger_events = list(
        Event.objects.select_related("zone")
        .filter(risk_level__in=["elevated_risk", "hazard", "emergency"])
        .order_by("-updated_at")[:50]
    )

    danger_by_zone: dict[int, Event] = {}
    for e in danger_events:
        if e.zone_id not in danger_by_zone:
            danger_by_zone[e.zone_id] = e

    pending_by_zone: dict[int, dict] = {}
    recent_seeded = Event.objects.select_related("zone").order_by("-created_at")[:120]
    for ev in recent_seeded:
        if ev.zone_id in pending_by_zone:
            continue
        stage = ev.stage_outputs_json or {}
        sim = stage.get("sim_fire_seed") or {}
        fire_cell = sim.get("fire_cell")
        if isinstance(fire_cell, list) and len(fire_cell) == 2:
            pending_by_zone[ev.zone_id] = {
                "event_id": ev.event_id,
                "fire_cell": [int(fire_cell[0]), int(fire_cell[1])],
            }

    for zone in zones:
        layout = get_or_create_layout(zone)
        cells = normalize_layout(layout.rows, layout.cols, layout.cells_json)
        alert_event = danger_by_zone.get(zone.id)
        alert_camera = _event_alert_camera(alert_event) if alert_event else ""
        zone_overviews.append(
            {
                "zone_id": zone.id,
                "zone_code": zone.code,
                "building_name": zone.building.name,
                "rows": layout.rows,
                "cols": layout.cols,
                "cells": cells,
                "camera_points": layout.camera_points_json,
                "alert_camera": alert_camera,
                "alert_level": alert_event.risk_level if alert_event else "",
                "alert_event_id": alert_event.event_id if alert_event else "",
                "alert_redirect_url": (
                    reverse("firewatch:event_footage_view", kwargs={"event_id": alert_event.event_id}) if alert_event else ""
                ),
                "pending_event_id": (pending_by_zone.get(zone.id) or {}).get("event_id", ""),
                "pending_fire_cell": (pending_by_zone.get(zone.id) or {}).get("fire_cell"),
                "edit_url": reverse("firewatch:zone_layout_editor", kwargs={"zone_id": zone.id}),
            }
        )

    for ev in danger_events:
        risk = _risk_presentation(ev)
        focus = _event_footage_focus(ev)
        add_notification(
            {
                "event_id": ev.event_id,
                "zone": ev.zone.code,
                "camera_id": focus["camera_id"] or _event_alert_camera(ev) or "video_feed",
                "risk_label": risk["risk_label"],
                "detail_url": reverse("firewatch:event_footage_view", kwargs={"event_id": ev.event_id}),
            }
        )

    recent_events = Event.objects.select_related("zone").order_by("-updated_at")[:80]
    for ev in recent_events:
        logs = list(ev.authority_notifications_json or [])
        for log in reversed(logs):
            if log.get("type") != "scenario_status_change":
                continue
            details = log.get("details") or {}
            add_notification(
                {
                    "event_id": ev.event_id,
                    "zone": ev.zone.code,
                    "camera_id": details.get("camera_id") or _event_alert_camera(ev) or "video_feed",
                    "risk_label": log.get("to_scenario") or ev.risk_level,
                    "detail_url": reverse("firewatch:event_footage_view", kwargs={"event_id": ev.event_id}),
                }
            )
            break

    return {
        "zone_overviews": zone_overviews,
        "active_notifications": active_notifications[:40],
        "generated_at": timezone.now().isoformat(),
    }


def _video_file_response(request: HttpRequest, file_path: Path) -> HttpResponse:
    data = file_path.read_bytes()
    response = HttpResponse(data, content_type="video/mp4")
    response["Content-Length"] = str(len(data))
    response["Accept-Ranges"] = "none"
    response["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response["Pragma"] = "no-cache"
    response["Expires"] = "0"
    return response


@require_GET
def health(request: HttpRequest) -> JsonResponse:
    return JsonResponse({"status": "ok", "service": "firewatch"})


@csrf_exempt
@require_http_methods(["POST"])
def stage1_event_trigger(request: HttpRequest) -> JsonResponse:
    payload = _json_body(request)

    trigger_type = payload.get("trigger_type", "manual")
    sensor_id = payload.get("sensor_id", "")
    zone_code = payload.get("zone")
    if not zone_code:
        return JsonResponse({"error": "zone is required"}, status=400)

    zone = Zone.objects.filter(code=zone_code).first()
    if not zone:
        building, _ = Building.objects.get_or_create(name="Default Building")
        zone = Zone.objects.create(code=zone_code, building=building, location_type="corridor")

    event_id = payload.get("event_id") or timezone.now().strftime("evt_%Y%m%d_%H%M%S")
    if Event.objects.filter(event_id=event_id).exists():
        event_id = f"{event_id}_{Event.objects.count() + 1}"

    event = Event.objects.create(
        event_id=event_id,
        trigger_type=trigger_type,
        sensor_id=sensor_id,
        trigger_time=parse_iso_dt(payload.get("trigger_time")),
        zone=zone,
    )

    response = {
        "event_id": event.event_id,
        "trigger_type": event.trigger_type,
        "sensor_id": event.sensor_id,
        "trigger_time": event.trigger_time.isoformat(),
        "zone": event.zone.code,
    }
    _save_stage_output(event, "stage1_event_trigger", response)
    return JsonResponse(response, status=201)


@csrf_exempt
@require_http_methods(["POST"])
def stage2_camera_selection(request: HttpRequest) -> JsonResponse:
    payload = _json_body(request)
    event = get_object_or_404(Event, event_id=payload.get("event_id", ""))

    cameras = list(event.zone.cameras.filter(active=True))
    if not cameras:
        cameras = [Camera.objects.create(camera_id=f"{event.zone.code}_CAM1", zone=event.zone, default_fps=10)]

    sampling_fps = int(payload.get("sampling_fps", 10))
    capture_seconds = int(payload.get("capture_seconds", 15))
    num_frames = max(1, sampling_fps * capture_seconds)

    now = timezone.now()
    out_cameras = []
    out_frames = {}

    for camera in cameras:
        out_cameras.append({"camera_id": camera.camera_id, "zone": event.zone.code, "fps": sampling_fps})
        frame_indices = list(range(num_frames))
        timestamps = [
            (now + timedelta(milliseconds=(i * 1000 / max(sampling_fps, 1)))).isoformat()
            for i in frame_indices
        ]
        frame_paths = [f"frames/{camera.camera_id}_{i:04d}.jpg" for i in frame_indices]
        out_frames[camera.camera_id] = {
            "frame_indices": frame_indices,
            "timestamps": timestamps,
            "frame_paths": frame_paths,
        }

    response = {"event_id": event.event_id, "cameras": out_cameras, "frames": out_frames}
    _save_stage_output(event, "stage2_camera_selection", response)
    return JsonResponse(response)


@csrf_exempt
@require_http_methods(["POST"])
def stage3_yolo_detection_ingest(request: HttpRequest) -> JsonResponse:
    payload = _json_body(request)
    required = ["event_id", "camera_id", "frame_index"]
    missing = [k for k in required if k not in payload]
    if missing:
        return JsonResponse({"error": f"missing fields: {', '.join(missing)}"}, status=400)

    frame = stage3_ingest_frame_detection(payload)

    response = {
        "event_id": frame.event.event_id,
        "camera_id": frame.camera.camera_id,
        "frame_index": frame.frame_index,
        "timestamp": frame.timestamp.isoformat(),
        "detections": frame.detections_json,
    }

    ingested_count = frame.event.frame_detections.count()
    _save_stage_output(
        frame.event,
        "stage3_yolo_detection",
        {
            "ingested_frames": ingested_count,
            "last_frame": response,
        },
    )
    return JsonResponse(response, status=201)


@csrf_exempt
@require_http_methods(["POST"])
def stage4_temporal_aggregation(request: HttpRequest) -> JsonResponse:
    payload = _json_body(request)
    event = get_object_or_404(Event, event_id=payload.get("event_id", ""))
    camera = get_object_or_404(Camera, camera_id=payload.get("camera_id", ""))
    window_seconds = int(payload.get("window_seconds", 15))

    summary = aggregate_temporal_evidence(event, camera, window_seconds=window_seconds)

    if summary.best_fire_frame is not None:
        event.keyframe_path = f"frames/{camera.camera_id}_{summary.best_fire_frame:04d}.jpg"
        event.annotated_frame_path = f"annotated/{camera.camera_id}_{summary.best_fire_frame:04d}.jpg"
        event.save(update_fields=["keyframe_path", "annotated_frame_path", "updated_at"])

    response = {
        "event_id": event.event_id,
        "camera_id": camera.camera_id,
        "window_seconds": window_seconds,
        "summary": {
            "fire_frames": summary.fire_frames,
            "smoke_frames": summary.smoke_frames,
            "controlled_flame_frames": summary.controlled_flame_frames,
            "fire_persistence": round(summary.fire_persistence, 4),
            "max_fire_conf": round(summary.max_fire_conf, 4),
            "mean_fire_conf": round(summary.mean_fire_conf, 4),
        },
        "trends": {
            "fire_bbox_area_growth_rate": round(summary.fire_bbox_area_growth_rate, 4),
            "smoke_conf_trend": summary.smoke_conf_trend,
        },
        "keyframes": {
            "best_fire_frame": summary.best_fire_frame,
            "best_smoke_frame": summary.best_smoke_frame,
            "crop_paths": [f"crops/fire_{summary.best_fire_frame}.jpg"] if summary.best_fire_frame is not None else [],
        },
    }
    _save_stage_output(event, "stage4_temporal_aggregation", response)
    return JsonResponse(response)


@csrf_exempt
@require_http_methods(["POST"])
def stage5_context_package(request: HttpRequest) -> JsonResponse:
    payload = _json_body(request)
    event = get_object_or_404(Event, event_id=payload.get("event_id", ""))
    response = build_context_package(event)
    _save_stage_output(event, "stage5_context_package", response)
    return JsonResponse(response)


@csrf_exempt
@require_http_methods(["POST"])
def stage6_advisory_reasoner(request: HttpRequest) -> JsonResponse:
    payload = _json_body(request)
    event = get_object_or_404(Event, event_id=payload.get("event_id", ""))

    advisory = payload.get("advisory") or {
        "scenario": "uncertain",
        "risk_level": "medium",
        "threat_score": event.max_fire_conf,
        "confidence": 0.5,
        "rationale": ["Advisory placeholder until external reasoner is connected."],
        "recommended_action": "request_human_check",
    }

    apply_advisory_reasoner(event, advisory)
    response = {
        "scenario": advisory.get("scenario", "uncertain"),
        "risk_level": advisory.get("risk_level", "medium"),
        "threat_score": advisory.get("threat_score", 0.0),
        "confidence": advisory.get("confidence", 0.0),
        "rationale": advisory.get("rationale", []),
        "recommended_action": advisory.get("recommended_action", "request_human_check"),
    }
    _save_stage_output(event, "stage6_advisory_reasoner", response)
    return JsonResponse(response)


@csrf_exempt
@require_http_methods(["POST"])
def stage7_risk_score(request: HttpRequest) -> JsonResponse:
    payload = _json_body(request)
    event = get_object_or_404(Event, event_id=payload.get("event_id", ""))
    config_name = payload.get("risk_config", "default")
    config = RiskConfig.objects.filter(name=config_name).first()
    response = score_event(event, config)
    _save_stage_output(event, "stage7_risk_score", response)
    return JsonResponse(response)


@csrf_exempt
@require_http_methods(["POST"])
def stage8_response_packet(request: HttpRequest) -> JsonResponse:
    payload = _json_body(request)
    event = get_object_or_404(Event, event_id=payload.get("event_id", ""))
    camera = None
    camera_id = payload.get("camera_id")
    if camera_id:
        camera = Camera.objects.filter(camera_id=camera_id).first()
    response = build_response_packet(event, camera)
    _save_stage_output(event, "stage8_response_packet", response)
    return JsonResponse(response)


@require_GET
def dashboard_home(request: HttpRequest) -> HttpResponse:
    status_filter = request.GET.get("status", "")
    zone_filter = request.GET.get("zone", "")
    camera_filter = request.GET.get("camera", "")

    events = Event.objects.select_related("zone", "zone__building").all()
    if status_filter:
        events = events.filter(status=status_filter)
    if zone_filter:
        events = events.filter(zone__code__icontains=zone_filter)
    if camera_filter:
        events = events.filter(zone__cameras__camera_id__icontains=camera_filter).distinct()

    dashboard_live = _dashboard_live_payload()
    for event in events[:200]:
        if not event.risk_level:
            level, action = map_risk_level(event.final_risk_score)
            event.risk_level = level
            event.risk_action = action

    context = {
        "events": events[:200],
        "status_filter": status_filter,
        "zone_filter": zone_filter,
        "camera_filter": camera_filter,
        "statuses": [k for k, _ in Event.STATUS],
        "zones": Zone.objects.select_related("building").all().order_by("building__name", "code"),
        "zone_overviews_json": json.dumps(dashboard_live["zone_overviews"]),
        "active_notifications": dashboard_live["active_notifications"],
        "active_notifications_json": json.dumps(dashboard_live["active_notifications"]),
        "new_event_id": request.GET.get("new_event", ""),
    }
    return render(request, "firewatch/dashboard_home.html", context)


@require_GET
def dashboard_live_state_api(request: HttpRequest) -> JsonResponse:
    response = JsonResponse(_dashboard_live_payload())
    response["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response["Pragma"] = "no-cache"
    response["Expires"] = "0"
    return response


@require_GET
def zone_layout_editor(request: HttpRequest, zone_id: int) -> HttpResponse:
    zone = get_object_or_404(Zone, id=zone_id)
    layout = get_or_create_layout(zone)
    layout.cells_json = normalize_layout(layout.rows, layout.cols, layout.cells_json)

    context = {
        "zone": zone,
        "layout_json": json.dumps(
            {
                "rows": layout.rows,
                "cols": layout.cols,
                "cells": layout.cells_json,
                "camera_points": layout.camera_points_json,
            }
        ),
        "camera_ids": [c.camera_id for c in zone.cameras.all()],
    }
    return render(request, "firewatch/layout_editor.html", context)


@require_GET
def admin_page(request: HttpRequest) -> HttpResponse:
    zone_id = request.GET.get("zone_id", "")
    zone = None
    if zone_id:
        zone = Zone.objects.filter(id=int(zone_id)).first()
    if not zone:
        zone = Zone.objects.order_by("id").first()
    if not zone:
        building, _ = Building.objects.get_or_create(name="Default Building")
        zone = Zone.objects.create(code="L3_Corridor_C12", building=building, location_type="corridor")

    layout = get_or_create_layout(zone)
    layout.cells_json = normalize_layout(layout.rows, layout.cols, layout.cells_json)
    zones = list(Zone.objects.select_related("building").order_by("building__name", "code"))

    context = {
        "zone": zone,
        "layout_json": json.dumps(
            {
                "rows": layout.rows,
                "cols": layout.cols,
                "cells": layout.cells_json,
                "camera_points": layout.camera_points_json,
            }
        ),
        "camera_ids": [c.camera_id for c in zone.cameras.all()],
        "admin_mode": True,
        "zones": zones,
        "selected_zone_id": zone.id,
    }
    return render(request, "firewatch/layout_editor.html", context)


@require_http_methods(["POST"])
def admin_reset_events(request: HttpRequest) -> HttpResponse:
    zone_id = request.POST.get("zone_id", "")
    count = Event.objects.count()
    Event.objects.all().delete()

    with MONITOR_LOCK:
        for key in list(EVENT_MONITORS.keys()):
            EVENT_MONITORS[key]["enabled"] = False
        EVENT_MONITORS.clear()

    messages.success(request, f"Reset completed. Deleted {count} event(s).")
    target = reverse("firewatch:admin_page")
    if zone_id:
        target = f"{target}?zone_id={zone_id}"
    return redirect(target)


@require_GET
def zone_layout_data(request: HttpRequest, zone_id: int) -> JsonResponse:
    zone = get_object_or_404(Zone, id=zone_id)
    layout = get_or_create_layout(zone)
    cells = normalize_layout(layout.rows, layout.cols, layout.cells_json)
    return JsonResponse(
        {
            "zone": zone.code,
            "rows": layout.rows,
            "cols": layout.cols,
            "cells": cells,
            "camera_points": layout.camera_points_json,
            "camera_ids": [c.camera_id for c in zone.cameras.all()],
        }
    )


@csrf_exempt
@require_http_methods(["POST"])
def zone_layout_save(request: HttpRequest, zone_id: int) -> JsonResponse:
    zone = get_object_or_404(Zone, id=zone_id)
    payload = _json_body(request)

    rows = int(payload.get("rows", 20))
    cols = int(payload.get("cols", 20))
    rows = max(5, min(rows, 80))
    cols = max(5, min(cols, 80))

    cells = normalize_layout(rows, cols, payload.get("cells", []))

    camera_points = payload.get("camera_points", {})
    if not isinstance(camera_points, dict):
        return JsonResponse({"error": "camera_points must be object"}, status=400)

    clean_points = {}
    for camera_id, point in camera_points.items():
        if not isinstance(point, list) or len(point) != 2:
            continue
        r, c = int(point[0]), int(point[1])
        if 0 <= r < rows and 0 <= c < cols:
            clean_points[str(camera_id)] = [r, c]

    layout = get_or_create_layout(zone)
    layout.rows = rows
    layout.cols = cols
    layout.cells_json = cells
    layout.camera_points_json = clean_points
    layout.save(update_fields=["rows", "cols", "cells_json", "camera_points_json", "updated_at"])

    return JsonResponse({"ok": True, "rows": rows, "cols": cols, "camera_points": clean_points})


@require_GET
def event_detail(request: HttpRequest, event_id: str) -> HttpResponse:
    # Event page is removed; route remains as a compatibility redirect.
    get_object_or_404(Event, event_id=event_id)
    return redirect(reverse("firewatch:event_footage_view", kwargs={"event_id": event_id}))


@csrf_exempt
@require_http_methods(["POST"])
def event_routes(request: HttpRequest, event_id: str) -> JsonResponse:
    event = get_object_or_404(Event, event_id=event_id)
    payload = _json_body(request)
    blocked_cells = payload.get("blocked_cells", [])
    data = compute_routes_for_event(event, blocked_cells=blocked_cells, max_alternatives=3)
    if "error" in data:
        return JsonResponse(data, status=400)
    return JsonResponse(data)


@require_GET
def event_flow_validation(request: HttpRequest, event_id: str) -> JsonResponse:
    event = get_object_or_404(Event, event_id=event_id)
    return JsonResponse(_build_flow_validation(event))


@require_GET
def event_live_feed_meta(request: HttpRequest, event_id: str) -> JsonResponse:
    event = get_object_or_404(Event, event_id=event_id)
    risk_info = _risk_presentation(event)
    latest = event.frame_detections.select_related("camera").order_by("-timestamp").first()
    if latest:
        camera = latest.camera
        payload = {
            "enabled": risk_info["risk_level"] in {"elevated_risk", "hazard"},
            "risk_level": risk_info["risk_level"],
            "camera_id": camera.camera_id,
            "rtsp_url": camera.rtsp_url,
            "note": "RTSP web playback requires bridge (e.g., WebRTC/HLS).",
        }
    else:
        payload = {
            "enabled": False,
            "risk_level": risk_info["risk_level"],
            "camera_id": "",
            "rtsp_url": "",
            "note": "No camera detections yet.",
        }
    return JsonResponse(payload)


@csrf_exempt
@require_http_methods(["POST"])
def event_notify_whatsapp_api(request: HttpRequest, event_id: str) -> JsonResponse:
    event = get_object_or_404(Event, event_id=event_id)
    payload = _json_body(request)
    phone_number = str(payload.get("phone_number", "6590000000")).replace("+", "")
    data = _build_whatsapp_payload(event, phone_number)

    logs = list(event.authority_notifications_json or [])
    logs.append({"channel": "whatsapp_mock", **data})
    event.authority_notifications_json = logs
    event.save(update_fields=["authority_notifications_json", "updated_at"])

    return JsonResponse({"ok": True, **data})


@require_http_methods(["POST"])
def event_notify_whatsapp_ui(request: HttpRequest, event_id: str) -> HttpResponse:
    event = get_object_or_404(Event, event_id=event_id)
    phone_number = str(request.POST.get("phone_number", "6590000000")).replace("+", "")
    data = _build_whatsapp_payload(event, phone_number)

    logs = list(event.authority_notifications_json or [])
    logs.append({"channel": "whatsapp_mock", **data})
    event.authority_notifications_json = logs
    event.save(update_fields=["authority_notifications_json", "updated_at"])

    messages.success(request, f"WhatsApp mock prepared for {phone_number}. Link is shown in Authority Notifications.")
    return redirect(reverse("firewatch:event_footage_view", kwargs={"event_id": event.event_id}))


@require_http_methods(["POST"])
def home_add_event(request: HttpRequest) -> HttpResponse:
    cameras = list(Camera.objects.select_related("zone").filter(active=True))
    if cameras:
        camera = random.choice(cameras)
        zone = camera.zone
    else:
        building, _ = Building.objects.get_or_create(name="Default Building")
        zone, _ = Zone.objects.get_or_create(code="L3_Corridor_C12", defaults={"building": building, "location_type": "corridor"})
        camera = Camera.objects.create(camera_id=f"{zone.code}_CAM1", zone=zone, default_fps=10, active=True)

    event_id = timezone.now().strftime("evt_%Y%m%d_%H%M%S")
    if Event.objects.filter(event_id=event_id).exists():
        event_id = f"{event_id}_{Event.objects.count() + 1}"

    event = Event.objects.create(
        event_id=event_id,
        trigger_type="manual",
        sensor_id="home_add_event",
        trigger_time=timezone.now(),
        zone=zone,
        status="under_review",
    )
    layout = get_or_create_layout(zone)
    camera_point = layout.camera_points_json.get(camera.camera_id)
    if isinstance(camera_point, list) and len(camera_point) == 2:
        fire_cell = [int(camera_point[0]), int(camera_point[1])]
    else:
        fire_cell = [max(0, layout.rows // 2), max(0, layout.cols // 2)]

    _save_stage_output(
        event,
        "stage1_event_trigger",
        {
            "event_id": event.event_id,
            "trigger_type": event.trigger_type,
            "sensor_id": event.sensor_id,
            "trigger_time": event.trigger_time.isoformat(),
            "zone": zone.code,
        },
    )
    _save_stage_output(
        event,
        "sim_fire_seed",
        {
            "camera_id": camera.camera_id,
            "fire_cell": fire_cell,
            "created_at": timezone.now().isoformat(),
        },
    )

    interval_seconds = 5.0
    start_seconds = 0.0
    video_duration = get_video_duration_seconds()

    with MONITOR_LOCK:
        EVENT_MONITORS[event.event_id] = {
            "enabled": True,
            "camera_id": camera.camera_id,
            "interval_seconds": interval_seconds,
            "next_start_seconds": start_seconds,
            "video_duration": video_duration,
            "last_tick": "",
            "last_error": "",
        }
        thread = threading.Thread(target=_monitor_loop, args=(event.event_id,), daemon=True)
        EVENT_MONITORS[event.event_id]["thread"] = thread
        thread.start()

    messages.success(
        request,
        f"Event {event.event_id} created. Continuous 5s interval monitoring started on camera {camera.camera_id}.",
    )
    return redirect(f"{reverse('firewatch:dashboard_home')}?new_event={event.event_id}")


@csrf_exempt
@require_http_methods(["POST"])
def event_run_classification(request: HttpRequest, event_id: str) -> JsonResponse:
    event = get_object_or_404(Event, event_id=event_id)
    payload = _json_body(request)
    start_seconds = float(payload.get("start_seconds", 0.0))
    analyze_seconds = float(payload.get("analyze_seconds", 10.0))
    sample_fps = float(payload.get("sample_fps", 2.0))
    conf = float(payload.get("conf", 0.25))
    video_path = payload.get("video_path")
    location_type = payload.get("location_type") or event.zone.location_type
    device = str(payload.get("device", "cpu"))

    try:
        result = run_classification_window(
            event=event,
            start_seconds=start_seconds,
            analyze_seconds=analyze_seconds,
            sample_fps=sample_fps,
            conf=conf,
            location_type=location_type,
            video_path=video_path,
            camera_id=payload.get("camera_id", "video_feed"),
            device=device,
        )
    except ClassificationRunError as exc:
        return JsonResponse({"ok": False, "error": str(exc)}, status=400)

    return JsonResponse(result)


@require_GET
def event_video_footage(request: HttpRequest, event_id: str) -> HttpResponse:
    event = get_object_or_404(Event, event_id=event_id)
    focus = _event_footage_focus(event)
    start_at = max(0.0, float(focus.get("detection_start_s", 0.0)))
    clip_path = _build_5s_clip_for_event(event, start_at, duration_seconds=5.0)
    return _video_file_response(request, clip_path)


@require_GET
def event_footage_view(request: HttpRequest, event_id: str) -> HttpResponse:
    event = get_object_or_404(Event.objects.select_related("zone", "zone__building"), event_id=event_id)
    focus = _event_footage_focus(event)
    n_seconds = 5
    start_at = max(0.0, float(focus.get("detection_start_s", 0.0)))
    emergency_notice = _latest_emergency_notification(event)
    evacuation_notice = _latest_evacuation_announcement(event)
    alarm_started_at = _alarm_started_at(event)

    context = {
        "event": event,
        "risk_info": _risk_presentation(event),
        "focus": focus,
        "time_since_alarm": _time_since_alarm_text(event),
        "n_seconds": n_seconds,
        "start_at": 0.0,
        "classification_runs": list(event.classification_runs_json or []),
        "emergency_notice": emergency_notice,
        "evacuation_notice": evacuation_notice,
        "alarm_started_at_iso": alarm_started_at.isoformat() if alarm_started_at else "",
        "footage_stream_url": (
            reverse("firewatch:event_video_footage", kwargs={"event_id": event.event_id})
            + f"?v={int(event.updated_at.timestamp())}_{uuid4().hex[:8]}"
        ),
    }
    return render(request, "firewatch/event_footage_view.html", context)


@csrf_exempt
@require_http_methods(["POST"])
def emergency_decision_api(request: HttpRequest, event_id: str) -> JsonResponse:
    event = get_object_or_404(Event, event_id=event_id)
    payload = _json_body(request)
    action = payload.get("action", "")
    try:
        result = resolve_emergency_decision(event, action)
    except ValueError as exc:
        return JsonResponse({"ok": False, "error": str(exc)}, status=400)
    return JsonResponse(result)


@csrf_exempt
@require_http_methods(["POST"])
def authorities_escalate_api(request: HttpRequest, event_id: str) -> JsonResponse:
    event = get_object_or_404(Event, event_id=event_id)
    if event.risk_level == "no_fire_risk":
        return JsonResponse({"ok": False, "error": "No Fire Risk cannot be escalated."}, status=400)

    payload = _json_body(request)
    mode = str(payload.get("mode", "manual")).strip().lower()
    now = timezone.now()

    if mode == "auto_timeout":
        status = "auto_escalation_triggered"
    else:
        status = "manual_escalation_requested"

    telegram_result = _send_telegram_authority_notification(event, mode=mode)
    if not telegram_result.get("ok"):
        status = f"{status}_telegram_partial_failure"

    logs = list(event.authority_notifications_json or [])
    logs.append(
        {
            "type": "authorities_escalation_request",
            "mode": mode,
            "risk_level": event.risk_level,
            "created_at": now.isoformat(),
            "telegram": telegram_result,
        }
    )
    event.authority_notifications_json = logs

    if event.risk_level == "emergency":
        event.emergency_call_pending = False
    event.emergency_call_status = status
    event.save(update_fields=["authority_notifications_json", "emergency_call_pending", "emergency_call_status", "updated_at"])

    return JsonResponse(
        {
            "ok": True,
            "event_id": event.event_id,
            "mode": mode,
            "risk_level": event.risk_level,
            "escalation_status": status,
            "telegram": telegram_result,
        }
    )


@require_http_methods(["POST"])
@transaction.atomic
def event_action(request: HttpRequest, event_id: str, action: str) -> HttpResponse:
    event = get_object_or_404(Event, event_id=event_id)

    if action not in {"ack", "false_alarm", "escalate"}:
        messages.error(request, "Invalid action")
        return redirect(reverse("firewatch:event_footage_view", kwargs={"event_id": event.event_id}))

    EventAction.objects.create(event=event, action=action, actor="dashboard_user")

    if action == "ack":
        event.status = "acknowledged"
    elif action == "false_alarm":
        event.status = "false_alarm"
    elif action == "escalate":
        event.status = "escalated"

    event.save(update_fields=["status", "updated_at"])
    messages.success(request, f"Action recorded: {action.upper()}")
    return redirect(reverse("firewatch:event_footage_view", kwargs={"event_id": event.event_id}))
