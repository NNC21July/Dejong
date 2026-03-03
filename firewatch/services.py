import statistics
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from django.utils import timezone
from django.utils.dateparse import parse_datetime

from .models import Camera, Event, FrameDetection, RiskConfig


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def map_risk_level(score: float) -> tuple[str, str]:
    s = clamp01(score)
    if s < 0.25:
        return "no_fire_risk", "log_watch"
    if s < 0.55:
        return "elevated_risk", "notify_tier1_tier2_live_feed"
    if s < 0.80:
        return "hazard", "notify_tier1_tier2_live_feed"
    return "emergency", "dispatch_and_notify_authorities"


def parse_iso_dt(value: str | None) -> datetime:
    if not value:
        return timezone.now()
    parsed = parse_datetime(value)
    if not parsed:
        return timezone.now()
    if timezone.is_naive(parsed):
        return timezone.make_aware(parsed, timezone.get_current_timezone())
    return parsed


def stage3_ingest_frame_detection(payload: dict[str, Any]) -> FrameDetection:
    event_id = payload["event_id"]
    camera_id = payload["camera_id"]
    event = Event.objects.get(event_id=event_id)
    camera = Camera.objects.get(camera_id=camera_id)

    return FrameDetection.objects.create(
        event=event,
        camera=camera,
        frame_index=int(payload["frame_index"]),
        timestamp=parse_iso_dt(payload.get("timestamp")),
        detections_json=payload.get("detections", []),
    )


@dataclass
class Stage4Summary:
    fire_frames: int
    smoke_frames: int
    controlled_flame_frames: int
    fire_persistence: float
    max_fire_conf: float
    mean_fire_conf: float
    fire_bbox_area_growth_rate: float
    smoke_conf_trend: str
    best_fire_frame: int | None
    best_smoke_frame: int | None


def _bbox_area(bbox_xyxy: list[float]) -> float:
    x1, y1, x2, y2 = bbox_xyxy
    return max(0.0, float(x2) - float(x1)) * max(0.0, float(y2) - float(y1))


def aggregate_temporal_evidence(event: Event, camera: Camera, window_seconds: int = 15) -> Stage4Summary:
    rows = list(event.frame_detections.filter(camera=camera).order_by("frame_index"))
    total_frames = len(rows)
    if total_frames == 0:
        return Stage4Summary(0, 0, 0, 0.0, 0.0, 0.0, 0.0, "stable", None, None)

    fire_frames = 0
    smoke_frames = 0
    controlled_frames = 0
    fire_conf_values: list[float] = []
    fire_areas: list[float] = []
    smoke_conf_by_frame: list[float] = []

    best_fire_frame = None
    best_smoke_frame = None
    best_fire_conf = -1.0
    best_smoke_conf = -1.0

    for row in rows:
        frame_fire_max = 0.0
        frame_smoke_max = 0.0
        has_fire = False
        has_smoke = False
        has_controlled = False

        for det in row.detections_json:
            class_id = int(det.get("class_id", -1))
            conf = float(det.get("confidence", 0.0))
            bbox = det.get("bbox_xyxy", [0, 0, 0, 0])

            if class_id == 0:
                has_fire = True
                frame_fire_max = max(frame_fire_max, conf)
                fire_conf_values.append(conf)
                fire_areas.append(_bbox_area(bbox))
                if conf > best_fire_conf:
                    best_fire_conf = conf
                    best_fire_frame = row.frame_index
            elif class_id == 1:
                has_smoke = True
                frame_smoke_max = max(frame_smoke_max, conf)
                if conf > best_smoke_conf:
                    best_smoke_conf = conf
                    best_smoke_frame = row.frame_index
            elif class_id == 2:
                has_controlled = True

        smoke_conf_by_frame.append(frame_smoke_max)

        if has_fire:
            fire_frames += 1
        if has_smoke:
            smoke_frames += 1
        if has_controlled:
            controlled_frames += 1

    fire_persistence = fire_frames / total_frames if total_frames else 0.0
    max_fire_conf = max(fire_conf_values) if fire_conf_values else 0.0
    mean_fire_conf = statistics.fmean(fire_conf_values) if fire_conf_values else 0.0

    if len(fire_areas) >= 2:
        start_area = max(fire_areas[0], 1e-6)
        fire_bbox_area_growth_rate = clamp01((fire_areas[-1] - start_area) / start_area)
    else:
        fire_bbox_area_growth_rate = 0.0

    if len(smoke_conf_by_frame) >= 2:
        delta = smoke_conf_by_frame[-1] - smoke_conf_by_frame[0]
        if delta > 0.05:
            smoke_conf_trend = "increasing"
        elif delta < -0.05:
            smoke_conf_trend = "decreasing"
        else:
            smoke_conf_trend = "stable"
    else:
        smoke_conf_trend = "stable"

    event.fire_frames = fire_frames
    event.smoke_frames = smoke_frames
    event.controlled_flame_frames = controlled_frames
    event.total_frames = total_frames
    event.fire_persistence = clamp01(fire_persistence)
    event.max_fire_conf = clamp01(max_fire_conf)
    event.mean_fire_conf = clamp01(mean_fire_conf)
    event.fire_bbox_area_growth_rate = clamp01(fire_bbox_area_growth_rate)
    event.smoke_conf_trend = smoke_conf_trend
    event.save(update_fields=[
        "fire_frames",
        "smoke_frames",
        "controlled_flame_frames",
        "total_frames",
        "fire_persistence",
        "max_fire_conf",
        "mean_fire_conf",
        "fire_bbox_area_growth_rate",
        "smoke_conf_trend",
        "updated_at",
    ])

    return Stage4Summary(
        fire_frames=fire_frames,
        smoke_frames=smoke_frames,
        controlled_flame_frames=controlled_frames,
        fire_persistence=clamp01(fire_persistence),
        max_fire_conf=clamp01(max_fire_conf),
        mean_fire_conf=clamp01(mean_fire_conf),
        fire_bbox_area_growth_rate=clamp01(fire_bbox_area_growth_rate),
        smoke_conf_trend=smoke_conf_trend,
        best_fire_frame=best_fire_frame,
        best_smoke_frame=best_smoke_frame,
    )


def build_context_package(event: Event) -> dict[str, Any]:
    now = timezone.localtime()
    hour = now.hour
    time_of_day = "day" if 6 <= hour < 18 else "night"
    day_of_week = now.strftime("%a")

    return {
        "event_id": event.event_id,
        "context": {
            "location_type": event.zone.location_type,
            "building_type": event.zone.building.building_type,
            "known_cooking_zone": event.zone.known_cooking_zone,
            "time_of_day": time_of_day,
            "day_of_week": day_of_week,
            "scheduled_event": False,
        },
        "evidence": {
            "yolo_summary": {
                "fire_frames": event.fire_frames,
                "smoke_frames": event.smoke_frames,
                "controlled_flame_frames": event.controlled_flame_frames,
                "fire_persistence": event.fire_persistence,
                "max_fire_conf": event.max_fire_conf,
                "mean_fire_conf": event.mean_fire_conf,
            },
            "keyframe_paths": [event.keyframe_path] if event.keyframe_path else [],
        },
    }


def apply_advisory_reasoner(event: Event, advisory_payload: dict[str, Any]) -> None:
    event.scenario = advisory_payload.get("scenario", "")
    event.advisory_risk_level = advisory_payload.get("risk_level", "")
    event.advisory_threat_score = clamp01(float(advisory_payload.get("threat_score", 0.0)))
    event.advisory_confidence = clamp01(float(advisory_payload.get("confidence", 0.0)))
    event.save(update_fields=[
        "scenario",
        "advisory_risk_level",
        "advisory_threat_score",
        "advisory_confidence",
        "updated_at",
    ])


def score_event(event: Event, config: RiskConfig | None = None) -> dict[str, Any]:
    cfg = config or RiskConfig.objects.first() or RiskConfig.objects.create(name="default")

    yolo_detection_strength = clamp01((event.max_fire_conf + event.mean_fire_conf) / 2)
    temporal_persistence = clamp01(event.fire_persistence)
    smoke_presence = clamp01(event.smoke_frames / max(event.total_frames, 1))
    growth_rate = clamp01(event.fire_bbox_area_growth_rate)

    context_modifier = 0.7 if event.zone.known_cooking_zone else 1.0
    if event.zone.location_type in {"corridor", "warehouse", "carpark"}:
        context_modifier = min(1.0, context_modifier + 0.15)

    openai_threat = clamp01(event.advisory_threat_score)

    score = (
        cfg.w1_yolo_detection_strength * yolo_detection_strength
        + cfg.w2_fire_persistence * temporal_persistence
        + cfg.w3_smoke_presence * smoke_presence
        + cfg.w4_growth_rate * growth_rate
        + cfg.w5_context_modifier * context_modifier
        + cfg.w6_openai_threat_score * openai_threat
    )
    score = clamp01(score)
    risk_level, risk_action = map_risk_level(score)

    if score >= cfg.high_threshold:
        decision = "dispatch"
    elif score >= cfg.medium_threshold:
        decision = "request_human_verification"
    else:
        decision = "monitor"

    explainability = [
        f"Fire detected in {event.fire_frames}/{max(event.total_frames, 1)} frames.",
        f"Smoke trend is {event.smoke_conf_trend}.",
        "Non-cooking zone." if not event.zone.known_cooking_zone else "Cooking zone context detected.",
    ]

    breakdown = {
        "yolo_detection_strength": round(cfg.w1_yolo_detection_strength * yolo_detection_strength, 4),
        "temporal_persistence": round(cfg.w2_fire_persistence * temporal_persistence, 4),
        "smoke_presence": round(cfg.w3_smoke_presence * smoke_presence, 4),
        "growth_rate": round(cfg.w4_growth_rate * growth_rate, 4),
        "context_modifier": round(cfg.w5_context_modifier * context_modifier, 4),
        "openai_threat_score": round(cfg.w6_openai_threat_score * openai_threat, 4),
    }

    event.final_risk_score = score
    event.decision = decision
    event.risk_level = risk_level
    event.risk_action = risk_action
    event.score_breakdown_json = breakdown
    event.explainability_json = explainability
    if decision == "dispatch":
        event.status = "dispatched"
    elif decision == "request_human_verification":
        event.status = "under_review"
    event.save(update_fields=[
        "final_risk_score",
        "decision",
        "risk_level",
        "risk_action",
        "score_breakdown_json",
        "explainability_json",
        "status",
        "updated_at",
    ])

    return {
        "event_id": event.event_id,
        "final_risk_score": round(score, 4),
        "risk_level": risk_level,
        "risk_action": risk_action,
        "decision": decision,
        "score_breakdown": breakdown,
        "explainability": explainability,
    }


def build_response_packet(event: Event, camera: Camera | None = None) -> dict[str, Any]:
    camera_obj = camera or event.zone.cameras.first()
    packet = {
        "event_id": event.event_id,
        "action_taken": "sent_to_dashboard",
        "response_packet": {
            "camera_id": camera_obj.camera_id if camera_obj else "",
            "zone": event.zone.code,
            "floorplan_marker": {
                "x": event.zone.floorplan_x,
                "y": event.zone.floorplan_y,
            },
            "attachments": [
                path
                for path in [event.keyframe_path or "keyframe.jpg", event.annotated_frame_path or "annotated_frame.jpg", "incident_summary.pdf"]
                if path
            ],
        },
    }
    event.response_packet_json = packet
    event.save(update_fields=["response_packet_json", "updated_at"])
    return packet
