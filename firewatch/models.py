from django.db import models
from django.utils import timezone


class Building(models.Model):
    BUILDING_TYPES = [
        ("residential", "Residential"),
        ("commercial", "Commercial"),
        ("industrial", "Industrial"),
    ]

    name = models.CharField(max_length=120, unique=True)
    building_type = models.CharField(max_length=20, choices=BUILDING_TYPES, default="commercial")

    def __str__(self):
        return self.name


class Zone(models.Model):
    LOCATION_TYPES = [
        ("corridor", "Corridor"),
        ("pantry", "Pantry"),
        ("carpark", "Carpark"),
        ("stage", "Stage"),
        ("warehouse", "Warehouse"),
        ("kitchen", "Kitchen"),
        ("other", "Other"),
    ]

    code = models.CharField(max_length=80, unique=True)
    building = models.ForeignKey(Building, on_delete=models.CASCADE, related_name="zones")
    floor_level = models.CharField(max_length=20, blank=True, default="")
    location_type = models.CharField(max_length=20, choices=LOCATION_TYPES, default="other")
    known_cooking_zone = models.BooleanField(default=False)
    floorplan_x = models.FloatField(default=0.5)
    floorplan_y = models.FloatField(default=0.5)

    def __str__(self):
        return self.code


class ZoneLayout(models.Model):
    zone = models.OneToOneField(Zone, on_delete=models.CASCADE, related_name="layout")
    rows = models.PositiveIntegerField(default=20)
    cols = models.PositiveIntegerField(default=20)
    cells_json = models.JSONField(default=list, blank=True)
    camera_points_json = models.JSONField(default=dict, blank=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Layout:{self.zone.code}"


class Camera(models.Model):
    camera_id = models.CharField(max_length=40, unique=True)
    zone = models.ForeignKey(Zone, on_delete=models.CASCADE, related_name="cameras")
    rtsp_url = models.CharField(max_length=500, blank=True, default="")
    default_fps = models.PositiveIntegerField(default=10)
    active = models.BooleanField(default=True)

    def __str__(self):
        return self.camera_id


class ModelVersion(models.Model):
    name = models.CharField(max_length=100)
    version = models.CharField(max_length=40)
    notes = models.TextField(blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ("name", "version")

    def __str__(self):
        return f"{self.name}:{self.version}"


class RiskConfig(models.Model):
    name = models.CharField(max_length=60, unique=True, default="default")
    w1_yolo_detection_strength = models.FloatField(default=0.35)
    w2_fire_persistence = models.FloatField(default=0.20)
    w3_smoke_presence = models.FloatField(default=0.15)
    w4_growth_rate = models.FloatField(default=0.10)
    w5_context_modifier = models.FloatField(default=0.10)
    w6_openai_threat_score = models.FloatField(default=0.10)
    medium_threshold = models.FloatField(default=0.55)
    high_threshold = models.FloatField(default=0.80)

    def __str__(self):
        return self.name


class Event(models.Model):
    TRIGGER_TYPES = [
        ("smoke_sensor", "Smoke Sensor"),
        ("heat_sensor", "Heat Sensor"),
        ("manual", "Manual"),
        ("periodic_scan", "Periodic Scan"),
    ]
    STATUS = [
        ("new", "New"),
        ("under_review", "Under Review"),
        ("acknowledged", "Acknowledged"),
        ("false_alarm", "False Alarm"),
        ("escalated", "Escalated"),
        ("dispatched", "Dispatched"),
        ("closed", "Closed"),
    ]

    event_id = models.CharField(max_length=80, unique=True)
    trigger_type = models.CharField(max_length=20, choices=TRIGGER_TYPES)
    sensor_id = models.CharField(max_length=40, blank=True, default="")
    trigger_time = models.DateTimeField(default=timezone.now)
    zone = models.ForeignKey(Zone, on_delete=models.PROTECT, related_name="events")
    status = models.CharField(max_length=20, choices=STATUS, default="new")

    fire_frames = models.PositiveIntegerField(default=0)
    smoke_frames = models.PositiveIntegerField(default=0)
    controlled_flame_frames = models.PositiveIntegerField(default=0)
    total_frames = models.PositiveIntegerField(default=0)
    fire_persistence = models.FloatField(default=0.0)
    max_fire_conf = models.FloatField(default=0.0)
    mean_fire_conf = models.FloatField(default=0.0)
    fire_bbox_area_growth_rate = models.FloatField(default=0.0)
    smoke_conf_trend = models.CharField(max_length=20, blank=True, default="stable")

    scenario = models.CharField(max_length=50, blank=True, default="")
    advisory_risk_level = models.CharField(max_length=10, blank=True, default="")
    advisory_threat_score = models.FloatField(default=0.0)
    advisory_confidence = models.FloatField(default=0.0)

    final_risk_score = models.FloatField(default=0.0)
    decision = models.CharField(max_length=40, blank=True, default="monitor")
    risk_level = models.CharField(max_length=30, blank=True, default="no_fire_risk")
    risk_action = models.CharField(max_length=80, blank=True, default="log_watch")

    keyframe_path = models.CharField(max_length=255, blank=True, default="")
    annotated_frame_path = models.CharField(max_length=255, blank=True, default="")

    explainability_json = models.JSONField(default=list, blank=True)
    score_breakdown_json = models.JSONField(default=dict, blank=True)
    response_packet_json = models.JSONField(default=dict, blank=True)
    stage_outputs_json = models.JSONField(default=dict, blank=True)
    flow_validation_json = models.JSONField(default=dict, blank=True)
    authority_notifications_json = models.JSONField(default=list, blank=True)
    classification_runs_json = models.JSONField(default=list, blank=True)
    first_hazard_detected_at = models.DateTimeField(null=True, blank=True)
    first_emergency_detected_at = models.DateTimeField(null=True, blank=True)
    emergency_call_pending = models.BooleanField(default=False)
    emergency_call_deadline = models.DateTimeField(null=True, blank=True)
    emergency_call_status = models.CharField(max_length=40, blank=True, default="")

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return self.event_id


class FrameDetection(models.Model):
    event = models.ForeignKey(Event, on_delete=models.CASCADE, related_name="frame_detections")
    camera = models.ForeignKey(Camera, on_delete=models.PROTECT, related_name="frame_detections")
    frame_index = models.PositiveIntegerField()
    timestamp = models.DateTimeField(default=timezone.now)
    detections_json = models.JSONField(default=list)

    class Meta:
        indexes = [models.Index(fields=["event", "camera", "frame_index"])]
        ordering = ["frame_index"]


class EventAction(models.Model):
    ACTIONS = [
        ("ack", "ACK"),
        ("false_alarm", "FALSE ALARM"),
        ("escalate", "ESCALATE"),
    ]

    event = models.ForeignKey(Event, on_delete=models.CASCADE, related_name="actions")
    action = models.CharField(max_length=20, choices=ACTIONS)
    actor = models.CharField(max_length=60, blank=True, default="dashboard_user")
    notes = models.TextField(blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
