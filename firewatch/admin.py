from django.contrib import admin

from .models import Building, Camera, Event, EventAction, FrameDetection, ModelVersion, RiskConfig, Zone, ZoneLayout


@admin.register(Building)
class BuildingAdmin(admin.ModelAdmin):
    list_display = ("name", "building_type")


@admin.register(Zone)
class ZoneAdmin(admin.ModelAdmin):
    list_display = ("code", "building", "floor_level", "location_type", "known_cooking_zone")
    list_filter = ("location_type", "known_cooking_zone", "building")


@admin.register(ZoneLayout)
class ZoneLayoutAdmin(admin.ModelAdmin):
    list_display = ("zone", "rows", "cols", "updated_at")


@admin.register(Camera)
class CameraAdmin(admin.ModelAdmin):
    list_display = ("camera_id", "zone", "default_fps", "active")
    list_filter = ("active", "zone")


@admin.register(Event)
class EventAdmin(admin.ModelAdmin):
    list_display = ("event_id", "zone", "trigger_type", "status", "final_risk_score", "decision", "created_at")
    list_filter = ("status", "trigger_type", "zone")
    search_fields = ("event_id", "zone__code", "sensor_id")


@admin.register(FrameDetection)
class FrameDetectionAdmin(admin.ModelAdmin):
    list_display = ("event", "camera", "frame_index", "timestamp")
    list_filter = ("camera",)


@admin.register(EventAction)
class EventActionAdmin(admin.ModelAdmin):
    list_display = ("event", "action", "actor", "created_at")
    list_filter = ("action",)


@admin.register(RiskConfig)
class RiskConfigAdmin(admin.ModelAdmin):
    list_display = (
        "name",
        "w1_yolo_detection_strength",
        "w2_fire_persistence",
        "w3_smoke_presence",
        "w4_growth_rate",
        "w5_context_modifier",
        "w6_openai_threat_score",
        "medium_threshold",
        "high_threshold",
    )


@admin.register(ModelVersion)
class ModelVersionAdmin(admin.ModelAdmin):
    list_display = ("name", "version", "created_at")
