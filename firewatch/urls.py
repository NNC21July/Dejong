from django.urls import path

from . import views

app_name = "firewatch"

urlpatterns = [
    path("health/", views.health, name="health"),
    path("", views.dashboard_home, name="dashboard_home"),
    path("admin-panel/", views.admin_page, name="admin_page"),
    path("admin-panel/reset-events/", views.admin_reset_events, name="admin_reset_events"),
    path("api/dashboard/live-state/", views.dashboard_live_state_api, name="dashboard_live_state_api"),
    path("events/add/", views.home_add_event, name="home_add_event"),
    path("zones/<int:zone_id>/layout/", views.zone_layout_editor, name="zone_layout_editor"),
    path("api/zones/<int:zone_id>/layout/", views.zone_layout_data, name="zone_layout_data"),
    path("api/zones/<int:zone_id>/layout/save/", views.zone_layout_save, name="zone_layout_save"),
    path("events/<str:event_id>/", views.event_detail, name="event_detail"),
    path("events/<str:event_id>/action/<str:action>/", views.event_action, name="event_action"),
    path("events/<str:event_id>/notify-whatsapp/", views.event_notify_whatsapp_ui, name="event_notify_whatsapp_ui"),
    path("events/<str:event_id>/footage/", views.event_video_footage, name="event_video_footage"),
    path("events/<str:event_id>/footage/view/", views.event_footage_view, name="event_footage_view"),
    path("api/events/<str:event_id>/routes/", views.event_routes, name="event_routes"),
    path("api/events/<str:event_id>/flow-validation/", views.event_flow_validation, name="event_flow_validation"),
    path("api/events/<str:event_id>/live-feed-meta/", views.event_live_feed_meta, name="event_live_feed_meta"),
    path("api/events/<str:event_id>/notify-authorities-whatsapp/", views.event_notify_whatsapp_api, name="event_notify_whatsapp_api"),
    path("api/events/<str:event_id>/classification/run/", views.event_run_classification, name="event_run_classification"),
    path("api/events/<str:event_id>/emergency/decision/", views.emergency_decision_api, name="emergency_decision_api"),
    path("api/events/<str:event_id>/authorities/escalate/", views.authorities_escalate_api, name="authorities_escalate_api"),
    path("api/stage1/event-trigger/", views.stage1_event_trigger, name="stage1_event_trigger"),
    path("api/stage2/camera-selection/", views.stage2_camera_selection, name="stage2_camera_selection"),
    path("api/stage3/yolo-detection/", views.stage3_yolo_detection_ingest, name="stage3_yolo_detection"),
    path("api/yolo/frame-detection/", views.stage3_yolo_detection_ingest, name="yolo_frame_detection"),
    path("api/stage4/temporal-aggregation/", views.stage4_temporal_aggregation, name="stage4_temporal_aggregation"),
    path("api/stage5/context-package/", views.stage5_context_package, name="stage5_context_package"),
    path("api/stage6/advisory-reasoner/", views.stage6_advisory_reasoner, name="stage6_advisory_reasoner"),
    path("api/stage7/risk-score/", views.stage7_risk_score, name="stage7_risk_score"),
    path("api/stage8/response-packet/", views.stage8_response_packet, name="stage8_response_packet"),
]
