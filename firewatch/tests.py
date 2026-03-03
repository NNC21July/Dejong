import json
from unittest.mock import patch

from django.test import Client, TestCase

from .models import Building, Camera, Event, RiskConfig, Zone


class FirewatchPipelineTests(TestCase):
    def setUp(self) -> None:
        self.client = Client()
        building = Building.objects.create(name="Test Building", building_type="commercial")
        self.zone = Zone.objects.create(code="L3_Corridor_C12", building=building, location_type="corridor")
        self.camera = Camera.objects.create(camera_id="C12", zone=self.zone, default_fps=10)
        RiskConfig.objects.create(name="default")

    def test_stage_flow_minimal(self):
        stage1 = self.client.post(
            "/api/stage1/event-trigger/",
            data=json.dumps({"trigger_type": "manual", "zone": self.zone.code}),
            content_type="application/json",
        )
        self.assertEqual(stage1.status_code, 201)
        event_id = stage1.json()["event_id"]

        for idx in range(4):
            payload = {
                "event_id": event_id,
                "camera_id": self.camera.camera_id,
                "frame_index": idx,
                "timestamp": "2026-03-01T10:00:00+08:00",
                "detections": [
                    {
                        "class_id": 0,
                        "class_name": "fire",
                        "confidence": 0.8,
                        "bbox_xyxy": [10, 10, 30 + idx, 30 + idx],
                    }
                ],
            }
            stage3 = self.client.post("/api/yolo/frame-detection/", data=json.dumps(payload), content_type="application/json")
            self.assertEqual(stage3.status_code, 201)

        stage4 = self.client.post(
            "/api/stage4/temporal-aggregation/",
            data=json.dumps({"event_id": event_id, "camera_id": self.camera.camera_id}),
            content_type="application/json",
        )
        self.assertEqual(stage4.status_code, 200)

        stage7 = self.client.post(
            "/api/stage7/risk-score/",
            data=json.dumps({"event_id": event_id}),
            content_type="application/json",
        )
        self.assertEqual(stage7.status_code, 200)
        self.assertIn("decision", stage7.json())
        self.assertIn("risk_level", stage7.json())

        stage8 = self.client.post(
            "/api/stage8/response-packet/",
            data=json.dumps({"event_id": event_id, "camera_id": self.camera.camera_id}),
            content_type="application/json",
        )
        self.assertEqual(stage8.status_code, 200)

        flow = self.client.get(f"/api/events/{event_id}/flow-validation/")
        self.assertEqual(flow.status_code, 200)
        self.assertIn("all_required_stages_ok", flow.json())

        notify = self.client.post(
            f"/api/events/{event_id}/notify-authorities-whatsapp/",
            data=json.dumps({"phone_number": "6591111111"}),
            content_type="application/json",
        )
        self.assertEqual(notify.status_code, 200)
        self.assertTrue(notify.json()["ok"])

        event = Event.objects.get(event_id=event_id)
        self.assertGreaterEqual(len(event.authority_notifications_json), 1)

    @patch("firewatch.views._send_telegram_authority_notification")
    def test_authorities_escalation_triggers_telegram_payload(self, mock_send):
        mock_send.return_value = {
            "ok": True,
            "results": {
                "message": {"ok": True},
                "video": {"ok": True},
                "map": {"ok": True},
            },
            "route_data": {"event_id": "evt_x"},
        }

        event = Event.objects.create(
            event_id="evt_test_escalate",
            trigger_type="manual",
            sensor_id="unit_test",
            zone=self.zone,
            status="under_review",
            risk_level="emergency",
            decision="dispatch",
            emergency_call_pending=True,
        )

        res = self.client.post(
            f"/api/events/{event.event_id}/authorities/escalate/",
            data=json.dumps({"mode": "manual"}),
            content_type="application/json",
        )
        self.assertEqual(res.status_code, 200)
        payload = res.json()
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["escalation_status"], "manual_escalation_requested")

        mock_send.assert_called_once()

        event.refresh_from_db()
        self.assertFalse(event.emergency_call_pending)
        self.assertEqual(event.emergency_call_status, "manual_escalation_requested")
        self.assertTrue(any(x.get("type") == "authorities_escalation_request" for x in event.authority_notifications_json))

