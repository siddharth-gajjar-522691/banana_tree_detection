"""
Tests for Pydantic schemas.

WHY test schemas?
  schemas.py is the contract between the backend and the frontend JavaScript.
  If a field name changes or a type changes, the JS breaks silently.
  These tests catch that before the code reaches production.
"""

import pytest
from pydantic import ValidationError

from schemas import DetectionResponse, HealthResponse


class TestDetectionResponse:
    def test_minimal_valid(self):
        r = DetectionResponse(
            result_image_url="/static/results/detected/test.jpg",
            object_count=5,
            detection_summary={"banana": 3, "leaf": 2},
            processing_time_ms=123.4,
        )
        assert r.success is True
        assert r.object_count == 5
        assert r.message == "Detection complete"
        assert r.id is None

    def test_with_db_id(self):
        r = DetectionResponse(
            id=42,
            result_image_url="/static/results/detected/test.jpg",
            object_count=1,
            detection_summary={"banana": 1},
            processing_time_ms=50.0,
        )
        assert r.id == 42

    def test_zero_detections(self):
        r = DetectionResponse(
            result_image_url="/static/results/detected/empty.jpg",
            object_count=0,
            detection_summary={},
            processing_time_ms=10.0,
        )
        assert r.object_count == 0
        assert r.detection_summary == {}

    def test_missing_required_field_raises(self):
        with pytest.raises(ValidationError):
            DetectionResponse(
                object_count=1,
                detection_summary={"banana": 1},
                processing_time_ms=50.0,
                # result_image_url is missing — must raise
            )

    def test_detection_summary_is_dict_of_ints(self):
        r = DetectionResponse(
            result_image_url="/static/results/detected/test.jpg",
            object_count=3,
            detection_summary={"banana": 2, "cluster": 1},
            processing_time_ms=75.5,
        )
        assert isinstance(r.detection_summary, dict)
        for k, v in r.detection_summary.items():
            assert isinstance(k, str)
            assert isinstance(v, int)


class TestHealthResponse:
    def test_ok_status(self):
        r = HealthResponse(status="ok", model_loaded=True, db_connected=True)
        assert r.status == "ok"
        assert r.model_loaded is True
        assert r.db_connected is True

    def test_degraded_status(self):
        r = HealthResponse(status="degraded", model_loaded=True, db_connected=False)
        assert r.status == "degraded"
        assert r.db_connected is False

    def test_missing_fields_raise(self):
        with pytest.raises(ValidationError):
            HealthResponse(status="ok")
            # model_loaded and db_connected are required
