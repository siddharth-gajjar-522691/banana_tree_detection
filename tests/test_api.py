"""
Tests for FastAPI endpoints.

WHY mock the YOLO model and database?
  In CI there is no GPU, no model file, and no MySQL server.
  We use unittest.mock to replace those dependencies with controlled fakes.
  This tests the routing, validation, and response structure — which is
  exactly what can break when you refactor the app.

  The real model performance is tested in model_2/evaluate.py separately.
"""

import io
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# ── Stub out ultralytics before importing main ──────────────────────────────
# WHY: ultralytics imports torch, which is huge. In CI we only install
# CPU torch to keep install time reasonable. We stub the YOLO class so
# detector.py can be imported without any model file present.
def _make_ultralytics_stub():
    stub = types.ModuleType("ultralytics")

    class FakeYOLO:
        def __init__(self, path, task=None):
            self.names = {0: "banana", 1: "leaf", 2: "cluster"}

        def __call__(self, *args, **kwargs):
            return []

    stub.YOLO = FakeYOLO
    return stub


sys.modules.setdefault("ultralytics", _make_ultralytics_stub())

# ── Patch database before import ────────────────────────────────────────────
import database as _db_module  # noqa: E402

_db_module.db_available = False  # Simulate no DB in CI
_db_module.get_recent_detections = lambda limit=10: []
_db_module.save_detection = lambda **kwargs: None
_db_module.wait_for_db = lambda *a, **kw: None
_db_module.init_pool = lambda *a, **kw: None

# ── Now import the app ───────────────────────────────────────────────────────
import main as _main_module  # noqa: E402

# Build test client WITHOUT triggering the real lifespan
# (which would try to load Models/my_model.pt and connect to MySQL)

client = TestClient(_main_module.app, raise_server_exceptions=True)


# ── Fixture: inject a mock detector ─────────────────────────────────────────
@dataclass
class FakeDetectionResult:
    object_count: int
    detection_summary: dict[str, int]
    processing_time_ms: float


@pytest.fixture()
def mock_detector(tmp_path):
    """Replace the global detector with a fake that returns predictable results."""
    fake = MagicMock()
    fake.detect.return_value = FakeDetectionResult(
        object_count=7,
        detection_summary={"banana": 5, "leaf": 2},
        processing_time_ms=42.0,
    )
    # Patch the module-level `detector` variable in main.py
    with patch.object(_main_module, "detector", fake):
        # Also patch RESULTS_DIR and UPLOADS_DIR to use tmp_path
        with patch.object(_main_module, "RESULTS_DIR", tmp_path / "results"):
            with patch.object(_main_module, "UPLOADS_DIR", tmp_path / "uploads"):
                (tmp_path / "results").mkdir()
                (tmp_path / "uploads").mkdir()
                yield fake


# ── Health endpoint ──────────────────────────────────────────────────────────
class TestHealthEndpoint:
    def test_health_returns_200(self):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_shape(self):
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "db_connected" in data

    def test_health_degraded_without_model(self):
        # By default no model is loaded in tests
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "degraded"
        assert data["model_loaded"] is False
        assert data["db_connected"] is False

    def test_health_ok_with_model(self, mock_detector):
        response = client.get("/health")
        data = response.json()
        # model_loaded is True because mock_detector is not None
        assert data["model_loaded"] is True


# ── Detect endpoint ──────────────────────────────────────────────────────────
class TestDetectEndpoint:
    def _make_image_bytes(self) -> bytes:
        """Create a minimal valid PNG in memory (1x1 white pixel)."""
        import struct
        import zlib

        def chunk(name: bytes, data: bytes) -> bytes:
            c = name + data
            return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c))

        sig = b"\x89PNG\r\n\x1a\n"
        ihdr = chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
        raw = b"\x00\xff\xff\xff"
        idat = chunk(b"IDAT", zlib.compress(raw))
        iend = chunk(b"IEND", b"")
        return sig + ihdr + idat + iend

    def test_detect_without_model_returns_503(self):
        data = {"confidence": "0.30"}
        files = {"image_file": ("test.png", self._make_image_bytes(), "image/png")}
        response = client.post("/detect", data=data, files=files)
        assert response.status_code == 503

    def test_detect_with_model_returns_200(self, mock_detector, tmp_path):
        # Make the fake detector also write an output file (detector.detect is mocked)
        # We need to patch shutil.copyfileobj so the upload path is created
        image_bytes = self._make_image_bytes()

        with patch("shutil.copyfileobj"):
            with patch("main.UPLOADS_DIR", tmp_path):
                data = {"confidence": "0.30"}
                files = {"image_file": ("test.png", image_bytes, "image/png")}
                response = client.post("/detect", data=data, files=files)

        assert response.status_code == 200
        body = response.json()
        assert body["success"] is True
        assert body["object_count"] == 7
        assert body["detection_summary"] == {"banana": 5, "leaf": 2}
        assert body["processing_time_ms"] == 42.0

    def test_detect_rejects_invalid_file_type(self, mock_detector):
        files = {"image_file": ("malware.exe", b"not an image", "application/octet-stream")}
        data = {"confidence": "0.30"}
        response = client.post("/detect", data=data, files=files)
        assert response.status_code == 400

    def test_detect_rejects_low_confidence(self, mock_detector):
        files = {"image_file": ("test.png", self._make_image_bytes(), "image/png")}
        data = {"confidence": "0.05"}  # Below 0.10 minimum
        response = client.post("/detect", data=data, files=files)
        assert response.status_code == 422  # FastAPI validation error

    def test_detect_rejects_high_confidence(self, mock_detector):
        files = {"image_file": ("test.png", self._make_image_bytes(), "image/png")}
        data = {"confidence": "0.99"}  # Above 0.95 maximum
        response = client.post("/detect", data=data, files=files)
        assert response.status_code == 422


# ── Homepage ─────────────────────────────────────────────────────────────────
class TestHomepage:
    def test_homepage_returns_200(self):
        response = client.get("/")
        assert response.status_code == 200

    def test_homepage_is_html(self):
        response = client.get("/")
        assert "text/html" in response.headers["content-type"]
