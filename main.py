import shutil
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import database
from detector import YOLODetector
from schemas import DetectionResponse, HealthResponse

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
MODEL_PATH  = BASE_DIR / "Models" / "my_model.pt"
STATIC_DIR  = BASE_DIR / "static"
RESULTS_DIR = STATIC_DIR / "results" / "detected"
UPLOADS_DIR = STATIC_DIR / "uploads"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

# ── App state ──────────────────────────────────────────────────────────────────
detector: YOLODetector | None = None


# ── Lifespan ───────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global detector

    print(f"Loading YOLO model from: {MODEL_PATH}")
    if not MODEL_PATH.exists():
        print(f"  WARNING: model file not found at {MODEL_PATH}")
    else:
        detector = YOLODetector(str(MODEL_PATH))
        print("  YOLO model loaded successfully.")

    try:
        database.wait_for_db()
        database.init_pool()
        print("  Database connection pool ready.")
    except Exception as exc:
        print(f"  WARNING: Database unavailable — running without persistence. ({exc})")

    yield
    print("Shutdown complete.")


# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Banana Tree Detection API",
    description=(
        "AI-powered detection and counting of bananas and leaves using Advance detection Technology. "
        "Upload an image and receive annotated results with per-class counts."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    history = database.get_recent_detections(limit=8) if database.db_available else []
    return templates.TemplateResponse(request, "index.html", {
        "model_loaded": detector is not None,
        "history":      history,
    })


@app.post("/detect", response_model=DetectionResponse)
async def detect(
    image_file: UploadFile = File(..., description="Image to analyse (.jpg / .png / .webp)"),
    confidence: float      = Form(0.30, ge=0.10, le=0.95, description="Detection confidence threshold"),
):
    if detector is None:
        raise HTTPException(503, "YOLO model is not loaded. Check that Models/my_model.pt exists.")

    ext = Path(image_file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type '{ext}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}")

    uid             = str(uuid.uuid4())[:8]
    upload_path     = UPLOADS_DIR / f"{uid}{ext}"
    output_filename = f"detected_{uid}{ext}"
    output_path     = RESULTS_DIR / output_filename

    try:
        with upload_path.open("wb") as f:
            shutil.copyfileobj(image_file.file, f)

        result = detector.detect(str(upload_path), str(output_path), confidence=confidence)

        result_url   = f"/static/results/detected/{output_filename}"
        detection_id = None

        if database.db_available:
            try:
                detection_id = database.save_detection(
                    input_filename    = image_file.filename or "unknown",
                    result_filepath   = result_url,
                    object_count      = result.object_count,
                    detection_summary = result.detection_summary,
                )
            except Exception as exc:
                print(f"DB save failed (non-fatal): {exc}")

        return DetectionResponse(
            id                 = detection_id,
            result_image_url   = result_url,
            object_count       = result.object_count,
            detection_summary  = result.detection_summary,
            processing_time_ms = result.processing_time_ms,
        )

    finally:
        if upload_path.exists():
            upload_path.unlink()


@app.get("/health", response_model=HealthResponse)
async def health():
    db_ok = False
    if database.db_available:
        try:
            database.get_recent_detections(limit=1)
            db_ok = True
        except Exception:  # noqa: S110
            pass

    return HealthResponse(
        status       = "ok" if (detector is not None and db_ok) else "degraded",
        model_loaded = detector is not None,
        db_connected = db_ok,
    )


@app.get("/history")
async def get_history(limit: int = 10):
    if not database.db_available:
        raise HTTPException(503, "Database not available.")
    return database.get_recent_detections(limit=min(limit, 50))
