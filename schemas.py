from pydantic import BaseModel


class DetectionResponse(BaseModel):
    success: bool = True
    id: int | None = None
    result_image_url: str
    object_count: int
    detection_summary: dict[str, int]
    processing_time_ms: float
    message: str = "Detection complete"


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    db_connected: bool
