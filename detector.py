import time
import cv2
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

from ultralytics import YOLO

# OpenCV uses BGR order
_BBOX_COLORS = [
    (164, 120,  87), ( 68, 148, 228), ( 93,  97, 209), (178, 182, 133),
    ( 88, 159, 106), ( 96, 202, 231), (159, 124, 168), (169, 162, 241),
    ( 98, 118, 150), (172, 176, 184),
]


@dataclass
class DetectionResult:
    object_count: int
    detection_summary: Dict[str, int]
    processing_time_ms: float
    output_path: str


class YOLODetector:
    """Loads the YOLO model once and exposes a detect() method reused across requests."""

    def __init__(self, model_path: str) -> None:
        self._model = YOLO(model_path, task="detect")
        self._labels: Dict[int, str] = self._model.names

    def detect(self, image_path: str, output_path: str, confidence: float = 0.30) -> DetectionResult:
        start = time.perf_counter()

        frame = cv2.imread(image_path)
        if frame is None:
            raise ValueError(f"Cannot read image: {image_path}")

        results = self._model(frame, verbose=False)
        boxes = results[0].boxes

        object_count = 0
        class_summary: Dict[str, int] = {}

        for box in boxes:
            conf = float(box.conf)
            if conf < confidence:
                continue

            xmin, ymin, xmax, ymax = box.xyxy.cpu().numpy().squeeze().astype(int)
            class_idx = int(box.cls)
            class_name = self._labels[class_idx]

            object_count += 1
            class_summary[class_name] = class_summary.get(class_name, 0) + 1

            color = _BBOX_COLORS[class_idx % len(_BBOX_COLORS)]
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)

            label = f"{class_name}: {int(conf * 100)}%"
            (lw, lh), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_y = max(ymin, lh + 10)

            cv2.rectangle(
                frame,
                (xmin, label_y - lh - 10),
                (xmin + lw, label_y + baseline - 10),
                color, cv2.FILLED,
            )
            cv2.putText(frame, label, (xmin, label_y - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        cv2.putText(frame, f"Total: {object_count}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(output_path, frame):
            raise IOError(f"Failed to write output image: {output_path}")

        elapsed_ms = round((time.perf_counter() - start) * 1000, 1)
        return DetectionResult(
            object_count=object_count,
            detection_summary=class_summary,
            processing_time_ms=elapsed_ms,
            output_path=output_path,
        )
