"""
mPowered Banana Detection — Standalone Prediction Script

WHY predict.py?
  The FastAPI app is great for web requests, but sometimes you need to:
    - Quickly test a newly trained model on a folder of images
    - Run batch inference on a large dataset without spinning up Docker
    - Verify that class names, confidence scores, and box counts look right
      before deploying to production

  This script mirrors exactly what detector.py does in the FastAPI app,
  so if results look good here, they will look the same in production.

Usage:
  # Single image:
  python predict.py --source /path/to/image.jpg

  # Directory (all images):
  python predict.py --source /path/to/images/

  # Use a specific model (default: new model_2 best.pt):
  python predict.py --source /path --weights ../Models/my_model.pt

  # Adjust confidence threshold:
  python predict.py --source /path --conf 0.50

  # Do not save annotated images (just print counts):
  python predict.py --source /path --no-save

Output:
  results/predictions/
  └── <run_timestamp>/
      ├── annotated images (with bounding boxes drawn)
      └── detections.json  (counts and confidence per image)
"""

import argparse
import json
import time
from pathlib import Path

BASE    = Path(__file__).parent
RESULTS = BASE / "results" / "predictions"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


def parse_args():
    default_weights = BASE / "results" / "banana_v1" / "weights" / "best.pt"
    fallback_weights = BASE.parent / "Models" / "my_model_yolo8n.pt"

    p = argparse.ArgumentParser(
        description="Run mPowered banana detection on images or a directory",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--source",
        required=True,
        help="Path to an image file or a directory of images",
    )
    p.add_argument(
        "--weights",
        default=str(default_weights if default_weights.exists() else fallback_weights),
        help="Path to .pt model weights",
    )
    p.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size (px)",
    )
    p.add_argument(
        "--conf",
        type=float,
        default=0.30,
        help="Confidence threshold (0.0-1.0)",
    )
    p.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="NMS IoU threshold",
    )
    p.add_argument(
        "--device",
        default="cpu",
        help="cpu | 0 (CUDA) | mps (Apple Silicon)",
    )
    p.add_argument(
        "--max_det",
        type=int,
        default=300,
        help="Maximum detections per image",
    )
    p.add_argument(
        "--save",
        action="store_true",
        default=True,
        help="Save annotated output images",
    )
    p.add_argument(
        "--no-save",
        dest="save",
        action="store_false",
        help="Only print results, do not save annotated images",
    )
    return p.parse_args()


def collect_images(source: str) -> list[Path]:
    p = Path(source)
    if p.is_file():
        if p.suffix.lower() in IMAGE_EXTENSIONS:
            return [p]
        print(f"[ERROR] Not a supported image file: {p}")
        raise SystemExit(1)
    if p.is_dir():
        imgs = sorted(
            f for f in p.rglob("*")
            if f.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not imgs:
            print(f"[ERROR] No images found in {p}")
            raise SystemExit(1)
        return imgs
    print(f"[ERROR] Source not found: {p}")
    raise SystemExit(1)


def main():
    args   = parse_args()
    images = collect_images(args.source)

    weights = Path(args.weights)
    if not weights.exists():
        print(f"[ERROR] Weights not found: {weights}")
        print("  → Run train.py first, or pass --weights path/to/existing.pt")
        raise SystemExit(1)

    # Create timestamped output directory
    run_ts  = time.strftime("%Y%m%d_%H%M%S")
    out_dir = RESULTS / run_ts
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("  mPowered — Banana Detection (predict.py)")
    print("=" * 60)
    print(f"  Model   : {weights}")
    print(f"  Source  : {args.source}")
    print(f"  Images  : {len(images)}")
    print(f"  Conf    : {args.conf}")
    print(f"  IoU     : {args.iou}")
    print(f"  Output  : {out_dir if args.save else 'not saved'}")
    print("=" * 60 + "\n")

    from ultralytics import YOLO
    model       = YOLO(str(weights))
    class_names = model.names

    all_results = []
    total_dets  = 0

    for idx, img_path in enumerate(images, 1):
        t0 = time.perf_counter()
        preds = model.predict(
            source=str(img_path),
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            max_det=args.max_det,
            verbose=False,
            save=args.save,
            project=str(out_dir),
            name="annotated",
            exist_ok=True,
        )
        ms = (time.perf_counter() - t0) * 1000

        r    = preds[0]
        boxes = r.boxes

        class_counts: dict[str, int] = {}
        if boxes is not None and len(boxes) > 0:
            for i in range(len(boxes)):
                cls_id   = int(boxes.cls[i].item())
                cls_name = class_names.get(cls_id, str(cls_id))
                class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

        n = sum(class_counts.values())
        total_dets += n

        result_entry = {
            "image":       img_path.name,
            "path":        str(img_path),
            "total":       n,
            "latency_ms":  round(ms, 1),
            "class_counts": class_counts,
        }
        all_results.append(result_entry)

        # Pretty per-image output
        counts_str = "  ".join(f"{k}: {v}" for k, v in class_counts.items())
        print(f"  [{idx:>3}/{len(images)}]  {img_path.name:<35}  {n:>3} det  {ms:>6.1f}ms  {counts_str}")

    # ── Summary ──────────────────────────────────────────────
    avg_ms = sum(r["latency_ms"] for r in all_results) / len(all_results)
    avg_dt = total_dets / len(all_results)

    print("\n" + "-" * 60)
    print(f"  Images processed : {len(all_results)}")
    print(f"  Total detections : {total_dets}")
    print(f"  Avg per image    : {avg_dt:.1f}")
    print(f"  Avg latency      : {avg_ms:.1f}ms  ({1000/avg_ms:.1f} FPS)")
    if args.save:
        print(f"  Annotated images : {out_dir}/annotated/")
    print("-" * 60 + "\n")

    # ── Save JSON ────────────────────────────────────────────
    report = {
        "timestamp":      run_ts,
        "model":          str(weights),
        "source":         args.source,
        "conf":           args.conf,
        "iou":            args.iou,
        "summary": {
            "images":          len(all_results),
            "total_dets":      total_dets,
            "avg_dets":        round(avg_dt, 2),
            "avg_latency_ms":  round(avg_ms, 1),
            "fps":             round(1000 / avg_ms, 1),
        },
        "per_image": all_results,
    }
    json_path = out_dir / "detections.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Full results: {json_path}\n")


if __name__ == "__main__":
    main()
