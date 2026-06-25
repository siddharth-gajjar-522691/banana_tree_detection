"""
mPowered Banana Detection — Benchmark Script

WHY benchmark.py?
  You have two existing models in production:
    - my_model.pt       (2.59M params, classes: Banana/Bunch/Cluster/Male Flower)
    - my_model_yolo8n.pt (9.43M params, classes: Banana/Leaf/Cluster/Male Flower)
  And a new trained model from model_2/train.py.

  This script runs all three on the same test images and produces a
  side-by-side comparison of speed (ms/image) and accuracy (object count
  and confidence), so you can decide with data whether to promote the new
  model to production.

  It does NOT require a labelled dataset — it runs inference on raw images
  and reports detection counts + average confidence. If you do have labels,
  run evaluate.py instead for mAP metrics.

Usage:
  # Compare all three models on a folder of images:
  python benchmark.py --test_dir /path/to/test/images

  # Only compare specific models:
  python benchmark.py --test_dir /path --models ../Models/my_model.pt results/banana_v1/weights/best.pt

  # Save annotated images from every model (for visual comparison):
  python benchmark.py --test_dir /path --save_images

Output:
  results/benchmark/
  ├── comparison.json       ← full per-image results for all models
  ├── summary_table.txt     ← ASCII table for quick reading
  └── annotated/            ← (if --save_images) side-by-side comparisons
"""

import argparse
import json
import time
from pathlib import Path

BASE    = Path(__file__).parent
MODELS  = BASE.parent / "Models"
RESULTS = BASE / "results" / "benchmark"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


def parse_args():
    default_models = [
        str(MODELS / "my_model.pt"),
        str(MODELS / "my_model_yolo8n.pt"),
    ]
    p = argparse.ArgumentParser(
        description="Benchmark multiple YOLO models on the same images",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--test_dir",
        required=True,
        help="Directory containing test images",
    )
    p.add_argument(
        "--models",
        nargs="+",
        default=default_models,
        help="List of .pt model paths to benchmark",
    )
    p.add_argument(
        "--new_model",
        default=str(BASE / "results" / "banana_v1" / "weights" / "best.pt"),
        help="Path to the new model_2 trained weights (added automatically)",
    )
    p.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size",
    )
    p.add_argument(
        "--conf",
        type=float,
        default=0.30,
        help="Confidence threshold",
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
        help="cpu | 0 (GPU) | mps",
    )
    p.add_argument(
        "--save_images",
        action="store_true",
        help="Save annotated output images from every model",
    )
    p.add_argument(
        "--max_images",
        type=int,
        default=100,
        help="Maximum number of images to benchmark (for speed)",
    )
    return p.parse_args()


def find_images(directory: str, max_images: int) -> list[Path]:
    d = Path(directory)
    if not d.exists():
        print(f"[ERROR] Test directory not found: {d}")
        raise SystemExit(1)
    images = [
        f for f in sorted(d.rglob("*"))
        if f.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if not images:
        print(f"[ERROR] No images found in {d}")
        raise SystemExit(1)
    return images[:max_images]


def benchmark_model(
    model_path: str,
    images: list[Path],
    imgsz: int,
    conf: float,
    iou: float,
    device: str,
    save_dir: Path | None = None,
) -> dict:
    from ultralytics import YOLO

    path = Path(model_path)
    if not path.exists():
        return {"error": f"File not found: {model_path}", "skipped": True}

    model = YOLO(str(path))
    class_names = model.names

    per_image = []
    latencies  = []

    for img in images:
        t0 = time.perf_counter()
        results = model.predict(
            source=str(img),
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            device=device,
            verbose=False,
            save=save_dir is not None,
            project=str(save_dir) if save_dir else None,
            name=path.stem if save_dir else None,
            exist_ok=True,
        )
        ms = (time.perf_counter() - t0) * 1000
        latencies.append(ms)

        r = results[0]
        boxes = r.boxes

        detections = []
        if boxes is not None and len(boxes) > 0:
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                conf_v = float(boxes.conf[i].item())
                detections.append({
                    "class":      class_names.get(cls_id, str(cls_id)),
                    "confidence": round(conf_v, 3),
                })

        per_image.append({
            "image":      img.name,
            "detections": len(detections),
            "latency_ms": round(ms, 1),
            "classes":    detections,
        })

    avg_lat  = sum(latencies) / len(latencies)
    fps      = 1000.0 / avg_lat if avg_lat > 0 else 0
    avg_dets = sum(d["detections"] for d in per_image) / len(per_image)

    return {
        "model":             str(path),
        "params_M":          round(sum(p.numel() for p in model.model.parameters()) / 1e6, 2),
        "classes":           list(class_names.values()),
        "images_tested":     len(images),
        "avg_latency_ms":    round(avg_lat, 1),
        "fps":               round(fps, 1),
        "p50_latency_ms":    round(sorted(latencies)[len(latencies) // 2], 1),
        "p95_latency_ms":    round(sorted(latencies)[int(len(latencies) * 0.95)], 1),
        "avg_detections":    round(avg_dets, 2),
        "per_image":         per_image,
    }


def print_summary_table(results: list[dict]) -> str:
    lines = [
        "",
        "=" * 78,
        "  mPowered — Model Benchmark Summary",
        "=" * 78,
        f"  {'Model':<30} {'Params':>8}  {'Lat(ms)':>8}  {'FPS':>6}  {'Avg Det':>8}  {'Classes'}",
        "  " + "-" * 76,
    ]
    for r in results:
        if r.get("skipped"):
            lines.append(f"  {Path(r['model']).name:<30}  SKIPPED — {r['error']}")
            continue
        name   = Path(r["model"]).name
        params = f"{r['params_M']}M"
        lat    = f"{r['avg_latency_ms']}ms"
        fps    = f"{r['fps']}"
        dets   = f"{r['avg_detections']}"
        cls    = ", ".join(r["classes"][:3])
        if len(r["classes"]) > 3:
            cls += "…"
        lines.append(f"  {name:<30} {params:>8}  {lat:>8}  {fps:>6}  {dets:>8}  {cls}")

    lines += [
        "=" * 78,
        "",
        "  HOW TO READ THIS TABLE",
        "  Params  : model size — larger = potentially more accurate",
        "  Lat(ms) : average inference time per image",
        "  FPS     : frames per second (throughput)",
        "  Avg Det : average detections per image at the given conf threshold",
        "",
        "  DECISION GUIDE",
        "  If new model has ≥ same avg detections AND lower latency → clear win",
        "  If new model has more detections → check false positive rate via evaluate.py",
        "=" * 78,
    ]
    return "\n".join(lines)


def main():
    args = parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    save_dir = RESULTS / "annotated" if args.save_images else None

    # Build model list: default production models + new model
    model_paths = list(args.models)
    new_model   = Path(args.new_model)
    if new_model.exists() and str(new_model) not in model_paths:
        model_paths.append(str(new_model))
    elif not new_model.exists():
        print(f"  [INFO] New model not found yet: {new_model}")
        print("         Run train.py first, then re-run benchmark.py")

    images = find_images(args.test_dir, args.max_images)
    print(f"\n  Found {len(images)} test images in {args.test_dir}")

    all_results = []
    for mp in model_paths:
        print(f"\n  Benchmarking: {mp}")
        r = benchmark_model(
            mp, images,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            save_dir=save_dir,
        )
        all_results.append(r)
        if not r.get("skipped"):
            print(f"    ✓ avg {r['avg_latency_ms']}ms  |  {r['fps']} FPS  |  {r['avg_detections']} dets/image")

    # Save full JSON
    json_path = RESULTS / "comparison.json"
    payload = {
        "timestamp":   time.strftime("%Y-%m-%dT%H:%M:%S"),
        "test_dir":    args.test_dir,
        "imgsz":       args.imgsz,
        "conf":        args.conf,
        "iou":         args.iou,
        "device":      args.device,
        "models":      all_results,
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    table = print_summary_table(all_results)
    print(table)

    txt_path = RESULTS / "summary_table.txt"
    with open(txt_path, "w") as f:
        f.write(table + "\n")

    print(f"  Full JSON : {json_path}")
    print(f"  Table     : {txt_path}\n")


if __name__ == "__main__":
    main()
