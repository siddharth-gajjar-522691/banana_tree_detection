"""
mPowered Banana Detection — Evaluation Script

WHY a separate evaluate.py?
  train.py validates after every epoch (quick online validation).
  This script does a *thorough* offline evaluation on a held-out test set,
  produces a human-readable report, and saves everything to JSON so results
  can be tracked and compared across model versions.

Usage:
  # After training completes:
  python evaluate.py --weights results/banana_v1/weights/best.pt

  # Compare against the old production models:
  python evaluate.py --weights ../Models/my_model.pt
  python evaluate.py --weights ../Models/my_model_yolo8n.pt

  # Evaluate on a specific split:
  python evaluate.py --weights best.pt --split test
  python evaluate.py --weights best.pt --split val

Output:
  results/eval_<run_name>/
  ├── metrics.json          ← machine-readable results (for CI/CD)
  ├── confusion_matrix.png  ← class confusion matrix
  ├── PR_curve.png          ← precision-recall curve per class
  ├── F1_curve.png          ← F1 vs confidence threshold
  └── summary.txt           ← human-readable report
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from ultralytics import YOLO

BASE = Path(__file__).parent
RESULTS = BASE / "results"


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate a YOLO model on the banana detection dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--weights",
        required=True,
        help="Path to .pt model weights (e.g. results/banana_v1/weights/best.pt)",
    )
    p.add_argument(
        "--data",
        default=str(BASE / "dataset.yaml"),
        help="Path to dataset.yaml",
    )
    p.add_argument(
        "--split",
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate on (use 'test' for final reporting)",
    )
    p.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size (match what you trained with)",
    )
    p.add_argument(
        "--conf",
        type=float,
        default=0.30,
        help="Confidence threshold for detections",
    )
    p.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="NMS IoU threshold (match hyperparams.yaml)",
    )
    p.add_argument(
        "--device",
        default="cpu",
        help="Device: 0 (GPU), cpu, mps (Apple Silicon)",
    )
    p.add_argument(
        "--name",
        default=None,
        help="Output directory name under results/ (default: eval_<model_name>)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    weights_path = Path(args.weights)
    if not weights_path.exists():
        print(f"[ERROR] Weights file not found: {weights_path}")
        raise SystemExit(1)

    run_name = args.name or f"eval_{weights_path.stem}"
    out_dir = RESULTS / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("  mPowered — Model Evaluation")
    print("=" * 60)
    print(f"  Model   : {weights_path}")
    print(f"  Dataset : {args.data} [{args.split}]")
    print(f"  Conf    : {args.conf}")
    print(f"  IoU     : {args.iou}")
    print(f"  Output  : {out_dir}")
    print("=" * 60 + "\n")

    model = YOLO(str(weights_path))

    # ── Run validation ──────────────────────────────────────
    # model.val() returns a Results object with all standard YOLO metrics:
    #   box.map   = mAP50-95 (the primary metric for academic comparison)
    #   box.map50 = mAP50    (more intuitive, used in most agri papers)
    #   box.maps  = per-class mAP50-95
    #   box.mp    = mean precision
    #   box.mr    = mean recall
    t0 = time.perf_counter()
    metrics = model.val(
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        project=str(RESULTS),
        name=run_name,
        plots=True,  # Generates confusion matrix, PR curve, F1 curve
        save_json=True,  # COCO-format JSON for external tools
        verbose=True,
        exist_ok=True,
    )
    elapsed = time.perf_counter() - t0

    # ── Extract per-class metrics ───────────────────────────
    class_names = model.names  # {0: 'banana_bunch', 1: 'leaf', ...}
    per_class_ap = metrics.box.maps  # list of mAP50-95 per class

    per_class: dict = {}
    for idx, name in class_names.items():
        per_class[name] = {
            "mAP50-95": round(float(per_class_ap[idx]), 4) if idx < len(per_class_ap) else None,
        }

    # ── Build report dict ───────────────────────────────────
    report = {
        "timestamp": datetime.now().isoformat(),
        "model": str(weights_path),
        "dataset": args.data,
        "split": args.split,
        "image_size": args.imgsz,
        "conf_threshold": args.conf,
        "iou_threshold": args.iou,
        "eval_time_s": round(elapsed, 2),
        "overall": {
            "mAP50": round(float(metrics.box.map50), 4),
            "mAP50-95": round(float(metrics.box.map), 4),
            "precision": round(float(metrics.box.mp), 4),
            "recall": round(float(metrics.box.mr), 4),
            "F1": round(
                2
                * float(metrics.box.mp)
                * float(metrics.box.mr)
                / (float(metrics.box.mp) + float(metrics.box.mr) + 1e-9),
                4,
            ),
        },
        "per_class": per_class,
    }

    # ── Save JSON ───────────────────────────────────────────
    json_path = out_dir / "metrics.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)

    # ── Write human-readable summary ────────────────────────
    summary_lines = [
        "=" * 60,
        "  mPowered Banana Detection — Evaluation Report",
        "=" * 60,
        f"  Model     : {weights_path}",
        f"  Dataset   : {args.data} [{args.split}]",
        f"  Timestamp : {report['timestamp']}",
        f"  Eval time : {elapsed:.1f}s",
        "",
        "  OVERALL METRICS",
        "  ---------------",
        f"  mAP50       : {report['overall']['mAP50']:.4f}",
        f"  mAP50-95    : {report['overall']['mAP50-95']:.4f}",
        f"  Precision   : {report['overall']['precision']:.4f}",
        f"  Recall      : {report['overall']['recall']:.4f}",
        f"  F1          : {report['overall']['F1']:.4f}",
        "",
        "  PER-CLASS mAP50-95",
        "  -------------------",
    ]

    for cls, vals in per_class.items():
        ap = vals["mAP50-95"]
        bar = "█" * int((ap or 0) * 40)
        summary_lines.append(f"  {cls:<20} {ap:.4f}  {bar}")

    summary_lines += [
        "",
        "  OUTPUT FILES",
        "  ------------",
        f"  JSON report       : {json_path}",
        f"  Confusion matrix  : {out_dir}/confusion_matrix.png",
        f"  PR curve          : {out_dir}/PR_curve.png",
        f"  F1 curve          : {out_dir}/F1_curve.png",
        "",
        "  HOW TO INTERPRET",
        "  ----------------",
        "  mAP50     : detection accuracy at IoU ≥ 0.5 (more lenient)",
        "             > 0.60 = production ready for counting",
        "             > 0.75 = excellent",
        "  mAP50-95  : stricter — averages over IoU 0.5→0.95",
        "             > 0.45 = production ready",
        "  Precision  : of what the model detected, % that were correct",
        "  Recall     : of all real objects, % the model found",
        "  F1        : balance of precision and recall (harmonic mean)",
        "=" * 60,
    ]

    summary_text = "\n".join(summary_lines)
    print("\n" + summary_text)

    summary_path = out_dir / "summary.txt"
    with open(summary_path, "w") as f:
        f.write(summary_text + "\n")

    print(f"\n  Full report saved: {json_path}")
    print(f"  Summary saved    : {summary_path}\n")

    return report


if __name__ == "__main__":
    main()
