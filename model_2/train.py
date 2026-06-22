"""
mPowered Banana Detection — Training Script
Model: YOLOv11m  (recommended) | fallback: YOLOv8m

WHY YOLOv11m over the current models?
  - Current my_model.pt:      2.59M params  → too small, low mAP ceiling
  - Current my_model_yolo8n:  9.43M params  → still limited accuracy
  - YOLOv11m:                20.1M params   → 51.5 mAP on COCO
  - YOLOv8m  (alternative):  25.9M params   → 50.2 mAP on COCO

YOLOv11m gives us ~10-15% more mAP on domain data vs the current nano
models, while still running in <50ms on a CPU and <10ms on a T4 GPU.
It fits comfortably in the FastAPI Docker container.

Usage:
  python train.py                                # uses all defaults
  python train.py --weights yolo11l.pt           # larger model
  python train.py --weights yolov8m.pt           # alternative architecture
  python train.py --imgsz 1280                   # default; required for individual banana counting
  python train.py --resume                       # resume interrupted run
  python train.py --device cpu                   # no GPU available
  python train.py --device mps                   # Apple Silicon Mac
"""

import argparse
import sys
from pathlib import Path

import yaml
from ultralytics import YOLO

BASE       = Path(__file__).parent
HYPER_FILE = BASE / "configs" / "hyperparams.yaml"
RESULTS    = BASE / "results"


def load_hyperparams() -> dict:
    with open(HYPER_FILE) as f:
        return yaml.safe_load(f)


def parse_args():
    hp = load_hyperparams()
    p  = argparse.ArgumentParser(
        description="Train mPowered banana detection model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--weights",
        default="yolo11m.pt",
        help=(
            "Starting weights. Options:\n"
            "  yolo11m.pt  (recommended — best accuracy/speed trade-off)\n"
            "  yolo11l.pt  (more accurate, needs more VRAM)\n"
            "  yolov8m.pt  (older but very stable alternative)\n"
            "  yolov8l.pt  (older, higher accuracy)\n"
            "  A local .pt file to fine-tune from a previous run."
        ),
    )
    p.add_argument("--data",    default=str(BASE / "dataset.yaml"),
                   help="Path to dataset.yaml")
    p.add_argument("--epochs",  type=int,   default=150)
    p.add_argument("--imgsz",   type=int,   default=hp.get("imgsz", 1280),
                   help="Input image size. 1280=default (individual bananas are small objects), 640=faster")
    p.add_argument("--batch",   type=int,   default=16,
                   help="Batch size. Reduce to 8 if VRAM < 8 GB")
    p.add_argument("--device",  default="0",
                   help="0 (GPU), cpu, mps (Apple Silicon), 0,1 (multi-GPU)")
    p.add_argument("--name",    default="banana_v1",
                   help="Run name — results saved to results/<name>/")
    p.add_argument("--resume",  action="store_true",
                   help="Resume the last interrupted training run")
    p.add_argument("--no-cache", dest="cache", action="store_false",
                   help="Disable RAM image caching (use if RAM < 16 GB)")
    p.set_defaults(cache=True)
    return p.parse_args(), hp


def main():
    args, hp = parse_args()

    # ── Validate dataset config exists ───────────────────────
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"[ERROR] dataset.yaml not found: {data_path}")
        print("  → Edit model_2/dataset.yaml and set the correct `path:`")
        sys.exit(1)

    # ── Print plan ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  mPowered Banana Detection — Training")
    print("=" * 60)
    print(f"  Architecture : {args.weights}")
    print(f"  Dataset      : {args.data}")
    print(f"  Epochs       : {args.epochs}")
    print(f"  Image size   : {args.imgsz}px")
    print(f"  Batch size   : {args.batch}")
    print(f"  Device       : {args.device}")
    print(f"  Output dir   : {RESULTS / args.name}")
    print(f"  RAM cache    : {'yes' if args.cache else 'no'}")
    print("=" * 60 + "\n")

    # ── Load model ───────────────────────────────────────────
    # YOLO() with a model name (e.g. "yolo11m.pt") automatically downloads
    # COCO-pretrained weights on first run. This gives us a strong starting
    # point and means we only need to fine-tune, not train from scratch.
    model = YOLO(args.weights)

    # ── Train ────────────────────────────────────────────────
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(RESULTS),
        name=args.name,
        resume=args.resume,

        # ── Optimizer (from hyperparams.yaml) ─────────────────
        optimizer=hp["optimizer"],
        lr0=hp["lr0"],
        lrf=hp["lrf"],
        momentum=hp["momentum"],
        weight_decay=hp["weight_decay"],
        warmup_epochs=hp["warmup_epochs"],
        warmup_momentum=hp["warmup_momentum"],
        warmup_bias_lr=hp["warmup_bias_lr"],

        # ── Loss weights ──────────────────────────────────────
        box=hp["box"],
        cls=hp["cls"],
        dfl=hp["dfl"],

        # ── Augmentation ──────────────────────────────────────
        hsv_h=hp["hsv_h"],
        hsv_s=hp["hsv_s"],
        hsv_v=hp["hsv_v"],
        degrees=hp["degrees"],
        translate=hp["translate"],
        scale=hp["scale"],
        shear=hp["shear"],
        perspective=hp["perspective"],
        flipud=hp["flipud"],
        fliplr=hp["fliplr"],
        mosaic=hp["mosaic"],
        close_mosaic=hp["close_mosaic"],
        copy_paste=hp["copy_paste"],
        copy_paste_mode=hp["copy_paste_mode"],
        mixup=hp["mixup"],

        # ── Training quality ──────────────────────────────────
        label_smoothing=hp["label_smoothing"],
        patience=hp["patience"],

        # ── Misc ──────────────────────────────────────────────
        val=True,           # Validate after every epoch
        save=True,          # Save best.pt and last.pt
        save_period=-1,     # Only save best (not every epoch checkpoint)
        plots=True,         # Generate training curve plots
        verbose=True,
        seed=42,
        deterministic=True,
        amp=True,           # Mixed-precision — faster on modern GPUs
        cache=args.cache,
        single_cls=False,
    )

    # ── Summary ───────────────────────────────────────────────
    best = RESULTS / args.name / "weights" / "best.pt"
    print("\n" + "=" * 60)
    print("  Training complete!")
    print(f"  Best model  : {best}")
    print(f"  Results dir : {RESULTS / args.name}/")
    print(f"  mAP50       : {results.results_dict.get('metrics/mAP50(B)', '?'):.4f}")
    print(f"  mAP50-95    : {results.results_dict.get('metrics/mAP50-95(B)', '?'):.4f}")
    print("=" * 60)

    # ── Export to ONNX ───────────────────────────────────────
    # ONNX lets you run the model without PyTorch and is the standard
    # format for production deployment / serving with ONNX Runtime.
    if best.exists():
        print("\n  Exporting best.pt → ONNX …")
        export_model = YOLO(str(best))
        onnx_path = export_model.export(
            format="onnx",
            imgsz=args.imgsz,
            simplify=True,
            dynamic=False,
        )
        print(f"  ONNX model  : {onnx_path}")

    print("\n  Next steps:")
    print(f"    Evaluate : python evaluate.py --weights {best}")
    print(f"    Benchmark: python benchmark.py --test_dir /path/to/test/images")
    print(f"    Deploy   : copy {best} to ../Models/ and restart Docker\n")

    return results


if __name__ == "__main__":
    main()
