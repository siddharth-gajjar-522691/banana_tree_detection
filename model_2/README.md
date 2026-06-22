# mPowered Banana Detection — Model v2

Production-grade YOLOv11m model for banana plantation detection and counting.

---

## Why a new model?

| | Current `my_model.pt` | Current `my_model_yolo8n.pt` | **New model_2** |
|---|---|---|---|
| Architecture | YOLOv8n (nano) | YOLOv8s (small) | **YOLOv11m (medium)** |
| Parameters | 2.59M | 9.43M | **20.1M** |
| COCO mAP50-95 | ~37% | ~44% | **51.5%** |
| Classes | Banana, Bunch, Cluster, Male Flower | Banana, Leaf, Cluster, Male Flower | **banana_bunch, leaf, cluster, male_flower** |
| Class naming | Inconsistent across models | Inconsistent | **Unified, snake_case** |

The nano and small models leave significant accuracy on the table. YOLOv11m gives ~10–15% more mAP on domain data while still running in <50ms on CPU and <10ms on a T4 GPU — fully compatible with the FastAPI Docker container.

---

## Folder structure

```
model_2/
├── dataset.yaml              ← Dataset paths and class definitions (edit path before training)
├── configs/
│   └── hyperparams.yaml      ← All training hyperparameters (justified for banana domain)
├── train.py                  ← Main training script
├── evaluate.py               ← Post-training evaluation (mAP, precision, recall, plots)
├── benchmark.py              ← Speed + accuracy comparison vs production models
├── predict.py                ← Standalone inference on images or directories
├── requirements.txt          ← Python deps for training environment
└── results/                  ← Auto-created; all training outputs go here
    ├── banana_v1/            ← Training run outputs
    │   ├── weights/
    │   │   ├── best.pt       ← Best checkpoint (use this in production)
    │   │   └── last.pt       ← Last epoch checkpoint
    │   ├── results.csv       ← Per-epoch metrics
    │   ├── confusion_matrix.png
    │   ├── PR_curve.png
    │   └── F1_curve.png
    ├── eval_banana_v1/       ← evaluate.py outputs
    └── benchmark/            ← benchmark.py outputs
```

---

## Step 1 — Prepare your dataset

Your dataset must be in **YOLO format**:

```
dataset/
├── images/
│   ├── train/    ← .jpg / .png images
│   ├── val/      ← 15-20% of total
│   └── test/     ← 10% held-out (never used in training)
└── labels/
    ├── train/    ← .txt files (same names as images)
    ├── val/
    └── test/
```

Each `.txt` label file has one row per object:
```
<class_id> <x_center> <y_center> <width> <height>
```
All values are normalised to `[0, 1]` relative to the image dimensions.

**Class IDs:**
| ID | Name | Use for |
|---|---|---|
| 0 | banana_bunch | Full hanging bunch (yield counting) |
| 1 | leaf | Individual leaf (health/disease) |
| 2 | cluster | Fruit hand within a bunch |
| 3 | male_flower | Male inflorescence (growth stage) |

**Recommended dataset size:**
- Minimum: 800 train / 150 val / 150 test
- Target: 2000+ train images
- Include: different times of day, weather, distances, plantation varieties

**Tip:** Use [Roboflow](https://roboflow.com) to annotate and export directly in YOLO format. It handles splitting automatically.

Then update `dataset.yaml`:
```yaml
path: /absolute/path/to/your/dataset
```

---

## Step 2 — Set up the environment

```bash
cd model_2
pip install -r requirements.txt

# GPU (NVIDIA) — replace cu118 with your CUDA version:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

## Step 3 — Train

```bash
# Standard run (recommended)
python train.py

# Apple Silicon Mac
python train.py --device mps

# No GPU (CPU only — slow but works)
python train.py --device cpu

# Less RAM (reduce batch if you get OOM)
python train.py --batch 8 --no-cache

# Higher resolution for better counting accuracy
python train.py --imgsz 1280 --batch 8

# Resume a run that was interrupted
python train.py --resume
```

Training saves to `results/banana_v1/`. The best checkpoint is at `results/banana_v1/weights/best.pt`.

At the end of training, the script automatically exports `best.pt` → `best.onnx` (for deployment without PyTorch).

---

## Step 4 — Evaluate

Run the full evaluation on your held-out test set:

```bash
python evaluate.py --weights results/banana_v1/weights/best.pt
```

This produces:
- `results/eval_banana_v1/metrics.json` — machine-readable metrics
- `results/eval_banana_v1/summary.txt` — human-readable report
- Confusion matrix, PR curve, F1 curve plots

**How to read the results:**

| Metric | Minimum for production | Excellent |
|---|---|---|
| mAP50 | > 0.60 | > 0.75 |
| mAP50-95 | > 0.45 | > 0.60 |
| Precision | > 0.70 | > 0.85 |
| Recall | > 0.65 | > 0.80 |

---

## Step 5 — Benchmark vs production models

```bash
python benchmark.py --test_dir /path/to/test/images
```

This runs all three models (the two existing production models + new model_2) on the same images and outputs a comparison table:

```
Model                          Params   Lat(ms)     FPS   Avg Det
my_model.pt                      2.59M     38ms    26.3       4.2
my_model_yolo8n.pt               9.43M     52ms    19.2       5.1
best.pt  (model_2)              20.1M     48ms    20.8       6.8
```

---

## Step 6 — Deploy to production

Once you're happy with benchmark and evaluation results:

```bash
# Copy best.pt to the FastAPI app's model directory
cp results/banana_v1/weights/best.pt ../Models/banana_v2.pt
```

Then update `detector.py` in the main app:
```python
# In main.py or .env:
MODEL_PATH = "Models/banana_v2.pt"
```

Restart the Docker container:
```bash
docker compose restart app
```

The FastAPI app will load the new model on startup (check the `/health` endpoint to confirm `model_loaded: true`).

---

## Quick test on a single image

```bash
# Test the new model on one image
python predict.py --source /path/to/photo.jpg

# Test on a whole folder
python predict.py --source /path/to/images/

# Compare directly against old model
python predict.py --source /path/to/image.jpg --weights ../Models/my_model.pt
```

---

## Hyperparameter summary

Key settings in `configs/hyperparams.yaml` and why they matter for banana plantations:

| Setting | Value | Why |
|---|---|---|
| `optimizer` | AdamW | Faster convergence when fine-tuning from COCO pretrained |
| `lr0` | 0.001 | Lower than YOLO default; protects COCO backbone features |
| `hsv_s` | 0.7 | Extreme lighting variation in plantations |
| `hsv_v` | 0.4 | Deep canopy shadows vs harsh midday sun |
| `degrees` | 20 | Plantation rows shot at angles |
| `scale` | 0.5 | Close-up to full-row shot distance variation |
| `copy_paste` | 0.15 | Synthesises dense bunch scenes for counting |
| `mosaic` | 1.0 | Critical for small datasets — 4 scenes per forward pass |
| `label_smoothing` | 0.1 | Prevents overconfident outputs |
| `patience` | 40 | Small datasets can plateau; needs time to recover |
| `iou_threshold` | 0.45 | Tighter NMS — banana bunches cluster closely |

---

## Troubleshooting

**Out of memory (OOM) during training:**
```bash
python train.py --batch 8 --no-cache
```

**Training is very slow (CPU):**
```bash
# On Apple Silicon, MPS is much faster than CPU:
python train.py --device mps --batch 16
```

**"dataset.yaml not found":**
Edit `dataset.yaml` and set `path:` to your dataset's absolute path.

**Model downloads failing:**
`yolo11m.pt` weights are ~40MB and downloaded from GitHub on first run. If behind a proxy, pre-download and pass the local path:
```bash
python train.py --weights /path/to/yolo11m.pt
```

**Detections look wrong / too many false positives:**
Raise the confidence threshold:
```bash
python predict.py --source /path --conf 0.50
```
Or lower it to catch more detections at the cost of more false positives (`--conf 0.25`).
