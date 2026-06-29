FROM python:3.11-slim

WORKDIR /app

# System libs needed by OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libwebp7 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ── Layer 1: PyTorch CPU (heaviest — isolated so OOM doesn't redo everything) ─
# CPU-only wheel is ~200 MB vs ~2 GB for the default GPU build.
RUN pip install --no-cache-dir \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

# ── Layer 2: ultralytics (needs torch already present to avoid re-downloading) ─
COPY requirements.txt .
RUN pip install --no-cache-dir ultralytics>=8.3.0

# ── Layer 3: remaining lightweight packages ────────────────────────────────────
RUN pip install --no-cache-dir \
    "fastapi>=0.115.0" \
    "uvicorn[standard]>=0.30.0" \
    "python-multipart>=0.0.9" \
    "jinja2>=3.1.4" \
    "python-dotenv>=1.0.0" \
    "opencv-python-headless>=4.10.0" \
    "mysql-connector-python>=8.3.0"

# Copy source code
COPY main.py detector.py database.py schemas.py ./
COPY templates/        templates/
COPY Models/           Models/
COPY static/images/    static/images/

# Pre-create output directories (volumes will overlay static/results at runtime)
RUN mkdir -p static/results/detected static/uploads

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
