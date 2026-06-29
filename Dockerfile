FROM python:3.11-slim

WORKDIR /app

# System libs: OpenCV + build tools for any C-extension wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libwebp7 \
        curl \
        gcc \
        g++ \
    && rm -rf /var/lib/apt/lists/*

# ── Layer 1: PyTorch CPU ───────────────────────────────────────────────────────
# Installed before ultralytics so it never pulls the 2 GB GPU wheel.
RUN pip install --no-cache-dir \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

# ── Layer 2: ultralytics ───────────────────────────────────────────────────────
# Quoted so /bin/sh does not treat > as a file-redirect operator.
# --extra-index-url lets pip verify CPU torch compatibility without re-downloading.
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    "ultralytics>=8.3.0"

# ── Layer 3: remaining app dependencies ────────────────────────────────────────
RUN pip install --no-cache-dir \
    "fastapi>=0.115.0" \
    "uvicorn[standard]>=0.30.0" \
    "python-multipart>=0.0.9" \
    "jinja2>=3.1.4" \
    "python-dotenv>=1.0.0" \
    "opencv-python-headless>=4.10.0" \
    "mysql-connector-python>=8.3.0"

# ── Copy source ────────────────────────────────────────────────────────────────
COPY requirements.txt .
COPY main.py detector.py database.py schemas.py ./
COPY templates/        templates/
COPY Models/           Models/
COPY static/images/    static/images/

RUN mkdir -p static/results/detected static/uploads

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
