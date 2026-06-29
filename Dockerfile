FROM python:3.11-slim

WORKDIR /app

# System libs needed by OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libwebp7 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (layer-cached until requirements change)
COPY requirements.txt .
# --extra-index-url lets pip resolve ALL deps together in one pass.
# torch/torchvision come from the CPU wheel index (~200 MB vs 2 GB GPU build).
# ultralytics sees torch already satisfied and skips the GPU download.
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    torch torchvision \
    -r requirements.txt

# Copy source code
COPY main.py detector.py database.py schemas.py ./
COPY templates/        templates/
COPY Models/           Models/
COPY static/images/    static/images/

# Pre-create output directories (volumes will overlay static/results at runtime)
RUN mkdir -p static/results/detected static/uploads

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
