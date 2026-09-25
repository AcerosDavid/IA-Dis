# ─────────────────────────────────────────────────────────────────────────────
# Mining Dispatch API — Dockerfile con soporte CUDA/GPU
#
# Imagen base: PyTorch oficial con CUDA 11.8 + cuDNN 8 sobre Ubuntu 22.04
# Soporta cualquier GPU NVIDIA con compute capability >= 3.7
#   RTX 3060/3070/3080/3090, RTX 4070/4080/4090,
#   GTX 1060/1070/1080, RTX 2070/2080, Quadro, A100, etc.
#
# Build:
#   docker build -t mining-dispatch-api .
#
# Run (GPU):
#   docker run --gpus all -p 8000:8000 mining-dispatch-api
# ─────────────────────────────────────────────────────────────────────────────

FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime

LABEL maintainer="Mining Dispatch"
LABEL description="IA-Dis API — YOLO + ByteTrack + FastAPI con GPU NVIDIA"

# ── Variables de entorno de Python ───────────────────────────────────────────
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Ultralytics: no mostrar updates ni telemetría
    YOLO_AUTOINSTALL=0 \
    ULTRALYTICS_AUTO_UPDATE=0 \
    # CUDA visible para PyTorch
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

WORKDIR /app

# ── Dependencias del sistema (OpenCV headless + utilidades) ──────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgl1-mesa-glx \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ── Dependencias Python ───────────────────────────────────────────────────────
# Copiamos requirements primero para aprovechar el cache de capas Docker
COPY requirements.txt .

# PyTorch ya viene en la imagen base — instalamos el resto sin reinstalar torch
RUN pip install --no-cache-dir \
        ultralytics>=8.3.0 \
        opencv-python-headless>=4.10.0 \
        sqlalchemy>=2.0.0 \
        alembic>=1.13.0 \
        fastapi>=0.115.0 \
        "uvicorn[standard]>=0.30.0" \
        pydantic-settings>=2.5.0 \
        python-dotenv>=1.0.0 \
        httpx>=0.27.0

# ── Código fuente ─────────────────────────────────────────────────────────────
COPY . .

# ── Directorios de datos (persistidos como volúmenes) ────────────────────────
RUN mkdir -p data models runs reports

# ── Puerto de la API ──────────────────────────────────────────────────────────
EXPOSE 8000

# ── Healthcheck ──────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/eventos/health || exit 1

# ── Entrypoint ────────────────────────────────────────────────────────────────
# Usa 2 workers: 1 para requests normales + 1 libre mientras YOLO procesa video
CMD ["uvicorn", "src.api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--loop", "asyncio"]
