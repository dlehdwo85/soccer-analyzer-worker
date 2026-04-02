FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

WORKDIR /app

# 시스템 패키지
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python 패키지
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# YOLO 모델 미리 다운로드 (빌드 시 캐시)
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# Worker 코드 복사
COPY . .

CMD ["python", "-u", "handler.py"]
