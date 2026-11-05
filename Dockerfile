# RunPod Serverless GPU 워커 이미지
#   docker build -f backend/Dockerfile.worker -t <registry>/soccer-analyzer-worker backend
# RunPod 콘솔 → Serverless → New Endpoint → GitHub 저장소 연결 시:
#   Dockerfile 경로 backend/Dockerfile.worker, 빌드 컨텍스트 backend
# 엔드포인트 환경변수: R2_ENDPOINT, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET (결과 업로드용)
# CUDA 12.1 이미지는 최신 세대 GPU(Blackwell)에서 "no kernel image is available"로 워커가 바로 꺼진다.
# 엔드포인트 GPU를 '이전 세대'(A5000·4090 등)로만 고를 것. CUDA 12.8 이미지(2.7.1-cuda12.8)는 기본 층이
# 4GB가 넘어 RunPod GitHub 빌드가 받는 도중에 끊겼다 (2026-11-02).
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    YOLO_MODEL=yolov8m.pt \
    YOLO_IMGSZ=1280 \
    YOLO_CONFIG_DIR=/app/.ultralytics \
    YOLO_BALL_MODEL=/app/models/ball.onnx \
    YOLO_BALL_IMGSZ=1280

RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-worker.txt fetch_ball_model.py ./
# GPU에서 공 모델을 돌리도록 onnxruntime을 GPU판으로 바꾼다. 최신판은 CUDA 13(1.27+)·12.8용으로 빌드돼
# 이 이미지(CUDA 12.1·cuDNN 9)에서 CUDA를 못 써 공 검출이 CPU로 돌았다 (2026-11-03 로그) → 1.20.2로 고정하고,
# PyTorch가 pip으로 깔아 둔 CUDA·cuDNN 라이브러리를 onnxruntime도 찾도록 라이브러리 경로에 등록한다.
RUN pip install --no-cache-dir -r requirements-worker.txt \
    && pip uninstall -y opencv-python onnxruntime \
    && pip install --no-cache-dir --force-reinstall --no-deps "opencv-python-headless>=4.9,<4.11" \
    && pip install --no-cache-dir "onnxruntime-gpu==1.20.2" \
    && python -c "import glob,site; d=[p for s in site.getsitepackages() for p in glob.glob(s+'/nvidia/*/lib')]; open('/etc/ld.so.conf.d/nvidia-pip.conf','w').write(chr(10).join(d)+chr(10))" \
    && ldconfig

# 모델 가중치를 이미지에 미리 받아 둔다 (콜드스타트 단축). 고정밀(quality=max)은 yolov8x를 쓴다.
RUN python -c "from ultralytics import YOLO; YOLO('${YOLO_MODEL}'); YOLO('yolov8x.pt')" \
    && mkdir -p /app/models && python fetch_ball_model.py /app/models/ball.onnx

COPY app/__init__.py app/__init__.py
COPY app/vision app/vision
COPY worker_handler.py train_ball.py .

CMD ["python", "-u", "worker_handler.py"]
