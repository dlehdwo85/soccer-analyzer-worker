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
    YOLO_BALL_MODEL=/app/models/ball_night.onnx \
    YOLO_BALL_IMGSZ=1920 \
    YOLO_BALL_TILES=0 \
    YOLO_BALL_CONF=0.15 \
    YOLO_BALL_CONF_NEW=0.25 \
    YOLO_BALL_GAP_S=1.5 \
    YOLO_PITCH_MODEL=/app/models/pitch.pt \
    YOLO_PITCH_EVERY=2

RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-worker.txt fetch_ball_model.py fetch_pitch_model.py ./
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

# 경기장 선 자동 인식 모델 (32점 키포인트). 장면마다 화면 → 경기장(m) 변환을 구한다.
# 2쿼터 밤 경기 시험(2026-11-15): 전 장면에서 변환은 나오지만, 방송 경기장 비율을 가정한 모델이라 미드필드 장면에서
# 하프라인이 4~5m 어긋난다. 그래서 오차에 강한 판정에만 쓴다 — 골키퍼(자기 골문 앞에 머무는 선수), 공격 방향,
# 경기장 밖 사람 제외, 평균 위치, 이동 거리. 코너킥·드로잉·프리킥(파울)·센터서클 킥오프처럼 선 근처 정밀도가 필요한
# 판정은 YOLO_PITCH_PRECISE=1일 때만 켠다 (우리 구장 영상으로 모델을 다시 학습한 뒤에 켤 것).
# 모델 파일이 없으면 분석은 예전 방식(화면 좌표 근사)으로 그대로 돈다.
RUN python fetch_pitch_model.py /app/models/pitch.pt

# 공 모델: 기존 모델을 우리 밤 경기(바모스 2·3쿼터, 대표님이 공 위치를 표시한 264장면)로 이어서 학습한 것.
# 1080p 원본 크기(1920)로 학습·검출한다 — 줄이면 7~10px 밤 공이 사라진다. 화면 분할 없이 한 번에 본다.
# 검증 장면(학습에 안 쓴 묶음)에서 진짜 공을 찾은 비율: 밤 8% → 31%, 낮 53% → 65%.
# 2판(2026-11-13): 1판 모델의 후보 424개를 확대해 검수한 273장면을 더해 재학습 — 찾는 비율은 비슷(밤 33%)하지만
# 엉뚱한 곳을 공으로 잡는 장면이 44장면 중 23 → 8로 줄었다.
# 5판(2026-11-21): 3경기(낮 1·밤 2)에서 4판이 놓친 구간을 골라 확대 검수한 386장면을 더한 데이터(학습 866장면)로
# 4판에서 이어서 학습. 학습에 안 쓴 시간 구간 510장면을 고르게 뽑아 두 모델이 엇갈린 곳을 눈으로 판정하니
# (확신도 0.25) 5판만 찾은 진짜 공 37 vs 4판만 11, 엉뚱한 검출 35 vs 37 — 낮 경기에서 크게 낫고 밤은 비슷하거나 조금 낫다.
# 파일은 워커 저장소 models/ball_night.onnx (원본·학습 데이터: R2 balltrain/20261121_v3/).
COPY models/ball_night.onnx /app/models/ball_night.onnx
COPY app/__init__.py app/__init__.py
COPY app/vision app/vision
COPY worker_handler.py train_ball.py .

CMD ["python", "-u", "worker_handler.py"]
