"""공 검출 모델 추가 학습 (GPU 워커에서 실행 — worker_handler의 mode=train_ball).

출발점: acatorcini/yolov9-soccer-ball v9b_best.pt (아마추어 측면 카메라 1080p, 3~5px 공).
우리 영상(밤 조명·높은 측면 카메라)에서 사람이 공 위치를 표시한 장면으로 이어서 학습한다.
분석과 같은 조건(1080p 화면 전체 → 1280)으로 학습해야 공 크기가 맞는다.

데이터셋 zip: images/{train,val}/*.jpg, labels/{train,val}/*.txt (YOLO 형식, 클래스 0 = 공)
로컬 실행: python train_ball.py <dataset.zip> [--epochs 60]
"""
import argparse
import os
import sys
import urllib.request
import zipfile
from typing import Dict

BASE_PT = ("https://huggingface.co/acatorcini/yolov9-soccer-ball/resolve/"
           "b30df5abc9f3eda4a9d326d953be12b3541a14b4/v9b_best.pt")


def train(data_zip: str, work: str, epochs: int = 60, batch: int = 8) -> Dict:
    """학습하고 {onnx, pt, map50, recall, precision, base_*}를 돌려준다."""
    work = os.path.abspath(work)   # ultralytics는 상대 경로 project를 runs/detect 아래에 둔다
    os.makedirs(work, exist_ok=True)
    data = os.path.join(work, "data")
    if not os.path.exists(data):
        zipfile.ZipFile(data_zip).extractall(data)
    base = os.path.join(work, "v9b_best.pt")
    if not os.path.exists(base):
        urllib.request.urlretrieve(BASE_PT, base)
    yaml = os.path.join(work, "ball.yaml")
    with open(yaml, "w") as f:
        f.write(f"path: {data}\ntrain: images/train\nval: images/val\nnames:\n  0: ball\n")

    from ultralytics import YOLO
    before = YOLO(base).val(data=yaml, imgsz=1280, batch=batch, project=os.path.join(work, "runs"), name="base")
    # 공은 작고 화면 전체 맥락(선수 발밑)이 중요해 크게 자르는 모자이크는 약하게, 밤 조명 차이는 밝기 변형으로
    model = YOLO(base)
    model.train(data=yaml, imgsz=1280, epochs=epochs, batch=batch, patience=15,
                mosaic=0.3, scale=0.2, translate=0.05, fliplr=0.5, hsv_v=0.5, hsv_s=0.5, degrees=0,
                close_mosaic=10, lr0=0.002, warmup_epochs=1,
                project=os.path.join(work, "runs"), name="ball", exist_ok=True)
    best = str(model.trainer.best)
    after = YOLO(best).val(data=yaml, imgsz=1280, batch=batch, project=os.path.join(work, "runs"), name="after")
    onnx = YOLO(best).export(format="onnx", imgsz=1280, opset=17, simplify=True, dynamic=False)
    return {"onnx": onnx, "pt": best,
            "map50": float(after.box.map50), "recall": float(after.box.mr), "precision": float(after.box.mp),
            "base_map50": float(before.box.map50), "base_recall": float(before.box.mr),
            "base_precision": float(before.box.mp)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("data_zip")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--work", default="balltrain")
    a = ap.parse_args()
    print(train(a.data_zip, a.work, a.epochs, a.batch))
    sys.exit(0)
