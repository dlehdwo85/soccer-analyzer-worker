"""공 검출 모델 추가 학습 (GPU 워커에서 실행 — worker_handler의 mode=train_ball).

출발점: 기본은 acatorcini/yolov9-soccer-ball v9b_best.pt (아마추어 측면 카메라 1080p, 3~5px 공).
base 에 우리가 학습한 .pt 경로를 주면 거기서 이어서 학습한다 (예: 지금 쓰는 밤 경기 모델).
분석과 같은 조건(1080p 화면 전체 → imgsz)으로 학습해야 공 크기가 맞는다.

데이터셋 zip: images/{train,val[,test]}/*.jpg, labels/{train,val[,test]}/*.txt (YOLO 형식, 클래스 0 = 공)
test 가 있으면 학습에 전혀 쓰지 않은 그 장면들로 학습 전·후 모델을 같은 조건에서 비교한다
(val 은 학습 중 가장 좋은 모델을 고르는 데 쓰여 점수가 낙관적으로 나온다).
로컬 실행: python train_ball.py <dataset.zip> [--epochs 60] [--base 모델.pt]
"""
import argparse
import os
import sys
import urllib.request
import zipfile
from typing import Dict, Optional

BASE_PT = ("https://huggingface.co/acatorcini/yolov9-soccer-ball/resolve/"
           "b30df5abc9f3eda4a9d326d953be12b3541a14b4/v9b_best.pt")


def _scores(model, yaml: str, split: str, imgsz: int, batch: int, project: str, name: str) -> Dict:
    r = model.val(data=yaml, split=split, imgsz=imgsz, batch=batch, project=project, name=name, exist_ok=True)
    return {"map50": float(r.box.map50), "recall": float(r.box.mr), "precision": float(r.box.mp)}


def train(data_zip: str, work: str, epochs: int = 60, batch: int = 8, imgsz: int = 1280,
          mosaic: float = 0.3, patience: int = 15, base: Optional[str] = None, lr0: float = 0.002,
          cos_lr: bool = False, scale: float = 0.2) -> Dict:
    """학습하고 {onnx, pt, map50, recall, precision, base_*, test_*}를 돌려준다.
    imgsz=1920이면 1080p 화면을 줄이지 않고 원본 크기로 학습한다(밤 경기 7~10px 공이 줄면 5px로 사라진다)."""
    work = os.path.abspath(work)   # ultralytics는 상대 경로 project를 runs/detect 아래에 둔다
    os.makedirs(work, exist_ok=True)
    data = os.path.join(work, "data")
    if not os.path.exists(data):
        zipfile.ZipFile(data_zip).extractall(data)
    if not base:
        base = os.path.join(work, "v9b_best.pt")
        if not os.path.exists(base):
            urllib.request.urlretrieve(BASE_PT, base)
    has_test = os.path.isdir(os.path.join(data, "images", "test"))
    yaml = os.path.join(work, "ball.yaml")
    with open(yaml, "w") as f:
        f.write(f"path: {data}\ntrain: images/train\nval: images/val\n"
                + ("test: images/test\n" if has_test else "") + "names:\n  0: ball\n")

    from ultralytics import YOLO
    runs = os.path.join(work, "runs")
    before = _scores(YOLO(base), yaml, "val", imgsz, batch, runs, "base")
    before_test = _scores(YOLO(base), yaml, "test", imgsz, batch, runs, "base_test") if has_test else None
    # 공은 작고 화면 전체 맥락(선수 발밑)이 중요해 크게 자르는 모자이크는 약하게, 밤 조명 차이는 밝기 변형으로
    model = YOLO(base)
    model.train(data=yaml, imgsz=imgsz, epochs=epochs, batch=batch, patience=patience,
                mosaic=mosaic, scale=scale, translate=0.05, fliplr=0.5, hsv_v=0.5, hsv_s=0.5, degrees=0,
                close_mosaic=10, lr0=lr0, warmup_epochs=1, cos_lr=cos_lr,
                project=runs, name="ball", exist_ok=True)
    best = str(model.trainer.best)
    after = _scores(YOLO(best), yaml, "val", imgsz, batch, runs, "after")
    onnx = YOLO(best).export(format="onnx", imgsz=imgsz, opset=17, simplify=True, dynamic=False)
    out = {"onnx": onnx, "pt": best, **after, **{"base_" + k: v for k, v in before.items()}}
    if has_test:
        out.update({"test_" + k: v for k, v in _scores(YOLO(best), yaml, "test", imgsz, batch, runs, "after_test").items()})
        out.update({"base_test_" + k: v for k, v in before_test.items()})
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("data_zip")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--base", default=None)
    ap.add_argument("--work", default="balltrain")
    a = ap.parse_args()
    print(train(a.data_zip, a.work, a.epochs, a.batch, a.imgsz, base=a.base))
    sys.exit(0)
