"""축구공 전용 검출 모델(ONNX)을 받아 지문(sha256)을 확인한다. Docker 빌드에서 호출.

모델: acatorcini/yolov9-soccer-ball (AGPL-3.0, 아마추어 측면 카메라 1080p 영상의 3~5px 공에 맞춰 학습)
ONNX 형식이라 불러올 때 임의 코드가 실행되지 않는다(.pt 피클과 달리). 버전·지문을 고정해 둔다.

사용: python fetch_ball_model.py /app/models/ball.onnx
"""
import hashlib
import sys
import urllib.request

URL = ("https://huggingface.co/acatorcini/yolov9-soccer-ball/resolve/"
       "b30df5abc9f3eda4a9d326d953be12b3541a14b4/v9b_best.onnx")
SHA256 = "9fd2031e5bced9dff47a48bae8c6809dd56493124ef8924ae28a3ce26a17a441"


def main(dest: str) -> None:
    data = urllib.request.urlopen(URL, timeout=120).read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != SHA256:
        raise SystemExit(f"공 모델 지문이 다릅니다: {digest}")
    with open(dest, "wb") as f:
        f.write(data)
    print(f"공 모델 저장: {dest} ({len(data) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "ball.onnx")
