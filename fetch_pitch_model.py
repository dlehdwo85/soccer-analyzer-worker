"""경기장 키포인트 모델(YOLOv8 pose, 32점)을 받아 지문(sha256)을 확인한다. Docker 빌드에서 호출.

모델: roboflow/sports 저장소(examples/soccer/setup.sh)가 안내하는 football-pitch-detection.pt.
코너·페널티 박스·골 에어리어·하프라인·센터서클 교점 32개를 찾는다 — app/vision/pitchreg.py가 이 점들로 장면마다
화면 → 경기장(m) 변환을 구한다. .pt는 불러올 때 코드가 실행될 수 있는 형식이라 버전(파일 id)과 지문을 고정해 둔다.

사용: python fetch_pitch_model.py /app/models/pitch.pt
"""
import hashlib
import sys
import urllib.request

URL = "https://drive.usercontent.google.com/download?id=1Ma5Kt86tgpdjCTKfum79YMgNnSjcoOyf&export=download&confirm=t"
SHA256 = "28f68f7c4056d6d9b137efd2e7ab5f3c494039380c63831649126ced25628b36"


def main(dest: str) -> None:
    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    data = urllib.request.urlopen(req, timeout=600).read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != SHA256:
        raise SystemExit(f"경기장 모델 지문이 다릅니다: {digest} ({len(data)} bytes)")
    with open(dest, "wb") as f:
        f.write(data)
    print(f"경기장 모델 저장: {dest} ({len(data) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "pitch.pt")
