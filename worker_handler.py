"""RunPod Serverless 워커 — GPU에서 영상 분석을 실행한다.

배포: backend/Dockerfile.worker 로 이미지를 빌드해 RunPod Serverless 엔드포인트에 올린다
(RunPod 콘솔에서 GitHub 저장소를 연결하면 이 Dockerfile로 직접 빌드할 수 있다).
백엔드는 runpod_service가 작업을 제출하고, 상태를 조회할 때 완료된 결과를 가져가 DB에 저장한다.

입력 (job["input"])
  video_url   : 영상 URL (R2 공개 URL 등)
  config      : AnalysisConfig 필드 (팀 색, 보정, 공격 방향, quality 등)
  result_key  : 결과를 올릴 R2 객체 키 (예: results/12/<uuid>.json.gz). 워커에 R2 환경변수가 있을 때만 쓴다.
출력
  {"status": "done", "result_key": "...", "summary": {...}}          — R2에 올린 경우 (권장, 큰 결과도 안전)
  {"status": "done", "result_gz": <gzip+base64 JSON>}                — R2가 없을 때(작은 결과만)
  {"status": "failed", "error": "..."}

워커 환경변수: R2_ENDPOINT, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET (결과 업로드용)
"""

import gzip
import json
import logging
import os
import tempfile

import requests
import runpod

from app.vision.pipeline import AnalysisConfig, analyze_video, json_default, pack_result

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("worker")


def _download(url: str, dest: str) -> None:
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                f.write(chunk)


def _r2_client():
    keys = ("R2_ENDPOINT", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY")
    if not all(os.environ.get(k) for k in keys):
        return None
    import boto3
    return boto3.client("s3", endpoint_url=os.environ["R2_ENDPOINT"], region_name="auto",
                        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
                        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"])


def _summary(result) -> dict:
    meta = result.get("meta") or {}
    return {"players": len(result.get("players", [])), "events": len(result.get("events", [])),
            "ball_detected_pct": meta.get("ball_detected_pct"), "duration_s": meta.get("duration_s")}


def handler(job):
    inp = job.get("input") or {}
    url = inp.get("video_url")
    if not url:
        return {"status": "failed", "error": "video_url이 없습니다"}
    cfg = AnalysisConfig.from_dict(inp.get("config") or {})

    def progress(pct: float, msg: str):
        try:
            runpod.serverless.progress_update(job, {"percent": round(pct, 1), "message": msg})
        except Exception:  # 진행률 보고 실패는 분석을 막지 않는다
            pass

    fd, path = tempfile.mkstemp(suffix=os.path.splitext(url.split("?")[0])[1] or ".mp4")
    os.close(fd)
    try:
        progress(1, "영상 내려받는 중")
        _download(url, path)
        result = analyze_video(path, cfg, progress=progress)
        logger.info("분석 완료: 선수 %d명, 이벤트 %d개", len(result["players"]), len(result["events"]))
        key = inp.get("result_key")
        s3 = _r2_client()
        if key and s3 is not None:
            progress(99, "결과 저장 중")
            body = gzip.compress(json.dumps(result, ensure_ascii=False, default=json_default).encode(), 6)
            s3.put_object(Bucket=os.environ.get("R2_BUCKET", "soccer-videos"), Key=key, Body=body,
                          ContentType="application/json", ContentEncoding="gzip")
            return {"status": "done", "result_key": key, "summary": _summary(result)}
        return {"status": "done", "result_gz": pack_result(result), "summary": _summary(result)}
    except Exception as e:
        logger.exception("분석 실패")
        return {"status": "failed", "error": str(e)}
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
