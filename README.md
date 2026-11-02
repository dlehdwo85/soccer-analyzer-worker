# soccer-analyzer-worker

RunPod Serverless GPU 워커. **직접 고치지 마세요** — [dlehdwo85/soccer-analyzer](https://github.com/dlehdwo85/soccer-analyzer)의
`backend/`에서 복사한 파일입니다 (원본 커밋 326f12e).

| 이 저장소 | 원본 |
|---|---|
| Dockerfile | backend/Dockerfile.worker |
| worker_handler.py, fetch_ball_model.py, requirements-worker.txt | backend/ 같은 이름 |
| app/vision/ | backend/app/vision/ |

입력: `{"video_url", "config": {..., "quality": "standard"|"max"}, "result_key"}`
출력: 결과를 R2(`result_key`)에 gzip JSON으로 올리고 `{"status": "done", "result_key", "summary"}`.

엔드포인트 환경변수: R2_ENDPOINT, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET.
