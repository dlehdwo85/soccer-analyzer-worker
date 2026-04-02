# Soccer Analyzer Worker

RunPod Serverless Worker for soccer video analysis.

## 기능
- YOLO v8 선수 감지
- 색상 기반 팀 분류 (홈/어웨이/심판)
- 선수 위치 추적 (Simple Tracker)
- YouTube URL 및 직접 URL 영상 다운로드 지원

## 입력 형식
```json
{
  "input": {
    "match_id": 1,
    "video_url": "https://...",
    "home_color": "#56b4d3",
    "away_color": "#ef4444",
    "referee_color": "#facc15",
    "field_type": "soccer"
  }
}
```

## 출력 형식
```json
{
  "status": "success",
  "match_id": 1,
  "players": [...],
  "frames": [...],
  "player_count": 14
}
```
