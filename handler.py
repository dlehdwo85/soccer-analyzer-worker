"""
RunPod Serverless Worker
=========================
입력:
  - video_url: 영상 URL (Supabase Storage 또는 YouTube)
  - match_id: 경기 ID
  - home_color: 홈팀 색상 HEX
  - away_color: 원정팀 색상 HEX
  - referee_color: 심판 색상 HEX
  - field_type: soccer | futsal

출력:
  - players: 선수별 요약 데이터
  - frames: 프레임별 추적 데이터
  - status: success | failed
"""

import os
import math
import json
import logging
import tempfile
import requests
import runpod
import cv2
import numpy as np
from collections import defaultdict
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# YOLO 모델 전역 로드 (Cold start 최소화)
MODEL = None

def get_model():
    global MODEL
    if MODEL is None:
        logger.info("YOLO 모델 로딩...")
        MODEL = YOLO("yolov8n.pt")
        logger.info("YOLO 모델 로딩 완료")
    return MODEL


# ══════════════════════════════════════════════
#  색상 기반 팀 분류
# ══════════════════════════════════════════════

def hex_to_hsv_range(hex_color: str, tolerance: int = 25):
    """HEX 색상을 HSV 범위로 변환."""
    h = hex_color.lstrip('#')
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    bgr = np.array([[[b, g, r]]], dtype=np.uint8)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[0][0]
    hue, sat, val = int(hsv[0]), int(hsv[1]), int(hsv[2])

    lower = np.array([max(0, hue - tolerance), max(0, sat - 80), max(0, val - 80)])
    upper = np.array([min(180, hue + tolerance), 255, 255])
    return lower, upper


def classify_player_team(frame, bbox, home_color, away_color, referee_color):
    """
    bounding box 내 상의 색상으로 팀 분류.
    Returns: 'home' | 'away' | 'referee' | 'unknown'
    """
    x1, y1, x2, y2 = map(int, bbox)
    h_box = y2 - y1

    # 상의 영역 (상단 40%)
    torso_y1 = y1 + int(h_box * 0.1)
    torso_y2 = y1 + int(h_box * 0.5)
    torso_x1 = x1 + int((x2 - x1) * 0.2)
    torso_x2 = x2 - int((x2 - x1) * 0.2)

    # 영역 유효성 검사
    if torso_y2 <= torso_y1 or torso_x2 <= torso_x1:
        return 'unknown'
    if torso_y2 > frame.shape[0] or torso_x2 > frame.shape[1]:
        return 'unknown'

    torso = frame[torso_y1:torso_y2, torso_x1:torso_x2]
    if torso.size == 0:
        return 'unknown'

    hsv_torso = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)

    scores = {}
    for label, hex_col in [('home', home_color), ('away', away_color), ('referee', referee_color)]:
        lower, upper = hex_to_hsv_range(hex_col)

        # 빨강은 두 범위 처리
        h_val = int(cv2.cvtColor(
            np.array([[[int(hex_col[1:3], 16), int(hex_col[3:5], 16), int(hex_col[5:7], 16)]]], dtype=np.uint8),
            cv2.COLOR_RGB2HSV
        )[0][0][0])
        if h_val < 15 or h_val > 165:
            lower2 = np.array([0, lower[1], lower[2]])
            upper2 = np.array([min(15, lower[0] + 25), 255, 255])
            lower3 = np.array([max(155, upper[0] - 25), lower[1], lower[2]])
            upper3 = np.array([180, 255, 255])
            mask = cv2.inRange(hsv_torso, lower2, upper2) | cv2.inRange(hsv_torso, lower3, upper3)
        else:
            mask = cv2.inRange(hsv_torso, lower, upper)

        pixel_count = cv2.countNonZero(mask)
        total_pixels = torso.shape[0] * torso.shape[1]
        scores[label] = pixel_count / total_pixels if total_pixels > 0 else 0

    best = max(scores, key=scores.get)
    if scores[best] < 0.05:
        return 'unknown'
    return best


# ══════════════════════════════════════════════
#  간단한 위치 기반 트래커
# ══════════════════════════════════════════════

class SimpleTracker:
    def __init__(self, max_dist=0.15, max_missed=15):
        self.tracks = {}
        self.next_id = 1
        self.max_dist = max_dist
        self.max_missed = max_missed

    def update(self, detections):
        """detections: [(x_norm, y_norm, team_side), ...]"""
        results = []
        matched_track = set()
        matched_det = set()

        # 거리 기반 매칭
        dist_list = []
        for di, (dx, dy, dteam) in enumerate(detections):
            for tid, track in self.tracks.items():
                if track['team'] != dteam:
                    continue
                d = math.sqrt((dx - track['x'])**2 + (dy - track['y'])**2)
                dist_list.append((d, di, tid))

        dist_list.sort(key=lambda x: x[0])
        for dist, di, tid in dist_list:
            if di in matched_det or tid in matched_track:
                continue
            if dist > self.max_dist:
                break
            self.tracks[tid]['x'] = detections[di][0]
            self.tracks[tid]['y'] = detections[di][1]
            self.tracks[tid]['missed'] = 0
            results.append((tid, detections[di]))
            matched_track.add(tid)
            matched_det.add(di)

        # 미매칭 → 새 트랙
        for di, det in enumerate(detections):
            if di in matched_det:
                continue
            new_id = f"track_{self.next_id}"
            self.next_id += 1
            self.tracks[new_id] = {'x': det[0], 'y': det[1], 'team': det[2], 'missed': 0}
            results.append((new_id, det))

        # 미검출 트랙 제거
        to_del = []
        for tid, track in self.tracks.items():
            if tid not in matched_track:
                track['missed'] += 1
                if track['missed'] > self.max_missed:
                    to_del.append(tid)
        for tid in to_del:
            del self.tracks[tid]

        return results


# ══════════════════════════════════════════════
#  영상 다운로드
# ══════════════════════════════════════════════

def download_video(video_url: str, tmp_path: str) -> bool:
    """영상 URL에서 파일 다운로드."""
    try:
        # YouTube URL 처리
        if 'youtube.com' in video_url or 'youtu.be' in video_url:
            import yt_dlp
            ydl_opts = {
                'format': 'best[ext=mp4][height<=720]',
                'outtmpl': tmp_path,
                'quiet': True,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([video_url])
            return os.path.exists(tmp_path)

        # 일반 URL (Supabase 등)
        response = requests.get(video_url, stream=True, timeout=300)
        response.raise_for_status()
        with open(tmp_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True

    except Exception as e:
        logger.error(f"영상 다운로드 실패: {e}")
        return False


# ══════════════════════════════════════════════
#  통계 계산
# ══════════════════════════════════════════════

def calc_stats(frames_list, match_id, fps):
    """트랙별 통계 계산."""
    track_map = defaultdict(list)
    for f in frames_list:
        track_map[f['track_id']].append(f)

    players = []
    half_ms = max((f['timestamp_ms'] for f in frames_list), default=0) / 2

    for tid, flist in track_map.items():
        if len(flist) < 5:  # 너무 짧은 트랙 제외
            continue

        team = flist[0]['team_side']
        sorted_f = sorted(flist, key=lambda x: x['frame_index'])
        positions = [(f['x'], f['y']) for f in sorted_f]

        # 총 이동거리
        total_m = 0.0
        for i in range(1, len(positions)):
            dx = (positions[i][0] - positions[i-1][0]) * 105
            dy = (positions[i][1] - positions[i-1][1]) * 68
            total_m += math.sqrt(dx*dx + dy*dy)

        # 전반/후반
        first_pos  = [(f['x'], f['y']) for f in sorted_f if f['timestamp_ms'] < half_ms]
        second_pos = [(f['x'], f['y']) for f in sorted_f if f['timestamp_ms'] >= half_ms]

        def seg_dist(seg):
            d = 0.0
            for i in range(1, len(seg)):
                dx = (seg[i][0]-seg[i-1][0])*105
                dy = (seg[i][1]-seg[i-1][1])*68
                d += math.sqrt(dx*dx+dy*dy)
            return d

        d1 = seg_dist(first_pos)
        d2 = seg_dist(second_pos)
        avg_x = sum(p[0] for p in positions) / len(positions)
        avg_y = sum(p[1] for p in positions) / len(positions)
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        spread = (max(xs)-min(xs))*105 + (max(ys)-min(ys))*68
        active = min(spread * 1.2, 100.0)
        stamina = d2/d1 if d1 > 0 else 1.0

        # AI 코멘트
        comment = _make_comment(total_m, avg_x, avg_y, stamina, active)
        badge   = _make_badge(avg_x, avg_y, total_m, active)

        players.append({
            "match_id": match_id,
            "track_id": tid,
            "team_side": team,
            "total_distance_m": round(total_m, 2),
            "avg_position_x": round(avg_x, 4),
            "avg_position_y": round(avg_y, 4),
            "active_area_score": round(active, 1),
            "first_half_distance_m": round(d1, 2),
            "second_half_distance_m": round(d2, 2),
            "comment": comment,
            "style_badge": badge,
            "fatigue_index": max(0, round((1-stamina)*100, 1)),
            "press_count": 0,
            "zone_left_pct": 33.0,
            "zone_center_pct": 34.0,
            "zone_right_pct": 33.0,
            "role_type": _make_role(avg_x, avg_y),
            "sprint_count": 0,
        })

    return players


def _make_comment(total_m, avg_x, avg_y, stamina, active):
    parts = []
    if total_m > 10000: parts.append("경기 전반에 걸쳐 높은 활동량을 보였습니다.")
    elif total_m > 7000: parts.append("안정적인 이동거리를 기록했습니다.")
    else: parts.append("이동거리가 다소 적었습니다.")
    if avg_x < 0.3: parts.append("측면 왼쪽 활동 비중이 높았습니다.")
    elif avg_x > 0.7: parts.append("측면 오른쪽 활동 비중이 높았습니다.")
    else: parts.append("중앙 구역 중심으로 활동했습니다.")
    if stamina < 0.7: parts.append("후반 활동량이 크게 줄었습니다.")
    elif stamina >= 0.9: parts.append("전후반 균등한 체력 유지력이 돋보였습니다.")
    if active > 65: parts.append("넓은 커버 범위로 팀에 기여했습니다.")
    return " ".join(parts)


def _make_badge(avg_x, avg_y, total_m, active):
    if total_m > 9500: return "활동량형"
    if active > 65: return "넓은커버"
    if avg_x < 0.3: return "좌측윙어"
    if avg_x > 0.7: return "우측윙어"
    if avg_y < 0.38: return "공격형"
    if avg_y > 0.62: return "수비형"
    return "플메이커"


def _make_role(avg_x, avg_y):
    lateral = "좌측 " if avg_x < 0.3 else "우측 " if avg_x > 0.7 else "중앙 "
    vertical = "공격수" if avg_y < 0.35 else "미드필더" if avg_y < 0.65 else "수비수"
    return lateral + vertical


# ══════════════════════════════════════════════
#  메인 분석 함수
# ══════════════════════════════════════════════

def analyze_video(video_path, match_id, home_color, away_color, referee_color, field_type):
    """YOLO + 색상 기반 선수 감지 및 추적."""
    model = get_model()
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"영상을 열 수 없습니다: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    logger.info(f"영상 정보: {width}x{height}, {fps}fps, {total_frames}프레임")

    # 샘플링: 5fps (너무 느리면 10fps 단위)
    sample_n = max(1, int(fps / 5))
    tracker = SimpleTracker()

    all_frames = []
    frame_idx = 0
    processed = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_n == 0:
            # YOLO 사람 감지 (class 0 = person)
            results = model(frame, classes=[0], conf=0.4, verbose=False)
            detections = []

            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    # 너무 작거나 큰 bbox 제외
                    w_box = x2 - x1
                    h_box = y2 - y1
                    if w_box < 20 or h_box < 30 or w_box > width*0.3:
                        continue

                    team = classify_player_team(frame, (x1,y1,x2,y2), home_color, away_color, referee_color)
                    if team == 'unknown' or team == 'referee':
                        continue

                    # 정규화 좌표 (중심점)
                    cx = ((x1 + x2) / 2) / width
                    cy = ((y1 + y2) / 2) / height
                    detections.append((cx, cy, team))

            track_results = tracker.update(detections)
            ts_ms = int((frame_idx / fps) * 1000)

            for tid, (cx, cy, team) in track_results:
                all_frames.append({
                    "match_id":     match_id,
                    "frame_index":  frame_idx,
                    "timestamp_ms": ts_ms,
                    "track_id":     tid,
                    "team_side":    team,
                    "x":            round(cx, 4),
                    "y":            round(cy, 4),
                    "bbox_x": 0.0, "bbox_y": 0.0, "bbox_w": 0.0, "bbox_h": 0.0,
                })

            processed += 1
            if processed % 50 == 0:
                progress = frame_idx / total_frames * 100
                logger.info(f"진행: {progress:.1f}% ({frame_idx}/{total_frames})")

        frame_idx += 1

    cap.release()
    logger.info(f"분석 완료: 총 {len(all_frames)}개 감지")

    if len(all_frames) < 10:
        raise ValueError("선수 감지 결과가 너무 적습니다. 색상 설정을 확인해주세요.")

    players = calc_stats(all_frames, match_id, fps)
    logger.info(f"선수 {len(players)}명 추적 완료")

    return {"players": players, "frames": all_frames[:10000]}


# ══════════════════════════════════════════════
#  RunPod Handler
# ══════════════════════════════════════════════

def handler(job):
    """RunPod Serverless 진입점."""
    job_input = job["input"]

    match_id      = job_input.get("match_id")
    video_url     = job_input.get("video_url")
    home_color    = job_input.get("home_color", "#56b4d3")
    away_color    = job_input.get("away_color", "#ef4444")
    referee_color = job_input.get("referee_color", "#facc15")
    field_type    = job_input.get("field_type", "soccer")

    logger.info(f"작업 시작: match_id={match_id}, url={video_url}")

    try:
        # 영상 다운로드
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_path = tmp.name

        logger.info("영상 다운로드 중...")
        if not download_video(video_url, tmp_path):
            return {"error": "영상 다운로드 실패", "status": "failed"}

        logger.info("영상 분석 시작...")
        result = analyze_video(tmp_path, match_id, home_color, away_color, referee_color, field_type)

        # 임시 파일 삭제
        os.unlink(tmp_path)

        logger.info("분석 완료!")
        return {
            "status": "success",
            "match_id": match_id,
            "players": result["players"],
            "frames":  result["frames"],
            "player_count": len(result["players"]),
        }

    except Exception as e:
        logger.error(f"분석 실패: {e}")
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        return {"error": str(e), "status": "failed"}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
