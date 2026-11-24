"""영상 → 분석 결과 파이프라인.

analyze_video()   : 실제 영상 (YOLO 필요 — requirements-vision.txt)
analyze_simulated(): 가상 경기 (샘플 모드)
둘 다 postprocess()로 같은 결과 형식(version 2)을 만든다.

결과 형식
{
  "version": 2,
  "meta": {...},                 # 길이, fps, 보정 여부, 공격 방향, 검출된 유니폼 색, 경고
  "players": [{track_id, team, label, color_hex, thumbnail, stats}],
  "events": [...],               # possession / pass / shot / recovery
  "team_stats": {...}, "pass_network": {...}, "possession_timeline": [...]
}
선수 stats는 합산 가능한 원시값이다(stats.finalize로 파생 지표 계산).
"""

from __future__ import annotations

import base64
import gzip
import json
import logging
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Protocol, Sequence, Tuple

import numpy as np

from . import color as C
from .ball import (drop_weak_at_feet, in_play_candidates, interpolate_gaps, select_ball_positions,
                   select_ball_positions_bidir)
from .camera import CameraMotion, stationary
from .events import EventConfig, Frame, PlayerObs, detect_events, find_kickoff, fit_to_extents
from .pitch import GOAL_WIDTH, POSSESSION_RADIUS_H, POSSESSION_RADIUS_M, PitchMapper
from .pitchreg import PitchKeypointModel, PitchRegistrar
from .situations import centre_kickoffs, find_goalkeepers, foul_player, home_goal_side, restart_type
from .scoreboard import ScoreboardReader
from .audio import detect_whistles, load_audio
from .pitchlines import TouchlineFilter, halfway_line
from .stats import compute_player_stats, pass_network, possession_timeline, team_stats
from .tracker import (Detection, IoUTracker, Tracklet, assign_roster_slots, pieces_from_tracklets, relink_chains,
                      stitch_tracklets,
                      suggest_same_player)

logger = logging.getLogger(__name__)

RESULT_VERSION = 2
MIN_ANALYSIS_FPS = 5.0
ProgressFn = Callable[[float, str], None]


@dataclass
class AnalysisConfig:
    field_type: str = "soccer"
    home_color: str = "#ef4444"
    away_color: str = "#3b82f6"
    referee_color: Optional[str] = "#facc15"
    home_gk_color: Optional[str] = None
    away_gk_color: Optional[str] = None
    calibration: Optional[Dict] = None
    analysis_fps: float = 6.0
    home_attacks: str = "auto"
    halftime_s: Optional[float] = None
    max_seconds: Optional[float] = None
    start_seconds: float = 0.0
    color_calib_s: float = 30.0
    minor_track_s: float = 10.0
    # standard: CPU에서도 돌 수 있는 설정 / max: GPU 전용 고정밀 (초당 15장면, 대형 선수 모델, 공 4분할 검출)
    quality: str = "standard"

    @classmethod
    def from_dict(cls, d: Dict) -> "AnalysisConfig":
        keys = cls.__dataclass_fields__.keys()
        return cls(**{k: v for k, v in (d or {}).items() if k in keys})

    def effective_fps(self) -> float:
        """고정밀은 장면 수를 늘려 빠른 패스·슈팅의 공 궤적이 끊기지 않게 한다 (명시한 값이 더 크면 그 값)."""
        return max(self.analysis_fps, QUALITY_PRESETS.get(self.quality, {}).get("analysis_fps", 0.0))

    def event_config(self, length: float, width: float, calibrated: bool = False) -> EventConfig:
        return EventConfig(length=length, width=width,
                           goal_width=GOAL_WIDTH.get(self.field_type, 7.32),
                           home_attacks=self.home_attacks, calibrated=calibrated,
                           # halftime_s는 영상 전체 기준 시각, 이벤트 시각은 분석 시작점 기준
                           halftime_s=(self.halftime_s - self.start_seconds) if self.halftime_s is not None else None)


# 품질 단계별 검출 설정. 표준은 환경변수(YOLO_MODEL 등)를 따르고, 고정밀은 이 값이 우선한다.
QUALITY_PRESETS: Dict[str, Dict] = {
    "standard": {},
    # 공 화면 분할은 쓰지 않는다. 운영 공 모델은 1920 고정 입력이고 영상도 1920이라 전체 화면이 이미 원본 해상도다.
    # 9분할을 켰던 2쿼터 재분석(2026-11-23, v2.6.1)은 조각이 2.7배로 확대돼 공을 직접 찾은 비율이 50% → 15%로 떨어졌다.
    # ball_conf: 공 후보를 0.07까지 받는다. 새 궤적 시작은 그대로 0.25↑라 약한 후보는 이어지는 궤적 근처에서만 쓰인다.
    "max": {"analysis_fps": 15.0, "model": "yolov8x.pt", "imgsz": 1280, "ball_full_every": 1, "ball_conf": 0.07},
}

# 긴 공백 대략 위치를 잇는 최대 길이(초) — 화면 표시·'공 위치 파악' 비율용, 이벤트 판정에는 안 쓴다
BALL_ROUGH_GAP_S = 8.0


# ── 검출기 ──────────────────────────────────────────────────────────────────

class Detector(Protocol):
    def detect(self, frame_rgb: np.ndarray) -> Tuple[List[Tuple[Tuple[float, float, float, float], float]],
                                                     List[Tuple[float, float, float]]]:
        """반환: (사람 [(box, conf)], 공 [(cx, cy, conf)])"""


class YoloDetector:
    """ultralytics YOLO (COCO: 0=person, 32=sports ball). 모델·해상도는 환경변수로 조정.

    공은 화면에서 몇 픽셀밖에 안 될 때가 많아 COCO 범용 모델로는 잘 안 잡힌다.
    축구공으로 학습한 YOLO 가중치가 있으면 YOLO_BALL_MODEL로 지정한다(클래스 이름에 ball/football 포함).
    """

    PERSON_CONF = 0.2
    BALL_CONF = 0.05   # 낮게 잡고, 시간 연속성으로 거른다 (ball.select_ball_positions)
    # 축구공 전용 모델은 진짜 공도 신뢰도 0.05~0.35로 낮게 나온다 → 새 궤적 시작 기준도 낮춘다.
    BALL_MODEL_MIN_CONF_NEW = 0.08
    BALL_FOCUS_CONF = 0.08     # 이 이상이면 다음 프레임 확대 영역의 중심으로 삼는다
    BALL_FOCUS_MAX_AGE = 12    # 공을 놓친 뒤 이 프레임 수까지는 마지막 위치 주변을 계속 본다
    BALL_FULL_EVERY = 3        # 공을 추적 중일 때 전체 화면 검출은 이 프레임마다 한 번 (속도 ↑)

    def __init__(self, model: Optional[str] = None, imgsz: Optional[int] = None, device: Optional[str] = None,
                 ball_model: Optional[str] = None, ball_imgsz: Optional[int] = None,
                 ball_roi: Optional[bool] = None):
        try:
            from ultralytics import YOLO
        except ImportError as e:  # pragma: no cover - 환경 의존
            raise RuntimeError("영상 분석에는 ultralytics가 필요합니다: pip install -r requirements-worker.txt") from e
        self.model = YOLO(model or os.environ.get("YOLO_MODEL", "yolov8s.pt"))
        self.imgsz = imgsz or int(os.environ.get("YOLO_IMGSZ", "1280"))
        self.device = device or os.environ.get("YOLO_DEVICE") or None
        bm = ball_model or os.environ.get("YOLO_BALL_MODEL")
        self.ball_model = YOLO(bm, task="detect") if bm else None
        self.ball_classes = None
        # 공 전용 모델은 학습 해상도(보통 1280)로 돌린다. 사람 검출 해상도(YOLO_IMGSZ)와 따로 둔다.
        self.ball_imgsz = ball_imgsz or int(os.environ.get("YOLO_BALL_IMGSZ", "1280"))
        # 확대 영역: 직전 공 위치(없으면 선수들이 모인 곳) 주변을 원본 해상도로 한 번 더 본다.
        self.ball_roi = (os.environ.get("YOLO_BALL_ROI", "1") != "0") if ball_roi is None else ball_roi
        # 타일 검출: 공을 놓쳤을 때 전체 화면을 n×n 조각으로 나눠 원본 해상도로 본다 (느리지만 작은 공을 더 찾음).
        # 0이면 끔. GPU나 넉넉한 시간이 있을 때 YOLO_BALL_TILES=2 권장.
        self.ball_tiles = int(os.environ.get("YOLO_BALL_TILES", "0"))
        self.ball_min_conf_new = 0.3
        self.ball_conf = self.BALL_CONF
        self.ball_gap_s = 2.5
        self._focus: Optional[Tuple[float, float]] = None
        self._focus_age = 0
        self._frame_no = 0
        if self.ball_model is not None:
            names = self.ball_model.names
            self.ball_classes = [k for k, v in names.items() if "ball" in str(v).lower()] or list(names)
            # 공 모델마다 신뢰도 눈금이 달라 환경변수로 맞춘다. 밤 경기로 추가 학습한 모델은 0.15 아래 후보가 대부분
            # 축구화라, 낮은 후보까지 받아 2.5초씩 이으면 못 찾은 장면에 틀린 위치가 들어갔다. 검증 39장면(사람 표시,
            # 학습에 안 씀): 후보 0.05↑·시작 0.08·잇기 2.5초 → 맞음 18·틀림 21 / 0.15↑·0.25·1.5초 → 맞음 26·틀림 3·없음 10.
            self.ball_conf = float(os.environ.get("YOLO_BALL_CONF", self.BALL_CONF))
            self.ball_min_conf_new = float(os.environ.get("YOLO_BALL_CONF_NEW", self.BALL_MODEL_MIN_CONF_NEW))
            self.ball_gap_s = float(os.environ.get("YOLO_BALL_GAP_S", "2.5"))

    @classmethod
    def for_quality(cls, quality: str = "standard") -> "YoloDetector":
        """품질 단계에 맞춘 검출기. 표준은 환경변수(YOLO_MODEL 등) 그대로, 고정밀은 단계 설정이 우선한다."""
        p = QUALITY_PRESETS.get(quality or "standard", {})
        det = cls(model=p.get("model") or os.environ.get("YOLO_MODEL"),
                  imgsz=p.get("imgsz") or (int(os.environ["YOLO_IMGSZ"]) if os.environ.get("YOLO_IMGSZ") else None))
        # 워커 이미지는 YOLO_BALL_TILES=0을 기본으로 깔아 두므로 '0'은 지정 안 한 것으로 본다
        # (그동안 고정밀 분석의 공 분할이 한 번도 적용되지 않았다 — 2026-11-23 2쿼터 재분석에서 발견)
        if "ball_tiles" in p and int(os.environ.get("YOLO_BALL_TILES") or 0) == 0:
            det.ball_tiles = p["ball_tiles"]
        if "ball_full_every" in p:
            det.BALL_FULL_EVERY = p["ball_full_every"]
        if "ball_conf" in p and det.ball_model is not None:
            det.ball_conf = min(det.ball_conf, p["ball_conf"])
        return det

    def _ball_predict(self, img, x0: int = 0, y0: int = 0) -> List[Tuple[float, float, float]]:
        rb = self.ball_model.predict(img, imgsz=self.ball_imgsz, classes=self.ball_classes, conf=self.ball_conf,
                                     device=self.device, verbose=False)[0]
        out = []
        if rb.boxes is not None and len(rb.boxes):
            for b, c in zip(rb.boxes.xyxy.cpu().numpy(), rb.boxes.conf.cpu().numpy()):
                out.append((float((b[0] + b[2]) / 2) + x0, float((b[1] + b[3]) / 2) + y0, float(c)))
        return out

    def _roi(self, w: int, h: int, persons) -> Optional[Tuple[int, int, int, int]]:
        """확대해서 다시 볼 영역 (x0, y0, x1, y1). 원본 해상도 그대로 ball_imgsz 정도 크기."""
        rw, rh = min(w, self.ball_imgsz), min(h, int(self.ball_imgsz * 9 / 16))
        if rw >= w and rh >= h:
            return None  # 영상이 작으면 전체 화면 검출과 같다
        if self._focus is not None and self._focus_age <= self.BALL_FOCUS_MAX_AGE:
            cx, cy = self._focus
        elif persons:
            cx = float(np.median([(b[0] + b[2]) / 2 for b, _ in persons]))
            cy = float(np.median([(b[1] + b[3]) / 2 for b, _ in persons]))
        else:
            return None
        x0 = int(min(max(cx - rw / 2, 0), w - rw))
        y0 = int(min(max(cy - rh / 2, 0), h - rh))
        return x0, y0, x0 + rw, y0 + rh

    @staticmethod
    def _held_by_person(cx: float, cy: float, persons) -> bool:
        """사람 상체·손 높이에 있는 후보(들고 있는 봉지·물병 등). 발밑 공과 머리 위 공은 남긴다."""
        for (x1, y1, x2, y2), _ in persons:
            h = y2 - y1
            if x1 <= cx <= x2 and y1 + 0.15 * h <= cy <= y1 + 0.7 * h:
                return True
        return False

    def detect(self, frame_rgb):
        classes = [0] if self.ball_model is not None else [0, 32]
        # ultralytics는 numpy 이미지를 OpenCV처럼 BGR로 받는다. RGB를 그대로 넣으면 색이 뒤바뀌어
        # 공 모델이 밤 경기 공을 거의 못 잡고 낮 경기도 덜 잡았다(검증 장면: 낮 53% → BGR로 82%).
        frame = np.ascontiguousarray(frame_rgb[..., ::-1])
        res = self.model.predict(frame, imgsz=self.imgsz, classes=classes, conf=self.BALL_CONF,
                                 device=self.device, verbose=False)[0]
        persons, balls = [], []
        if res.boxes is not None and len(res.boxes):
            xyxy = res.boxes.xyxy.cpu().numpy()
            conf = res.boxes.conf.cpu().numpy()
            cls = res.boxes.cls.cpu().numpy().astype(int)
            for b, c, k in zip(xyxy, conf, cls):
                if k == 0 and c >= self.PERSON_CONF:
                    persons.append(((float(b[0]), float(b[1]), float(b[2]), float(b[3])), float(c)))
                elif k == 32:
                    balls.append((float((b[0] + b[2]) / 2), float((b[1] + b[3]) / 2), float(c)))
        if self.ball_model is not None:
            h, w = frame_rgb.shape[:2]
            self._frame_no += 1
            roi = self._roi(w, h, persons) if self.ball_roi else None
            tracking = roi is not None and self._focus is not None and self._focus_age <= self.BALL_FOCUS_MAX_AGE
            cands: List[Tuple[float, float, float]] = []
            # 추적 중이면 확대 영역만 보고, 전체 화면은 가끔 확인한다 (공이 멀리 튀었을 때 대비).
            if not tracking or self._frame_no % self.BALL_FULL_EVERY == 0:
                cands += self._ball_predict(frame)
                # 분할은 전체 화면 검출에 더해서만 본다. 전체 화면이 공 모델 입력보다 커서 줄여 볼 때만 의미가 있다
                # (조각을 모델 입력 크기로 키우면 공이 학습 때보다 커 보여 오히려 놓친다).
                if self.ball_tiles >= 2 and w > self.ball_imgsz and (not tracking or self.BALL_FULL_EVERY == 1):
                    n, ov = self.ball_tiles, 0.12
                    tw, th = int(w / n * (1 + ov)), int(h / n * (1 + ov))
                    for iy in range(n):
                        for ix in range(n):
                            x0 = 0 if ix == 0 else min(int(ix * w / n - (tw - w / n) / 2), w - tw)
                            y0 = 0 if iy == 0 else min(int(iy * h / n - (th - h / n) / 2), h - th)
                            cands += self._ball_predict(frame[y0:y0 + th, x0:x0 + tw], x0, y0)
            if roi is not None:
                x0, y0, x1, y1 = roi
                cands += self._ball_predict(frame[y0:y1, x0:x1], x0, y0)
            # 두 번 잡힌 같은 공은 하나로 (신뢰도 높은 쪽)
            merged: List[Tuple[float, float, float]] = []
            for c in sorted(cands, key=lambda c: -c[2]):
                if all((c[0] - m[0]) ** 2 + (c[1] - m[1]) ** 2 > 12 ** 2 for m in merged):
                    merged.append(c)
            merged = [c for c in merged if not self._held_by_person(c[0], c[1], persons)]
            balls.extend(merged)
            best = max(merged, key=lambda c: c[2], default=None)
            if best is not None and best[2] >= self.BALL_FOCUS_CONF:
                self._focus, self._focus_age = (best[0], best[1]), 0
            else:
                self._focus_age += 1
        return persons, balls


# ── 검출 캐시 ────────────────────────────────────────────────────────────────

class _DetectionCache:
    """프레임별 (사람, 공) 검출을 pickle로 저장/재생한다. 파일 이름에 영상·구간·stride가 들어간다."""

    def __init__(self, path: str, detector: Detector, log: Optional[List] = None, meta: Optional[Dict] = None):
        self.path = path
        self._inner = detector
        self._log: List = log if log is not None else []
        self._replay = log is not None
        self._meta: Dict = meta or {}
        self._i = 0
        # 뒤 단계가 참고하는 검출기 속성은 그대로 흉내 낸다
        self.ball_min_conf_new = getattr(detector, "ball_min_conf_new", 0.3)
        self.ball_gap_s = getattr(detector, "ball_gap_s", 2.5)
        self.ball_model = getattr(detector, "ball_model", None)
        if self._replay:
            self.ball_min_conf_new = self._meta.get("ball_min_conf_new", self.ball_min_conf_new)
            self.ball_model = True if self._meta.get("dedicated_ball") else None

    @property
    def detector(self) -> "Detector":
        return self

    @classmethod
    def for_run(cls, source: str, start_f: int, end_f: Optional[int], stride: int,
                detector: Detector) -> Optional["_DetectionCache"]:
        cache_dir = os.environ.get("ANALYSIS_CACHE_DIR")
        if not cache_dir or not isinstance(source, str) or not os.path.exists(source):
            return None
        import pickle
        os.makedirs(cache_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(source))[0]
        size = os.path.getsize(source)
        path = os.path.join(cache_dir, f"{base}_{size}_{start_f}_{end_f or 0}_{stride}.det.pkl")
        if os.path.exists(path):
            with open(path, "rb") as f:
                data = pickle.load(f)
            obj = cls(path, detector, log=data["log"], meta=data.get("meta", {}))
            logger.info("검출 캐시 재생: %s (%d프레임)", path, len(data["log"]))
            return obj
        return cls(path, detector)

    def detect(self, frame_rgb):
        if self._replay:
            if self._i >= len(self._log):
                raise RuntimeError("검출 캐시가 영상보다 짧습니다 — 캐시 파일을 지우고 다시 분석하세요")
            out = self._log[self._i]
            self._i += 1
            return out
        out = self._inner.detect(frame_rgb)
        self._log.append(out)
        return out

    def save(self) -> None:
        if self._replay:
            return
        import pickle
        meta = {"ball_min_conf_new": self.ball_min_conf_new, "dedicated_ball": self.ball_model is not None}
        with open(self.path, "wb") as f:
            pickle.dump({"meta": meta, "log": self._log}, f)
        logger.info("검출 캐시 저장: %s (%d프레임)", self.path, len(self._log))


# ── 실제 영상 ────────────────────────────────────────────────────────────────

@dataclass
class _TrackInfo:
    votes: List[Tuple[str, float]] = field(default_factory=list)
    labs: List[np.ndarray] = field(default_factory=list)
    apps: List[np.ndarray] = field(default_factory=list)   # 머리·반바지·양말 색 (팀원 구분용)
    hist_sum: Optional[np.ndarray] = None   # 상체·하체 HSV 히스토그램 합 (끊긴 선수 다시 잇기용, 평균만 필요해 합으로)
    hist_n: int = 0
    # 초마다 외형 합 {초: [LAB 합(9), LAB 수, 히스토그램 합, 히스토그램 수]} — 조각의 처음·마지막 몇 초 외형을 따로 내려고
    secs: Dict[int, list] = field(default_factory=dict)
    best_area: float = 0.0
    thumb: Optional[bytes] = None


def _app_vector(ti: "_TrackInfo") -> Optional[np.ndarray]:
    """트랙 외형: 머리·반바지·양말 LAB 중앙값(9) + 상체·하체 HSV 히스토그램 평균(144). 없는 부분은 NaN."""
    # 길이는 늘 153 — 조각마다 길이가 다르면 여러 조각을 모아 중앙값·평균을 낼 수 없다
    lab = np.nanmedian(np.array(ti.apps), axis=0) if ti.apps else None
    if lab is None and ti.hist_n == 0:
        return None
    hist = (ti.hist_sum / ti.hist_n).astype(float) if ti.hist_n else np.full(C.HIST_LEN, np.nan)
    return np.concatenate([lab if lab is not None else np.full(9, np.nan), hist])


EDGE_APPEARANCE_S = 15.0


def _edge_vectors(ti: "_TrackInfo", window_s: float = EDGE_APPEARANCE_S
                  ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """트랙 처음·마지막 window_s초의 외형(LAB 평균 9 + 히스토그램 평균 144). 끊긴 순간 가까이의 외형으로 잇기 위해."""
    secs = sorted(s for s in ti.secs if s >= 0)
    if not secs:
        return None, None

    def vec(keys):
        lab = sum((ti.secs[s][0] for s in keys), np.zeros(9))
        ln = sum(ti.secs[s][1] for s in keys)
        hs = [ti.secs[s][2] for s in keys if ti.secs[s][2] is not None]
        hn = sum(ti.secs[s][3] for s in keys)
        if not ln and not hn:
            return None
        return np.concatenate([lab / ln if ln else np.full(9, np.nan),
                               (sum(hs) / hn).astype(float) if hn else np.full(C.HIST_LEN, np.nan)])
    return (vec([s for s in secs if s <= secs[0] + window_s]), vec([s for s in secs if s >= secs[-1] - window_s]))


def _encode_thumb(frame_rgb: np.ndarray, box) -> Optional[bytes]:
    try:
        import cv2
    except ImportError:  # pragma: no cover
        return None
    h, w = frame_rgb.shape[:2]
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    pad = int(0.1 * (y2 - y1))
    x1, y1, x2, y2 = max(0, x1 - pad), max(0, y1 - pad), min(w, x2 + pad), min(h, y2 + pad)
    if x2 - x1 < 4 or y2 - y1 < 8:
        return None
    crop = cv2.cvtColor(frame_rgb[y1:y2, x1:x2], cv2.COLOR_RGB2BGR)
    scale = 128 / crop.shape[0]
    crop = cv2.resize(crop, (max(1, int(crop.shape[1] * scale)), 128))
    ok, buf = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    return buf.tobytes() if ok else None


def _pitch_registrar(source: str, start_f: int, end_f: Optional[int], stride: int, frame_w: int, frame_h: int,
                     length: float, width: float) -> Tuple[Optional[PitchRegistrar], Optional[str]]:
    """경기장 선 자동 인식기. 키포인트 모델(YOLO_PITCH_MODEL)이 있거나, 같은 영상·구간의 키포인트를 저장해 둔
    파일(ANALYSIS_CACHE_DIR의 *.pitch.pkl)이 있을 때만 만든다. 반환: (인식기, 새로 저장할 파일 경로)"""
    path = None
    cache_dir = os.environ.get("ANALYSIS_CACHE_DIR")
    if cache_dir and isinstance(source, str) and os.path.exists(source):
        base = os.path.splitext(os.path.basename(source))[0]
        path = os.path.join(cache_dir, f"{base}_{os.path.getsize(source)}_{start_f}_{end_f or 0}_{stride}.pitch.pkl")
        if os.path.exists(path):
            import pickle
            with open(path, "rb") as f:
                data = pickle.load(f)
            reg = PitchRegistrar(frame_w, frame_h, length, width, model=None, every=int(data.get("every", 1)))
            for k, kps in data["kps"].items():
                reg.add_keypoints(int(k), kps)
            logger.info("경기장 키포인트 캐시 재생: %s (%d장면)", path, len(reg.kps))
            return reg, None
    model = PitchKeypointModel.from_env()
    if model is None:
        return None, None
    return PitchRegistrar(frame_w, frame_h, length, width, model=model,
                          every=int(os.environ.get("YOLO_PITCH_EVERY", "1"))), path


def _registered_geometry(reg: PitchRegistrar, pts: Sequence, length: float, width: float, smooth_n: int,
                         margin: float = 8.0) -> Optional[List[Tuple]]:
    """트랙의 발 위치(박스 아래 가운데, 원본 화면 좌표)를 경기장 m로 옮긴다. 절반 넘게 경기장 밖이면 None
    (벤치·관중·라인 밖에서 몸 푸는 사람). 장면마다 변환이 조금씩 흔들리므로 짧게 평균을 내 떨림을 줄인다.
    반환 항목: (장면, x, y, 점유 반경 m, 축척, 화면 발 x, 화면 발 y, 화면 점유 반경 px)"""
    raw = []
    for p in pts:
        b = p.payload[3] if isinstance(p.payload, tuple) and len(p.payload) > 3 else p.box
        fx, fy = (b[0] + b[2]) / 2, b[3]
        q = reg.to_pitch(p.frame, fx, fy)
        if q is not None and np.isfinite(q[0]) and np.isfinite(q[1]):
            raw.append((p.frame, q[0], q[1], fx, fy, POSSESSION_RADIUS_H * max(4.0, b[3] - b[1])))
    if len(raw) < 3:
        return None
    ok = [(-margin <= r[1] <= length + margin and -margin <= r[2] <= width + margin) for r in raw]
    if sum(ok) < 0.5 * len(raw):
        return None
    raw = [r for r, o in zip(raw, ok) if o]
    h = max(0, smooth_n // 2)
    out = []
    for i, (f, x, y, fx, fy, fr) in enumerate(raw):
        seg = [r for r in raw[max(0, i - h): i + h + 1] if abs(r[0] - f) <= 2 * h]
        out.append((f, float(np.mean([r[1] for r in seg])), float(np.mean([r[2] for r in seg])),
                    POSSESSION_RADIUS_M, 1.0, fx, fy, fr))
    return out


def analyze_video(source: str, cfg: AnalysisConfig, detector: Optional[Detector] = None,
                  progress: Optional[ProgressFn] = None) -> Dict:
    import cv2

    progress = progress or (lambda p, m: None)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"영상을 열 수 없습니다: {source}")
    fps_video = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    a_fps = max(MIN_ANALYSIS_FPS, min(cfg.effective_fps(), fps_video))
    stride = max(1, int(round(fps_video / a_fps)))
    eff_fps = fps_video / stride
    start_f = int(cfg.start_seconds * fps_video)
    end_f = total if not cfg.max_seconds else min(total or 10 ** 9, start_f + int(cfg.max_seconds * fps_video))
    if start_f:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)

    mapper = PitchMapper.build(frame_w, frame_h, cfg.field_type, cfg.calibration)
    # 보정이 없으면 카메라가 움직인다고 보고 화면 이동을 보정한다 (보정된 고정 카메라는 그대로)
    cam = None if mapper.calibrated else CameraMotion(frame_w, frame_h)
    clf = C.TeamColorClassifier.from_match(cfg.home_color, cfg.away_color, cfg.referee_color,
                                           cfg.home_gk_color, cfg.away_gk_color)
    detector = detector or YoloDetector.for_quality(cfg.quality)
    # 검출 캐시: ANALYSIS_CACHE_DIR가 있으면 프레임별 검출을 저장해 두고, 같은 영상·구간이면 YOLO 없이 재생한다.
    # 규칙(점유·패스·세트피스)을 바꿀 때 몇 시간짜리 재검출 없이 몇 초 만에 다시 계산하기 위한 것.
    det_cache = _DetectionCache.for_run(source, start_f, end_f, stride, detector)
    if det_cache is not None:
        detector = det_cache.detector
    tracker = IoUTracker(team_of=lambda pl: (C.TEAM_OF_LABEL.get(pl[0]), pl[1]) if pl else (None, 0.0))
    info: Dict[int, _TrackInfo] = defaultdict(_TrackInfo)
    times: List[float] = []
    ball_cands: List[List[Tuple[float, float, float]]] = []
    warnings: List[str] = []
    rng = np.random.default_rng(0)

    # 색 보정 전까지 검출(프레임 이미지는 제외 — 메모리)을 모아 두었다가 보정 후 한꺼번에 분류
    buffered: List[Tuple[int, List]] = []
    cam_T: List[np.ndarray] = []   # 프레임별 카메라 변환(현재→기준) — 영상 검수 오버레이에서 원래 화면 좌표로 되돌릴 때 쓴다
    calibrated_colors = False
    calib_frames = max(10, int(cfg.color_calib_s * eff_fps))
    board = ScoreboardReader(frame_w, frame_h)   # 영상에 박힌 점수 자막 → 골 시각
    # 보정 없는 영상: 카메라 쪽 터치라인을 찾아 그 바깥(벤치·몸 푸는 사람)을 선수에서 뺀다
    touch = None if mapper.calibrated else TouchlineFilter()
    halfway_x: List[float] = []   # 하프라인의 경기장 x (안정화 좌표) — 경기장 가운데 찾기
    removed_outside = 0
    # 경기장 선 자동 인식: 사용자가 경기장을 찍지 않아도 장면마다 화면 → 경기장(m) 변환을 구한다
    reg, reg_cache = (None, None)
    if not mapper.calibrated and cfg.field_type != "futsal":
        try:
            reg, reg_cache = _pitch_registrar(source, start_f, end_f, stride, frame_w, frame_h,
                                              mapper.length, mapper.width)
        except Exception:
            logger.exception("경기장 선 인식 준비 실패")

    def run_tracker(k: int, frame_rgb: Optional[np.ndarray], dets: List):
        tracked = []
        hists: Dict[int, np.ndarray] = {}   # 히스토그램은 추적 기록(payload)에 넣지 않는다 — 장면마다 쌓이면 메모리가 커서
        for box, conf, sample, orig in dets:
            label, lconf = clf.classify(sample)
            det = Detection(box, conf, (label, lconf, sample.median_lab if sample else None, orig,
                                        sample.appearance if sample else None))
            if sample is not None and sample.hist is not None:
                hists[id(det)] = sample.hist
            tracked.append(det)
        for tid, det in tracker.update(k, tracked):
            ti = info[tid]
            hh = hists.get(id(det))
            sec = ti.secs.setdefault(int(times[k]) if k < len(times) else -1, [np.zeros(9), 0, None, 0])
            if hh is not None:
                hv = np.nan_to_num(hh)
                ti.hist_sum = hv.copy() if ti.hist_sum is None else ti.hist_sum + hv
                ti.hist_n += 1
                sec[2] = hv.copy() if sec[2] is None else sec[2] + hv
                sec[3] += 1
            ap = det.payload[4]
            if ap is not None and np.isfinite(ap).all():
                sec[0] = sec[0] + ap
                sec[1] += 1
            ti.votes.append((det.payload[0], det.payload[1]))
            if det.payload[2] is not None:
                ti.labs.append(det.payload[2])
            if det.payload[4] is not None and len(ti.apps) < 400:
                ti.apps.append(det.payload[4])
            ob = det.payload[3]
            area = (ob[2] - ob[0]) * (ob[3] - ob[1])
            if frame_rgb is not None and area > ti.best_area and det.payload[0] in C.TEAM_OF_LABEL:
                thumb = _encode_thumb(frame_rgb, ob)
                if thumb:
                    ti.best_area, ti.thumb = area, thumb

    k = 0
    f_idx = start_f
    t_started = time.time()
    while True:
        if end_f and f_idx >= end_f:
            break
        ok = cap.grab()
        if not ok:
            break
        if (f_idx - start_f) % stride:
            f_idx += 1
            continue
        ok, bgr = cap.retrieve()
        if not ok:
            break
        frame_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        t = (f_idx - start_f) / fps_video
        persons, balls = detector.detect(frame_rgb)
        if touch is not None:
            try:
                persons = touch.update(t, frame_rgb, persons)
            except Exception:  # 부가 기능 — 실패하면 거르지 않는다
                logger.exception("터치라인 인식 실패")
                touch = None
        balls = in_play_candidates(balls, persons, frame_h)   # 벤치 옆 여분의 공·조명 같은 경기 밖 후보 제거
        balls = drop_weak_at_feet(balls, persons)              # 약한 후보 중 선수 발(흰 축구화) 자리는 버린다
        try:
            board.add(t, frame_rgb)
        except Exception:  # 부가 기능 — 실패해도 분석은 계속
            logger.exception("점수 자막 읽기 실패")
            board.failed = True
        if reg is not None and reg.model is not None:
            try:
                reg.add(k, frame_rgb)
            except Exception:  # 부가 기능 — 실패하면 위치는 예전 방식(화면 좌표 근사)으로
                logger.exception("경기장 선 인식 실패")
                reg = None
        if cam is not None:
            cam_T.append(cam.update(frame_rgb, [b for b, _ in persons]).copy())
            if touch is not None and k % 5 == 0:
                try:
                    hl = halfway_line(frame_rgb)
                    if hl is not None:
                        gx, gy = (hl[0], hl[1]) if hl[1] > hl[3] else (hl[2], hl[3])   # 선의 아래 끝 (땅)
                        halfway_x.append(mapper.to_pitch(*cam.point(gx, gy))[0])
                except Exception:
                    logger.exception("하프라인 인식 실패")
            balls = [(*cam.point(bx, by), bc) for bx, by, bc in balls]
        dets = []
        for box, conf in persons:
            if mapper.calibrated:
                x, y, _, _ = mapper.player_geometry(box)
                if not mapper.inside(x, y):
                    removed_outside += 1
                    continue  # 관중·벤치
            ref = cam.box(box) if cam is not None else box
            dets.append((ref, conf, C.sample_jersey(frame_rgb, box, rng=rng), box))
        times.append(t)
        ball_cands.append(balls)

        if not calibrated_colors:
            buffered.append((k, dets))
            if len(buffered) >= calib_frames:
                clf.calibrate([d[2] for _, ds in buffered for d in ds if d[2] is not None])
                calibrated_colors = True
                for bk, bdets in buffered:
                    run_tracker(bk, None, bdets)
                buffered = []
        else:
            run_tracker(k, frame_rgb, dets)

        k += 1
        f_idx += 1
        if k % 25 == 0:
            span = (end_f - start_f) if end_f else 0
            pct = min(95.0, (f_idx - start_f) / span * 90) if span else 50.0
            progress(pct, f"선수·공 검출 중 {int(t // 60)}분 {int(t % 60)}초 지점")
    cap.release()

    if not calibrated_colors:
        clf.calibrate([d[2] for _, ds in buffered for d in ds if d[2] is not None])
        for bk, bdets in buffered:
            run_tracker(bk, None, bdets)
    if not times:
        raise RuntimeError("영상에서 프레임을 읽지 못했습니다")
    logger.info("검출 완료: %d프레임, %.0f초 소요", len(times), time.time() - t_started)
    progress(92, "선수 궤적 정리 및 이벤트 판정 중")

    # ── 경기장 선 인식 결과: 충분히 잡혔으면 모든 위치를 경기장 m로 계산한다 ────────────
    reg_info: Optional[Dict] = None
    if reg is not None:
        try:
            if reg_cache and reg.kps:
                import pickle
                with open(reg_cache, "wb") as f:
                    pickle.dump({"every": reg.every, "kps": reg.kps}, f)
            reg_info = reg.finish(cam_T, times)
            reg_info["used"] = reg_info["covered_pct"] >= 60.0 and reg_info["anchors"] >= 20
            logger.info("경기장 선 인식: %s", reg_info)
        except Exception:
            logger.exception("경기장 선 인식 정리 실패")
            reg_info = None
    registered = bool(reg_info and reg_info.get("used"))
    # 정밀 단계: 선 인식 위치를 '경기장 지정'과 같은 수준으로 믿고 코너킥·드로잉·프리킥(파울)·센터서클 킥오프까지 위치로
    # 판정한다. 공개 키포인트 모델은 방송 경기장 비율을 가정해, 우리 구장 미드필드 장면에서 하프라인이 4~5m 어긋났다
    # (2026-11-15, 2쿼터 화면에 도면을 겹쳐 확인). 그 오차로는 선 근처 판정이 틀리므로 기본은 끄고, 오차에 강한 것만 쓴다:
    # 골키퍼(박스 안에 머무는 선수), 공격 방향, 경기장 밖 사람 제외, 평균 위치, 이동 거리.
    precise = registered and os.environ.get("YOLO_PITCH_PRECISE", "0") == "1"
    if reg_info is not None:
        reg_info["precise"] = precise
    length, width = (reg.length, reg.width) if registered else (mapper.length, mapper.width)
    smooth_n = max(1, int(round(0.4 * eff_fps)) | 1)

    # ── 트랙 → 팀 확정, 경기장 좌표 ───────────────────────────────────────────
    tracklets: List[Tracklet] = []
    labels: Dict[str, str] = {}
    geo: Dict[str, List[Tuple[int, float, float, float, float]]] = {}
    removed_static = 0
    for tid, pts in tracker.history.items():
        lbl, _ = C.vote_track_labels(info[tid].votes)
        if lbl not in C.TEAM_OF_LABEL:
            continue
        if cam is not None and lbl in (C.HOME, C.AWAY) and stationary(
                [(times[p.frame], (p.box[0] + p.box[2]) / 2, p.box[3]) for p in pts],
                [p.box[3] - p.box[1] for p in pts]):
            removed_static += 1
            continue  # 오래 제자리에 있는 사람 — 관중·벤치
        key = f"T{tid}"
        if registered:
            g = _registered_geometry(reg, pts, length, width, smooth_n)
            if g is None:
                removed_outside += 1
                continue  # 경기장 밖에 있던 사람 (벤치·관중·라인 밖에서 몸 푸는 사람)
        else:
            g = []
            for p in pts:
                x, y, r, s = mapper.player_geometry(p.box)
                g.append((p.frame, x, y, r, s))
        labels[key] = lbl
        geo[key] = g
        lab = np.median(np.array(info[tid].labs), axis=0) if info[tid].labs else None
        tracklets.append(Tracklet(key, C.TEAM_OF_LABEL[lbl], [(times[q[0]], q[1], q[2]) for q in g], lab))

    mapping = stitch_tracklets(tracklets)
    # 골키퍼: 유니폼 색이 필드 선수와 같아도, 화면에 잡힌 동안 내내 자기 골문 앞 박스에 있던 선수다
    gk_info: Optional[Dict] = None
    if registered:
        try:
            chains: Dict[str, Tuple[str, List[Tuple[float, float, float]]]] = {}
            for tr in tracklets:
                head = mapping.get(tr.tid, tr.tid)
                chains.setdefault(head, (tr.team, []))[1].extend(tr.points)
            for v in chains.values():
                v[1].sort(key=lambda p: p[0])
            ht = (cfg.halftime_s - cfg.start_seconds) if cfg.halftime_s is not None else None
            side = None
            if cfg.home_attacks in ("right", "left"):
                side = "left" if cfg.home_attacks == "right" else "right"
            else:
                side = home_goal_side(chains, length, halftime_s=ht)
            gks, gk_info = find_goalkeepers(chains, length, width, halftime_s=ht, home_side=side)
            for tr in tracklets:
                team = gks.get(mapping.get(tr.tid, tr.tid))
                if team and labels.get(tr.tid) in (C.HOME, C.AWAY, C.HOME_GK, C.AWAY_GK):
                    # 골키퍼 팀은 지키는 골문으로 정한다 (옷 색이 상대 팀과 비슷해도)
                    labels[tr.tid] = C.HOME_GK if team == "home" else C.AWAY_GK
                    tr.team = team
            if gk_info:
                gk_info["pieces"] = len(gks)
        except Exception:
            logger.exception("골키퍼 찾기 실패")
    # 카메라가 돌아간 사이 화면 밖에 있다 돌아온 선수는 자동으로 잇기엔 불확실해 '같은 선수 후보'만 남긴다
    merge_hints: Dict[str, List[str]] = {}
    try:
        def at_edge(pt) -> bool:
            raw = pt.payload[3] if isinstance(pt.payload, tuple) and len(pt.payload) > 3 else pt.box
            return raw[0] < 0.03 * frame_w or raw[2] > 0.97 * frame_w
        keys = [t.tid for t in tracklets]
        apps = {key: (_app_vector(info[int(key[1:])]), len(info[int(key[1:])].apps)) for key in keys}
        exits = {key: at_edge(tracker.history[int(key[1:])][-1]) for key in keys}
        enters = {key: at_edge(tracker.history[int(key[1:])][0]) for key in keys}
        hints = suggest_same_player(tracklets, mapping, apps, exits, enters, C.appearance_distance)
        merge_hints = {"P" + h[1:]: ["P" + o[1:] for o in lst] for h, lst in hints.items()}
    except Exception:  # 부가 정보 — 실패해도 분석 결과는 살린다
        logger.exception("같은 선수 후보 계산 실패")
    # 경기장 안에는 팀당 필드 10 + GK 1명 — 화면 밖에 나갔다 돌아온 조각을 정원 칸에 맞춰 묶는다.
    # 선수 행은 조각별로 남기고 묶음만 meta로 넘겨, 저장할 때 '자동 합치기'(되돌리기 가능)로 반영한다.
    roster_slots: Dict[str, str] = {}
    roster_extra = 0
    if cfg.field_type != "futsal":
        try:
            edges = {t.tid: _edge_vectors(info[int(t.tid[1:])]) for t in tracklets}
            roster_slots, roster_extra = roster_from_tracklets(
                tracklets, mapping, apps, labels, clf.reference_lab, edges=edges)
        except Exception:
            logger.exception("팀당 정원 정리 실패")
    players_obs: Dict[int, List[PlayerObs]] = defaultdict(list)
    merged_labels: Dict[str, str] = {}
    thumbs: Dict[str, Optional[bytes]] = {}
    thumb_area: Dict[str, float] = {}
    color_by_player: Dict[str, List[np.ndarray]] = defaultdict(list)
    for key, g in geo.items():
        pid = "P" + mapping.get(key, key)[1:]
        merged_labels.setdefault(pid, labels[key])
        team = C.TEAM_OF_LABEL[labels[key]]
        for q in g:
            players_obs[q[0]].append(PlayerObs(pid, team, *q[1:]))
        tid = int(key[1:])
        if info[tid].thumb and info[tid].best_area > thumb_area.get(pid, 0):
            thumbs[pid], thumb_area[pid] = info[tid].thumb, info[tid].best_area
        color_by_player[pid].extend(info[tid].labs[:50])

    if det_cache is not None:
        det_cache.save()

    # 공 전용 모델은 검출이 촘촘해 궤적 기억·보간 구간을 더 길게 잡아도 엉뚱한 점을 잇지 않는다.
    # 실측(1080p 조기축구 1분): 보간 0.8초 → 공 위치 파악 66%, 2.5초 → 91%. 발밑에 가려진 공이 대부분이라
    # 양 끝 이동 속도가 화면 폭의 0.6배/초를 넘는 공백(시야 밖으로 나간 경우)만 잇지 않는다.
    dedicated_ball = getattr(detector, "ball_model", None) is not None
    # 공 전용 모델은 앞뒤 양방향으로 추적한다 — 다시 잡힌 공에서 거슬러 올라가며 궤적 근처의 약한 후보를 받는다
    # (2쿼터 50초 실험: 후보 0.07↑ + 발 자리 약한 후보 제거 + 양방향 → 직접 검출 +2.3%p, 새 점 19개 중 맞음 15개)
    select = select_ball_positions_bidir if dedicated_ball else select_ball_positions
    ball_sel = select(
        ball_cands, times, frame_w, min_conf_new=getattr(detector, "ball_min_conf_new", 0.3),
        memory_s=1.5 if dedicated_ball else 1.0)
    # 카메라가 공을 따라 돌아가는 사이 놓친 공은 화면 이동을 따라 잇는다 (화면 중심의 안정화 좌표)
    cam_centers = [tuple((T @ np.array([frame_w / 2, frame_h / 2, 1.0]))[:2]) for T in cam_T] if cam_T else None
    ball_px = interpolate_gaps(ball_sel, times,
                               max_gap_s=getattr(detector, "ball_gap_s", 2.5) if dedicated_ball else 0.8,
                               max_speed_px_s=0.6 * frame_w if dedicated_ball else None,
                               cam_centers=cam_centers if dedicated_ball else None,
                               cam_min_shift_px=0.05 * frame_w)
    # 긴 공백(최대 BALL_ROUGH_GAP_S초)의 대략 위치 — 화면 표시와 '공 위치 파악' 비율에만 쓰고 이벤트 판정에는 안 쓴다.
    # 2쿼터 공백은 대부분 경기 중(긴 패스·발밑 가림)이었다. 잡힌 공을 일부러 가리고 직선으로 이어 본 실측:
    # 3초 공백 67%·5초 50%·8초 41%가 실제 위치 60px(화면 폭 3%) 안 — 짧은 공백보다 부정확해서 따로 표시한다.
    ball_est = ball_px
    if dedicated_ball:
        ball_est = interpolate_gaps(ball_px, times, max_gap_s=BALL_ROUGH_GAP_S, cam_centers=cam_centers,
                                    cam_max_gap_s=BALL_ROUGH_GAP_S, cam_min_shift_px=0.05 * frame_w)
    frames = []
    for i, t in enumerate(times):
        b = ball_px[i]
        if registered:
            bp = bi = None
            if b is not None:
                # 공 위치는 안정화 좌표라 그 장면의 원래 화면 좌표로 되돌린 뒤 경기장으로 옮긴다
                q = np.linalg.inv(cam_T[i]) @ np.array([b[0], b[1], 1.0]) if i < len(cam_T) else (b[0], b[1], 1.0)
                bi = (float(q[0] / q[2]), float(q[1] / q[2]))
                bp = reg.to_pitch(i, bi[0], bi[1])
                if bp is not None and not (-15 <= bp[0] <= length + 15 and -15 <= bp[1] <= width + 15):
                    bp = None      # 높이 뜬 공·잘못 잡은 점 — 경기장에서 한참 벗어난 위치는 버린다
            frames.append(Frame(t, players_obs.get(i, []), bp, bi if bp is not None else None))
            continue
        bp = mapper.to_pitch(*b) if b is not None else None
        frames.append(Frame(t, players_obs.get(i, []), bp))

    if cam is not None and not mapper.calibrated and not registered:
        # 하프라인이 충분히 잡혔으면 그 위치를 경기장 가운데로 맞춘다 (양쪽 골대 거리가 같아진다)
        center = float(np.median(halfway_x)) if len(halfway_x) >= 20 else None
        frames = fit_to_extents(frames, length, width, center_x=center)
    ball_seen = sum(1 for b in ball_px if b is not None) / len(times)       # 이벤트 판정에 쓴 위치
    ball_rough = sum(1 for b in ball_est if b is not None) / len(times)     # 긴 공백 대략 위치까지
    if ball_seen < 0.3:
        warnings.append(f"공이 영상의 {ball_seen * 100:.0f}% 구간에서만 검출되어 볼 이벤트가 적게 잡혔을 수 있습니다.")
    camera_moving = cam is not None and cam.travel() > 0.3 * frame_w
    if precise:
        pass    # 경기장 선으로 위치를 정밀하게 계산했다 — 카메라가 움직여도 위치·슈팅·세트피스 판정을 쓴다
    elif camera_moving:
        warnings.append("카메라가 움직이는 영상입니다. 화면 이동을 보정해 추적했지만 공격 방향·슈팅 위치 판정은 "
                        "부정확할 수 있으니 '이벤트 검수'에서 슈팅을 확인해 주세요.")
    elif not mapper.calibrated:
        warnings.append("경기장 지정이 없어 경기장 밖 사람(벤치·관중)이 일부 섞일 수 있습니다. '선수 정리'에서 제외하세요.")
    if len(tracker.history) and not labels:
        warnings.append("유니폼 색으로 팀을 구분하지 못했습니다. 경기 설정의 팀 색상을 확인하세요.")

    meta = {
        "source": "video",
        "start_s": cfg.start_seconds,
        "duration_s": round(times[-1], 1),
        "fps_video": round(fps_video, 2),
        "analysis_fps": round(eff_fps, 2),
        "frame_w": frame_w, "frame_h": frame_h,
        "calibrated": mapper.calibrated or precise,
        "auto_calibrated": registered,          # 경기장 선 자동 인식으로 위치(m)를 계산함 (사용자 지정 없음, 오차 몇 m)
        "metric_positions": mapper.calibrated or registered,
        "registration": reg_info,
        "goalkeepers": gk_info,
        "ball_detected_pct": round(ball_rough * 100, 1),         # 위치를 정한 장면 (긴 공백의 대략 위치까지 포함)
        "ball_events_pct": round(ball_seen * 100, 1),            # 이벤트 판정에 쓴 장면 (직접 + 짧은 공백 이음)
        "ball_direct_pct": round(sum(1 for b in ball_sel if b is not None) / len(times) * 100, 1),  # 모델이 직접 잡은 장면
        "raw_tracks": len(tracker.history),
        "removed_outside_pitch": removed_outside,
        "removed_static": removed_static,
        "removed_outside_touchline": touch.removed if touch is not None else 0,
        "halfway_samples": len(halfway_x),
        "camera_moving": camera_moving,
        "snapped_colors": dict(clf.snapped_hex),
        "merge_hints": merge_hints,
        "roster_slots": roster_slots,
        "roster_over_capacity": roster_extra,
        "warnings": warnings,
    }
    colors = {pid: C.rgb_to_hex(C.lab_to_rgb(np.median(np.array(v), axis=0))[0])
              for pid, v in color_by_player.items() if v}
    board_out = {"found": False}
    try:
        board_out = board.finish()
    except Exception:
        logger.exception("점수 자막 읽기 실패")
    # 킥오프: 영상 시작과 골 뒤에 양 팀이 각자 진영에 갈라선 순간 → 공격 방향, 그리고 실점한 팀(킥오프 팀)
    kicks = kickoffs(frames, board_out, cfg, pitch=(length, width) if precise else None)
    meta["kickoffs"] = kicks
    if cfg.home_attacks == "auto" and gk_info and cfg.halftime_s is None:
        # 골키퍼가 지키는 골문이 가장 확실한 근거다: 홈 골키퍼가 왼쪽 골문이면 홈은 오른쪽으로 공격한다
        cfg.home_attacks = "right" if gk_info["home_side"] == "left" else "left"
        meta["attack_from_goalkeeper"] = True
    elif cfg.home_attacks == "auto" and kicks:
        first_half = [k["home_dir"] * (-1 if cfg.halftime_s is not None and k["t"] + cfg.start_seconds >= cfg.halftime_s else 1)
                      for k in kicks]
        cfg.home_attacks = "right" if sum(first_half) > 0 else "left"
        meta["attack_from_kickoff"] = True
    result = postprocess(frames, cfg, length, width, meta, merged_labels, thumbs, colors, progress)
    try:
        result["meta"]["scoreboard"] = scoreboard_goals(board_out, result["events"],
                                                        left_team=scoreboard_left_team(kicks, result["events"]))
        if precise and not result["meta"]["scoreboard"].get("found"):
            # 점수 자막이 없는 영상: 경기 도중의 센터서클 킥오프로 골을 찾는다
            result["meta"]["scoreboard"] = goals_from_kickoffs(kicks, result["events"])
            result["meta"]["kickoffs"] = kicks
    except Exception:
        logger.exception("점수 자막 골 판정 실패")
    try:
        whistles = detect_whistles(load_audio(source, cfg.start_seconds, cfg.max_seconds))
        result["meta"]["whistles"] = [[round(t, 2), round(d, 2), round(s)] for t, d, s in whistles]
        result["events"] = whistle_restarts(result["events"], whistles, result["meta"].get("scoreboard") or {},
                                            frames=frames if precise else None,
                                            pitch=(length, width) if precise else None,
                                            attack=result["meta"].get("attack"),
                                            halftime_s=(cfg.halftime_s - cfg.start_seconds)
                                            if cfg.halftime_s is not None else None)
    except Exception:  # 소리가 없거나 읽지 못해도 분석은 그대로
        logger.exception("호루라기 판정 실패")
    try:
        result["overlay"] = build_overlay(
            tracker.history, {int(k[1:]): "P" + mapping.get(k, k)[1:] for k in geo},
            ball_sel, ball_px, cam_T, frame_w, frame_h, cfg.start_seconds, stride / fps_video, len(times),
            ball_est=ball_est)
    except Exception:  # 검수용 부가 데이터 — 실패해도 분석 결과는 살린다
        logger.exception("영상 검수 오버레이 생성 실패")
    return result


def kickoffs(frames: Sequence[Frame], board: Dict, cfg: "AnalysisConfig",
             pitch: Optional[Tuple[float, float]] = None) -> List[Dict]:
    """영상 시작(0~40초)과 점수 자막이 바뀐 골 뒤(−20~+120초)의 킥오프 배치.
    pitch(경기장 길이·폭)가 있으면 위치를 미터로 아는 것이므로, 공이 센터 스폿에 놓이고 양 팀이 하프라인으로 갈라선
    진짜 킥오프를 먼저 찾는다. 점수 자막과 짝이 없는 경기 도중 킥오프는 video_only로 남긴다(자막 없는 영상의 골 근거)."""
    out: List[Dict] = []
    centre = centre_kickoffs(frames, *pitch) if pitch else []
    used: set = set()
    first = next((i for i, k in enumerate(centre) if k["t"] <= 60.0), None)
    if first is not None:
        used.add(first)
        out.append({**centre[first], "after_goal": None, "by": "centre"})
    else:
        k = find_kickoff(frames, 0.0, 40.0, min_sep=0.9)
        if k:
            out.append({**k, "after_goal": None})
    for c in (board.get("changes") or []) if board.get("found") else []:
        i = next((i for i, k in enumerate(centre) if i not in used and c["t"] - 20.0 <= k["t"] <= c["t"] + 150.0), None)
        if i is not None:
            used.add(i)
            out.append({**centre[i], "after_goal": c["t"], "goal_side": c["side"], "by": "centre"})
            continue
        k = find_kickoff(frames, c["t"] - 20.0, c["t"] + 120.0, min_sep=0.85)
        if k:
            out.append({**k, "after_goal": c["t"], "goal_side": c["side"]})
    for i, k in enumerate(centre):
        if i not in used and k["t"] > 60.0:
            out.append({**k, "after_goal": None, "video_only": True, "by": "centre"})
    return [{kk: (round(v, 2) if isinstance(v, float) else v) for kk, v in k.items()} for k in out]


def goals_from_kickoffs(kicks: List[Dict], events: Sequence[Dict], window_s: float = 150.0) -> Dict:
    """점수 자막이 없을 때: 경기 도중 센터서클 킥오프마다 그 직전에 골이 있었다고 본다.
    킥오프를 한(직후 처음 공을 잡은) 팀이 실점한 팀이고, 넣은 팀의 마지막 슈팅을 득점 장면으로 붙인다."""
    other = {"home": "away", "away": "home"}
    spells = sorted((e for e in events if e["kind"] == "possession"), key=lambda e: e["t"])
    goals = []
    for k in kicks:
        if not k.get("video_only"):
            continue
        sp = next((s for s in spells if k["t"] - 1.0 <= s["t"] <= k["t"] + 15.0), None)
        if sp is None:
            continue
        team = other[sp["team"]]
        t_ref = k.get("t0", k["t"])
        shots = [e for e in events if e["kind"] == "shot" and e["team"] == team and t_ref - window_s <= e["t"] <= t_ref]
        shot = max(shots, key=lambda e: e["t"]) if shots else None
        tb = round(max(0.0, t_ref - 1.0), 2)
        k["after_goal"] = tb       # 결과 조회 때 '킥오프로 확인된 골'로 짝지어진다
        goals.append({"t_board": tb, "team": team, "side": None, "source": "video",
                      "t": shot["t"] if shot else tb, "player": shot["player"] if shot else None})
    if not goals:
        return {"found": False}
    score = {"home": sum(g["team"] == "home" for g in goals), "away": sum(g["team"] == "away" for g in goals)}
    return {"found": True, "source": "video", "goals": goals, "score": score}


def scoreboard_left_team(kicks: Sequence[Dict], events: Sequence[Dict]) -> str:
    """점수 자막 왼쪽이 어느 팀인가. 골 뒤 킥오프를 한 팀(킥오프 직후 처음 공을 잡은 팀)이 실점한 팀이므로
    골을 넣은 팀은 그 반대다. 근거가 없으면 홈(관례상 왼쪽)."""
    votes = {"home": 0, "away": 0}
    other = {"home": "away", "away": "home"}
    spells = sorted((e for e in events if e["kind"] == "possession"), key=lambda e: e["t"])
    for k in kicks:
        if k.get("after_goal") is None:
            continue
        sp = next((s for s in spells if k["t"] - 1.0 <= s["t"] <= k["t"] + 15.0), None)
        if sp is None:
            continue
        scorer = other[sp["team"]]
        votes[scorer if k["goal_side"] == "left" else other[scorer]] += 1
    return "away" if votes["away"] > votes["home"] else "home"


def whistle_restarts(events: List[Dict], whistles: Sequence[Tuple[float, float, float]], board: Dict,
                     window_s: float = 25.0, dup_s: float = 3.0, kickoff_after_goal_s: float = 120.0,
                     frames: Optional[Sequence[Frame]] = None, pitch: Optional[Tuple[float, float]] = None,
                     attack: Optional[Dict] = None, halftime_s: Optional[float] = None) -> List[Dict]:
    """호루라기 뒤 처음 공을 잡은 점유를 '재개'(파울 뒤 프리킥 등) 세트피스로 더한다.
    골(점수 자막) 뒤 kickoff_after_goal_s 안의 첫 호루라기 재개는 킥오프로 본다. 이미 있는 세트피스와 겹치면 더하지 않고
    그 세트피스에 호루라기 시각만 붙인다.

    pitch(경기장 길이·폭)가 있으면 위치를 미터로 아는 것이므로 재개 위치로 종류를 나눈다: 코너킥·드로잉·골킥·킥오프,
    그 밖의 경기장 안 재개는 프리킥(페널티 스폿이면 페널티킥)이고, 이때 재개하지 않은 쪽이 파울(또는 오프사이드)을
    한 것이다. 호루라기 직전 공에 가장 가까웠던 그 팀 선수를 파울한 선수로 추정해 target에 적는다."""
    other = {"home": "away", "away": "home"}
    spells = sorted((e for e in events if e["kind"] == "possession"), key=lambda e: e["t"])
    existing = [e for e in events if e["kind"] == "set_piece"]
    goal_ts = sorted(g["t_board"] for g in (board.get("goals") or []))
    used_goal = set()
    out = list(events)

    def typed(sp: Dict, wt: float, detail: Dict) -> Tuple[Dict, Optional[str]]:
        """위치로 재개 종류를 정하고, 프리킥·페널티킥이면 파울 정보를 붙인다. 반환: (detail, 파울한 선수)"""
        if pitch is None or sp.get("x") is None or sp.get("y") is None:
            return detail, None
        half = "second" if halftime_s is not None and sp["t"] >= halftime_s else "first"
        d = ((attack or {}).get(half) or {}).get(sp["team"])
        own_goal_x = None if d is None else (0.0 if d > 0 else pitch[0])
        detail["type"] = restart_type(sp["x"], sp["y"], pitch[0], pitch[1], own_goal_x)
        if detail["type"] in ("free_kick", "penalty"):
            detail["foul"] = True
            return detail, (foul_player(frames, wt, other[sp["team"]]) if frames is not None else None)
        return detail, None

    for wt, wd, ws in whistles:
        end = wt + wd
        sp = next((s for s in spells if end + 0.3 < s["t"] <= end + window_s), None)
        if sp is None:
            continue
        dup = next((e for e in existing if abs(sp["t"] - e["t"]) <= dup_s), None)
        if dup is not None:
            d = dup["detail"] = dict(dup.get("detail") or {})
            if d.get("whistle_t") is None:
                d["whistle_t"], d["whistle_strength"] = round(wt, 2), round(ws)
                if d.get("type") in ("free_kick", "penalty") and pitch is not None:
                    d["foul"] = True
                    if dup.get("target") is None and frames is not None:
                        dup["target"] = foul_player(frames, wt, other[sp["team"]])
            continue
        g = next((gt for gt in goal_ts if gt not in used_goal and gt - 30 <= wt <= gt + kickoff_after_goal_s), None)
        detail = {"type": "kick_off" if g is not None else "whistle",
                  "whistle_t": round(wt, 2), "whistle_strength": round(ws)}
        target = None
        if g is not None:
            used_goal.add(g)
        else:
            detail, target = typed(sp, wt, detail)
        ev = {"kind": "set_piece", "t": sp["t"], "team": sp["team"], "player": sp["player"], "t_end": None,
              "target": target, "x": sp.get("x"), "y": sp.get("y"), "x2": None, "y2": None, "success": None,
              "detail": detail}
        out.append(ev)
        existing.append(ev)
    out.sort(key=lambda e: (e["t"], e["kind"] != "possession"))
    return out


def scoreboard_goals(board: Dict, events: List[Dict], left_team: str = "home", window_s: float = 60.0) -> Dict:
    """점수 자막이 바뀐 시각을 골로 바꾼다. 자막은 보통 골 직후 바뀌므로 그 팀의 직전 슈팅(window_s 안)을
    득점 장면으로 보고 시각·선수를 붙인다. 자막 왼쪽 팀은 홈팀으로 본다(경기 설정에서 홈·원정을 맞추면 된다)."""
    if not board.get("found"):
        return {"found": False}
    other = {"home": "away", "away": "home"}
    goals = []
    for c in board.get("changes", []):
        team = left_team if c["side"] == "left" else other[left_team]
        shots = [e for e in events if e["kind"] == "shot" and e["team"] == team and c["t"] - window_s <= e["t"] <= c["t"]]
        shot = max(shots, key=lambda e: e["t"]) if shots else None
        goals.append({"t_board": c["t"], "team": team, "side": c["side"],
                      "t": shot["t"] if shot else c["t"], "player": shot["player"] if shot else None})
    score = {"home": sum(g["team"] == "home" for g in goals), "away": sum(g["team"] == "away" for g in goals)}
    return {"found": True, "box": board.get("box"), "goals": goals, "score": score}


def roster_from_tracklets(tracklets: Sequence[Tracklet], mapping: Dict[str, str],
                          apps: Dict[str, Tuple[Optional[np.ndarray], int]], labels: Dict[str, str],
                          reference_lab: Dict[str, np.ndarray],
                          edges: Optional[Dict[str, Tuple[Optional[np.ndarray], Optional[np.ndarray]]]] = None
                          ) -> Tuple[Dict[str, str], int]:
    """이어 붙인 조각을 팀당 필드 10 + GK 1 칸으로 묶는다. 반환: ({조각 P번호: 칸 대표 P번호}, 정원 초과 수).
    상의 색이 자기 팀보다 상대·심판에 가깝거나 형광 조끼인 조각(심판·조끼 입은 GK)은 칸에 넣지 않는다."""
    gk_keys = {k: labels.get(k) in (C.HOME_GK, C.AWAY_GK) for k in labels}
    refs = {k: v for k, v in reference_lab.items() if k in (C.HOME, C.AWAY, C.REFEREE)}
    bad = {mapping.get(t.tid, t.tid) for t in tracklets
           if not gk_keys.get(t.tid) and C.kit_mismatch(t.color_lab, t.team, refs)}
    # 1) 카메라가 끊긴 순간 한꺼번에 다시 나타난 조각을 직전에 끝난 조각과 한 번에 짝짓는다(속도 예측 + 색)
    links = relink_chains(tracklets, mapping, apps, gk_keys, C.reid_distance, skip=bad, edges=edges)

    def root(h: str) -> str:
        seen = set()
        while h in links and h not in seen:
            seen.add(h)
            h = links[h]
        return h
    linked = {t.tid: root(mapping.get(t.tid, t.tid)) for t in tracklets}
    # 2) 그래도 남은 조각(오래 화면 밖에 있다 돌아온 선수)을 팀당 정원 칸에 담는다
    pieces = [p for p in pieces_from_tracklets(tracklets, linked, apps, gk_keys) if p.head not in bad]
    slot_map, extra = assign_roster_slots(pieces, C.appearance_distance)
    out: Dict[str, str] = {}
    for t in tracklets:
        h = mapping.get(t.tid, t.tid)
        lead = slot_map.get(linked[t.tid], linked[t.tid])
        if h != lead and h not in bad:
            out["P" + h[1:]] = "P" + lead[1:]
    return out, len(extra)


def build_overlay(history: Dict[int, List], tid_to_pid: Dict[int, str], ball_sel: Sequence, ball_px: Sequence,
                  cam_T: Sequence[np.ndarray], frame_w: int, frame_h: int, start_s: float, step: float,
                  n: int, ball_est: Optional[Sequence] = None) -> Dict:
    """영상 위에 그릴 프레임별 선수 박스와 공 위치. 모두 원래 화면 픽셀 좌표(카메라 보정 전)로 되돌린다.
    공 표시: 1 직접 검출 · 0 짧은 공백 이음(이벤트 판정에 씀) · 2 긴 공백 대략 위치(표시만)."""
    tracks: Dict[str, List[List[int]]] = defaultdict(list)
    for tid, pts in history.items():
        pid = tid_to_pid.get(tid)
        if pid is None:
            continue  # 심판·관중 등 버린 트랙
        for p in pts:
            raw = p.payload[3] if isinstance(p.payload, tuple) and len(p.payload) > 3 else p.box
            tracks[pid].append([int(p.frame)] + [int(round(v)) for v in raw])
    for pid in tracks:
        tracks[pid].sort(key=lambda r: r[0])

    def to_raw(k: int, x: float, y: float) -> Tuple[float, float]:
        if not cam_T or k >= len(cam_T):
            return x, y
        q = np.linalg.inv(cam_T[k]) @ np.array([x, y, 1.0])
        return float(q[0] / q[2]), float(q[1] / q[2])

    ball = []
    for k, b in enumerate(ball_est if ball_est is not None else ball_px):
        if b is None:
            continue
        x, y = to_raw(k, b[0], b[1])
        flag = 1 if ball_sel[k] is not None else (0 if ball_px[k] is not None else 2)
        ball.append([k, int(round(x)), int(round(y)), flag])
    return {"v": 1, "w": int(frame_w), "h": int(frame_h), "start_s": float(start_s), "step": float(step),
            "n": int(n), "tracks": dict(tracks), "ball": ball}


# ── 가상 경기 (샘플) ─────────────────────────────────────────────────────────

def analyze_simulated(cfg: AnalysisConfig, duration_s: float = 1200.0, seed: int = 7,
                      players_per_team: Optional[int] = None, progress: Optional[ProgressFn] = None) -> Dict:
    from .simulate import simulate_match
    n = players_per_team or (5 if cfg.field_type == "futsal" else 11)
    sim = simulate_match(duration_s=duration_s, fps=5.0, players_per_team=n,
                         field_type=cfg.field_type, seed=seed)
    labels = {}
    for f in sim.frames[:1]:
        for p in f.players:
            gk = p.tid.endswith("1") and len(p.tid) == 2
            labels[p.tid] = (C.HOME_GK if p.team == "home" else C.AWAY_GK) if gk else p.team
    meta = {"source": "sample", "duration_s": duration_s, "fps_video": sim.fps, "analysis_fps": sim.fps,
            "calibrated": True, "ball_detected_pct": None, "raw_tracks": len(labels), "warnings": []}
    colors = {tid: (cfg.home_color if p_team == "home" else cfg.away_color)
              for tid, p_team in ((p.tid, p.team) for p in sim.frames[0].players)}
    return postprocess(sim.frames, cfg, sim.length, sim.width, meta, labels, {}, colors, progress)


# ── 공통 후처리 ──────────────────────────────────────────────────────────────

def postprocess(frames: Sequence[Frame], cfg: AnalysisConfig, length: float, width: float, meta: Dict,
                labels: Dict[str, str], thumbs: Dict[str, Optional[bytes]], colors: Dict[str, str],
                progress: Optional[ProgressFn] = None) -> Dict:
    progress = progress or (lambda p, m: None)
    calibrated = bool(meta.get("calibrated"))
    ecfg = cfg.event_config(length, width, calibrated=calibrated)
    # 보정 없이 카메라까지 움직이면 화면 가장자리가 경기장 경계가 아니다 → '아웃' 판정을 끈다.
    ecfg.trust_bounds = calibrated or not meta.get("camera_moving", False)
    # 경기장 선 자동 인식 위치(정밀 판정 전이어도)면 골라인을 넘긴 슈팅 판정은 쓴다 — 바모스전 2쿼터 8:39
    ecfg.metric_positions = calibrated or bool(meta.get("metric_positions"))
    if meta.get("auto_calibrated"):
        ecfg.out_margin_m = 2.5
        ecfg.out_restart_within_m = 7.0
    events, spells, dirs = detect_events(frames, ecfg)
    ps = compute_player_stats(frames, events, length, width,
                              metric=calibrated or bool(meta.get("metric_positions")))
    progress(97, "통계 집계 중")
    duration = meta.get("duration_s") or (frames[-1].t if frames else 0)

    players = []
    for tid, st in sorted(ps.items(), key=lambda kv: (kv[1]["team"], kv[0])):
        st = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in st.items()}
        players.append({
            "track_id": tid,
            "team": st.pop("team"),
            "label": labels.get(tid, ""),
            "color_hex": colors.get(tid),
            "thumbnail": base64.b64encode(thumbs[tid]).decode() if thumbs.get(tid) else None,
            "minor": st["seconds_tracked"] < cfg.minor_track_s and st["touches"] == 0,
            "stats": st,
        })

    meta = dict(meta)
    meta.update({"field": {"type": cfg.field_type, "length": length, "width": width},
                 "attack": dirs, "result_version": RESULT_VERSION})
    return {
        "version": RESULT_VERSION,
        "meta": meta,
        "players": players,
        "events": [e.to_dict() for e in events],
        "team_stats": team_stats(ps, spells),
        "pass_network": pass_network(events, ps),
        "possession_timeline": possession_timeline(spells, duration),
    }


# ── 직렬화 (RunPod 응답 크기 절감) ───────────────────────────────────────────

def json_default(o):
    """numpy 숫자·참거짓은 파이썬 값으로 (str로 바꾸면 False가 "False"라는 참 문자열이 된다)."""
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def pack_result(result: Dict) -> str:
    return base64.b64encode(gzip.compress(json.dumps(result, ensure_ascii=False, default=json_default).encode())).decode()


def unpack_result(packed: str) -> Dict:
    return json.loads(gzip.decompress(base64.b64decode(packed)).decode())
