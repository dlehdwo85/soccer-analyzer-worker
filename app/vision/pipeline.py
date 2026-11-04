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
from .ball import in_play_candidates, interpolate_gaps, select_ball_positions
from .camera import CameraMotion, stationary
from .events import EventConfig, Frame, PlayerObs, detect_events, fit_to_extents
from .pitch import GOAL_WIDTH, PitchMapper
from .scoreboard import ScoreboardReader
from .audio import detect_whistles, load_audio
from .stats import compute_player_stats, pass_network, possession_timeline, team_stats
from .tracker import Detection, IoUTracker, Tracklet, stitch_tracklets, suggest_same_player

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
    "max": {"analysis_fps": 15.0, "model": "yolov8x.pt", "imgsz": 1280, "ball_tiles": 2, "ball_full_every": 1},
}


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
        self._focus: Optional[Tuple[float, float]] = None
        self._focus_age = 0
        self._frame_no = 0
        if self.ball_model is not None:
            names = self.ball_model.names
            self.ball_classes = [k for k, v in names.items() if "ball" in str(v).lower()] or list(names)
            self.ball_min_conf_new = self.BALL_MODEL_MIN_CONF_NEW

    @classmethod
    def for_quality(cls, quality: str = "standard") -> "YoloDetector":
        """품질 단계에 맞춘 검출기. 표준은 환경변수(YOLO_MODEL 등) 그대로, 고정밀은 단계 설정이 우선한다."""
        p = QUALITY_PRESETS.get(quality or "standard", {})
        det = cls(model=p.get("model") or os.environ.get("YOLO_MODEL"),
                  imgsz=p.get("imgsz") or (int(os.environ["YOLO_IMGSZ"]) if os.environ.get("YOLO_IMGSZ") else None))
        if "ball_tiles" in p and not os.environ.get("YOLO_BALL_TILES"):
            det.ball_tiles = p["ball_tiles"]
        if "ball_full_every" in p:
            det.BALL_FULL_EVERY = p["ball_full_every"]
        return det

    def _ball_predict(self, img, x0: int = 0, y0: int = 0) -> List[Tuple[float, float, float]]:
        rb = self.ball_model.predict(img, imgsz=self.ball_imgsz, classes=self.ball_classes, conf=self.BALL_CONF,
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
        res = self.model.predict(frame_rgb, imgsz=self.imgsz, classes=classes, conf=self.BALL_CONF,
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
                # 고정밀(BALL_FULL_EVERY=1)은 추적 중에도 매 장면 4분할로 본다 (GPU 전제)
                if self.ball_tiles >= 2 and (not tracking or self.BALL_FULL_EVERY == 1):
                    n, ov = self.ball_tiles, 0.12
                    tw, th = int(w / n * (1 + ov)), int(h / n * (1 + ov))
                    for iy in range(n):
                        for ix in range(n):
                            x0 = 0 if ix == 0 else min(int(ix * w / n - (tw - w / n) / 2), w - tw)
                            y0 = 0 if iy == 0 else min(int(iy * h / n - (th - h / n) / 2), h - th)
                            cands += self._ball_predict(frame_rgb[y0:y0 + th, x0:x0 + tw], x0, y0)
                else:
                    cands += self._ball_predict(frame_rgb)
            if roi is not None:
                x0, y0, x1, y1 = roi
                cands += self._ball_predict(frame_rgb[y0:y1, x0:x1], x0, y0)
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
    best_area: float = 0.0
    thumb: Optional[bytes] = None


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
    removed_outside = 0

    def run_tracker(k: int, frame_rgb: Optional[np.ndarray], dets: List):
        tracked = []
        for box, conf, sample, orig in dets:
            label, lconf = clf.classify(sample)
            tracked.append(Detection(box, conf, (label, lconf, sample.median_lab if sample else None, orig,
                                                 sample.appearance if sample else None)))
        for tid, det in tracker.update(k, tracked):
            ti = info[tid]
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
        balls = in_play_candidates(balls, persons, frame_h)   # 벤치 옆 여분의 공·조명 같은 경기 밖 후보 제거
        try:
            board.add(t, frame_rgb)
        except Exception:  # 부가 기능 — 실패해도 분석은 계속
            logger.exception("점수 자막 읽기 실패")
            board.failed = True
        if cam is not None:
            cam_T.append(cam.update(frame_rgb, [b for b, _ in persons]).copy())
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
        labels[key] = lbl
        g = []
        for p in pts:
            x, y, r, s = mapper.player_geometry(p.box)
            g.append((p.frame, x, y, r, s))
        geo[key] = g
        lab = np.median(np.array(info[tid].labs), axis=0) if info[tid].labs else None
        tracklets.append(Tracklet(key, C.TEAM_OF_LABEL[lbl], [(times[f], x, y) for f, x, y, _, _ in g], lab))

    mapping = stitch_tracklets(tracklets)
    # 카메라가 돌아간 사이 화면 밖에 있다 돌아온 선수는 자동으로 잇기엔 불확실해 '같은 선수 후보'만 남긴다
    merge_hints: Dict[str, List[str]] = {}
    try:
        def at_edge(pt) -> bool:
            raw = pt.payload[3] if isinstance(pt.payload, tuple) and len(pt.payload) > 3 else pt.box
            return raw[0] < 0.03 * frame_w or raw[2] > 0.97 * frame_w
        keys = [t.tid for t in tracklets]
        apps = {key: (np.nanmedian(np.array(info[int(key[1:])].apps), axis=0) if info[int(key[1:])].apps else None,
                      len(info[int(key[1:])].apps)) for key in keys}
        exits = {key: at_edge(tracker.history[int(key[1:])][-1]) for key in keys}
        enters = {key: at_edge(tracker.history[int(key[1:])][0]) for key in keys}
        hints = suggest_same_player(tracklets, mapping, apps, exits, enters, C.appearance_distance)
        merge_hints = {"P" + h[1:]: ["P" + o[1:] for o in lst] for h, lst in hints.items()}
    except Exception:  # 부가 정보 — 실패해도 분석 결과는 살린다
        logger.exception("같은 선수 후보 계산 실패")
    players_obs: Dict[int, List[PlayerObs]] = defaultdict(list)
    merged_labels: Dict[str, str] = {}
    thumbs: Dict[str, Optional[bytes]] = {}
    thumb_area: Dict[str, float] = {}
    color_by_player: Dict[str, List[np.ndarray]] = defaultdict(list)
    for key, g in geo.items():
        pid = "P" + mapping.get(key, key)[1:]
        merged_labels.setdefault(pid, labels[key])
        team = C.TEAM_OF_LABEL[labels[key]]
        for f, x, y, r, s in g:
            players_obs[f].append(PlayerObs(pid, team, x, y, r, s))
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
    ball_sel = select_ball_positions(
        ball_cands, times, frame_w, min_conf_new=getattr(detector, "ball_min_conf_new", 0.3),
        memory_s=1.5 if dedicated_ball else 1.0)
    # 카메라가 공을 따라 돌아가는 사이 놓친 공은 화면 이동을 따라 잇는다 (화면 중심의 안정화 좌표)
    cam_centers = [tuple((T @ np.array([frame_w / 2, frame_h / 2, 1.0]))[:2]) for T in cam_T] if cam_T else None
    ball_px = interpolate_gaps(ball_sel, times,
                               max_gap_s=2.5 if dedicated_ball else 0.8,
                               max_speed_px_s=0.6 * frame_w if dedicated_ball else None,
                               cam_centers=cam_centers if dedicated_ball else None,
                               cam_min_shift_px=0.05 * frame_w)
    frames = []
    for i, t in enumerate(times):
        b = ball_px[i]
        bp = mapper.to_pitch(*b) if b is not None else None
        frames.append(Frame(t, players_obs.get(i, []), bp))

    if cam is not None and not mapper.calibrated:
        frames = fit_to_extents(frames, mapper.length, mapper.width)
    ball_seen = sum(1 for b in ball_px if b is not None) / len(times)
    if ball_seen < 0.3:
        warnings.append(f"공이 영상의 {ball_seen * 100:.0f}% 구간에서만 검출되어 볼 이벤트가 적게 잡혔을 수 있습니다.")
    camera_moving = cam is not None and cam.travel() > 0.3 * frame_w
    if camera_moving:
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
        "calibrated": mapper.calibrated,
        "ball_detected_pct": round(ball_seen * 100, 1),
        "raw_tracks": len(tracker.history),
        "removed_outside_pitch": removed_outside,
        "removed_static": removed_static,
        "camera_moving": camera_moving,
        "snapped_colors": dict(clf.snapped_hex),
        "merge_hints": merge_hints,
        "warnings": warnings,
    }
    colors = {pid: C.rgb_to_hex(C.lab_to_rgb(np.median(np.array(v), axis=0))[0])
              for pid, v in color_by_player.items() if v}
    result = postprocess(frames, cfg, mapper.length, mapper.width, meta, merged_labels, thumbs, colors, progress)
    try:
        result["meta"]["scoreboard"] = scoreboard_goals(board.finish(), result["events"])
    except Exception:
        logger.exception("점수 자막 골 판정 실패")
    try:
        whistles = detect_whistles(load_audio(source, cfg.start_seconds, cfg.max_seconds))
        result["meta"]["whistles"] = [[round(t, 2), round(d, 2), round(s)] for t, d, s in whistles]
        result["events"] = whistle_restarts(result["events"], whistles, result["meta"].get("scoreboard") or {})
    except Exception:  # 소리가 없거나 읽지 못해도 분석은 그대로
        logger.exception("호루라기 판정 실패")
    try:
        result["overlay"] = build_overlay(
            tracker.history, {int(k[1:]): "P" + mapping.get(k, k)[1:] for k in geo},
            ball_sel, ball_px, cam_T, frame_w, frame_h, cfg.start_seconds, stride / fps_video, len(times))
    except Exception:  # 검수용 부가 데이터 — 실패해도 분석 결과는 살린다
        logger.exception("영상 검수 오버레이 생성 실패")
    return result


def whistle_restarts(events: List[Dict], whistles: Sequence[Tuple[float, float, float]], board: Dict,
                     window_s: float = 25.0, dup_s: float = 3.0, kickoff_after_goal_s: float = 120.0) -> List[Dict]:
    """호루라기 뒤 처음 공을 잡은 점유를 '재개'(파울 뒤 프리킥 등) 세트피스로 더한다.
    골(점수 자막) 뒤 kickoff_after_goal_s 안의 첫 호루라기 재개는 킥오프로 본다. 이미 있는 세트피스와 겹치면 더하지 않는다."""
    spells = sorted((e for e in events if e["kind"] == "possession"), key=lambda e: e["t"])
    existing = [e["t"] for e in events if e["kind"] == "set_piece"]
    goal_ts = sorted(g["t_board"] for g in (board.get("goals") or []))
    used_goal = set()
    out = list(events)
    for wt, wd, ws in whistles:
        end = wt + wd
        sp = next((s for s in spells if end + 0.3 < s["t"] <= end + window_s), None)
        if sp is None or any(abs(sp["t"] - t) <= dup_s for t in existing):
            continue
        g = next((gt for gt in goal_ts if gt not in used_goal and gt - 30 <= wt <= gt + kickoff_after_goal_s), None)
        kind = "kick_off" if g is not None else "whistle"
        if g is not None:
            used_goal.add(g)
        out.append({"kind": "set_piece", "t": sp["t"], "team": sp["team"], "player": sp["player"], "t_end": None,
                    "target": None, "x": sp.get("x"), "y": sp.get("y"), "x2": None, "y2": None, "success": None,
                    "detail": {"type": kind, "whistle_t": round(wt, 2), "whistle_strength": round(ws)}})
        existing.append(sp["t"])
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


def build_overlay(history: Dict[int, List], tid_to_pid: Dict[int, str], ball_sel: Sequence, ball_px: Sequence,
                  cam_T: Sequence[np.ndarray], frame_w: int, frame_h: int, start_s: float, step: float,
                  n: int) -> Dict:
    """영상 위에 그릴 프레임별 선수 박스와 공 위치. 모두 원래 화면 픽셀 좌표(카메라 보정 전)로 되돌린다."""
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
    for k, b in enumerate(ball_px):
        if b is None:
            continue
        x, y = to_raw(k, b[0], b[1])
        ball.append([k, int(round(x)), int(round(y)), 1 if ball_sel[k] is not None else 0])
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
    events, spells, dirs = detect_events(frames, ecfg)
    ps = compute_player_stats(frames, events, length, width)
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
