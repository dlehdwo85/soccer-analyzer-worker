"""경기장 선 자동 인식 → 장면마다 화면 좌표를 경기장 좌표(m)로 바꾸는 변환.

사용자가 경기장 네 모서리를 찍지 않아도, 카메라가 움직여도 위치를 미터로 알기 위한 것이다. 경기장 키포인트 모델
(YOLO pose, 32점: 코너·페널티 박스·골 에어리어·하프라인·센터서클)이 장면마다 보이는 점을 찾고, 그 점들과 경기장 도면의
대응으로 호모그래피를 구한다. 점이 모자란 장면은 가까운 장면의 변환을 카메라 움직임(camera.CameraMotion)으로 옮겨 쓴다.

이 위치가 있어야 가능한 판정: 골키퍼(자기 골문 앞에 머무는 선수), 슈팅 위치와 xG, 코너킥·드로잉·골킥·프리킥 구분,
센터서클 킥오프(영상만으로 골 확인), 이동 거리와 속도.

경기장 좌표계는 pitch.py와 같다: x = 왼쪽 골라인→오른쪽 골라인, y = 먼 터치라인→가까운 터치라인.
"""

from __future__ import annotations

import bisect
import logging
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

N_KP = 32
BOX_W, BOX_L = 40.32, 16.5        # 페널티 박스
GA_W, GA_L = 18.32, 5.5           # 골 에어리어
CIRCLE_R, SPOT = 9.15, 11.0
ARC_HALF = float(np.sqrt(CIRCLE_R ** 2 - (BOX_L - SPOT) ** 2))   # 페널티 아크가 박스 선과 만나는 점 (7.31m)

Kp = Tuple[int, float, float, float]   # (점 번호 0~31, x, y, 신뢰도) — 원본 픽셀


def template(length: float, width: float, arc_half: float = ARC_HALF) -> np.ndarray:
    """키포인트 32개의 경기장 좌표(m). 번호 순서는 공개 모델(roboflow sports)의 학습 규칙을 따른다.
    박스와 서클 크기는 규격 고정, 경기장 길이와 폭만 바뀐다."""
    L, W, c = float(length), float(width), float(width) / 2
    b, g, a = BOX_W / 2, GA_W / 2, arc_half
    return np.array([
        (0, 0), (0, c - b), (0, c - g), (0, c + g), (0, c + b), (0, W),
        (GA_L, c - g), (GA_L, c + g), (SPOT, c),
        (BOX_L, c - b), (BOX_L, c - a), (BOX_L, c + a), (BOX_L, c + b),
        (L / 2, 0), (L / 2, c - CIRCLE_R), (L / 2, c + CIRCLE_R), (L / 2, W),
        (L - BOX_L, c - b), (L - BOX_L, c - a), (L - BOX_L, c + a), (L - BOX_L, c + b),
        (L - SPOT, c), (L - GA_L, c - g), (L - GA_L, c + g),
        (L, 0), (L, c - b), (L, c - g), (L, c + g), (L, c + b), (L, W),
        (L / 2 - CIRCLE_R, c), (L / 2 + CIRCLE_R, c),
    ], dtype=np.float64)


def _apply(h: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    q = np.hstack([pts, np.ones((len(pts), 1))]) @ h.T
    return q[:, :2] / q[:, 2:3]


class Fit:
    """한 장면의 변환. h: 화면 픽셀 → 경기장 m, h_inv: 경기장 m → 화면 픽셀."""
    __slots__ = ("h", "h_inv", "n", "err_px", "ids")

    def __init__(self, h, h_inv, n, err_px, ids):
        self.h, self.h_inv, self.n, self.err_px, self.ids = h, h_inv, n, err_px, ids


def fit_frame(kps: Sequence[Kp], tmpl: np.ndarray, frame_w: float, min_conf: float = 0.5,
              min_inliers: int = 5, thr_frac: float = 0.008) -> Optional[Fit]:
    """믿을 만한 키포인트로 호모그래피를 구한다. 조건(점 수, 퍼짐, 오차)을 못 채우면 None.
    점 4개는 여유가 없어 틀린 점 하나로 변환이 통째로 틀어지므로 5개 이상 일치할 때만 받는다."""
    import cv2
    good = [(i, x, y) for i, x, y, c in kps if c >= min_conf and 0 <= i < len(tmpl)]
    if len(good) < min_inliers:
        return None
    ids = np.array([g[0] for g in good])
    img = np.array([(g[1], g[2]) for g in good], dtype=np.float64)
    pit = tmpl[ids]
    h_inv, mask = cv2.findHomography(pit, img, cv2.RANSAC, thr_frac * frame_w)   # 경기장 → 화면 (오차를 픽셀로 잰다)
    if h_inv is None or mask is None:
        return None
    inl = mask.reshape(-1).astype(bool)
    if inl.sum() < min_inliers:
        return None
    # 일치한 점이 한 줄로 늘어서 있으면(하프라인 위 점들만 보일 때) 변환이 정해지지 않는다
    p = pit[inl] - pit[inl].mean(axis=0)
    if np.linalg.svd(p, compute_uv=False)[-1] < 3.0:
        return None
    try:
        h = np.linalg.inv(h_inv)
    except np.linalg.LinAlgError:
        return None
    err = float(np.median(np.linalg.norm(_apply(h_inv, pit[inl]) - img[inl], axis=1)))
    return Fit(h / h[2, 2], h_inv / h_inv[2, 2], int(inl.sum()), err, [int(i) for i in ids[inl]])


def estimate_dims(frames_kps: Sequence[Sequence[Kp]], frame_w: float, default: Tuple[float, float],
                  lengths: Sequence[float] = (90.0, 95.0, 100.0, 105.0, 110.0),
                  widths: Sequence[float] = (52.0, 56.0, 60.0, 64.0, 68.0, 72.0),
                  margin: float = 0.5) -> Tuple[float, float, float]:
    """경기장 길이와 폭 추정. 박스와 서클은 규격이 정해져 있으므로, 길이와 폭을 바꿔 가며 모든 장면의 점이 가장 잘
    맞는 크기를 고른다 (동네 구장은 105×68보다 작은 곳이 많다). 기본 크기보다 확실히(margin) 나을 때만 바꾼다 —
    한쪽 진영만 보이는 장면이 대부분이면 길이·폭은 약하게만 정해진다. 반환: (길이, 폭, 일치 점 평균 개수)."""
    sample = [k for k in frames_kps if sum(1 for q in k if q[3] >= 0.5) >= 7]
    if len(sample) < 8:
        return default[0], default[1], 0.0
    sample = sample[:: max(1, len(sample) // 40)]

    def score(L: float, W: float) -> Tuple[float, float]:
        t = template(L, W)
        fits = [fit_frame(k, t, frame_w) for k in sample]
        n = float(np.mean([f.n if f else 0 for f in fits]))
        e = float(np.mean([f.err_px if f else 0.01 * frame_w for f in fits]))
        return n - e / (0.004 * frame_w), n

    base, base_n = score(*default)
    best = (base + margin, float(default[0]), float(default[1]), base_n)
    for L in lengths:
        for W in widths:
            if (L, W) == tuple(default):
                continue
            s, n = score(L, W)
            if s > best[0]:
                best = (s, float(L), float(W), n)
    return best[1], best[2], best[3]


class PitchKeypointModel:
    """경기장 키포인트 YOLO pose 모델. 경로는 YOLO_PITCH_MODEL."""

    def __init__(self, path: str, imgsz: int = 640, device: Optional[str] = None):
        from ultralytics import YOLO
        self.model = YOLO(path, task="pose")
        self.imgsz, self.device = imgsz, device

    @classmethod
    def from_env(cls) -> Optional["PitchKeypointModel"]:
        path = os.environ.get("YOLO_PITCH_MODEL")
        if not path or not os.path.exists(path):
            return None
        try:
            return cls(path, int(os.environ.get("YOLO_PITCH_IMGSZ", "640")), os.environ.get("YOLO_DEVICE") or None)
        except Exception:  # 부가 기능 — 모델을 못 읽으면 위치 추정은 예전 방식으로
            logger.exception("경기장 키포인트 모델을 읽지 못했습니다")
            return None

    def detect(self, frame_rgb: np.ndarray) -> List[Kp]:
        frame = np.ascontiguousarray(frame_rgb[..., ::-1])       # ultralytics는 BGR
        r = self.model.predict(frame, imgsz=self.imgsz, device=self.device, verbose=False)[0]
        if r.keypoints is None or len(r.keypoints) == 0 or r.keypoints.conf is None:
            return []
        j = int(r.boxes.conf.argmax()) if r.boxes is not None and len(r.boxes) else 0
        xy = r.keypoints.xy[j].cpu().numpy()
        cf = r.keypoints.conf[j].cpu().numpy()
        return [(i, float(x), float(y), float(c)) for i, ((x, y), c) in enumerate(zip(xy, cf)) if c >= 0.3]


class PitchRegistrar:
    """장면별 키포인트를 모아 두었다가 finish()에서 경기장 크기를 정하고 장면마다 변환을 만든다."""

    def __init__(self, frame_w: int, frame_h: int, length: float, width: float, model=None, every: int = 1):
        self.frame_w, self.frame_h = frame_w, frame_h
        self.length, self.width = length, width
        self.model = model
        self.every = max(1, every)
        self.kps: Dict[int, List[Kp]] = {}
        self.fits: Dict[int, Fit] = {}
        self.anchors: List[int] = []
        self.cam_T: Sequence[np.ndarray] = []
        self.n_frames = 0
        self.max_bridge = 0
        self.dropped = 0
        self._hs: Dict[int, List[Tuple[float, np.ndarray]]] = {}

    def add(self, k: int, frame_rgb: np.ndarray) -> None:
        self.n_frames = max(self.n_frames, k + 1)
        if self.model is not None and k % self.every == 0:
            self.kps[k] = self.model.detect(frame_rgb)

    def add_keypoints(self, k: int, kps: List[Kp]) -> None:
        self.n_frames = max(self.n_frames, k + 1)
        self.kps[k] = list(kps)

    def finish(self, cam_T: Optional[Sequence[np.ndarray]], times: Sequence[float], fixed_dims: bool = False,
               max_bridge_s: float = 4.0) -> Dict:
        """경기장 크기 추정 → 장면별 변환 → 튀는 변환 제거. 반환: 요약(meta용)."""
        self.cam_T = cam_T if cam_T is not None else []
        self.n_frames = len(times)
        ks = sorted(self.kps)
        n_est = 0.0
        if not fixed_dims and ks:
            self.length, self.width, n_est = estimate_dims([self.kps[k] for k in ks], self.frame_w,
                                                           (self.length, self.width))
        tmpl = template(self.length, self.width)
        for k in ks:
            f = fit_frame(self.kps[k], tmpl, self.frame_w)
            if f is not None:
                self.fits[k] = f
        self._drop_jumps()
        self.anchors = sorted(self.fits)
        step = float(np.median(np.diff(times))) if len(times) > 1 else 0.2
        self.max_bridge = int(round(max_bridge_s / max(step, 1e-6)))
        covered = sum(1 for k in range(self.n_frames) if self._near(k) is not None) / max(1, self.n_frames)
        return {"length": self.length, "width": self.width,
                "direct_pct": round(len(self.anchors) / max(1, len(ks)) * 100, 1),
                "covered_pct": round(covered * 100, 1), "anchors": len(self.anchors), "dropped_jumps": self.dropped,
                "median_err_px": round(float(np.median([f.err_px for f in self.fits.values()])), 2) if self.fits else None,
                "median_points": float(np.median([f.n for f in self.fits.values()])) if self.fits else 0.0,
                "dims_estimated": bool(n_est)}

    def _probe(self) -> np.ndarray:
        return np.array([[self.frame_w / 2, self.frame_h * 0.6]])

    def _drop_jumps(self, max_m: float = 6.0) -> None:
        """앞뒤 장면과 비교해 화면 가운데의 경기장 위치가 혼자 크게 다른 변환(점 번호를 잘못 붙인 장면)을 버린다.
        카메라가 실제로 돌아간 만큼은 빼고 본다."""
        ks = sorted(self.fits)
        bad = []
        for a, k, b in zip(ks, ks[1:], ks[2:]):
            if b - a > 12 * self.every:
                continue
            pa, pb = self._moved(a, k), self._moved(b, k)
            if pa is None or pb is None:
                continue
            pk = _apply(self.fits[k].h, self._probe())[0]
            if np.linalg.norm(pa - pb) < max_m and min(np.linalg.norm(pk - pa), np.linalg.norm(pk - pb)) > max_m:
                bad.append(k)
        for k in bad:
            del self.fits[k]
        self.dropped = len(bad)

    def _h_from(self, j: int, k: int) -> Optional[np.ndarray]:
        """장면 j의 변환을 장면 k에서 쓰도록 옮긴다: k 화면 → (카메라 움직임) → j 화면 → 경기장."""
        f = self.fits.get(j)
        if f is None:
            return None
        if j == k:
            return f.h
        if not len(self.cam_T) or max(j, k) >= len(self.cam_T):
            return None
        return f.h @ np.linalg.inv(self.cam_T[j]) @ self.cam_T[k]

    def _moved(self, j: int, k: int) -> Optional[np.ndarray]:
        h = self._h_from(j, k)
        return None if h is None else _apply(h, self._probe())[0]

    def _near(self, k: int) -> Optional[Tuple[Optional[int], Optional[int]]]:
        i = bisect.bisect_left(self.anchors, k)
        nxt = self.anchors[i] if i < len(self.anchors) else None
        if nxt == k:
            return (k, k)
        prv = self.anchors[i - 1] if i > 0 else None
        if prv is not None and k - prv > self.max_bridge:
            prv = None
        if nxt is not None and nxt - k > self.max_bridge:
            nxt = None
        return None if prv is None and nxt is None else (prv, nxt)

    def has(self, k: int) -> bool:
        return self._near(k) is not None

    def _frame_hs(self, k: int) -> List[Tuple[float, np.ndarray]]:
        hs = self._hs.get(k)
        if hs is None:
            hs = []
            near = self._near(k)
            for j in (set(near) - {None}) if near is not None else ():
                h = self._h_from(j, k)
                if h is not None:
                    hs.append((1.0 / (abs(j - k) + 0.5), h))
            self._hs[k] = hs
        return hs

    def to_pitch(self, k: int, x: float, y: float) -> Optional[Tuple[float, float]]:
        """장면 k의 원본 픽셀 (x, y) → 경기장 m. 가까운 변환이 없으면 None.
        앞뒤 변환이 다 있으면 시간 거리로 섞어, 변환이 바뀌는 순간 위치가 튀지 않게 한다."""
        ox = oy = wsum = 0.0
        for w, h in self._frame_hs(k):
            d = h[2, 0] * x + h[2, 1] * y + h[2, 2]
            if abs(d) < 1e-9:
                continue
            ox += w * (h[0, 0] * x + h[0, 1] * y + h[0, 2]) / d
            oy += w * (h[1, 0] * x + h[1, 1] * y + h[1, 2]) / d
            wsum += w
        if wsum <= 0:
            return None
        return float(ox / wsum), float(oy / wsum)

    def to_image(self, k: int, x: float, y: float) -> Optional[Tuple[float, float]]:
        """경기장 m → 장면 k의 원본 픽셀 (가장 가까운 변환 하나로)."""
        near = self._near(k)
        if near is None:
            return None
        j = min(set(near) - {None}, key=lambda a: abs(a - k))
        h = self._h_from(j, k)
        if h is None:
            return None
        q = _apply(np.linalg.inv(h), np.array([[x, y]]))[0]
        return float(q[0]), float(q[1])
