"""카메라 움직임(팬·틸트·줌) 보정.

조기축구 영상은 대개 사람이 카메라를 들고 공을 따라 돌린다. 화면 좌표를 그대로 쓰면
- 프레임 사이에 모든 박스가 같이 밀려 추적(IoU 매칭)이 끊기고
- 카메라가 도는 속도가 공·선수 속도로 잘못 더해지고
- 가만히 서 있는 관중·벤치를 움직이는 선수와 구분할 수 없다.

배경(나무·펜스·관중석) 특징점의 광류로 프레임 간 닮음 변환(이동+회전+확대)을 추정하고,
첫 프레임 기준 좌표로 누적해 모든 검출을 '안정화된 좌표'로 옮긴다.
선수 박스 안의 특징점은 선수 자신의 움직임이라 제외하고, 화면에 고정된 자막·점수판은
카메라가 움직일 때 '움직이지 않는 점'으로 걸러낸다.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

Box = Tuple[float, float, float, float]


class CameraMotion:
    def __init__(self, frame_w: int, frame_h: int, work_w: int = 480):
        self.f = frame_w / float(work_w) if frame_w > work_w else 1.0
        self.work = (int(round(frame_w / self.f)), int(round(frame_h / self.f)))
        self.T = np.eye(3)            # 현재 프레임 좌표 → 기준(첫 프레임) 좌표
        self.prev_gray: Optional[np.ndarray] = None
        self.prev_boxes: List[Box] = []
        self.offsets: List[Tuple[float, float, float]] = []   # 프레임별 (tx, ty, scale) — 진단용
        self.failed = 0

    # ── 추정 ──────────────────────────────────────────────────────────────
    def _mask(self, boxes: Sequence[Box]) -> np.ndarray:
        w, h = self.work
        m = np.full((h, w), 255, np.uint8)
        for x1, y1, x2, y2 in boxes:
            pad = 0.15 * (y2 - y1)
            a, b = int(max(0, (x1 - pad) / self.f)), int(max(0, (y1 - pad) / self.f))
            c, d = int(min(w, (x2 + pad) / self.f)), int(min(h, (y2 + pad) / self.f))
            m[b:d, a:c] = 0
        return m

    def update(self, frame_rgb: np.ndarray, boxes: Sequence[Box]) -> np.ndarray:
        """현재 프레임과 그 프레임의 사람 박스로 변환을 갱신하고 T(현재→기준)를 돌려준다."""
        import cv2

        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        if self.f != 1.0:
            gray = cv2.resize(gray, self.work, interpolation=cv2.INTER_AREA)
        if self.prev_gray is not None:
            A = self._estimate(cv2, self.prev_gray, gray, self._mask(self.prev_boxes))
            if A is None:
                self.failed += 1
            else:
                S = np.diag([self.f, self.f, 1.0])
                A3 = S @ np.vstack([A, [0, 0, 1]]) @ np.linalg.inv(S)
                self.T = self.T @ np.linalg.inv(A3)
        self.prev_gray = gray
        self.prev_boxes = list(boxes)
        sc = float(np.sqrt(abs(np.linalg.det(self.T[:2, :2]))))
        self.offsets.append((float(self.T[0, 2]), float(self.T[1, 2]), sc))
        return self.T

    @staticmethod
    def _estimate(cv2, prev: np.ndarray, cur: np.ndarray, mask: np.ndarray) -> Optional[np.ndarray]:
        pts = cv2.goodFeaturesToTrack(prev, maxCorners=400, qualityLevel=0.01, minDistance=7, mask=mask)
        if pts is None or len(pts) < 12:
            return None
        nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev, cur, pts, None, winSize=(21, 21), maxLevel=3)
        ok = st.reshape(-1) == 1
        p0, p1 = pts.reshape(-1, 2)[ok], nxt.reshape(-1, 2)[ok]
        if len(p0) < 12:
            return None
        mag = np.linalg.norm(p1 - p0, axis=1)
        med = float(np.median(mag))
        if med > 1.0:
            moving = mag > 0.25 * med   # 화면에 고정된 자막·점수판 제외
            if moving.sum() >= 12:
                p0, p1 = p0[moving], p1[moving]
        A, inl = cv2.estimateAffinePartial2D(p0, p1, method=cv2.RANSAC, ransacReprojThreshold=1.0,
                                             maxIters=2000, confidence=0.995)
        if A is None or inl is None or inl.sum() < max(10, 0.3 * len(p0)):
            return None
        return A

    # ── 적용 ──────────────────────────────────────────────────────────────
    def point(self, x: float, y: float) -> Tuple[float, float]:
        v = self.T @ np.array([x, y, 1.0])
        return float(v[0]), float(v[1])

    def box(self, b: Box) -> Box:
        x1, y1, x2, y2 = b
        pts = np.array([[x1, y1, 1], [x2, y1, 1], [x2, y2, 1], [x1, y2, 1]], dtype=float) @ self.T.T
        return float(pts[:, 0].min()), float(pts[:, 1].min()), float(pts[:, 0].max()), float(pts[:, 1].max())

    def travel(self) -> float:
        """기준 대비 카메라가 움직인 최대 폭(원본 픽셀)."""
        if not self.offsets:
            return 0.0
        o = np.array(self.offsets)
        return float(max(np.ptp(o[:, 0]), np.ptp(o[:, 1])))


def stationary(points: Sequence[Tuple[float, float, float]], heights: Sequence[float],
               min_duration_s: float = 15.0, max_span_h: float = 0.8) -> bool:
    """안정화 좌표에서 오래 거의 움직이지 않은 트랙(관중·벤치)인가. points: [(t, x, y)]."""
    if len(points) < 3:
        return False
    ts = [p[0] for p in points]
    if ts[-1] - ts[0] < min_duration_s:
        return False
    xy = np.array([(p[1], p[2]) for p in points])
    med = np.median(xy, axis=0)
    span = float(np.percentile(np.linalg.norm(xy - med, axis=1), 90))
    h = float(np.median(heights)) if len(heights) else 1.0
    return span < max_span_h * h
