"""이미지 좌표 ↔ 경기장 좌표(m).

보정(calibration) 있음: 사용자가 영상 프레임에서 찍은 4점과 그 점의 실제 경기장 좌표로
호모그래피를 구해 발 위치(박스 하단 중앙)를 미터 단위로 변환한다. 고정 카메라에서 정확하다.

보정 없음: 화면 전체를 경기장으로 간주하는 근사 좌표를 쓴다. 원근 왜곡이 있으므로
- 공 점유 반경은 선수 박스 높이로 정해(멀리 있는 선수는 작게) 원근을 상쇄하고,
- 이동거리는 '사람 키 ≈ 1.75m'를 이용한 국소 축척으로 추정한다.
UI에는 '추정치'로 표시한다.

경기장 좌표계: x = 왼쪽 골라인→오른쪽 골라인 (0~길이), y = 먼 터치라인→가까운 터치라인 (0~폭).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

PERSON_HEIGHT_M = 1.75
POSSESSION_RADIUS_M = 1.6     # 보정 시 공 소유 판정 반경
# 미보정 시 박스 높이 대비 점유 반경. 1080p 광각 영상(선수 키 약 40px)에서 0.55는 발밑 공(≈1m)도 자주 놓쳤다:
# 실측 1분 구간에서 0.55 → 점유 12.5초, 0.8 → 19초(패스 성공 3→4). 그 이상은 잡음만 늘었다.
POSSESSION_RADIUS_H = 0.8

FIELD_DIMS = {"soccer": (105.0, 68.0), "futsal": (40.0, 20.0)}
GOAL_WIDTH = {"soccer": 7.32, "futsal": 3.0}


def field_dims(field_type: str, length: Optional[float] = None, width: Optional[float] = None) -> Tuple[float, float]:
    L, W = FIELD_DIMS.get(field_type, FIELD_DIMS["soccer"])
    return (float(length) if length else L, float(width) if width else W)


def homography(src: Sequence[Sequence[float]], dst: Sequence[Sequence[float]]) -> np.ndarray:
    """4점 이상 대응점으로 3x3 호모그래피(DLT, 최소제곱)."""
    if len(src) < 4 or len(src) != len(dst):
        raise ValueError("호모그래피에는 4개 이상의 대응점이 필요합니다")
    rows = []
    for (x, y), (u, v) in zip(src, dst):
        rows.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        rows.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    a = np.asarray(rows, dtype=np.float64)
    _, s, vt = np.linalg.svd(a)
    h = vt[-1].reshape(3, 3)
    if abs(h[2, 2]) < 1e-12:
        raise ValueError("대응점이 한 직선 위에 있는 등 변환을 만들 수 없습니다")
    h = h / h[2, 2]
    # 점들이 거의 일직선이면 조건수가 폭증한다
    if s[-2] < 1e-9 * s[0]:
        raise ValueError("대응점 배치가 퇴화되어 변환을 만들 수 없습니다")
    return h


def apply_h(h: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    hom = np.hstack([pts, np.ones((len(pts), 1))]) @ h.T
    return hom[:, :2] / hom[:, 2:3]


@dataclass
class PitchMapper:
    """프레임 크기(px)와 보정 정보로 좌표 변환을 담당한다."""
    frame_w: float
    frame_h: float
    length: float
    width: float
    h: Optional[np.ndarray] = None   # 이미지 정규화 좌표(0~1) → 경기장 m

    @property
    def calibrated(self) -> bool:
        return self.h is not None

    @classmethod
    def build(cls, frame_w: float, frame_h: float, field_type: str = "soccer",
              calibration: Optional[Dict] = None) -> "PitchMapper":
        """calibration = {image_points:[[u,v]x4 (0~1)], pitch_points:[[x,y]x4 (m)], length?, width?}"""
        length, width = field_dims(field_type,
                                   (calibration or {}).get("length"),
                                   (calibration or {}).get("width"))
        h = None
        if calibration and calibration.get("image_points") and calibration.get("pitch_points"):
            h = homography(calibration["image_points"], calibration["pitch_points"])
        return cls(frame_w=frame_w, frame_h=frame_h, length=length, width=width, h=h)

    def to_pitch(self, px: float, py: float) -> Tuple[float, float]:
        u, v = px / self.frame_w, py / self.frame_h
        if self.h is not None:
            x, y = apply_h(self.h, np.array([[u, v]]))[0]
            return float(x), float(y)
        return u * self.length, v * self.width

    def inside(self, x: float, y: float, margin: float = 3.0) -> bool:
        return -margin <= x <= self.length + margin and -margin <= y <= self.width + margin

    def player_geometry(self, box: Sequence[float]) -> Tuple[float, float, float, float]:
        """박스 → (x, y, 점유 반경 r, 국소 축척 s[m/좌표단위]) — 모두 경기장 좌표 단위."""
        x1, y1, x2, y2 = box
        fx, fy = (x1 + x2) / 2, y2
        x, y = self.to_pitch(fx, fy)
        if self.h is not None:
            return x, y, POSSESSION_RADIUS_M, 1.0
        h_px = max(4.0, y2 - y1)
        units_per_px = self.length / self.frame_w
        r = POSSESSION_RADIUS_H * h_px * units_per_px
        m_per_px = PERSON_HEIGHT_M / h_px
        s = m_per_px / units_per_px
        return x, y, r, s


def calibration_from_preset(preset: str, image_points: List[List[float]], length: float, width: float) -> Dict:
    """UI 프리셋 → pitch_points.

    image_points 순서: 화면상 왼쪽 위, 오른쪽 위, 오른쪽 아래, 왼쪽 아래.
    full       : 경기장 네 모서리
    left_half  : 왼쪽 절반(왼쪽 골라인 ~ 하프라인)
    right_half : 오른쪽 절반(하프라인 ~ 오른쪽 골라인)
    """
    half = length / 2
    presets = {
        "full": [[0, 0], [length, 0], [length, width], [0, width]],
        "left_half": [[0, 0], [half, 0], [half, width], [0, width]],
        "right_half": [[half, 0], [length, 0], [length, width], [half, width]],
    }
    if preset not in presets:
        raise ValueError(f"알 수 없는 보정 프리셋: {preset}")
    if len(image_points) != 4:
        raise ValueError("보정에는 4개의 점이 필요합니다")
    return {"preset": preset, "image_points": image_points, "pitch_points": presets[preset],
            "length": length, "width": width}
