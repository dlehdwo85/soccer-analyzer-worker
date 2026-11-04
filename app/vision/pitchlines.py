"""경기장 흰 선 인식 → 카메라 쪽 터치라인 바깥(벤치·몸 푸는 사람)을 선수에서 뺀다.

경기장 지정(보정) 없이 카메라가 움직이는 영상은 어디까지가 경기장인지 몰라, 라인 밖에서 몸을 푸는
같은 팀 유니폼 사람까지 선수로 셌다. 잔디 위 흰 픽셀에서 허프 직선을 찾아, 화면 아래쪽의 길고 거의
가로인 선을 카메라 쪽 터치라인으로 본다. 그 선보다 발이 확실히 아래(카메라 쪽)에 있는 사람은 경기장 밖이다.

잘못 고른 선(펜스·광고판)으로 진짜 선수를 지우지 않게, 사람 대부분(MIN_ON_PITCH)이 선 위쪽(경기장 쪽)에
있을 때만 그 선을 믿는다. 선이 안 보이는 장면은 직전 선을 HOLD_S초까지 쓰고, 그 뒤로는 거르지 않는다.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

Line = Tuple[float, float, float, float]   # x1, y1, x2, y2 (원본 픽셀)
WORK_W = 960
MAX_ANGLE = 35.0       # 터치라인 기울기 상한(도) — 원근으로 기울어도 대체로 가로
MIN_LEN_FRAC = 0.125   # 화면 폭 대비 최소 길이
MIN_ON_PITCH = 0.7     # 이 비율 이상의 사람이 선 위쪽에 있어야 그 선을 믿는다
OUT_MARGIN_H = 0.25    # 발이 선보다 키의 이만큼 더 아래면 경기장 밖
HOLD_S = 1.0


def _white_lines(frame_rgb: np.ndarray, min_len_frac: float):
    """잔디 위 흰 선 후보 [(x1, y1, x2, y2, 각도, 길이)] (원본 픽셀)."""
    import cv2
    h, w = frame_rgb.shape[:2]
    s = WORK_W / float(w)
    img = cv2.resize(frame_rgb, (WORK_W, int(round(h * s))), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    grass = cv2.inRange(hsv, (22, 15, 30), (95, 255, 255))   # 마른 잔디(채도 낮음)까지
    grass = cv2.morphologyEx(grass, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    white = cv2.inRange(hsv, (0, 0, 170), (180, 60, 255))
    white = cv2.bitwise_and(white, cv2.dilate(grass, np.ones((9, 9), np.uint8)))
    white[: int(0.08 * img.shape[0])] = 0          # 화면 맨 위 자막·조명
    lines = cv2.HoughLinesP(white, 1, np.pi / 360, threshold=60,
                            minLineLength=int(min_len_frac * WORK_W), maxLineGap=25)
    out = []
    for x1, y1, x2, y2 in ([] if lines is None else lines[:, 0]):
        ang = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        out.append((x1 / s, y1 / s, x2 / s, y2 / s, min(ang, 180 - ang), float(np.hypot(x2 - x1, y2 - y1)) / s))
    return out


def near_touchline(frame_rgb: np.ndarray) -> Optional[Line]:
    """화면 아래쪽의 길고 거의 가로인 흰 선 (원본 픽셀 좌표). 없으면 None."""
    best = None
    for x1, y1, x2, y2, ang, length in _white_lines(frame_rgb, MIN_LEN_FRAC):
        if ang > MAX_ANGLE:
            continue
        score = 0.5 * (y1 + y2) / 2 + length         # 아래쪽이면서 긴 선
        if best is None or score > best[0]:
            best = (score, (x1, y1, x2, y2))
    return best[1] if best else None


def halfway_line(frame_rgb: np.ndarray) -> Optional[Line]:
    """화면 높이의 35% 넘게 이어지는 세로에 가까운 흰 선 중 가장 긴 것 = 하프라인. 없으면 None.
    (높은 측면 카메라에서 하프라인만 화면을 위아래로 길게 가로지른다)"""
    h = frame_rgb.shape[0]
    best = None
    for x1, y1, x2, y2, ang, length in _white_lines(frame_rgb, 0.2):
        if ang < 55 or abs(y2 - y1) < 0.35 * h:
            continue
        if best is None or length > best[0]:
            best = (length, (x1, y1, x2, y2))
    return best[1] if best else None


def line_y(line: Line, x: float) -> float:
    x1, y1, x2, y2 = line
    if abs(x2 - x1) < 1e-6:
        return max(y1, y2)
    return y1 + (y2 - y1) * (x - x1) / (x2 - x1)


def outside(line: Line, box: Sequence[float]) -> bool:
    """발(박스 아래 가운데)이 선보다 키의 OUT_MARGIN_H배 넘게 아래(카메라 쪽)인가."""
    x1, y1, x2, y2 = box
    return y2 - line_y(line, (x1 + x2) / 2) > OUT_MARGIN_H * (y2 - y1)


class TouchlineFilter:
    """프레임마다 update(t, frame, persons) → 경기장 안 사람만. 걸러 낸 수는 removed."""

    def __init__(self) -> None:
        self.line: Optional[Line] = None
        self.line_t = -1e9
        self.removed = 0
        self.frames_with_line = 0

    def update(self, t: float, frame_rgb: np.ndarray, persons: List) -> List:
        cand = near_touchline(frame_rgb)
        if cand is not None and len(persons) >= 4:
            above = sum(1 for b, _ in persons if not outside(cand, b))
            if above / len(persons) >= MIN_ON_PITCH:
                self.line, self.line_t = cand, t
        if self.line is None or t - self.line_t > HOLD_S:
            return persons
        self.frames_with_line += 1
        keep = [p for p in persons if not outside(self.line, p[0])]
        self.removed += len(persons) - len(keep)
        return keep
