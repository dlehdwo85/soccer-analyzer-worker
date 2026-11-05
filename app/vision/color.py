"""유니폼 색상 기반 팀 분류.

선수 박스의 상체(가슴) 영역에서 잔디색을 걷어낸 픽셀만 모아 LAB 색공간에서
기준 색(홈/원정/홈GK/원정GK/심판)과 비교한다.

- 단색 유니폼: 픽셀 중앙값과 기준 색의 거리로 판정
- 줄무늬 유니폼: 기준 색에 가까운 픽셀 비율로 판정 (중앙값이 중간색으로 흐려지는 문제 회피)
- 사용자가 고른 색이 실제 영상과 다를 수 있으므로, 경기 초반 관측 색을 군집화해
  기준 색을 가장 가까운 실제 군집 색으로 보정한다(snap).
- 프레임 단위 판정은 흔들리므로 트랙 전체 다수결로 팀을 확정한다.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# 라벨: 팀 소속으로 집계되는 라벨과 제외 라벨
HOME, AWAY, HOME_GK, AWAY_GK, REFEREE, OTHER = "home", "away", "home_gk", "away_gk", "referee", "other"
TEAM_OF_LABEL = {HOME: "home", AWAY: "away", HOME_GK: "home", AWAY_GK: "away"}

# LAB 거리에서 밝기(L) 가중치를 낮춰 햇빛/그늘 차이에 덜 민감하게 한다.
L_WEIGHT = 0.5
PIXEL_MATCH_DE = 22.0      # 픽셀이 기준 색과 "같은 색"으로 볼 거리
MIN_MATCH_FRACTION = 0.18  # 줄무늬 판정에 필요한 최소 비율
MAX_MEDIAN_DE = 38.0       # 중앙값 판정 허용 거리 — 넘으면 OTHER
SNAP_MAX_DE = 55.0         # 기준 색을 관측 군집으로 보정할 최대 거리


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    h = hex_color.strip().lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def rgb_to_hex(rgb: Sequence[float]) -> str:
    r, g, b = (int(max(0, min(255, round(c)))) for c in rgb)
    return f"#{r:02x}{g:02x}{b:02x}"


def _srgb_to_linear(c: np.ndarray) -> np.ndarray:
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """(N,3) RGB 0-255 → (N,3) CIE LAB (L 0-100). OpenCV 없이 동작."""
    rgb = np.asarray(rgb, dtype=np.float64).reshape(-1, 3) / 255.0
    lin = _srgb_to_linear(rgb)
    m = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    xyz = lin @ m.T / np.array([0.95047, 1.0, 1.08883])
    eps, kappa = 216 / 24389, 24389 / 27
    f = np.where(xyz > eps, np.cbrt(xyz), (kappa * xyz + 16) / 116)
    L = 116 * f[:, 1] - 16
    a = 500 * (f[:, 0] - f[:, 1])
    b = 200 * (f[:, 1] - f[:, 2])
    return np.stack([L, a, b], axis=1)


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    lab = np.asarray(lab, dtype=np.float64).reshape(-1, 3)
    fy = (lab[:, 0] + 16) / 116
    fx = fy + lab[:, 1] / 500
    fz = fy - lab[:, 2] / 200
    eps, kappa = 216 / 24389, 24389 / 27

    def inv(f):
        f3 = f ** 3
        return np.where(f3 > eps, f3, (116 * f - 16) / kappa)

    xyz = np.stack([inv(fx), inv(fy), inv(fz)], axis=1) * np.array([0.95047, 1.0, 1.08883])
    m_inv = np.array([[3.2404542, -1.5371385, -0.4985314],
                      [-0.9692660, 1.8760108, 0.0415560],
                      [0.0556434, -0.2040259, 1.0572252]])
    lin = np.clip(xyz @ m_inv.T, 0, 1)
    srgb = np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * lin ** (1 / 2.4) - 0.055)
    return srgb * 255.0


def delta_e(lab_a: np.ndarray, lab_b: np.ndarray) -> np.ndarray:
    """L 가중 CIE76 거리. lab_a (N,3), lab_b (3,) 또는 (N,3)."""
    d = np.asarray(lab_a, dtype=np.float64) - np.asarray(lab_b, dtype=np.float64)
    d = np.atleast_2d(d)
    return np.sqrt((L_WEIGHT * d[:, 0]) ** 2 + d[:, 1] ** 2 + d[:, 2] ** 2)


# ── 상체 픽셀 추출 ───────────────────────────────────────────────────────────

@dataclass
class JerseySample:
    median_lab: np.ndarray            # (3,)
    pixels_lab: np.ndarray            # (K,3) 서브샘플
    valid_fraction: float             # 잔디 제외 후 남은 비율
    appearance: Optional[np.ndarray] = None  # (9,) 머리·반바지·양말 영역 LAB 중앙값 — 같은 팀 선수끼리 구분용


def torso_pixels_rgb(frame_rgb: np.ndarray, box: Sequence[float]) -> np.ndarray:
    """박스(x1,y1,x2,y2)의 가슴 영역 RGB 픽셀 (N,3). 팔·배경을 피하려 가운데만 자른다."""
    h_img, w_img = frame_rgb.shape[:2]
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    cx1 = int(round(x1 + 0.25 * w))
    cx2 = int(round(x2 - 0.25 * w))
    cy1 = int(round(y1 + 0.15 * h))
    cy2 = int(round(y1 + 0.50 * h))
    cx1, cx2 = max(0, cx1), min(w_img, max(cx1 + 1, cx2))
    cy1, cy2 = max(0, cy1), min(h_img, max(cy1 + 1, cy2))
    crop = frame_rgb[cy1:cy2, cx1:cx2]
    return crop.reshape(-1, 3)


def grass_mask_rgb(pixels_rgb: np.ndarray) -> np.ndarray:
    """잔디(초록) 픽셀이면 True. HSV 근사로 OpenCV 없이 계산."""
    p = pixels_rgb.astype(np.float64)
    r, g, b = p[:, 0], p[:, 1], p[:, 2]
    mx = p.max(axis=1)
    mn = p.min(axis=1)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0)
    green_dominant = (g >= r) & (g >= b) & (g - np.minimum(r, b) > 18)
    return green_dominant & (sat > 0.18) & (mx > 40)


BG_DE = 14.0  # 배경 색과 이만큼 가까운 상체 픽셀은 배경으로 본다


def background_pixels_rgb(frame_rgb: np.ndarray, box: Sequence[float]) -> np.ndarray:
    """박스 바로 바깥 좌우 띠(상체 높이)의 픽셀 — 선수 뒤 배경(나무·펜스·관중석) 색 추정용."""
    h_img, w_img = frame_rgb.shape[:2]
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    band = max(2, int(round(0.18 * w)))
    cy1, cy2 = max(0, int(round(y1 + 0.10 * h))), min(h_img, int(round(y1 + 0.55 * h)))
    parts = []
    lx2, rx1 = int(round(x1)), int(round(x2))
    if lx2 - band >= 0:
        parts.append(frame_rgb[cy1:cy2, lx2 - band:lx2].reshape(-1, 3))
    if rx1 + band <= w_img:
        parts.append(frame_rgb[cy1:cy2, rx1:rx1 + band].reshape(-1, 3))
    return np.concatenate(parts) if parts else np.zeros((0, 3), dtype=frame_rgb.dtype)


def _remove_background(valid_rgb: np.ndarray, bg_rgb: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    bg = bg_rgb[~grass_mask_rgb(bg_rgb)] if len(bg_rgb) else bg_rgb
    if len(bg) < 12:
        return valid_rgb
    if len(bg) > 400:
        bg = bg[rng.choice(len(bg), 400, replace=False)]
    centers, _ = kmeans(rgb_to_lab(bg), 3, iters=10)
    lab = rgb_to_lab(valid_rgb)
    near = np.min([delta_e(lab, c) for c in centers], axis=0) < BG_DE
    kept = valid_rgb[~near]
    # 옆에 같은 팀 선수가 붙어 있으면 '배경'이 곧 유니폼이라 다 지워진다 → 이때는 거르지 않는다
    return kept if len(kept) >= max(8, 0.3 * len(valid_rgb)) else valid_rgb


def sample_jersey(frame_rgb: np.ndarray, box: Sequence[float], max_pixels: int = 256,
                  rng: Optional[np.random.Generator] = None) -> Optional[JerseySample]:
    """상체 픽셀에서 잔디와 배경을 빼고 유니폼 색 표본을 만든다."""
    rng = rng or np.random.default_rng(0)
    px = torso_pixels_rgb(frame_rgb, box)
    if len(px) < 12:
        return None
    keep = ~grass_mask_rgb(px)
    valid = px[keep]
    frac = float(keep.mean()) if len(px) else 0.0
    if len(valid) < 8:
        return None
    valid = _remove_background(valid, background_pixels_rgb(frame_rgb, box), rng)
    if len(valid) > max_pixels:
        valid = valid[rng.choice(len(valid), max_pixels, replace=False)]
    lab = rgb_to_lab(valid)
    return JerseySample(median_lab=np.median(lab, axis=0), pixels_lab=lab, valid_fraction=frac,
                        appearance=body_appearance(frame_rgb, box))


# 유니폼 상의는 팀원이 모두 같으니, 사람마다 다른 머리(머리카락·피부)·반바지·양말·축구화 색을 본다.
APPEARANCE_BANDS = ((0.0, 0.14), (0.56, 0.76), (0.80, 1.0))


def body_appearance(frame_rgb: np.ndarray, box: Sequence[float]) -> Optional[np.ndarray]:
    """박스 가운데 세로 띠를 머리·반바지·양말 영역으로 나눠 잔디를 뺀 LAB 중앙값 (9,). 못 구한 영역은 NaN."""
    h_img, w_img = frame_rgb.shape[:2]
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    if h < 24:
        return None
    cx1, cx2 = max(0, int(round(x1 + 0.3 * w))), min(w_img, int(round(x2 - 0.3 * w)))
    if cx2 - cx1 < 2:
        return None
    out = np.full(9, np.nan)
    for i, (a, b) in enumerate(APPEARANCE_BANDS):
        ya, yb = max(0, int(round(y1 + a * h))), min(h_img, int(round(y1 + b * h)))
        px = frame_rgb[ya:yb, cx1:cx2].reshape(-1, 3)
        if len(px) < 6:
            continue
        px = px[~grass_mask_rgb(px)]
        if len(px) >= 4:
            out[3 * i:3 * i + 3] = np.median(rgb_to_lab(px), axis=0)
    return out if np.isfinite(out).any() else None


def appearance_distance(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> Optional[float]:
    """두 트랙 외형(body_appearance 중앙값)의 영역별 평균 색 거리. 비교할 영역이 없으면 None."""
    if a is None or b is None:
        return None
    ds = []
    for i in range(len(APPEARANCE_BANDS)):
        pa, pb = a[3 * i:3 * i + 3], b[3 * i:3 * i + 3]
        if np.isfinite(pa).all() and np.isfinite(pb).all():
            ds.append(float(delta_e(pa, pb)[0]))
    return float(np.mean(ds)) if ds else None


def kit_mismatch(lab: Optional[np.ndarray], team: str, refs: Dict[str, np.ndarray]) -> bool:
    """상의 색이 자기 팀보다 상대 팀이나 심판 색에 가까운가 (팀 판정이 흔들린 조각 — 심판·상대 선수·형광 조끼 GK)."""
    if lab is None or team not in refs:
        return False
    # 무채색(검정·흰색) 팀인데 몸통이 형광처럼 진한 색이면 조끼 입은 GK·심판이다 (밤 조명에선 형광이 어둡게 찍혀
    # 심판 기준색보다 팀 색에 더 가깝게 나오기도 한다)
    chroma = lambda v: float(np.hypot(v[1], v[2]))
    if chroma(lab) > 30 and chroma(refs[team]) < 20:
        return True
    own = float(delta_e(lab, refs[team])[0])
    others = [float(delta_e(lab, v)[0]) for k, v in refs.items() if k != team]
    return bool(others) and min(others) < own


# ── 군집화 (기준 색 보정용) ───────────────────────────────────────────────────

def kmeans(points: np.ndarray, k: int, iters: int = 30, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """작은 k-means (k-means++ 초기화). 반환: (centers (k,3), labels (N,))."""
    pts = np.asarray(points, dtype=np.float64)
    n = len(pts)
    k = max(1, min(k, n))
    rng = np.random.default_rng(seed)
    centers = [pts[rng.integers(n)]]
    for _ in range(1, k):
        d2 = np.min([delta_e(pts, c) ** 2 for c in centers], axis=0)
        probs = d2 / d2.sum() if d2.sum() > 0 else np.full(n, 1 / n)
        centers.append(pts[rng.choice(n, p=probs)])
    centers = np.array(centers)
    labels = np.zeros(n, dtype=int)
    for _ in range(iters):
        dists = np.stack([delta_e(pts, c) for c in centers], axis=1)
        new_labels = dists.argmin(axis=1)
        if np.array_equal(new_labels, labels) and _ > 0:
            break
        labels = new_labels
        for i in range(k):
            members = pts[labels == i]
            if len(members):
                centers[i] = np.median(members, axis=0)
    return centers, labels


# ── 분류기 ───────────────────────────────────────────────────────────────────

@dataclass
class TeamColorClassifier:
    """기준 색 사전 {라벨: hex}. home/away 필수, 나머지 선택."""
    reference_hex: Dict[str, str]
    reference_lab: Dict[str, np.ndarray] = field(default_factory=dict)
    snapped_hex: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if not self.reference_lab:
            self.reference_lab = {
                lbl: rgb_to_lab(np.array([hex_to_rgb(hx)]))[0]
                for lbl, hx in self.reference_hex.items() if hx
            }

    @classmethod
    def from_match(cls, home: str, away: str, referee: Optional[str] = None,
                   home_gk: Optional[str] = None, away_gk: Optional[str] = None) -> "TeamColorClassifier":
        refs = {HOME: home, AWAY: away}
        if referee:
            refs[REFEREE] = referee
        if home_gk:
            refs[HOME_GK] = home_gk
        if away_gk:
            refs[AWAY_GK] = away_gk
        return cls(reference_hex=refs)

    def calibrate(self, samples: Sequence[JerseySample], k: int = 5) -> Dict[str, str]:
        """관측 색 군집으로 기준 색을 보정한다. 반환: 보정된 {라벨: hex}.

        홈/원정이 같은 군집으로 끌려가면(사용자 색이 실제와 크게 다를 때),
        가장 큰 두 군집을 홈/원정에 거리 순으로 배정한다.
        """
        if len(samples) < 10:
            return {}
        medians = np.array([s.median_lab for s in samples])
        centers, labels = kmeans(medians, k=min(k, len(medians)))
        sizes = np.bincount(labels, minlength=len(centers))
        big = [i for i in np.argsort(-sizes) if sizes[i] >= max(3, 0.04 * len(medians))]
        if not big:
            return {}

        # 심판·골키퍼는 경기장에 한두 명뿐이라 군집이 작다 → 더 작은 군집까지 보정 후보로 (바모스전: 심판 색이
        # 보정되지 않아 형광 노랑 심판 일부가 팀 선수로 들어갔다)
        small = [i for i in np.argsort(-sizes) if sizes[i] >= max(3, 0.015 * len(medians))]
        snapped: Dict[str, int] = {}
        used: set = set()
        # 팀 색을 먼저(큰 군집 우선) 배정해 GK/심판이 팀 군집을 빼앗지 않게 한다.
        order = [lbl for lbl in (HOME, AWAY, REFEREE, HOME_GK, AWAY_GK) if lbl in self.reference_lab]
        for lbl in order:
            ref = self.reference_lab[lbl]
            pool = big if lbl in (HOME, AWAY) else small
            cands = [(float(delta_e(centers[i], ref)[0]), i) for i in pool if i not in used]
            if not cands:
                continue
            d, i = min(cands)
            if d <= SNAP_MAX_DE:
                snapped[lbl] = i
                used.add(i)

        if HOME not in snapped or AWAY not in snapped:
            # 사용자 색이 실제와 많이 다름 → 가장 큰 두 군집 사용
            top2 = big[:2]
            if len(top2) == 2:
                a, b = top2
                cost_ab = delta_e(centers[a], self.reference_lab[HOME])[0] + delta_e(centers[b], self.reference_lab[AWAY])[0]
                cost_ba = delta_e(centers[b], self.reference_lab[HOME])[0] + delta_e(centers[a], self.reference_lab[AWAY])[0]
                home_i, away_i = (a, b) if cost_ab <= cost_ba else (b, a)
                snapped[HOME], snapped[AWAY] = home_i, away_i

        for lbl, i in snapped.items():
            self.reference_lab[lbl] = centers[i]
            self.snapped_hex[lbl] = rgb_to_hex(lab_to_rgb(centers[i])[0])
        return dict(self.snapped_hex)

    def classify(self, sample: Optional[JerseySample]) -> Tuple[str, float]:
        """반환: (라벨, 신뢰도 0-1)."""
        if sample is None or not self.reference_lab:
            return OTHER, 0.0
        best_frac_lbl, best_frac = OTHER, 0.0
        best_med_lbl, best_med = OTHER, float("inf")
        for lbl, ref in self.reference_lab.items():
            frac = float((delta_e(sample.pixels_lab, ref) <= PIXEL_MATCH_DE).mean())
            if frac > best_frac:
                best_frac_lbl, best_frac = lbl, frac
            med = float(delta_e(sample.median_lab, ref)[0])
            if med < best_med:
                best_med_lbl, best_med = lbl, med

        if best_med <= MAX_MEDIAN_DE * 0.6:
            return best_med_lbl, 1.0 - best_med / MAX_MEDIAN_DE
        if best_frac >= MIN_MATCH_FRACTION:
            return best_frac_lbl, min(1.0, best_frac * 1.5)
        if best_med <= MAX_MEDIAN_DE:
            return best_med_lbl, max(0.05, 1.0 - best_med / MAX_MEDIAN_DE)
        return OTHER, 0.0


def vote_track_labels(votes: Sequence[Tuple[str, float]], min_share: float = 0.4) -> Tuple[str, float]:
    """트랙의 프레임별 (라벨, 신뢰도) 투표 → (최종 라벨, 득표율)."""
    if not votes:
        return OTHER, 0.0
    score: Dict[str, float] = defaultdict(float)
    for lbl, conf in votes:
        score[lbl] += max(conf, 0.05)
    total = sum(score.values())
    lbl, s = max(score.items(), key=lambda kv: kv[1])
    share = s / total if total else 0.0
    if lbl == OTHER or share < min_share:
        # OTHER가 최다라도 팀 라벨이 상당하면 팀으로 본다(부분 가림 대비)
        team_lbls = [(l, v) for l, v in score.items() if l in TEAM_OF_LABEL]
        if team_lbls:
            l2, v2 = max(team_lbls, key=lambda kv: kv[1])
            if v2 / total >= 0.35:
                return l2, v2 / total
        return (lbl, share) if lbl != OTHER and share >= min_share else (OTHER, share)
    return lbl, share


def summarize_detected_colors(samples_by_label: Dict[str, List[JerseySample]]) -> Dict[str, str]:
    """라벨별 실제 관측 대표 색(hex) — UI에 '검출된 유니폼 색'으로 보여준다."""
    out = {}
    for lbl, samples in samples_by_label.items():
        if samples:
            med = np.median(np.array([s.median_lab for s in samples]), axis=0)
            out[lbl] = rgb_to_hex(lab_to_rgb(med)[0])
    return out
