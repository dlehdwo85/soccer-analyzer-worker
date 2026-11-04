"""영상에 박힌 점수판(자막) 읽기 → 골 시각.

편집된 경기 영상은 화면 구석에 '월넛 0 : 0 바모스' 같은 점수 자막이 고정돼 있다. 글자를 OCR로 읽는 대신
1) 처음 수십 초 동안 화면 위·아래 띠에서 '움직이지 않는 흰 글자' 줄을 찾아 점수판 위치를 정하고
2) 그 영역의 흰 글자 모양을 1초마다 기록해 두었다가
3) 콜론(:)을 찾아 바로 왼쪽·오른쪽 숫자 글자 모양만 비교한다. 점수 글자 폭에 따라 가운데 정렬된 자막
   전체가 조금씩 밀리므로 영역 전체를 비교하면 안 된다. 한쪽 숫자 모양이 바뀌어 몇 초 이상 유지되면
   그쪽 팀 골이다. 점수는 늘기만 하므로 처음 보는 모양으로 바뀔 때마다 1골로 센다.
자막이 잠깐 사라지거나(리플레이·전환) 콜론을 못 찾는 장면은 건너뛴다. 이미 본 모양으로 되돌아가면(정정) 골로 세지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

BANDS = (("top", 0.0, 0.16), ("bottom", 0.86, 1.0))   # 자막이 있을 법한 화면 위·아래 띠 (높이 비율)
FIND_SAMPLES = 45       # 처음 이만큼(초) 모아 점수판 위치를 정한다
STATIC_FRAC = 0.9       # 이 비율 이상의 장면에서 흰색이면 '고정된 글자' 픽셀
READABLE_FRAC = 0.5     # 점수판 후보로 인정하려면 이 비율 이상의 장면에서 콜론·숫자가 읽혀야
HOLD_S = 20             # 바뀐 숫자 모양이 이만큼(초) 동안
HOLD_FRAC = 0.7         # 이 비율 이상이어야 점수 변경으로 인정


def white_mask(rgb: np.ndarray) -> np.ndarray:
    """밝고 채도 낮은 픽셀(자막의 흰 글자)."""
    mn = rgb.min(axis=2).astype(np.int16)
    mx = rgb.max(axis=2).astype(np.int16)
    return (mn >= 185) & (mx - mn <= 45)


def _components(mask: np.ndarray) -> List[Tuple[int, int, int, int, int]]:
    """연결 요소 [(x0, y0, x1, y1, 픽셀 수)]."""
    import cv2
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    return [(int(s[0]), int(s[1]), int(s[0] + s[2]), int(s[1] + s[3]), int(s[4])) for s in stats[1:n]]


def find_text_row(static: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """가장 글자가 많은 줄 (x0, y0, x1, y1)."""
    rows = find_text_rows(static)
    return rows[0] if rows else None


def find_text_rows(static: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """고정된 흰 픽셀에서 글자처럼 생긴 조각들이 가로로 늘어선 줄들 (글자 많은 순).
    경기장 선처럼 길고 가는 것, 너무 작은 점은 버린다. 점수판이 아닌 고정 글자(건물 간판·방송사 표시)도
    섞이므로, 고르는 쪽에서 콜론과 숫자가 실제로 읽히는지 확인한다."""
    h, w = static.shape
    comps = [c for c in _components(static)
             if c[4] >= 6 and 3 <= c[3] - c[1] <= 0.6 * h and c[2] - c[0] <= 0.12 * w]
    if len(comps) < 3:
        return []
    found = []
    for c in comps:
        cy, ch = (c[1] + c[3]) / 2, c[3] - c[1]
        row = [d for d in comps if abs((d[1] + d[3]) / 2 - cy) <= 0.5 * ch and 0.5 * ch <= d[3] - d[1] <= 2 * ch]
        if len(row) < 3:
            continue
        row.sort(key=lambda d: d[0])
        # 글자 사이 간격이 글자 높이의 3배를 넘으면 다른 덩어리
        groups, cur = [], [row[0]]
        for d in row[1:]:
            if d[0] - cur[-1][2] <= 3 * ch:
                cur.append(d)
            else:
                groups.append(cur)
                cur = [d]
        groups.append(cur)
        for g in groups:
            if len(g) >= 3:
                box = (min(d[0] for d in g), min(d[1] for d in g), max(d[2] for d in g), max(d[3] for d in g))
                found.append((len(g), box))
    out: List[Tuple[int, int, int, int]] = []
    for _, box in sorted(found, key=lambda x: -x[0]):
        if not any(abs(box[0] - o[0]) < 4 and abs(box[1] - o[1]) < 4 for o in out):
            out.append(box)
    return out


@dataclass
class ScoreChange:
    t: float
    side: str            # left | right


@dataclass
class ScoreboardReader:
    frame_w: int
    frame_h: int
    every_s: float = 1.0
    _last_t: float = -1e9
    _find: Dict[str, List[np.ndarray]] = field(default_factory=dict)   # 띠별 원본 해상도 흰 글자 마스크
    _find_t: List[float] = field(default_factory=list)
    box: Optional[Tuple[int, int, int, int]] = None     # 원본 픽셀 (x0, y0, x1, y1)
    failed: bool = False
    samples: List[Tuple[float, np.ndarray]] = field(default_factory=list)

    def _band_rows(self, a: float, b: float) -> Tuple[int, int]:
        return int(a * self.frame_h), int(b * self.frame_h)

    def add(self, t: float, frame_rgb: np.ndarray) -> None:
        if self.failed or t - self._last_t < self.every_s:
            return
        self._last_t = t
        if self.box is None:
            for name, a, b in BANDS:
                y0, y1 = self._band_rows(a, b)
                self._find.setdefault(name, []).append(white_mask(frame_rgb[y0:y1]))
            self._find_t.append(t)
            if len(self._find_t) >= FIND_SAMPLES:
                self._locate()
            return
        x0, y0, x1, y1 = self.box
        self.samples.append((t, white_mask(frame_rgb[y0:y1, x0:x1])))

    def _locate(self) -> None:
        for name, a, b in BANDS:
            masks = self._find.get(name) or []
            if not masks:
                continue
            static = np.stack(masks).mean(axis=0) >= STATIC_FRAC
            band_y0, _ = self._band_rows(a, b)
            for x0, y0, x1, y1 in find_text_rows(static)[:8]:
                pad = int(round(0.6 * (y1 - y0)))
                bx0, bx1 = max(0, x0 - pad), min(self.frame_w, x1 + pad)
                by0, by1 = max(0, y0 - pad), min(static.shape[0], y1 + pad)
                crops = [m[by0:by1, bx0:bx1] for m in masks]
                # 점수판이면 대부분의 장면에서 콜론과 양옆 숫자가 읽힌다 (경기 1: 오른쪽 위 다른 고정 글자를
                # 점수판으로 잡아 없던 골 3개를 기록했다)
                if sum(score_glyphs(c) is not None for c in crops) < READABLE_FRAC * len(crops):
                    continue
                self.box = (bx0, band_y0 + by0, bx1, band_y0 + by1)
                # 위치를 찾는 동안 모은 장면도 버리지 않는다
                self.samples = list(zip(self._find_t, crops))
                break
            if self.box is not None:
                break
        else:
            self.failed = True
        self._find.clear()

    def finish(self) -> Dict:
        """{"found": bool, "box": [...], "changes": [{"t", "side"}]}"""
        if self.box is None and not self.failed and self._find_t:
            self._locate()
        if self.box is None or len(self.samples) < HOLD_S:
            return {"found": False}
        return {"found": True, "box": list(self.box),
                "changes": [{"t": round(c.t, 1), "side": c.side} for c in detect_changes(self.samples)]}


GLYPH_H, GLYPH_W = 16, 24   # 두 자리까지 담는 칸


def _normalize(crop: np.ndarray) -> np.ndarray:
    """글자 높이를 GLYPH_H로 맞추되 폭은 비율대로 두고 가운데에 놓는다 ('1'처럼 좁은 글자를 늘리지 않는다)."""
    import cv2
    h, w = crop.shape
    nw = max(1, min(GLYPH_W, int(round(w * GLYPH_H / max(1, h)))))
    g = cv2.resize(crop.astype(np.uint8) * 255, (nw, GLYPH_H), interpolation=cv2.INTER_AREA) >= 128
    out = np.zeros((GLYPH_H, GLYPH_W), bool)
    x = (GLYPH_W - nw) // 2
    out[:, x:x + nw] = g
    return out


def score_glyphs(mask: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """흰 글자 마스크에서 콜론(작은 점 두 개가 위아래로)을 찾아 바로 왼쪽·오른쪽 숫자 글자 모양을 돌려준다.
    모양은 높이를 맞춘 이진 그림 (GLYPH_H×GLYPH_W). 콜론이나 숫자를 못 찾으면 None."""
    comps = [c for c in _components(mask) if c[4] >= 2]
    if len(comps) < 4:
        return None
    hs = sorted(c[3] - c[1] for c in comps if c[4] >= 8)
    if not hs:
        return None
    h = hs[len(hs) * 3 // 4]                      # 글자 높이 (큰 쪽 기준)
    dots = [c for c in comps if c[3] - c[1] <= 0.35 * h and c[2] - c[0] <= 0.35 * h]
    colon = None
    for a in dots:
        for b in dots:
            if b is a or b[1] <= a[3]:
                continue
            if abs((a[0] + a[2]) - (b[0] + b[2])) <= 3 and 0.3 * h <= b[1] - a[1] <= 0.9 * h:
                colon = (min(a[0], b[0]), a[1], max(a[2], b[2]), b[3])
                break
        if colon:
            break
    if colon is None:
        return None
    cx = (colon[0] + colon[2]) / 2
    tall = [c for c in comps if c[3] - c[1] >= 0.6 * h and c[1] <= colon[3] and c[3] >= colon[1]]

    def side(sign: int) -> Optional[np.ndarray]:
        cand = sorted((c for c in tall if (c[0] + c[2]) / 2 * sign > cx * sign), key=lambda c: abs((c[0] + c[2]) / 2 - cx))
        if not cand:
            return None
        group = [cand[0]]
        for c in cand[1:]:   # 두 자리 점수: 글자 사이 간격이 좁은 것까지
            edge = min(g[0] for g in group) if sign < 0 else max(g[2] for g in group)
            gap = edge - c[2] if sign < 0 else c[0] - edge
            if 0 <= gap <= 0.35 * h:
                group.append(c)
            else:
                break
        x0, y0 = min(g[0] for g in group), min(g[1] for g in group)
        x1, y1 = max(g[2] for g in group), max(g[3] for g in group)
        return _normalize(mask[y0:y1, x0:x1])

    left, right = side(-1), side(1)
    if left is None or right is None:
        return None
    return left, right


def _same(a: np.ndarray, b: np.ndarray, tol: float = 0.15) -> bool:
    """같은 숫자 모양인가. 자막이 다시 가운데 정렬되며 글자가 반 픽셀씩 밀려도 같다고 보도록 1칸 어긋남을 허용한다."""
    if a.shape != b.shape:
        return False
    h, w = a.shape
    best = 1.0
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            pa = a[max(0, dy):h + min(0, dy), max(0, dx):w + min(0, dx)]
            pb = b[max(0, -dy):h + min(0, -dy), max(0, -dx):w + min(0, -dx)]
            best = min(best, float((pa ^ pb).mean()))
    return best < tol


def _labels(glyphs: List[np.ndarray]) -> List[int]:
    """글자 모양을 비슷한 것끼리 묶어 번호를 붙인다 (처음 나온 순서)."""
    protos: List[np.ndarray] = []
    out = []
    for g in glyphs:
        for i, p in enumerate(protos):
            if _same(g, p):
                out.append(i)
                break
        else:
            protos.append(g)
            out.append(len(protos) - 1)
    return out


def detect_changes(samples: List[Tuple[float, np.ndarray]]) -> List[ScoreChange]:
    """1초 간격 자막 마스크 → 점수가 바뀐 순간들.

    한쪽 숫자 모양이 처음 보는 모양으로 바뀌고, 그 뒤 HOLD_S초 동안 그 모양이 다수(HOLD_FRAC)일 때만 골로 본다.
    점수는 몇 분씩 그대로 있으므로, 몇 초 흔들리는 글자 경계·가림은 이 조건을 넘지 못한다."""
    seq = []
    for t, m in samples:
        g = score_glyphs(m)
        if g is not None:
            seq.append((t, g))
    out: List[ScoreChange] = []
    for idx, name in ((0, "left"), (1, "right")):
        if len(seq) < HOLD_S:
            break
        labels = _labels([g[idx] for _, g in seq])
        cur = max(set(labels[:HOLD_S]), key=labels[:HOLD_S].count)
        seen = {cur}
        i = 0
        while i < len(labels):
            lb = labels[i]
            if lb == cur:
                i += 1
                continue
            win = labels[i:i + HOLD_S]
            if len(win) < HOLD_S * 0.5 or win.count(lb) < HOLD_FRAC * len(win):
                i += 1
                continue
            if lb not in seen:
                out.append(ScoreChange(seq[i][0], name))
                seen.add(lb)
            cur = lb
            i += 1
    out.sort(key=lambda c: c.t)
    return out
