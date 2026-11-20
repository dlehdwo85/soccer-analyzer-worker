"""경기장 위치(m)를 알 때만 가능한 상황 판정: 골키퍼, 센터서클 킥오프, 호루라기 뒤 재개 종류와 파울.

모두 pitchreg(경기장 선 자동 인식)이나 사용자의 경기장 지정으로 위치를 미터로 알 때만 쓴다.
좌표계: x = 왼쪽 골라인→오른쪽 골라인 (0~길이), y = 먼 터치라인→가까운 터치라인 (0~폭).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

from .events import Frame

BOX_L, BOX_HALF_W = 16.5, 20.16


# ── 골키퍼 ───────────────────────────────────────────────────────────────────

def _box_side(x: float, y: float, length: float, width: float, pad: float = 2.0) -> Optional[str]:
    """페널티 박스 안이면 'left'/'right', 아니면 None."""
    if abs(y - width / 2) > BOX_HALF_W + pad:
        return None
    if x <= BOX_L + pad:
        return "left"
    if x >= length - BOX_L - pad:
        return "right"
    return None


def home_goal_side(chains: Dict[str, Tuple[str, List[Tuple[float, float, float]]]], length: float,
                   halftime_s: Optional[float] = None, min_gap_m: float = 3.0) -> Optional[str]:
    """홈팀이 지키는 골문 쪽(전반 기준). 팀마다 화면에 잡힌 동안의 평균 x를 보고, 평균이 자기 골문 쪽으로 치우친
    팀이 그쪽을 지킨다고 본다 (공격 방향 판정과 같은 근거). 두 팀 평균 차이가 min_gap_m 미만이면 None."""
    s = {"home": [0.0, 0], "away": [0.0, 0]}
    for team, pts in chains.values():
        if team not in s:
            continue
        for t, x, _ in pts:
            flip = halftime_s is not None and t >= halftime_s
            s[team][0] += (length - x) if flip else x
            s[team][1] += 1
    if not s["home"][1] or not s["away"][1]:
        return None
    hm, am = s["home"][0] / s["home"][1], s["away"][0] / s["away"][1]
    if abs(hm - am) < min_gap_m:
        return None
    return "left" if hm < am else "right"


def find_goalkeepers(chains: Dict[str, Tuple[str, List[Tuple[float, float, float]]]], length: float, width: float,
                     halftime_s: Optional[float] = None, min_frac: float = 0.8, min_s: float = 6.0,
                     max_depth_m: float = 9.0, max_lateral_m: float = 11.0, home_side: Optional[str] = None
                     ) -> Tuple[Dict[str, str], Optional[Dict]]:
    """골키퍼는 골대 앞에 있는 사람이다. 유니폼 색이 필드 선수와 같아도 위치로 찾는다.

    chains: {선수 조각 id: (팀, [(t, x, y)])}. 화면에 잡힌 동안 거의 내내(min_frac) 한쪽 페널티 박스 안에 있던 조각이
    후보다. 수비수는 공을 따라 박스를 드나들지만 골키퍼는 박스에 남는다. 팀마다 그런 조각의 시간이 몰린 쪽이 그 팀
    골문이고(두 팀은 서로 반대쪽), 그중 골문 바로 앞(골라인에서 max_depth_m, 골문 가운데에서 옆으로 max_lateral_m 안)에
    주로 있던 조각만 골키퍼다 — 박스 모서리나 코너 쪽에 서 있던 수비수를 빼기 위해서다 (2쿼터 화면 대조: 코너 근처
    7초짜리 조각이 골키퍼로 잡혔다). 같은 시간에 후보가 여럿이면 골문에 가장 가까운 한 명만 골키퍼다.
    반환: ({조각 id: 팀}, {"home_side": 'left'|'right', "evidence_s": {...}}) — 근거가 없으면 ({}, None).
    후반(halftime_s 뒤)은 진영이 바뀐 것으로 본다.

    home_side(홈이 지키는 골문 쪽)를 알면 유니폼 색으로 정한 팀은 보지 않는다: 그 골문 앞에 머문 사람은 그 골문을 지키는
    팀의 골키퍼다. 골키퍼 옷은 필드 선수와 색이 달라 상대 팀으로 분류되기 쉽다 (2쿼터 재분석: 왼쪽 골대 앞에 내내 서 있던
    월넛 골키퍼가 어두운 옷 때문에 바모스로 분류됐고, 그 때문에 양 팀 골문을 거꾸로 판단했다)."""
    def half_sign(t: float) -> int:
        return -1 if halftime_s is not None and t >= halftime_s else 1

    cands = []
    evid = {"home": {"left": 0.0, "right": 0.0}, "away": {"left": 0.0, "right": 0.0}}
    for cid, (team, pts) in chains.items():
        if team not in evid or len(pts) < 3:
            continue
        dur = pts[-1][0] - pts[0][0]
        if dur < min_s:
            continue
        # 전반 기준 쪽으로 환산 (후반은 좌우를 뒤집어 센다)
        sides = []
        for t, x, y in pts:
            s = _box_side(x, y, length, width)
            if s is not None and half_sign(t) < 0:
                s = "left" if s == "right" else "right"
            sides.append(s)
        for side in ("left", "right"):
            frac = sum(s == side for s in sides) / len(sides)
            if frac >= min_frac:
                evid[team][side] += dur * frac
                depth = sorted(min(x, length - x) for _, x, _ in pts)[len(pts) // 2]
                lateral = sorted(abs(y - width / 2) for _, _, y in pts)[len(pts) // 2]
                cands.append((cid, team, side, pts[0][0], pts[-1][0], depth, lateral))
    a = evid["home"]["left"] + evid["away"]["right"]
    b = evid["home"]["right"] + evid["away"]["left"]
    if home_side in ("left", "right"):
        if a + b < 20.0:
            return {}, None
        own = {"home": home_side, "away": "right" if home_side == "left" else "left"}
        by_side = {v: k for k, v in own.items()}
        # 유니폼 색과 상관없이 그 골문을 지키는 팀으로
        cands = [(cid, by_side[side], side, t0, t1, depth, lateral) for cid, _, side, t0, t1, depth, lateral in cands]
    else:
        if max(a, b) < 20.0 or max(a, b) < 1.5 * min(a, b):
            return {}, None       # 근거가 약하다 — 골키퍼를 정하지 않는다
        own = {"home": "left", "away": "right"} if a >= b else {"home": "right", "away": "left"}
    out: Dict[str, str] = {}
    taken: Dict[str, List[Tuple[float, float]]] = {"home": [], "away": []}
    for cid, team, side, t0, t1, depth, lateral in sorted(cands, key=lambda c: c[5]):
        if side != own[team] or depth > max_depth_m or lateral > max_lateral_m:
            continue
        if any(min(t1, b1) - max(t0, b0) > 0.5 for b0, b1 in taken[team]):
            continue      # 같은 시간에 골문에 더 가까운 사람이 이미 골키퍼다
        taken[team].append((t0, t1))
        out[cid] = team
    return out, {"home_side": own["home"], "side_from": "positions" if home_side else "keepers",
                 "evidence_s": {"same": round(a, 1), "opposite": round(b, 1)}}


# ── 센터서클 킥오프 ──────────────────────────────────────────────────────────

def centre_kickoffs(frames: Sequence[Frame], length: float, width: float, ball_m: float = 3.0,
                    line_tol_m: float = 2.0, min_each: int = 4, min_own: float = 0.9, min_still_s: float = 1.0,
                    merge_s: float = 20.0) -> List[Dict]:
    """킥오프 순간들: 공이 센터 스폿에 멈춰 있고 양 팀이 하프라인을 사이에 두고 각자 진영에 서 있다가 공이 떠나는 때.
    골이 들어가면 반드시 이렇게 다시 시작하므로, 경기 시작 뒤의 킥오프는 그 직전에 골이 있었다는 영상 근거다.
    반환: [{t(공이 떠난 시각), t0(공이 놓인 시각), home_dir(+1: 홈이 오른쪽으로 공격), sep, split_x}]"""
    cx, cy = length / 2, width / 2
    runs: List[Dict] = []
    cur: Optional[Dict] = None
    for f in frames:
        ok = None
        if f.ball is not None and math.hypot(f.ball[0] - cx, f.ball[1] - cy) <= ball_m:
            hx = [p.x for p in f.players if p.team == "home"]
            ax = [p.x for p in f.players if p.team == "away"]
            if len(hx) >= min_each and len(ax) >= min_each:
                def own(xs, sign):   # sign +1: 왼쪽 진영이 자기 진영
                    return sum((cx - x) * sign >= -line_tol_m for x in xs) / len(xs)
                left = min(own(hx, 1), own(ax, -1))      # 홈이 왼쪽 진영
                right = min(own(hx, -1), own(ax, 1))
                if max(left, right) >= min_own:
                    ok = (1 if left >= right else -1, max(left, right))
        if ok is not None:
            if cur is not None and f.t - cur["t"] <= 1.0 and cur["home_dir"] == ok[0]:
                cur["t"], cur["sep"] = f.t, max(cur["sep"], ok[1])
            else:
                if cur is not None:
                    runs.append(cur)
                cur = {"t0": f.t, "t": f.t, "home_dir": ok[0], "sep": ok[1]}
    if cur is not None:
        runs.append(cur)
    out: List[Dict] = []
    for r in runs:
        if r["t"] - r["t0"] < min_still_s:
            continue
        if out and r["t0"] - out[-1]["t"] <= merge_s and out[-1]["home_dir"] == r["home_dir"]:
            out[-1]["t"], out[-1]["sep"] = r["t"], max(out[-1]["sep"], r["sep"])     # 다시 찬 킥오프
            continue
        out.append({"t0": round(r["t0"], 2), "t": round(r["t"], 2), "home_dir": r["home_dir"],
                    "sep": round(r["sep"], 3), "split_x": cx})
    return out


# ── 호루라기 뒤 재개 종류, 파울 ──────────────────────────────────────────────

def restart_type(x: float, y: float, length: float, width: float, own_goal_x: Optional[float] = None,
                 corner_m: float = 4.0, touchline_m: float = 2.5) -> str:
    """재개 위치로 종류를 정한다: 코너킥, 드로잉, 골킥, 페널티킥, 킥오프, 그 밖은 프리킥."""
    cy = width / 2
    if math.hypot(x - length / 2, y - cy) <= 3.0:
        return "kick_off"
    if min(x, length - x) <= corner_m and min(y, width - y) <= corner_m:
        return "corner"
    if min(y, width - y) <= touchline_m:
        return "throw_in"
    gx = min(x, length - x)
    if abs(gx - 11.0) <= 1.5 and abs(y - cy) <= 1.5:
        return "penalty"
    near_goal_x = 0.0 if x < length / 2 else length
    if gx <= 6.5 and abs(y - cy) <= 12.0 and (own_goal_x is None or abs(own_goal_x - near_goal_x) < 1e-6):
        return "goal_kick"
    return "free_kick"


def foul_player(frames: Sequence[Frame], t: float, team: str, max_m: float = 5.0, window_s: float = 1.5
                ) -> Optional[str]:
    """호루라기 직전(window_s) 공에 가장 가까이 있던 team 선수 — 파울한 선수 추정. 못 찾으면 None."""
    best: Optional[Tuple[float, str]] = None
    for f in frames:
        if f.t < t - window_s:
            continue
        if f.t > t + 0.3:
            break
        if f.ball is None:
            continue
        for p in f.players:
            if p.team != team:
                continue
            d = math.hypot(p.x - f.ball[0], p.y - f.ball[1])
            if d <= max_m and (best is None or d < best[0]):
                best = (d, p.tid)
    return best[1] if best else None
