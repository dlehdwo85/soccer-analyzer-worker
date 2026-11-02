"""선수·팀 통계 집계 — 볼 이벤트 중심.

모든 선수 통계는 '더할 수 있는' 형태로 저장한다(합계, 가중합).
그래서 끊긴 트랙 여러 개를 한 선수로 합칠 때 merge_player_stats()로 정확히 합산된다.

이동거리·속도·활동 영역은 영상만으로는 정확히 재기 어려워(카메라 시점·원근·가려짐)
계산하지 않는다. 위치는 출전 시간과 평균 위치(패스 네트워크용)에만 쓴다.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .events import Event, Frame, Spell

KEY_PASS_WINDOW_S = 8.0  # 패스를 받은 선수가 이 시간 안에 슈팅하면 키패스

BALL_KEYS = ("touches", "possession_s", "passes_attempted", "passes_completed", "passes_failed",
             "passes_received", "forward_passes", "key_passes", "shots", "shots_on_target", "shots_blocked",
             "interceptions", "recoveries", "dispossessed", "xg",
             # 세트피스(재개): 보정된 경기장에서만 종류가 나뉘고, 아니면 set_pieces만 센다
             "set_pieces", "corners", "free_kicks", "throw_ins", "goal_kicks", "kick_offs")
SET_PIECE_KEY = {"corner": "corners", "free_kick": "free_kicks", "throw_in": "throw_ins",
                 "goal_kick": "goal_kicks", "kick_off": "kick_offs"}
PRESENCE_KEYS = ("seconds_tracked",)
SUM_KEYS = BALL_KEYS + PRESENCE_KEYS + ("pos_sum_x", "pos_sum_y", "pos_weight", "pass_length_sum")


def empty_stats() -> Dict:
    d = {k: 0 for k in SUM_KEYS}
    d.update({"first_seen_s": None, "last_seen_s": None})
    return d


# ── 출전 시간·평균 위치 ──────────────────────────────────────────────────────

def presence_stats(points: Sequence[Tuple[float, float, float, float]], length: float, width: float,
                   max_gap_s: float = 1.5) -> Dict:
    """points: [(t, x, y, s)] — 화면에 잡힌 시간과 시간가중 평균 위치(0~1)."""
    st = empty_stats()
    if not points:
        return st
    pts = sorted(points, key=lambda p: p[0])
    st["first_seen_s"], st["last_seen_s"] = pts[0][0], pts[-1][0]
    for (t0, x0, y0, _), (t1, x1, y1, _) in zip(pts, pts[1:]):
        dt = t1 - t0
        if dt <= 0 or dt > max_gap_s:
            continue
        st["seconds_tracked"] += dt
        st["pos_sum_x"] += (x0 + x1) / 2 / length * dt
        st["pos_sum_y"] += (y0 + y1) / 2 / width * dt
        st["pos_weight"] += dt
    return st


# ── 볼 이벤트 ────────────────────────────────────────────────────────────────

def add_event_stats(stats: Dict[str, Dict], events: Iterable[Event]) -> None:
    def g(tid: str) -> Dict:
        return stats.setdefault(tid, empty_stats())

    pending: Optional[Tuple[str, str, float]] = None  # 최근 성공 패스 (패서, 받은 선수, 받은 시각)
    for e in sorted(events, key=lambda ev: ev.t):
        if e.kind == "shot":
            if pending and pending[1] == e.player and e.t - pending[2] <= KEY_PASS_WINDOW_S:
                g(pending[0])["key_passes"] += 1
            pending = None
        elif e.kind == "pass":
            pending = (e.player, e.target, e.t_end or e.t) if e.success and e.target else None
        elif e.kind == "recovery":
            pending = None

        if e.kind == "possession":
            s = g(e.player)
            s["touches"] += 1
            s["possession_s"] += (e.t_end or e.t) - e.t
        elif e.kind == "pass":
            s = g(e.player)
            s["passes_attempted"] += 1
            if e.success:
                s["passes_completed"] += 1
                s["pass_length_sum"] += e.detail.get("length_m", 0.0)
                if e.detail.get("forward"):
                    s["forward_passes"] += 1
                if e.target:
                    g(e.target)["passes_received"] += 1
            else:
                s["passes_failed"] += 1
                if e.target and e.detail.get("reason") == "intercepted":
                    g(e.target)["interceptions"] += 1
        elif e.kind == "shot":
            s = g(e.player)
            s["shots"] += 1
            if e.success:
                s["shots_on_target"] += 1
            s["xg"] += e.detail.get("xg", 0.0) or 0.0
            if e.detail.get("outcome") == "blocked":
                s["shots_blocked"] += 1
        elif e.kind == "recovery":
            g(e.player)["recoveries"] += 1
            if e.target:
                g(e.target)["dispossessed"] += 1
        elif e.kind == "set_piece":
            s = g(e.player)
            s["set_pieces"] += 1
            key = SET_PIECE_KEY.get(e.detail.get("type") or "")
            if key:
                s[key] += 1


def finalize(st: Dict) -> Dict:
    """파생 지표를 붙인 사본."""
    out = dict(st)
    att = st["passes_attempted"]
    out["pass_accuracy"] = round(st["passes_completed"] / att * 100, 1) if att else None
    out["avg_pass_length_m"] = round(st["pass_length_sum"] / st["passes_completed"], 1) if st["passes_completed"] else None
    w = st["pos_weight"]
    out["avg_x"] = round(st["pos_sum_x"] / w, 4) if w else None
    out["avg_y"] = round(st["pos_sum_y"] / w, 4) if w else None
    out["minutes_tracked"] = round(st["seconds_tracked"] / 60, 1)
    return out


def merge_player_stats(items: Sequence[Dict]) -> Dict:
    st = empty_stats()
    for it in items:
        for k in SUM_KEYS:
            st[k] += it.get(k, 0) or 0
        for key, fn in (("first_seen_s", min), ("last_seen_s", max)):
            vals = [v for v in (st[key], it.get(key)) if v is not None]
            st[key] = fn(vals) if vals else None
    return st


def compute_player_stats(frames: Sequence[Frame], events: Sequence[Event],
                         length: float, width: float) -> Dict[str, Dict]:
    """선수별 원시(합산 가능) 통계. 키에 'team' 포함."""
    traj: Dict[str, List[Tuple[float, float, float, float]]] = defaultdict(list)
    team_of: Dict[str, str] = {}
    for f in frames:
        for p in f.players:
            traj[p.tid].append((f.t, p.x, p.y, p.s))
            team_of[p.tid] = p.team
    stats: Dict[str, Dict] = {tid: presence_stats(pts, length, width) for tid, pts in traj.items()}
    add_event_stats(stats, events)
    for tid, st in stats.items():
        st["team"] = team_of.get(tid, "home")
    return stats


# ── 팀 단위 ─────────────────────────────────────────────────────────────────

def team_stats(player_stats: Dict[str, Dict], spells: Sequence[Spell]) -> Dict[str, Dict]:
    out = {}
    poss = {"home": 0.0, "away": 0.0}
    for sp in spells:
        if sp.team in poss:
            poss[sp.team] += sp.duration
    total_poss = sum(poss.values())
    for team in ("home", "away"):
        members = [s for s in player_stats.values() if s.get("team") == team]
        agg = merge_player_stats(members)
        fin = finalize(agg)
        out[team] = {
            "possession_pct": round(poss[team] / total_poss * 100, 1) if total_poss else None,
            "possession_s": round(poss[team], 1),
            "players": len(members),
            **{k: fin[k] for k in ("passes_attempted", "passes_completed", "passes_failed", "pass_accuracy",
                                    "forward_passes", "key_passes", "shots", "shots_on_target", "shots_blocked",
                                    "interceptions", "recoveries", "dispossessed", "touches",
                                    "set_pieces", "corners", "free_kicks", "throw_ins", "goal_kicks", "kick_offs")},
            "xg": round(agg["xg"], 2),
        }
    return out


def pass_network(events: Sequence[Event], player_stats: Dict[str, Dict], min_passes: int = 1) -> Dict[str, Dict]:
    """팀별 {nodes:[{id,x,y,touches}], edges:[{a,b,count}]} — 좌표는 0~1 정규화(평균 위치)."""
    edges: Dict[str, Dict[Tuple[str, str], int]] = {"home": defaultdict(int), "away": defaultdict(int)}
    for e in events:
        if e.kind == "pass" and e.success and e.target and e.team in edges:
            key = tuple(sorted((e.player, e.target)))
            edges[e.team][key] += 1
    out = {}
    for team in ("home", "away"):
        ids = {tid for pair, c in edges[team].items() if c >= min_passes for tid in pair}
        nodes = []
        for tid in sorted(ids):
            st = finalize(player_stats.get(tid, empty_stats()))
            if st["avg_x"] is None:
                continue
            nodes.append({"id": tid, "x": st["avg_x"], "y": st["avg_y"], "touches": st["touches"]})
        node_ids = {n["id"] for n in nodes}
        out[team] = {
            "nodes": nodes,
            "edges": [{"a": a, "b": b, "count": c} for (a, b), c in sorted(edges[team].items())
                      if c >= min_passes and a in node_ids and b in node_ids],
        }
    return out


def possession_timeline(spells: Sequence[Spell], duration_s: float, bins: Optional[int] = None) -> List[Dict]:
    """구간별 홈 점유율(%) — 구간 길이는 경기 길이에 맞춰 1~5분."""
    if duration_s <= 0:
        return []
    if bins is None:
        bin_s = 300.0 if duration_s >= 1800 else max(60.0, duration_s / 10)
    else:
        bin_s = duration_s / bins
    n = max(1, int(math.ceil(duration_s / bin_s)))
    home = [0.0] * n
    away = [0.0] * n
    for sp in spells:
        # 구간 번호를 하나씩 늘려 가며 나눈다 — 경계가 소수일 때 부동소수점 오차로 같은 자리를 맴돌지 않도록.
        # 마지막 구간은 남은 시간을 모두 받는다 (점유가 영상 끝을 조금 넘는 경우).
        t = sp.t0
        i = min(n - 1, max(0, int(sp.t0 // bin_s)))
        while t < sp.t1 and i < n:
            seg_end = sp.t1 if i == n - 1 else min(sp.t1, (i + 1) * bin_s)
            if seg_end > t:
                (home if sp.team == "home" else away)[i] += seg_end - t
                t = seg_end
            i += 1
    out = []
    for i in range(n):
        tot = home[i] + away[i]
        out.append({"start_s": round(i * bin_s, 1), "end_s": round(min(duration_s, (i + 1) * bin_s), 1),
                    "home_pct": round(home[i] / tot * 100, 1) if tot else None})
    return out
