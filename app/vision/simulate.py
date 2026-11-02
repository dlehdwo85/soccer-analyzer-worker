"""가상 경기 생성기.

선수 위치와 공 궤적을 물리적으로 그럴듯하게 만들고, 실제로 일어난 이벤트(정답)를 함께 반환한다.
- 샘플 분석: 영상 없이도 실제 이벤트 판정 엔진을 거친 일관된 결과를 보여준다.
- 테스트: 판정 엔진의 정확도(정밀도/재현율)를 정답과 비교해 검증한다.

관측 잡음(위치 흔들림, 공 미검출)을 넣어 실제 영상 추적과 비슷한 조건을 만든다.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from .events import Frame, PlayerObs
from .pitch import GOAL_WIDTH, POSSESSION_RADIUS_M, field_dims

# (x, y) 비율 — 홈팀이 오른쪽으로 공격할 때 기준. 첫 번째는 골키퍼.
FORMATIONS = {
    11: [(0.04, 0.5), (0.2, 0.15), (0.18, 0.38), (0.18, 0.62), (0.2, 0.85),
         (0.38, 0.3), (0.36, 0.5), (0.38, 0.7), (0.58, 0.18), (0.62, 0.5), (0.58, 0.82)],
    7: [(0.04, 0.5), (0.22, 0.25), (0.2, 0.5), (0.22, 0.75), (0.42, 0.35), (0.42, 0.65), (0.62, 0.5)],
    5: [(0.05, 0.5), (0.28, 0.3), (0.28, 0.7), (0.52, 0.35), (0.6, 0.65)],
}


@dataclass
class TruthEvent:
    kind: str                 # possession | pass | shot | recovery
    t: float
    team: str
    player: str
    target: Optional[str] = None
    success: Optional[bool] = None
    detail: Dict = field(default_factory=dict)


@dataclass
class SimResult:
    frames: List[Frame]
    truth: List[TruthEvent]
    length: float
    width: float
    duration_s: float
    fps: float


@dataclass
class _P:
    tid: str
    team: str
    base: Tuple[float, float]
    x: float
    y: float
    is_gk: bool = False
    target: Optional[Tuple[float, float]] = None
    wander: Tuple[float, float] = (0.0, 0.0)


class _MatchSim:
    def __init__(self, duration_s, fps, n, field_type, seed, pos_noise, ball_noise, ball_dropout, pass_success):
        self.rng = random.Random(seed)
        self.L, self.W = field_dims(field_type)
        self.goal_half = GOAL_WIDTH.get(field_type, 7.32) / 2
        self.scale = self.L / 105.0
        self.duration_s, self.fps, self.dt = duration_s, fps, 1.0 / fps
        self.pos_noise, self.ball_noise, self.ball_dropout = pos_noise, ball_noise, ball_dropout
        self.pass_success = pass_success
        self.direction = {"home": 1, "away": -1}
        form = FORMATIONS[n if n in FORMATIONS else 11]
        self.players: List[_P] = []
        for team, sign in (("home", 1), ("away", -1)):
            for i, (fx, fy) in enumerate(form):
                bx = fx * self.L if sign > 0 else (1 - fx) * self.L
                by = fy * self.W if sign > 0 else (1 - fy) * self.W
                tid = f"{'H' if team == 'home' else 'A'}{i + 1}"
                self.players.append(_P(tid, team, (bx, by), bx + self.rng.gauss(0, 1),
                                       by + self.rng.gauss(0, 1), is_gk=(i == 0)))
        self.by_id = {p.tid: p for p in self.players}
        self.truth: List[TruthEvent] = []
        self.frames: List[Frame] = []
        self.owner: Optional[_P] = None
        self.owner_t0 = 0.0
        self.phase_end = 0.0
        self.flight: Optional[Dict] = None
        self.dead_until: Optional[float] = None
        self.restart: Optional[Tuple[_P, Tuple[float, float]]] = None
        kick_off = self.by_id["H" + str(len(form))]
        kick_off.x, kick_off.y = self.L / 2 - 1, self.W / 2
        self.ball = [kick_off.x + 0.5, kick_off.y]
        self._take(kick_off, 0.0)

    # ── 상태 전환 ─────────────────────────────────────────────────────
    def _take(self, p: _P, t: float):
        self.owner, self.flight, self.owner_t0 = p, None, t
        self.phase_end = t + self.rng.uniform(1.2, 4.5)
        for q in self.players:
            q.target = None

    def _release(self, t: float):
        o = self.owner
        self.truth.append(TruthEvent("possession", self.owner_t0, o.team, o.tid, detail={"t_end": t}))
        self.owner = None

    def _fly(self, to: Tuple[float, float], speed: float, on_arrive: Callable[[float], None]):
        self.flight = {"to": to, "speed": speed, "on_arrive": on_arrive}

    def _dead(self, t: float, wait: float, p: _P, spot: Tuple[float, float]):
        self.dead_until = t + wait
        self.restart = (p, spot)
        p.target = spot

    def _nearest(self, cands: List[_P], x: float, y: float) -> _P:
        return min(cands, key=lambda q: math.hypot(q.x - x, q.y - y))

    # ── 동작 ──────────────────────────────────────────────────────────
    def _pass(self, t: float):
        o = self.owner
        d = self.direction[o.team]
        mates = [q for q in self.players if q.team == o.team and q is not o and not q.is_gk]
        mates = [q for q in mates if 6 * self.scale <= math.hypot(q.x - o.x, q.y - o.y) <= 35 * self.scale] or mates
        weights = [1.0 + max(0.0, (q.x - o.x) * d) / 10 for q in mates]
        recv = self.rng.choices(mates, weights)[0]
        speed = self.rng.uniform(11, 16)
        self._release(t)
        if self.rng.random() < self.pass_success:
            to = (recv.x + self.rng.gauss(0, 1), recv.y + self.rng.gauss(0, 1))
            recv.target = to
            self.truth.append(TruthEvent("pass", t, o.team, o.tid, target=recv.tid, success=True))
            self._fly(to, speed, lambda ta, r=recv: self._take(r, ta))
        else:
            f = self.rng.uniform(0.55, 0.85)
            to = (o.x + (recv.x - o.x) * f, o.y + (recv.y - o.y) * f)
            opps = [q for q in self.players if q.team != o.team and not q.is_gk]
            cut = self._nearest(opps, *to)
            cut.target = to
            self.truth.append(TruthEvent("pass", t, o.team, o.tid, target=cut.tid, success=False,
                                         detail={"reason": "intercepted"}))
            self._fly(to, speed, lambda ta, c=cut: self._arrive_at(c, ta))

    def _arrive_at(self, p: _P, t: float):
        p.x, p.y = self.ball[0] - 0.4 * self.direction[p.team], self.ball[1]
        self._take(p, t)

    def _tackle(self, t: float):
        o = self.owner
        opps = [q for q in self.players if q.team != o.team and not q.is_gk]
        tk = self._nearest(opps, o.x, o.y)
        tk.x, tk.y = o.x + 0.6, o.y + 0.4
        self._release(t)
        self.truth.append(TruthEvent("recovery", t, tk.team, tk.tid, target=o.tid))
        self._take(tk, t + self.dt)

    def _kick_out(self, t: float):
        o = self.owner
        side_y = -2.0 if o.y < self.W / 2 else self.W + 2.0
        to = (o.x + self.direction[o.team] * self.rng.uniform(10, 25), side_y)
        self._release(t)
        self.truth.append(TruthEvent("pass", t, o.team, o.tid, success=False, detail={"reason": "out"}))
        opp = self._nearest([q for q in self.players if q.team != o.team and not q.is_gk], *to)
        spot = (min(self.L - 1, max(1, to[0])), 0.5 if side_y < 0 else self.W - 0.5)
        self._fly(to, 15.0, lambda ta: self._dead(ta, 4.0, opp, spot))

    def _shoot(self, t: float):
        o = self.owner
        d = self.direction[o.team]
        goal_x = self.L if d > 0 else 0.0
        on = self.rng.random() < 0.45
        off = self.rng.uniform(self.goal_half * 0.1, self.goal_half * 0.85) if on else self.rng.uniform(
            self.goal_half + 1.5, self.goal_half + 3.5)
        aim_y = self.W / 2 + off * self.rng.choice((-1, 1))
        self._release(t)
        keeper = next(q for q in self.players if q.team != o.team and q.is_gk)
        self.truth.append(TruthEvent("shot", t, o.team, o.tid, success=on,
                                     detail={"outcome": "on_target" if on else "off_target"}))
        if on:
            keeper.target = (goal_x - d * 0.5, aim_y)
            self._fly((goal_x - d * 0.3, aim_y), self.rng.uniform(16, 22),
                      lambda ta, k=keeper: self._take(k, ta))
        else:
            spot = (goal_x - d * 5.5, self.W / 2)
            self._fly((goal_x + d * 1.5, aim_y), self.rng.uniform(16, 22),
                      lambda ta, k=keeper: self._dead(ta, 5.0, k, spot))

    # ── 메인 루프 ─────────────────────────────────────────────────────
    def run(self) -> SimResult:
        L, W, dt = self.L, self.W, self.dt
        t = 0.0
        while t <= self.duration_s:
            if self.dead_until is not None:
                if t >= self.dead_until:
                    p, (rx, ry) = self.restart
                    p.x, p.y = rx, ry
                    self.ball = [rx + 0.5 * self.direction[p.team], ry]
                    self.dead_until = self.restart = None
                    self._take(p, t)
            elif self.flight is not None:
                fx, fy = self.flight["to"]
                dx, dy = fx - self.ball[0], fy - self.ball[1]
                dist = math.hypot(dx, dy)
                step = self.flight["speed"] * dt
                if dist <= step:
                    self.ball = [fx, fy]
                    cb = self.flight["on_arrive"]
                    self.flight = None
                    cb(t)
                else:
                    self.ball = [self.ball[0] + dx / dist * step, self.ball[1] + dy / dist * step]
            elif self.owner is not None:
                o, d = self.owner, self.direction[self.owner.team]
                if t >= self.phase_end:
                    goal_x = L if d > 0 else 0.0
                    dist_goal = math.hypot(goal_x - o.x, W / 2 - o.y)
                    r = self.rng.random()
                    if dist_goal < 26 * self.scale and r < 0.5:
                        self._shoot(t)
                    elif r < 0.08:
                        self._tackle(t)
                    elif r < 0.12:
                        self._kick_out(t)
                    else:
                        self._pass(t)
                else:
                    o.target = (min(L - 3, max(3, o.x + d * 4.0)), min(W - 3, max(3, o.y + self.rng.gauss(0, 2))))

            self._move_players(t)
            if self.owner is not None and self.flight is None and self.dead_until is None:
                d = self.direction[self.owner.team]
                self.ball = [self.owner.x + 0.5 * d, self.owner.y + 0.1]
            self._observe(t)
            t = round(t + dt, 6)

        if self.owner is not None:
            self._release(self.duration_s)
        return SimResult(self.frames, self.truth, L, W, self.duration_s, self.fps)

    def _move_players(self, t: float):
        L, W, dt = self.L, self.W, self.dt
        shift = (self.ball[0] - L / 2) * 0.35
        for p in self.players:
            if p is self.owner and self.flight is None and self.dead_until is None:
                tgt, vmax = p.target or (p.x, p.y), 4.5
            elif p.target is not None:
                tgt, vmax = p.target, 7.5
            else:
                # 천천히 변하는 개인 움직임(위치 잡기) — 매 프레임 새로 뽑으면 제자리에서 떨게 된다
                amp = 0.3 if p.is_gk else 1.0
                p.wander = (0.97 * p.wander[0] + self.rng.gauss(0, 0.35 * amp),
                            0.97 * p.wander[1] + self.rng.gauss(0, 0.35 * amp))
                gx = p.base[0] + (shift * 0.15 if p.is_gk else shift) + p.wander[0]
                gy = p.base[1] + (self.ball[1] - W / 2) * (0.1 if p.is_gk else 0.25) + p.wander[1]
                tgt, vmax = (gx, gy), (2.5 if p.is_gk else 3.5)
            dx, dy = tgt[0] - p.x, tgt[1] - p.y
            dist = math.hypot(dx, dy)
            step = min(dist, vmax * dt)
            if dist > 1e-6:
                p.x += dx / dist * step
                p.y += dy / dist * step
            p.x = min(L, max(0.0, p.x))
            p.y = min(W, max(0.0, p.y))

    def _observe(self, t: float):
        g = self.rng.gauss
        obs = [PlayerObs(p.tid, p.team, p.x + g(0, self.pos_noise), p.y + g(0, self.pos_noise),
                         POSSESSION_RADIUS_M, 1.0) for p in self.players]
        ball = None
        # 경기가 멈춘 동안에도 공은 보인다(밖에 떨어져 있음)
        if self.rng.random() > self.ball_dropout:
            ball = (self.ball[0] + g(0, self.ball_noise), self.ball[1] + g(0, self.ball_noise))
        self.frames.append(Frame(round(t, 3), obs, ball))


def simulate_match(duration_s: float = 600.0, fps: float = 5.0, players_per_team: int = 11,
                   field_type: str = "soccer", seed: int = 7, pos_noise: float = 0.12,
                   ball_noise: float = 0.15, ball_dropout: float = 0.06,
                   pass_success: float = 0.78) -> SimResult:
    return _MatchSim(duration_s, fps, players_per_team, field_type, seed,
                     pos_noise, ball_noise, ball_dropout, pass_success).run()
