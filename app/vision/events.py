"""공 점유와 경기 이벤트 판정.

입력: 프레임 목록. 각 프레임 = 시간, 선수 관측(팀·경기장 좌표·점유 반경·축척), 공 위치(모르면 None).

1) 점유 구간(possession spell)
   공이 선수 점유 반경 안에 일정 시간 이상 머물면 그 선수가 공을 '잡은' 것으로 본다.
   반경을 벗어나 멀어지면(차거나 빼앗김) 구간이 끝난다. 공이 잠깐 안 보이는 동안은
   마지막 점유자를 유지한다. 같은 선수의 짧게 끊긴 구간(드리블 중 공을 앞으로 밀기)은 합친다.

2) 이벤트 (연속한 두 점유 구간 A → B 사이에서 판정)
   - 슈팅 : A가 공을 상대 골문 방향으로 빠르게 찼고, 궤적 연장선이 골문 근처를 지남
            → 골문 안쪽이면 유효 슈팅, 상대 필드 플레이어가 골문 앞에 닿기 전에 막으면 블록
   - 패스 성공 : B가 A와 같은 팀(다른 선수)
   - 패스 실패 : B가 상대 팀이고 공이 날아갔음(가로채기) / 아무도 못 잡고 오래 지남(아웃·유실)
   - 볼 탈취  : B가 상대 팀이고 공이 날아가지 않음(몸싸움으로 빼앗음) → A는 '볼 뺏김'

공이 오래 안 보인 구간을 사이에 둔 경우는 추측하지 않고 이벤트를 만들지 않는다.
"""

from __future__ import annotations

import math
from bisect import bisect_left, bisect_right
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass
class PlayerObs:
    tid: str
    team: str          # home | away
    x: float
    y: float
    r: float           # 점유 반경 (경기장 좌표 단위)
    s: float = 1.0     # 국소 축척 (m / 좌표 단위)


@dataclass
class Frame:
    t: float
    players: List[PlayerObs]
    ball: Optional[Tuple[float, float]] = None


@dataclass
class EventConfig:
    length: float = 105.0
    width: float = 68.0
    goal_width: float = 7.32
    home_attacks: str = "auto"          # auto | right | left (전반 기준)
    halftime_s: Optional[float] = None  # 주어지면 후반에 공격 방향을 뒤집는다
    confirm_s: float = 0.2              # 점유 인정까지 머문 시간
    lose_s: float = 0.6                 # 반경 밖으로 이만큼 지나면 점유 종료
    release_factor: float = 2.5         # 공이 반경의 이 배수 밖이면 즉시 점유 종료(킥)
    hold_unknown_s: float = 1.5         # 공이 안 보여도 점유 유지하는 시간
    hold_unknown_owner_s: float = 3.0   # 공은 안 보이지만 점유자는 계속 잡히고 있을 때(발밑 가림) 유지 시간
    merge_same_s: float = 1.5           # 같은 선수 구간 합치기 허용 공백
    max_link_s: float = 6.0             # A→B를 하나의 패스로 볼 최대 공백
    max_unknown_s: float = 2.5          # 공백 중 공 미검출이 이보다 길면 판정 보류
    pass_min_m: float = 2.0
    kick_speed: float = 5.0             # m/s
    shot_speed: float = 9.0             # m/s
    shot_max_dist: float = 35.0         # m
    forward_pass_m: float = 10.0
    blocked_min_goal_dist: float = 6.0  # 골라인에서 이보다 먼 상대가 막으면 블록
    max_control_speed: float = 7.0      # 공이 이보다 빠르게 지나가면 '잡은 것'으로 보지 않음 (m/s)
    blip_max_s: float = 0.3             # 같은 선수 점유 사이에 끼인 이보다 짧은 상대 점유는 잡음
    intercept_min_m: float = 4.0        # 상대가 가져갔을 때 공이 이만큼 날아갔으면 가로채기
    intercept_min_gap_s: float = 0.6    # 또는 공이 이만큼 떠 있었으면 가로채기
    # 경기장 경계를 믿을 수 있는가. 보정이 없고 카메라가 움직이면 화면 밖 = 경기장 밖이 아니므로
    # '아웃'으로 단정하지 않는다 (공이 시야에서 벗어난 것뿐일 수 있다).
    trust_bounds: bool = True
    # 세트피스(재개) 판정
    calibrated: bool = False            # 경기장 보정이 있어야 코너킥·골킥·스로인·프리킥을 위치로 구분한다
    still_s: float = 1.2                # 재개 전 공이 이만큼 멈춰 있어야 한다
    still_m: float = 0.75               # 멈춤으로 보는 최대 이동(m)
    dead_gap_s: float = 1.5             # 직전 점유 종료 후 이만큼 비어 있으면 경기 중단으로 본다
    corner_m: float = 3.0               # 코너 아크 반경 근사
    touchline_m: float = 2.0            # 스로인: 터치라인에서 이 거리 안
    goal_area_x_m: float = 6.5          # 골킥: 골라인에서 이 거리 안 (골에어리어 5.5m + 여유)
    goal_area_y_m: float = 12.0         # 골킥: 골문 중심에서 이 거리 안
    kickoff_m: float = 2.5              # 킥오프: 센터 스폿 반경


@dataclass
class Spell:
    tid: str
    team: str
    t0: float
    t1: float
    x0: float
    y0: float
    x1: float
    y1: float
    s: float = 1.0

    @property
    def duration(self) -> float:
        return max(0.0, self.t1 - self.t0)


@dataclass
class Event:
    kind: str                       # possession | pass | shot | recovery | set_piece
    t: float
    team: str
    player: str
    t_end: Optional[float] = None
    target: Optional[str] = None    # 패스 받은/가로챈 선수, 탈취당한 선수
    x: float = 0.0
    y: float = 0.0
    x2: Optional[float] = None
    y2: Optional[float] = None
    success: Optional[bool] = None
    detail: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        d = asdict(self)
        for k in ("t", "t_end", "x", "y", "x2", "y2"):
            if d.get(k) is not None:
                d[k] = round(d[k], 2)
        return d


def _dist(ax: float, ay: float, bx: float, by: float) -> float:
    return math.hypot(ax - bx, ay - by)


# ── 공격 방향 ────────────────────────────────────────────────────────────────

def resolve_attack_direction(frames: Sequence[Frame], cfg: EventConfig) -> Dict[str, Dict[str, int]]:
    """{'first': {'home': +1|-1, 'away': ...}, 'second': {...}}. +1 = x 증가 방향(오른쪽 골)을 공격."""
    def from_positions(fs: Sequence[Frame]) -> Dict[str, int]:
        sums = {"home": [0.0, 0], "away": [0.0, 0]}
        for f in fs:
            for p in f.players:
                if p.team in sums:
                    sums[p.team][0] += p.x
                    sums[p.team][1] += 1
        hm = sums["home"][0] / sums["home"][1] if sums["home"][1] else cfg.length / 2
        am = sums["away"][0] / sums["away"][1] if sums["away"][1] else cfg.length / 2
        # 자기 진영(골문 쪽)에 더 오래 머무르므로 평균 x가 작은 팀이 왼쪽 골을 지키고 오른쪽을 공격
        home_dir = 1 if hm <= am else -1
        return {"home": home_dir, "away": -home_dir}

    if cfg.home_attacks in ("right", "left"):
        d = 1 if cfg.home_attacks == "right" else -1
        first = {"home": d, "away": -d}
    else:
        fs1 = [f for f in frames if cfg.halftime_s is None or f.t < cfg.halftime_s]
        first = from_positions(fs1)
    second = {"home": -first["home"], "away": -first["away"]} if cfg.halftime_s is not None else dict(first)
    return {"first": first, "second": second}


def attack_dir(dirs: Dict[str, Dict[str, int]], team: str, t: float, cfg: EventConfig) -> int:
    half = "second" if cfg.halftime_s is not None and t >= cfg.halftime_s else "first"
    return dirs[half][team]


# ── 1) 점유 구간 ─────────────────────────────────────────────────────────────

def _controlled(t0: float, start: Tuple[float, float], t: float, bx: float, by: float,
                s: float, cfg: EventConfig) -> bool:
    """확인 구간 동안 공이 느리게 움직였는가 — 빠르게 스쳐 지나간 공은 점유가 아니다."""
    if t <= t0:
        return True
    return _dist(start[0], start[1], bx, by) * s / (t - t0) <= cfg.max_control_speed


def detect_spells(frames: Sequence[Frame], cfg: EventConfig) -> List[Spell]:
    spells: List[Spell] = []
    owner: Optional[PlayerObs] = None
    own_t0 = own_last = 0.0
    own_start = (0.0, 0.0)
    own_end = (0.0, 0.0)
    pending: Optional[PlayerObs] = None
    pend_t0 = 0.0
    pend_n = 0
    pend_start = (0.0, 0.0)

    def close(t_end: float):
        nonlocal owner
        if owner is not None and own_last - own_t0 >= 0:
            spells.append(Spell(owner.tid, owner.team, own_t0, max(own_t0, t_end),
                                own_start[0], own_start[1], own_end[0], own_end[1], owner.s))
        owner = None

    for f in frames:
        if f.ball is None:
            if owner is not None:
                # 공이 발밑에 가려진 경우가 흔하다: 점유자가 계속 보이면 더 오래 기다린다.
                owner_visible = any(p.tid == owner.tid for p in f.players)
                limit = cfg.hold_unknown_owner_s if owner_visible else cfg.hold_unknown_s
                if f.t - own_last > limit:
                    close(own_last)
            pending, pend_n = None, 0
            continue
        bx, by = f.ball
        by_id = {p.tid: p for p in f.players}
        cand: Optional[PlayerObs] = None
        best = float("inf")
        for p in f.players:
            d = _dist(p.x, p.y, bx, by)
            if d <= p.r and d / p.r < best:
                best, cand = d / p.r, p

        if owner is not None:
            cur = by_id.get(owner.tid)
            if cand is not None and cand.tid == owner.tid:
                own_last, own_end, owner = f.t, (bx, by), cand
                pending, pend_n = None, 0
                continue
            # 점유자가 잠깐 검출에서 빠진 경우는 시간 기준(lose_s)으로만 판단한다.
            far = cur is not None and _dist(cur.x, cur.y, bx, by) > cfg.release_factor * cur.r
            if cand is None:
                if far or f.t - own_last > cfg.lose_s:
                    close(own_last)
                continue
            # 다른 선수가 공 근처 — 확인되면 소유권 이전
            if pending is not None and pending.tid == cand.tid and _controlled(pend_t0, pend_start, f.t, bx, by, cand.s, cfg):
                pend_n += 1
            else:
                pending, pend_t0, pend_n, pend_start = cand, f.t, 1, (bx, by)
            if pend_n >= 2 and f.t - pend_t0 >= cfg.confirm_s - 1e-9:
                close(own_last)
                owner, own_t0, own_last, own_start, own_end = cand, pend_t0, f.t, pend_start, (bx, by)
                pending, pend_n = None, 0
            continue

        # 주인 없음
        if cand is None:
            pending, pend_n = None, 0
            continue
        if pending is not None and pending.tid == cand.tid and _controlled(pend_t0, pend_start, f.t, bx, by, cand.s, cfg):
            pend_n += 1
        else:
            pending, pend_t0, pend_n, pend_start = cand, f.t, 1, (bx, by)
        if pend_n >= 2 and f.t - pend_t0 >= cfg.confirm_s - 1e-9:
            owner, own_t0, own_last, own_start, own_end = cand, pend_t0, f.t, pend_start, (bx, by)
            pending, pend_n = None, 0

    if owner is not None:
        close(own_last)
    return merge_same_player(spells, cfg)


def merge_same_player(spells: List[Spell], cfg: EventConfig) -> List[Spell]:
    # A, (짧은 B), A 형태에서 B는 공이 스친 잡음으로 보고 제거한다.
    cleaned: List[Spell] = []
    for i, sp in enumerate(spells):
        if 0 < i < len(spells) - 1:
            prev, nxt = spells[i - 1], spells[i + 1]
            if (sp.duration < cfg.blip_max_s and prev.tid == nxt.tid != sp.tid
                    and nxt.t0 - prev.t1 <= cfg.merge_same_s + cfg.blip_max_s):
                continue
        cleaned.append(sp)
    out: List[Spell] = []
    for sp in cleaned:
        if out and out[-1].tid == sp.tid and sp.t0 - out[-1].t1 <= cfg.merge_same_s + cfg.blip_max_s:
            prev = out[-1]
            prev.t1, prev.x1, prev.y1 = sp.t1, sp.x1, sp.y1
        else:
            out.append(Spell(**asdict(sp)))
    return out


# ── 2) 이벤트 ────────────────────────────────────────────────────────────────

class _BallIndex:
    """시간 구간 질의를 이분 탐색으로 처리 (90분 경기에서도 빠르게)."""

    def __init__(self, frames: Sequence[Frame]):
        self.times = [f.t for f in frames]
        self.balls = [f.ball for f in frames]

    def path(self, t0: float, t1: float) -> List[Tuple[float, float, float]]:
        i, j = bisect_left(self.times, t0), bisect_right(self.times, t1)
        return [(self.times[k], b[0], b[1]) for k in range(i, j) if (b := self.balls[k]) is not None]

    def unknown_time(self, t0: float, t1: float) -> float:
        i, j = bisect_right(self.times, t0), bisect_left(self.times, t1)
        unknown, prev_t = 0.0, t0
        for k in range(i, j):
            if self.balls[k] is None:
                unknown += self.times[k] - prev_t
            prev_t = self.times[k]
        return unknown

    def release_velocity(self, t_rel: float, window: float = 0.6) -> Optional[Tuple[float, float]]:
        pts = self.path(t_rel, t_rel + window)
        if len(pts) < 2:
            return None
        (ta, xa, ya), (tb, xb, yb) = pts[0], pts[-1]
        if tb - ta <= 1e-6:
            return None
        return (xb - xa) / (tb - ta), (yb - ya) / (tb - ta)


def _shot_check(a: Spell, bi: _BallIndex, cfg: EventConfig, direction: int
                ) -> Optional[Dict]:
    """빠르고 골문 쪽(각도 40° 이내)으로 향한 킥이면 슈팅 정보를, 아니면 None."""
    vel = bi.release_velocity(a.t1, window=0.4)
    if vel is None:
        return None
    vx, vy = vel
    speed = math.hypot(vx, vy) * a.s
    if speed < cfg.shot_speed:
        return None
    goal_x = cfg.length if direction > 0 else 0.0
    goal_y = cfg.width / 2
    gx, gy = goal_x - a.x1, goal_y - a.y1
    gdist = math.hypot(gx, gy)
    if gdist * a.s > cfg.shot_max_dist or gdist < 1e-6:
        return None
    half = cfg.goal_width / 2
    y_cross = None
    if vx * direction > 1e-6:
        # 골라인 쪽으로 가는 공: 연장선이 골문 근처를 지나야 슈팅 (아니면 크로스/롱패스)
        y_cross = a.y1 + vy * (goal_x - a.x1) / vx
        if abs(y_cross - goal_y) > half + 4.0:
            return None
    else:
        # 골라인 옆 좁은 각도: 공이 골문 중심 쪽(40° 이내)으로 향해야 슈팅
        cos = (vx * gx + vy * gy) / (math.hypot(vx, vy) * gdist)
        if cos < math.cos(math.radians(40)):
            return None
    return {"speed_ms": round(speed, 1), "y_cross": y_cross, "goal_x": goal_x, "goal_y": goal_y, "half": half}


def shot_xg(x: float, y: float, goal_x: float, goal_y: float, goal_width_m: float, s: float = 1.0) -> float:
    """간이 xG — 거리·골문 각도 로지스틱 모형.

    대략 페널티 지점 0.2, 골 에어리어 정면 0.5, 페널티 박스 경계 0.07, 25m 0.03 수준.
    머리/발, 수비 위치, 패스 유형은 반영하지 못하므로 참고 지표다.
    """
    half = goal_width_m / 2 / max(s, 1e-6)
    dx, dy = abs(goal_x - x), y - goal_y
    d_m = math.hypot(dx, dy) * s
    angle = math.atan2(2 * half * dx, dx * dx + dy * dy - half * half)
    if angle < 0:
        angle += math.pi
    logit = -0.4 - 0.14 * d_m + 1.0 * angle
    return round(1 / (1 + math.exp(-logit)), 3)


def _went_out(bi: _BallIndex, t0: float, t1: float, cfg: EventConfig, s: float) -> bool:
    m = 1.0 / max(s, 1e-6)  # 1m를 좌표 단위로
    return any(x < -m or x > cfg.length + m or y < -m or y > cfg.width + m for _, x, y in bi.path(t0, t1))


def _ball_still_before(bi: _BallIndex, t: float, cfg: EventConfig, s: float) -> Optional[float]:
    """t 직전에 공이 멈춰 있던 시간(초). 멈춰 있지 않았거나 공을 거의 못 봤으면 None."""
    pts = bi.path(t - cfg.still_s, t)
    if len(pts) < 3 or pts[-1][0] - pts[0][0] < cfg.still_s * 0.6:
        return None
    xs, ys = [p[1] for p in pts], [p[2] for p in pts]
    spread = math.hypot(max(xs) - min(xs), max(ys) - min(ys)) * s
    return round(pts[-1][0] - pts[0][0], 2) if spread <= cfg.still_m else None


def _set_piece_type(a: Spell, prev: Optional[Spell], cfg: EventConfig, direction: int) -> Optional[str]:
    """재개 종류. 보정이 없으면 위치를 믿을 수 없어 'restart'로만 기록한다."""
    if not cfg.calibrated:
        return "restart"
    x, y = a.x0 * a.s, a.y0 * a.s
    L, W = cfg.length, cfg.width
    if prev is None and math.hypot(x - L / 2, y - W / 2) <= cfg.kickoff_m:
        return "kick_off"
    if min(x, L - x) <= cfg.corner_m and min(y, W - y) <= cfg.corner_m:
        return "corner"
    own_goal_x = 0.0 if direction > 0 else L          # 공격 방향의 반대편이 우리 골문
    if abs(x - own_goal_x) <= cfg.goal_area_x_m and abs(y - W / 2) <= cfg.goal_area_y_m:
        return "goal_kick"
    if min(y, W - y) <= cfg.touchline_m:
        return "throw_in"
    return "free_kick"


def detect_set_pieces(spells: Sequence[Spell], bi: _BallIndex, cfg: EventConfig,
                      dirs: Dict[str, Dict[str, int]]) -> List[Event]:
    """경기 중단 뒤 다시 시작하는 점유 = 세트피스.

    중단 근거: 직전 점유 뒤 공이 밖으로 나갔거나, 점유 공백이 dead_gap_s 이상이거나,
    공이 still_s 동안 멈춰 있었다. 종류 구분(코너·골킥·스로인·프리킥·킥오프)은 보정된 경기장에서만 한다.
    """
    out: List[Event] = []
    for i, a in enumerate(spells):
        prev = spells[i - 1] if i else None
        still = _ball_still_before(bi, a.t0, cfg, a.s)
        # 경계를 못 믿으면(이동 카메라·미보정) '밖으로 나감'도 판정하지 않는다 — 화면 밖으로 사라진 공이 모두
        # 아웃으로 잡혀 경기 1(21분)에서 세트피스가 102개나 나왔다. 이때는 공이 멈췄던 재개만 센다.
        went_out = cfg.trust_bounds and prev is not None and _went_out(bi, prev.t1, a.t0, cfg, prev.s)
        gap = (a.t0 - prev.t1) if prev is not None else float("inf")
        if still is None and not went_out:
            continue
        if still is None and gap < cfg.dead_gap_s:
            continue
        kind = _set_piece_type(a, prev, cfg, attack_dir(dirs, a.team, a.t0, cfg))
        if kind is None:
            continue
        out.append(Event("set_piece", a.t0, a.team, a.tid, x=a.x0, y=a.y0,
                         detail={"type": kind, "still_s": still, "after_out": went_out, "gap_s": round(min(gap, 999), 2)}))
    return out


def detect_events(frames: Sequence[Frame], cfg: Optional[EventConfig] = None
                  ) -> Tuple[List[Event], List[Spell], Dict[str, Dict[str, int]]]:
    cfg = cfg or EventConfig()
    frames = sorted(frames, key=lambda f: f.t)
    dirs = resolve_attack_direction(frames, cfg)
    spells = detect_spells(frames, cfg)
    bi = _BallIndex(frames)
    events: List[Event] = []

    for sp in spells:
        events.append(Event("possession", sp.t0, sp.team, sp.tid, t_end=sp.t1,
                            x=sp.x0, y=sp.y0, x2=sp.x1, y2=sp.y1,
                            detail={"duration_s": round(sp.duration, 2)}))
    events.extend(detect_set_pieces(spells, bi, cfg, dirs))

    for i, a in enumerate(spells):
        b = spells[i + 1] if i + 1 < len(spells) else None
        direction = attack_dir(dirs, a.team, a.t1, cfg)
        gap = (b.t0 - a.t1) if b else float("inf")
        t_next = b.t0 if b and gap <= cfg.max_link_s else a.t1 + cfg.max_link_s

        # 슈팅은 킥 직후 궤적만으로 판정하므로, 공이 나중에 사라져도 판정한다.
        shot = _shot_check(a, bi, cfg, direction)
        if shot:
            events.append(_shot_event(a, b, bi, cfg, shot))
            continue
        if bi.unknown_time(a.t1, t_next) > cfg.max_unknown_s:
            continue  # 공이 오래 안 보였다 — 추측하지 않는다

        vel = bi.release_velocity(a.t1)
        rel_speed = math.hypot(*vel) * a.s if vel else 0.0
        travel_m = _dist(a.x1, a.y1, b.x0, b.y0) * a.s if b else 0.0
        kicked = rel_speed >= cfg.kick_speed or travel_m >= 5.0

        if cfg.trust_bounds and _went_out(bi, a.t1, t_next, cfg, a.s):
            if kicked or b is None or b.team != a.team:
                events.append(Event("pass", a.t1, a.team, a.tid, x=a.x1, y=a.y1,
                                    success=False, detail={"reason": "out"}))
            continue

        if b is None or gap > cfg.max_link_s:
            # 경계를 믿을 수 없으면(이동 카메라·미보정) 공이 시야 밖으로 나간 것인지 알 수 없다 — 추측하지 않는다.
            if rel_speed >= cfg.kick_speed and cfg.trust_bounds:
                events.append(Event("pass", a.t1, a.team, a.tid, x=a.x1, y=a.y1,
                                    success=False, detail={"reason": "out"}))
            continue

        if b.team == a.team:
            if b.tid == a.tid or travel_m < cfg.pass_min_m:
                continue
            forward = (b.x0 - a.x1) * direction * a.s >= cfg.forward_pass_m
            events.append(Event("pass", a.t1, a.team, a.tid, t_end=b.t0, target=b.tid,
                                x=a.x1, y=a.y1, x2=b.x0, y2=b.y0, success=True,
                                detail={"reason": "completed", "length_m": round(travel_m, 1),
                                        "forward": forward}))
        elif travel_m >= cfg.intercept_min_m or gap >= cfg.intercept_min_gap_s:
            events.append(Event("pass", a.t1, a.team, a.tid, t_end=b.t0, target=b.tid,
                                x=a.x1, y=a.y1, x2=b.x0, y2=b.y0, success=False,
                                detail={"reason": "intercepted", "length_m": round(travel_m, 1)}))
        else:
            events.append(Event("recovery", b.t0, b.team, b.tid, target=a.tid, x=b.x0, y=b.y0))
    events.sort(key=lambda e: (e.t, e.kind != "possession"))
    return events, spells, dirs


def _shot_event(a: Spell, b: Optional[Spell], bi: _BallIndex, cfg: EventConfig, shot: Dict) -> Event:
    goal_x, goal_y, half = shot["goal_x"], shot["goal_y"], shot["half"]
    t_stop = b.t0 if b and b.t0 - a.t1 <= cfg.max_link_s else a.t1 + 3.0
    path = bi.path(a.t1, t_stop)
    reached = next(((t, x, y) for t, x, y in path if abs(x - goal_x) * a.s <= 1.5), None)
    blocked = (b is not None and b.team != a.team and b.t0 - a.t1 <= 2.0
               and abs(b.x0 - goal_x) * a.s > cfg.blocked_min_goal_dist)
    if reached is not None and not blocked:
        outcome = "on_target" if abs(reached[2] - goal_y) <= half + 0.5 else "off_target"
        end = (reached[1], reached[2])
    elif blocked:
        outcome, end = "blocked", (b.x0, b.y0)
    elif shot["y_cross"] is not None:
        outcome = "on_target" if abs(shot["y_cross"] - goal_y) <= half + 0.5 else "off_target"
        end = (goal_x, shot["y_cross"])
    else:
        outcome, end = "off_target", (a.x1, a.y1)
    return Event("shot", a.t1, a.team, a.tid, t_end=reached[0] if reached else t_stop,
                 target=b.tid if b and b.team != a.team else None,
                 x=a.x1, y=a.y1, x2=end[0], y2=end[1], success=outcome == "on_target",
                 detail={"outcome": outcome, "speed_ms": shot["speed_ms"],
                         "xg": shot_xg(a.x1, a.y1, goal_x, goal_y, cfg.goal_width, a.s)})


def fit_to_extents(frames: Sequence[Frame], length: float, width: float, q: float = 1.0,
                   center_x: Optional[float] = None) -> List[Frame]:
    """보정 없이 카메라가 움직인 영상: 선수들이 실제로 있던 범위(양 끝 q%)를 경기장 크기로 맞춘다.

    안정화 좌표는 첫 장면 화면 기준이라 카메라가 돈 만큼 경기장이 늘어나 있다 (경기 1: 가로 약 1.6배).
    그대로 쓰면 골대 거리·슈팅 거리가 틀려 xG가 7을 넘는 등 말이 안 되는 값이 나온다.
    좌표·점유 반경은 같이 늘리고 줄이며, 국소 축척(m/단위)은 반대로 고쳐 실제 거리는 그대로 둔다.
    center_x(하프라인 위치)를 주면 그곳을 경기장 가운데로 두고 양쪽으로 같은 축척을 쓴다 — 선수가 한쪽 진영에
    몰려 있던 영상에서 범위만으로 맞추면 가운데가 밀려, 한쪽 골대로만 슈팅이 잡혔다 (경기 1: 홈 0 : 원정 28).
    """
    xs = [p.x for f in frames for p in f.players]
    ys = [p.y for f in frames for p in f.players]
    if len(xs) < 50:
        return list(frames)
    import numpy as np
    # numpy 값이 섞이면 비교 결과가 numpy bool이 되어 결과 JSON 저장이 깨진다 → 파이썬 float로
    x0, x1 = (float(v) for v in np.percentile(xs, [q, 100 - q]))
    y0, y1 = (float(v) for v in np.percentile(ys, [q, 100 - q]))
    if x1 - x0 < 1e-6 or y1 - y0 < 1e-6:
        return list(frames)
    if center_x is not None and x0 < center_x < x1:
        half = max(center_x - x0, x1 - center_x)
        x0, x1 = center_x - half, center_x + half
    kx, ky = length / (x1 - x0), width / (y1 - y0)
    k = (kx + ky) / 2
    out = []
    for f in frames:
        ps = [PlayerObs(p.tid, p.team, (p.x - x0) * kx, (p.y - y0) * ky, p.r * k, p.s / k) for p in f.players]
        ball = ((f.ball[0] - x0) * kx, (f.ball[1] - y0) * ky) if f.ball else None
        out.append(Frame(f.t, ps, ball))
    return out
