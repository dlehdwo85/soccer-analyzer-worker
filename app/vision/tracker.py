"""선수 추적: IoU 기반 2단계 매칭 트래커 + 끊긴 트랙(tracklet) 이어붙이기.

ByteTrack의 핵심 아이디어(고신뢰 검출 먼저 매칭, 남은 트랙에 저신뢰 검출 재매칭)를
의존성 없이 구현했다. 등속 예측으로 빠른 선수도 놓치지 않게 하고, 매칭이 안 되면
중심 거리(박스 높이 대비)로 한 번 더 시도한다.

가림으로 ID가 끊기는 문제는 stitch_tracklets()로 완화한다: 같은 팀이고, 끊긴 시간
안에 달려갈 수 있는 거리에서 다시 나타난 트랙을 하나로 잇는다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

Box = Tuple[float, float, float, float]  # x1, y1, x2, y2 (px)


def iou(a: Box, b: Box) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


@dataclass
class Detection:
    box: Box
    conf: float
    payload: object = None  # 색상 샘플 등 호출자가 붙이는 데이터


@dataclass
class _Track:
    tid: int
    box: Box
    vel: Tuple[float, float] = (0.0, 0.0)  # 중심 이동 px/frame
    hits: int = 1
    missed: int = 0
    confirmed: bool = False
    team_votes: Dict[str, float] = field(default_factory=dict)

    def team(self, min_votes: float = 2.0) -> Optional[str]:
        if not self.team_votes:
            return None
        lbl, v = max(self.team_votes.items(), key=lambda kv: kv[1])
        total = sum(self.team_votes.values())
        return lbl if v >= min_votes and v / total >= 0.6 else None

    def predicted(self) -> Box:
        vx, vy = self.vel
        k = self.missed + 1
        x1, y1, x2, y2 = self.box
        return (x1 + vx * k, y1 + vy * k, x2 + vx * k, y2 + vy * k)

    def update(self, box: Box):
        ocx, ocy = (self.box[0] + self.box[2]) / 2, (self.box[1] + self.box[3]) / 2
        ncx, ncy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        steps = self.missed + 1
        nvx, nvy = (ncx - ocx) / steps, (ncy - ocy) / steps
        self.vel = (0.6 * self.vel[0] + 0.4 * nvx, 0.6 * self.vel[1] + 0.4 * nvy)
        self.box = box
        self.hits += 1
        self.missed = 0


@dataclass
class TrackPoint:
    frame: int
    box: Box
    conf: float
    payload: object = None


class IoUTracker:
    """프레임마다 update(dets) 호출 → [(track_id, Detection)] 반환(확정 트랙만).

    team_of(payload) → (팀, 신뢰도) 를 주면 팀이 확실히 다른 검출과는 매칭하지 않는다.
    다른 팀 선수가 엇갈려 지나갈 때 ID가 바뀌는 문제를 막는다.
    """

    def __init__(self, high_conf: float = 0.45, iou_match: float = 0.2,
                 max_missed: int = 15, min_hits: int = 2, center_gate: float = 0.6,
                 team_of: Optional[Callable[[object], Tuple[Optional[str], float]]] = None):
        self.team_of = team_of
        self.high_conf = high_conf
        self.iou_match = iou_match
        self.max_missed = max_missed
        self.min_hits = min_hits
        self.center_gate = center_gate  # 박스 높이 대비 허용 중심 거리
        self.tracks: Dict[int, _Track] = {}
        self._next = 1
        self.history: Dict[int, List[TrackPoint]] = {}
        self._pending: Dict[int, List[TrackPoint]] = {}

    def _team_conflict(self, tr: "_Track", det: Detection) -> bool:
        if self.team_of is None:
            return False
        t_team = tr.team()
        d_team, d_conf = self.team_of(det.payload)
        return bool(t_team and d_team and d_conf >= 0.5 and d_team != t_team)

    def _match(self, track_ids: List[int], dets: List[Tuple[int, Detection]]):
        pairs = []
        for tid in track_ids:
            tr = self.tracks[tid]
            pb = tr.predicted()
            ph = max(1.0, pb[3] - pb[1])
            pcx, pcy = (pb[0] + pb[2]) / 2, (pb[1] + pb[3]) / 2
            for di, det in dets:
                if self._team_conflict(tr, det):
                    continue
                s = iou(pb, det.box)
                if s >= self.iou_match:
                    pairs.append((s + 1.0, tid, di))  # IoU 매칭 우선
                    continue
                dcx, dcy = (det.box[0] + det.box[2]) / 2, (det.box[1] + det.box[3]) / 2
                dist = ((pcx - dcx) ** 2 + (pcy - dcy) ** 2) ** 0.5 / ph
                if dist <= self.center_gate:
                    pairs.append((1.0 - dist / self.center_gate, tid, di))
        pairs.sort(key=lambda p: -p[0])
        used_t, used_d, out = set(), set(), []
        for _, tid, di in pairs:
            if tid in used_t or di in used_d:
                continue
            used_t.add(tid)
            used_d.add(di)
            out.append((tid, di))
        return out, used_t, used_d

    def update(self, frame: int, detections: Sequence[Detection]) -> List[Tuple[int, Detection]]:
        indexed = list(enumerate(detections))
        high = [(i, d) for i, d in indexed if d.conf >= self.high_conf]
        low = [(i, d) for i, d in indexed if d.conf < self.high_conf]

        active = list(self.tracks.keys())
        m1, used_t, used_d = self._match(active, high)
        remaining = [t for t in active if t not in used_t]
        m2, used_t2, used_d2 = self._match(remaining, low)
        matches = m1 + m2
        used_t |= used_t2
        used_d |= used_d2

        results: List[Tuple[int, Detection]] = []
        for tid, di in matches:
            det = detections[di]
            tr = self.tracks[tid]
            tr.update(det.box)
            self._vote(tr, det)
            self._record(tid, frame, det)
            if not tr.confirmed and tr.hits >= self.min_hits:
                tr.confirmed = True
                self.history.setdefault(tid, []).extend(self._pending.pop(tid, []))
            if tr.confirmed:
                results.append((tid, det))

        for tid in active:
            if tid not in used_t:
                tr = self.tracks[tid]
                tr.missed += 1
                if tr.missed > self.max_missed or (not tr.confirmed and tr.missed > 1):
                    del self.tracks[tid]
                    self._pending.pop(tid, None)

        for i, det in high:
            if i in used_d:
                continue
            tid = self._next
            self._next += 1
            self.tracks[tid] = _Track(tid=tid, box=det.box)
            self._vote(self.tracks[tid], det)
            self._pending[tid] = [TrackPoint(frame, det.box, det.conf, det.payload)]
            if self.min_hits <= 1:
                self.tracks[tid].confirmed = True
                self.history.setdefault(tid, []).extend(self._pending.pop(tid))
                results.append((tid, det))
        return results

    def _vote(self, tr: "_Track", det: Detection):
        if self.team_of is None:
            return
        team, conf = self.team_of(det.payload)
        if team and conf >= 0.3:
            tr.team_votes[team] = tr.team_votes.get(team, 0.0) + conf

    def _record(self, tid: int, frame: int, det: Detection):
        pt = TrackPoint(frame, det.box, det.conf, det.payload)
        if self.tracks[tid].confirmed:
            self.history.setdefault(tid, []).append(pt)
        else:
            self._pending.setdefault(tid, []).append(pt)


# ── tracklet 이어붙이기 ───────────────────────────────────────────────────────

@dataclass
class Tracklet:
    """경기장 좌표로 변환된 트랙 조각. points: [(t초, x, y)]."""
    tid: str
    team: str
    points: List[Tuple[float, float, float]]
    color_lab: Optional[np.ndarray] = None
    members: List[str] = field(default_factory=list)

    @property
    def start(self) -> Tuple[float, float, float]:
        return self.points[0]

    @property
    def end(self) -> Tuple[float, float, float]:
        return self.points[-1]


def stitch_tracklets(tracklets: Sequence[Tracklet], max_gap_s: float = 4.0,
                     max_speed: float = 8.0, slack_m: float = 3.0,
                     max_color_de: float = 30.0) -> Dict[str, str]:
    """끊긴 트랙을 이어 {원래 tid: 대표 tid} 매핑을 반환한다.

    A가 끝난 뒤 B가 시작하고, 같은 팀이며, 공백 시간 안에 이동 가능한 거리이고,
    (색 정보가 있으면) 색이 비슷할 때 잇는다. 비용이 작은 쌍부터 탐욕적으로 연결하고,
    한 조각은 앞·뒤로 각각 한 번만 연결된다.
    """
    items = sorted(tracklets, key=lambda t: t.start[0])
    cands = []
    for a in items:
        for b in items:
            if a.tid == b.tid or a.team != b.team:
                continue
            gap = b.start[0] - a.end[0]
            if gap <= 0 or gap > max_gap_s:
                continue
            dist = ((a.end[1] - b.start[1]) ** 2 + (a.end[2] - b.start[2]) ** 2) ** 0.5
            reach = max_speed * gap + slack_m
            if dist > reach:
                continue
            color_pen = 0.0
            if a.color_lab is not None and b.color_lab is not None:
                d = a.color_lab - b.color_lab
                cde = float(np.sqrt((0.5 * d[0]) ** 2 + d[1] ** 2 + d[2] ** 2))
                if cde > max_color_de:
                    continue
                color_pen = cde / max_color_de
            cost = dist / reach + gap / max_gap_s + 0.5 * color_pen
            cands.append((cost, a.tid, b.tid))
    cands.sort()

    nxt: Dict[str, str] = {}
    prv: Dict[str, str] = {}

    for _, a, b in cands:
        # 연결은 항상 시간상 앞→뒤이므로 사이클이 생기지 않는다.
        if a in nxt or b in prv:
            continue
        nxt[a] = b
        prv[b] = a

    mapping: Dict[str, str] = {}
    for t in items:
        head = t.tid
        while head in prv:
            head = prv[head]
        mapping[t.tid] = head
    return mapping


# ── 화면 밖으로 나갔다 돌아온 선수 후보 ───────────────────────────────────────

@dataclass
class _Chain:
    head: str
    team: str
    start: Tuple[float, float, float]
    end: Tuple[float, float, float]
    appearance: Optional[np.ndarray]
    n_app: int
    exit_edge: bool    # 마지막 조각이 화면 가장자리에서 끝났다(카메라가 돌아가며 화면 밖으로 나감)
    enter_edge: bool   # 첫 조각이 화면 가장자리에서 시작했다


def suggest_same_player(tracklets: Sequence[Tracklet], mapping: Dict[str, str],
                        appearance: Dict[str, Tuple[Optional[np.ndarray], int]],
                        exit_edge: Dict[str, bool], enter_edge: Dict[str, bool],
                        appearance_distance: Callable[[Optional[np.ndarray], Optional[np.ndarray]], Optional[float]],
                        k: int = 3, max_gap_s: float = 90.0, avg_speed: float = 6.0, slack_m: float = 8.0,
                        max_app_de: float = 14.0, edge_only_gap_s: float = 12.0) -> Dict[str, List[str]]:
    """stitch_tracklets로 이은 선수(체인)마다 '같은 사람일 수 있는' 다른 선수 후보를 k명까지 고른다.

    카메라가 반대편으로 돌아간 사이 화면 밖에 있던 선수는 수십 초 뒤 새 트랙으로 나타난다. 같은 팀 상의는
    다 같아서 머리·반바지·양말 색(appearance)과 공백 동안 갈 수 있는 거리로 후보를 매긴다.
    자동으로 잇지 않는 이유: 경기 1(21분)에서 일부러 끊은 트랙으로 재 보니 1순위가 맞는 경우가 절반뿐이었다
    (3순위 안에는 75~80%). 그래서 '선수 정리' 화면에서 사람이 썸네일을 보고 고르도록 후보만 준다.
    화면 가장자리로 나가서 가장자리로 들어온 경우가 아니면 edge_only_gap_s보다 긴 공백은 후보로 두지 않는다.
    반환: {대표 tid: [후보 대표 tid, ...]} (가까운 순)
    """
    groups: Dict[str, List[Tracklet]] = {}
    for t in tracklets:
        groups.setdefault(mapping.get(t.tid, t.tid), []).append(t)
    chains: List[_Chain] = []
    for head, mem in groups.items():
        mem.sort(key=lambda t: t.start[0])
        apps = [(appearance.get(t.tid) or (None, 0)) for t in mem]
        vecs = [a for a, n in apps if a is not None for _ in range(min(n, 50))]
        app = np.nanmedian(np.array(vecs), axis=0) if vecs else None
        chains.append(_Chain(head, mem[0].team, mem[0].start, mem[-1].end, app, sum(n for _, n in apps),
                             exit_edge.get(mem[-1].tid, False), enter_edge.get(mem[0].tid, False)))

    scored: Dict[str, List[Tuple[float, str]]] = {}
    for a in chains:
        for b in chains:
            if a is b or a.team != b.team:
                continue
            gap = b.start[0] - a.end[0]
            if gap <= 0 or gap > max_gap_s:
                continue
            if gap > edge_only_gap_s and not (a.exit_edge and b.enter_edge):
                continue
            dist = ((a.end[1] - b.start[1]) ** 2 + (a.end[2] - b.start[2]) ** 2) ** 0.5
            reach = avg_speed * gap + slack_m
            if dist > reach:
                continue
            d = appearance_distance(a.appearance, b.appearance)
            if d is None or d > max_app_de:
                continue
            cost = d / max_app_de + 0.5 * dist / reach + 0.3 * gap / max_gap_s
            scored.setdefault(a.head, []).append((cost, b.head))
            scored.setdefault(b.head, []).append((cost, a.head))
    return {h: [o for _, o in sorted(lst)[:k]] for h, lst in scored.items()}


# ── 팀당 정원(필드 10 + GK 1)에 맞춰 선수 칸 채우기 ─────────────────────────────

@dataclass
class RosterPiece:
    """칸에 담을 선수 조각(이어 붙인 체인 하나)."""
    head: str
    team: str
    gk: bool
    t0: float
    t1: float
    p0: Tuple[float, float]          # 시작·끝 위치 (경기장 m, 또는 화면 픽셀 → pos_scale로 m 환산)
    p1: Tuple[float, float]
    appearance: Optional[np.ndarray]
    lateral: float                   # 평균 옆 위치(m) — 왼쪽·오른쪽 자리
    pos_scale: float = 1.0           # p0·p1 단위 → m
    pos_max_gap_s: float = float("inf")  # 이보다 긴 공백에서는 위치를 비교하지 않는다(화면 좌표는 카메라가 돌면 무의미)

    @property
    def seconds(self) -> float:
        return self.t1 - self.t0


@dataclass
class _Slot:
    pieces: List[RosterPiece] = field(default_factory=list)
    apps: List[Tuple[np.ndarray, float]] = field(default_factory=list)
    lat_sum: Tuple[float, float] = (0.0, 0.0)

    def appearance(self) -> Optional[np.ndarray]:
        if not self.apps:
            return None
        return np.average(np.array([a for a, _ in self.apps]), axis=0, weights=[w for _, w in self.apps])

    def lateral(self) -> float:
        s, w = self.lat_sum
        return s / w if w else 0.0

    def add(self, p: RosterPiece) -> None:
        self.pieces.append(p)
        w = max(p.seconds, 0.1)
        if p.appearance is not None and np.isfinite(p.appearance).any():
            self.apps.append((p.appearance, w))
        self.lat_sum = (self.lat_sum[0] + p.lateral * w, self.lat_sum[1] + w)


def pieces_from_tracklets(tracklets: Sequence[Tracklet], mapping: Dict[str, str],
                          appearance: Dict[str, Tuple[Optional[np.ndarray], int]],
                          is_gk: Dict[str, bool]) -> List[RosterPiece]:
    groups: Dict[str, List[Tracklet]] = {}
    for t in tracklets:
        groups.setdefault(mapping.get(t.tid, t.tid), []).append(t)
    out = []
    for head, mem in groups.items():
        pts = sorted((p for t in mem for p in t.points), key=lambda p: p[0])
        apps = [appearance.get(t.tid) or (None, 0) for t in mem]
        vecs = [a for a, n in apps if a is not None for _ in range(min(n, 50))]
        gk_n = sum(len(t.points) for t in mem if is_gk.get(t.tid))
        out.append(RosterPiece(head, mem[0].team, gk_n * 2 > len(pts), pts[0][0], pts[-1][0],
                               (pts[0][1], pts[0][2]), (pts[-1][1], pts[-1][2]),
                               np.nanmedian(np.array(vecs), axis=0) if vecs else None,
                               float(np.mean([p[2] for p in pts]))))
    return out


def assign_roster_slots(pieces: Sequence[RosterPiece],
                        appearance_distance: Callable[[Optional[np.ndarray], Optional[np.ndarray]], Optional[float]],
                        field_slots: int = 10, gk_slots: int = 1, min_seconds: float = 2.0,
                        overlap_tol_s: float = 0.5, avg_speed: float = 6.0, slack_m: float = 8.0,
                        max_app_de: float = 25.0, open_cost: float = 1.0, batch_s: float = 1.0
                        ) -> Tuple[Dict[str, str], List[str]]:
    """경기장 안에는 팀마다 필드 선수 10명 + GK 1명뿐이다. 선수 조각을 시간 순으로 보며 팀·역할별 정원만큼의
    칸에 담아, 화면 밖에 나갔다 돌아와 새 번호가 붙은 사람을 '지금 비어 있는 칸'(화면 밖에 나가 있는 선수) 중
    하나로 모은다.

    - 같은 칸의 조각은 시간이 겹치면 안 된다(동시에 두 곳에 있을 수 없다).
    - 공백 동안 평균 속도로 갈 수 있는 거리여야 한다(위치를 비교할 수 있을 때).
    - 비용 = 머리·반바지·양말 색 차이 + 옆 위치(왼쪽·오른쪽 자리) 차이 + 이동 거리.
    - 골키퍼를 따로 알아보지 못한 팀은 GK까지 필드 칸(10 + 1)에 담는다.
    - 정원이 차기 전에는 잘 맞는 칸이 없으면(open_cost 초과) 새 칸을 연다. 정원이 차면 빈 칸 중 가장 나은 곳에
      담고, 빈 칸이 없으면(동시에 정원보다 많이 보임) 정원 초과 — 팀을 잘못 본 심판·라인 밖 인원·중복 검출이다.
    정확도 한계: 같은 유니폼이라 색 구분이 약해, 오래 화면 밖에 있던 선수는 틀리게 담길 수 있다.
    그래서 결과는 '합치기'로 저장해 선수 정리 화면에서 되돌릴 수 있게 한다.
    반환: ({조각 head: 칸 대표 head}, 정원 초과 head 목록)
    """
    out: Dict[str, str] = {}
    extra: List[str] = []
    for team in sorted({p.team for p in pieces}):
        # 골키퍼를 따로 못 알아본 팀(GK 색 미지정)은 GK도 필드 칸에 들어온다
        has_gk = any(p.gk for p in pieces if p.team == team)
        for gk, cap in ((True, gk_slots), (False, field_slots + (0 if has_gk else gk_slots))):
            mine = sorted((p for p in pieces if p.team == team and p.gk == gk and p.seconds >= min_seconds),
                          key=lambda p: p.t0)
            slots: List[_Slot] = []
            i = 0
            while i < len(mine):
                j = i
                while j < len(mine) and mine[j].t0 - mine[i].t0 <= batch_s:
                    j += 1
                batch, i = mine[i:j], j
                pairs = []
                for bi, p in enumerate(batch):
                    for si, s in enumerate(slots):
                        last = s.pieces[-1]
                        if last.t1 > p.t0 + overlap_tol_s:
                            continue
                        gap = max(p.t0 - last.t1, 0.0)
                        move = 0.0
                        if gap <= min(p.pos_max_gap_s, last.pos_max_gap_s):
                            reach = avg_speed * gap + slack_m
                            d = float(np.hypot(last.p1[0] - p.p0[0], last.p1[1] - p.p0[1])) * p.pos_scale
                            if d > reach:
                                continue
                            move = d / reach
                        da = appearance_distance(s.appearance(), p.appearance)
                        cost = (0.5 if da is None else min(da / max_app_de, 1.5))                             + abs(s.lateral() - p.lateral) / 15.0 + 0.5 * move
                        pairs.append((cost, bi, si))
                pairs.sort()
                used_b, used_s = set(), set()
                for cost, bi, si in pairs:
                    if bi in used_b or si in used_s or (len(slots) < cap and cost > open_cost):
                        continue
                    used_b.add(bi)
                    used_s.add(si)
                    slots[si].add(batch[bi])
                for bi, p in enumerate(batch):
                    if bi in used_b:
                        continue
                    if len(slots) < cap:
                        slots.append(_Slot())
                        slots[-1].add(p)
                    else:
                        extra.append(p.head)
            for s in slots:
                lead = max(s.pieces, key=lambda q: q.seconds).head
                for q in s.pieces:
                    out[q.head] = lead
    return out, extra
