"""공 궤적 추출.

YOLO의 sports ball 검출은 작고 자주 놓치며, 관중석 공·하얀 점을 잘못 잡기도 한다.
1) 직전 궤적에서 예측한 위치 근처 후보를 우선 선택(게이팅)
2) 앞뒤와 동떨어진 튀는 점 제거
3) 짧은 공백은 선형 보간 (긴 공백은 비워 둔다 — 이벤트 판정이 '모름'으로 처리)
4) 카메라가 공을 따라 돌아가는 동안의 공백은 화면 이동을 따라 잇는다
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

BallCand = Tuple[float, float, float]   # cx, cy (px), conf
BallPos = Optional[Tuple[float, float]]


def select_ball_positions(cands_per_frame: Sequence[Sequence[BallCand]], times: Sequence[float],
                          frame_w: float, min_conf_new: float = 0.3,
                          max_speed_frac: float = 0.45, base_gate_frac: float = 0.03,
                          memory_s: float = 1.0) -> List[BallPos]:
    """프레임별 후보 → 프레임별 공 위치(px) 또는 None."""
    out: List[BallPos] = []
    last: Optional[Tuple[float, float, float]] = None  # (t, x, y)
    vel = (0.0, 0.0)
    for cands, t in zip(cands_per_frame, times):
        chosen = None
        tracking = last is not None and t - last[0] <= memory_s
        if tracking:
            dt = t - last[0]
            px, py = last[1] + vel[0] * dt, last[2] + vel[1] * dt
            gate = max_speed_frac * frame_w * dt + base_gate_frac * frame_w
            best = None
            for cx, cy, conf in cands:
                d = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
                if d <= gate and (best is None or d - 50 * conf < best[0]):
                    best = (d - 50 * conf, cx, cy)
            if best:
                chosen = (best[1], best[2])
        # 최근 궤적이 살아 있으면 멀리 떨어진 후보로 갈아타지 않는다(가림 중 오검출 방지).
        if chosen is None and not tracking:
            strong = [c for c in cands if c[2] >= min_conf_new]
            if strong:
                cx, cy, _ = max(strong, key=lambda c: c[2])
                chosen = (cx, cy)
        if chosen is not None:
            if last is not None and t - last[0] <= memory_s and t > last[0]:
                dt = t - last[0]
                nv = ((chosen[0] - last[1]) / dt, (chosen[1] - last[2]) / dt)
                vel = (0.5 * vel[0] + 0.5 * nv[0], 0.5 * vel[1] + 0.5 * nv[1])
            else:
                vel = (0.0, 0.0)
            last = (t, chosen[0], chosen[1])
        out.append(chosen)
    return drop_excursions(remove_spikes(out, times, frame_w), times, frame_w)


def select_ball_positions_bidir(cands_per_frame: Sequence[Sequence[BallCand]], times: Sequence[float],
                                frame_w: float, min_conf_new: float = 0.3, memory_s: float = 1.0) -> List[BallPos]:
    """앞으로 한 번, 뒤로 한 번 추적해서 합친다.

    앞으로만 추적하면 공을 놓친 뒤 강한 후보(min_conf_new)가 나올 때까지 약한 후보를 못 쓴다. 뒤로 추적하면
    다시 잡힌 공에서 거꾸로 거슬러 올라가며 그 사이의 약한 후보를 궤적 근처에서만 받는다.
    두 방향이 같은 장면에 다른 점을 고르면 앞쪽 결과를 둔다. 합친 뒤 튀는 점·묶음은 다시 지운다."""
    fwd = select_ball_positions(cands_per_frame, times, frame_w, min_conf_new=min_conf_new, memory_s=memory_s)
    rev_t = [-t for t in reversed(times)]
    rev = list(reversed(select_ball_positions(list(reversed(cands_per_frame)), rev_t, frame_w,
                                              min_conf_new=min_conf_new, memory_s=memory_s)))
    merged = [a if a is not None else b for a, b in zip(fwd, rev)]
    return drop_excursions(remove_spikes(merged, times, frame_w), times, frame_w)


def in_play_candidates(balls: Sequence[BallCand], persons: Sequence[Tuple[Tuple[float, float, float, float], float]],
                       frame_h: float, margin_h: float = 3.0, top_frac: float = 0.04) -> List[BallCand]:
    """경기가 벌어지는 곳 밖의 공 후보를 버린다 (원래 화면 좌표).

    야간 경기 실측(바모스전): 공 후보의 약 30%가 벤치 옆 터치라인 밖에 놓인 여분의 공, 약 25%가 화면 맨 위 조명이었다.
    선수 발 위치의 10~90% 범위를 선수 키의 margin_h배만큼 넓힌 상자를 경기 영역으로 보고, 그 밖의 후보와
    화면 맨 위(top_frac) 후보를 버린다. 벤치·코치 몇 명은 백분위로 걸러진다. 사람이 거의 없으면 그대로 둔다.
    """
    if len(persons) < 4:
        return [b for b in balls if b[1] >= top_frac * frame_h]
    xs = np.array([(b[0] + b[2]) / 2 for b, _ in persons])
    ys = np.array([b[3] for b, _ in persons])
    h = float(np.median([b[3] - b[1] for b, _ in persons]))
    x0, x1 = np.percentile(xs, 10) - margin_h * h, np.percentile(xs, 90) + margin_h * h
    y0, y1 = np.percentile(ys, 10) - margin_h * h, np.percentile(ys, 90) + margin_h * h
    return [b for b in balls if x0 <= b[0] <= x1 and y0 <= b[1] <= y1 and b[1] >= top_frac * frame_h]


def drop_weak_at_feet(balls: Sequence[BallCand], persons: Sequence[Tuple[Tuple[float, float, float, float], float]],
                      weak_below: float = 0.15) -> List[BallCand]:
    """신뢰도 weak_below 미만 후보 중 선수 발 자리(상자 아래 20%~아래로 10%)에 있는 것을 버린다.

    밤 경기 2쿼터 실측: 0.07~0.15 후보로 새로 잡힌 점 37개 중 11개가 걷는 선수의 흰 축구화였고 모두 발 자리였다.
    이 규칙을 넣으면 새 점 19개 중 틀림 4개(79% 정확). 강한 후보(발밑 드리블 공)는 그대로 둔다."""
    out = []
    for b in balls:
        if b[2] < weak_below:
            feet = False
            for (x1, y1, x2, y2), _ in persons:
                h, w = y2 - y1, x2 - x1
                if x1 - 0.15 * w <= b[0] <= x2 + 0.15 * w and y2 - 0.2 * h <= b[1] <= y2 + 0.1 * h:
                    feet = True
                    break
            if feet:
                continue
        out.append(b)
    return out


def remove_spikes(pos: List[BallPos], times: Sequence[float], frame_w: float,
                  spike_frac: float = 0.12) -> List[BallPos]:
    """앞뒤 점은 서로 가까운데 가운데만 멀리 튄 점을 제거한다."""
    out = list(pos)
    idx = [i for i, p in enumerate(pos) if p is not None]
    thr = spike_frac * frame_w
    for k in range(1, len(idx) - 1):
        i0, i1, i2 = idx[k - 1], idx[k], idx[k + 1]
        if times[i2] - times[i0] > 1.0:
            continue
        a, b, c = pos[i0], pos[i1], pos[i2]
        dac = ((a[0] - c[0]) ** 2 + (a[1] - c[1]) ** 2) ** 0.5
        dab = ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5
        dbc = ((b[0] - c[0]) ** 2 + (b[1] - c[1]) ** 2) ** 0.5
        if dab > thr and dbc > thr and dac < 0.5 * min(dab, dbc):
            out[i1] = None
    return out


def drop_excursions(pos: List[BallPos], times: Sequence[float], frame_w: float, max_len_s: float = 1.5,
                    speed_frac: float = 0.6, ret_ratio: float = 0.5, passes: int = 3) -> List[BallPos]:
    """짧게 딴 곳으로 튀었다가 돌아오는 묶음을 지운다 (remove_spikes는 한 점짜리만 지운다).

    가짜 공(신발·옆에 놓인 공)을 몇 장면 연속으로 잡으면 궤적이 그쪽으로 갔다가 돌아온다 — 밤 경기 2쿼터 실측 18분에 38번.
    묶음으로 들어갈 때와 나올 때 모두 공이 낼 수 없는 속도(화면 폭의 speed_frac배/초 초과)이고,
    묶음을 건너뛴 앞뒤 점은 자연스럽게 이어지면(그 속도의 ret_ratio배 이하) 묶음을 지운다.
    사람이 표시한 121장면 대조: 맞음 98 → 98, 직접 검출의 2.4%만 지워짐 (잃는 진짜 공 없음)."""
    out = list(pos)
    vmax = speed_frac * frame_w

    def speed(i: int, j: int) -> float:
        a, b = out[i], out[j]
        return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5 / max(1e-6, times[j] - times[i])

    for _ in range(passes):
        idx = [i for i, p in enumerate(out) if p is not None]
        bad: List[int] = []
        n = 1
        while n < len(idx) - 1:
            if speed(idx[n - 1], idx[n]) <= vmax:
                n += 1
                continue
            m = n   # 튄 묶음의 끝: 묶음 안에서는 자연스럽게 이어진다
            while m + 1 < len(idx) and times[idx[m + 1]] - times[idx[n]] <= max_len_s and speed(idx[m], idx[m + 1]) <= vmax:
                m += 1
            if m + 1 >= len(idx):
                break
            if speed(idx[m], idx[m + 1]) > vmax and speed(idx[n - 1], idx[m + 1]) <= vmax * ret_ratio:
                bad.extend(idx[n:m + 1])
                n = m + 1
            else:
                n += 1
        if not bad:
            break
        for i in bad:
            out[i] = None
    return out


def interpolate_gaps(pos: List[BallPos], times: Sequence[float], max_gap_s: float = 0.8,
                     max_speed_px_s: Optional[float] = None,
                     cam_centers: Optional[Sequence[Optional[Tuple[float, float]]]] = None,
                     cam_max_gap_s: float = 5.0, cam_follow: float = 0.7,
                     cam_min_shift_px: float = 0.0) -> List[BallPos]:
    """짧은 공백은 직선 보간. max_speed_px_s를 주면 양 끝이 그보다 빨리 이동한 공백(시야 밖으로 나갔다
    다른 곳에서 다시 보인 경우)은 잇지 않는다 — 가려진 공은 보통 거의 제자리에 있다.

    cam_centers(프레임별 화면 중심의 안정화 좌표)를 주면 카메라가 돌아가는 동안 놓친 공도 잇는다.
    사람이 찍는 영상은 카메라가 공을 따라가므로, 공백 동안의 화면 이동을 경로에 cam_follow만큼 더하고
    (cam_max_gap_s까지), 속도 판정도 화면 이동으로 설명되는 만큼은 빼고 한다 — 패스를 따라 화면을
    돌리는 사이 공이 흐려져 안 보인 경우다.
    """
    out = list(pos)
    known = [i for i, p in enumerate(pos) if p is not None]
    for a, b in zip(known, known[1:]):
        if b - a <= 1:
            continue
        gap = times[b] - times[a]
        if gap <= 0:
            continue
        (x0, y0), (x1, y1) = pos[a], pos[b]
        ca = cb = None
        if cam_centers is not None:
            ca, cb = cam_centers[a], cam_centers[b]
        cam_shift = 0.0
        if ca is not None and cb is not None:
            cam_shift = ((cb[0] - ca[0]) ** 2 + (cb[1] - ca[1]) ** 2) ** 0.5
        # 화면이 공백 동안 cam_min_shift_px 넘게 움직였으면 카메라가 공을 따라간 공백으로 본다
        follow = ca is not None and cb is not None and cam_shift > cam_min_shift_px
        if gap > (max(max_gap_s, cam_max_gap_s) if follow else max_gap_s):
            continue
        if max_speed_px_s is not None:
            dx, dy = x1 - x0, y1 - y0
            if follow:
                dx -= cam_follow * (cb[0] - ca[0])
                dy -= cam_follow * (cb[1] - ca[1])
            if (dx * dx + dy * dy) ** 0.5 / gap > max_speed_px_s:
                continue
        for i in range(a + 1, b):
            f = (times[i] - times[a]) / gap
            x, y = x0 + (x1 - x0) * f, y0 + (y1 - y0) * f
            c = cam_centers[i] if follow else None
            if c is not None:
                # 직선 경로에서 벗어난 화면 이동만큼 공도 따라 움직였다고 본다 (양 끝에서는 0)
                x += cam_follow * (c[0] - (ca[0] + (cb[0] - ca[0]) * f))
                y += cam_follow * (c[1] - (ca[1] + (cb[1] - ca[1]) * f))
            out[i] = (x, y)
    return out
