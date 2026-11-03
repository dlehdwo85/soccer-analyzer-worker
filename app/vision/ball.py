"""공 궤적 추출.

YOLO의 sports ball 검출은 작고 자주 놓치며, 관중석 공·하얀 점을 잘못 잡기도 한다.
1) 직전 궤적에서 예측한 위치 근처 후보를 우선 선택(게이팅)
2) 앞뒤와 동떨어진 튀는 점 제거
3) 짧은 공백은 선형 보간 (긴 공백은 비워 둔다 — 이벤트 판정이 '모름'으로 처리)
4) 카메라가 공을 따라 돌아가는 동안의 공백은 화면 이동을 따라 잇는다
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

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
    return remove_spikes(out, times, frame_w)


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
