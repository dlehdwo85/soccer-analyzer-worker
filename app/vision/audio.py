"""영상 소리에서 심판 호루라기 찾기 → 경기 중단·재개(파울 뒤 프리킥, 골 뒤 킥오프 등).

카메라가 움직이는 영상은 화면만으로는 파울·아웃을 알기 어렵다. 호루라기는 2~4.5kHz 대역에 뾰족한 음이
0.15초 이상 이어지는 소리라, 저음(말소리·발소리) 대비 그 대역이 크고 한 주파수에 몰려 있는 구간을 찾는다.
바모스전 2쿼터(18분): 후보 61개, 강한 것 약 25개. 9:10 골 직후 9:13~9:15에 가장 강한 소리(킥오프)가 잡혔다.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

SR = 16000
N_FFT = 1024
HOP = 256
BAND = (2000.0, 4500.0)
LOW = (150.0, 2000.0)
MIN_PEAK = 150.0      # 대역 안 가장 센 주파수가 대역 중앙값의 몇 배인가 (뾰족함)
MIN_RATIO_DB = -6.0   # 대역 에너지 / 저음 에너지
MIN_DUR = 0.15
MERGE_S = 1.5         # 이만큼 안에 다시 분 호루라기는 한 번으로 (짧게 여러 번 부는 경우)


def _ffmpeg() -> Optional[str]:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001
        return None


def load_audio(source: str, start_s: float = 0.0, max_s: Optional[float] = None) -> Optional[np.ndarray]:
    """영상의 소리를 16kHz 모노 float로. 소리가 없거나 ffmpeg가 없으면 None."""
    exe = _ffmpeg()
    if not exe:
        return None
    cmd = [exe, "-hide_banner", "-loglevel", "error", "-ss", str(start_s), "-i", source]
    if max_s:
        cmd += ["-t", str(max_s)]
    cmd += ["-vn", "-ac", "1", "-ar", str(SR), "-f", "s16le", "-"]
    try:
        out = subprocess.run(cmd, capture_output=True, timeout=600).stdout
    except (OSError, subprocess.SubprocessError) as e:
        logger.warning("소리 추출 실패: %s", e)
        return None
    if len(out) < SR * 2:
        return None
    return np.frombuffer(out, np.int16).astype(np.float32) / 32768.0


def detect_whistles(x: np.ndarray, sr: int = SR) -> List[Tuple[float, float, float]]:
    """[(시각 초, 길이 초, 세기)] — 세기는 뾰족함 점수(클수록 확실)."""
    if x is None or len(x) < N_FFT * 4:
        return []
    frames = (len(x) - N_FFT) // HOP
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(frames)[:, None]
    freqs = np.fft.rfftfreq(N_FFT, 1 / sr)
    band = (freqs >= BAND[0]) & (freqs <= BAND[1])
    low = (freqs >= LOW[0]) & (freqs < LOW[1])
    win = np.hanning(N_FFT).astype(np.float32)
    score = np.empty(frames, np.float32)
    ratio = np.empty(frames, np.float32)
    for a in range(0, frames, 4096):     # 메모리를 아끼려 나눠서
        S = np.abs(np.fft.rfft(x[idx[a:a + 4096]] * win, axis=1)) ** 2
        b = S[:, band]
        score[a:a + 4096] = b.max(axis=1) / (np.median(b, axis=1) + 1e-12)
        ratio[a:a + 4096] = 10 * np.log10(b.sum(axis=1) / (S[:, low].sum(axis=1) + 1e-12) + 1e-12)
    on = (score > MIN_PEAK * 0.4) & (ratio > MIN_RATIO_DB)
    t = np.arange(frames) * HOP / sr
    out: List[Tuple[float, float, float]] = []
    i = 0
    while i < frames:
        if not on[i]:
            i += 1
            continue
        j = i
        while j + 1 < frames and (on[j + 1] or (j + 2 < frames and on[j + 2])):
            j += 1
        dur = float(t[j] - t[i])
        strength = float(np.median(score[i:j + 1]))
        if dur >= MIN_DUR and strength >= MIN_PEAK:
            if out and t[i] - (out[-1][0] + out[-1][1]) <= MERGE_S:
                s0, d0, st0 = out[-1]
                out[-1] = (s0, float(t[j] - s0), max(st0, strength))
            else:
                out.append((float(t[i]), dur, strength))
        i = j + 1
    return out
