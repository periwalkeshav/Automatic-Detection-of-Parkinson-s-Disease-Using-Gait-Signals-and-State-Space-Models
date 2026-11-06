"""
Gait-cycle (stride-level) features from the VGRF signal.

TSFresh describes a window of force samples statistically; it has no notion of
stance, swing, heel-strike or toe-off. Parkinsonian gait, however, is
characterised by *stride-to-stride variability* — irregular stride timing and
inconsistent loading — which is a property of the gait cycle, not of an
arbitrary 5 s window.

This module segments each recording into individual strides and describes every
stride with clinically meaningful quantities (stride/stance/swing time, the
two loading peaks and the mid-stance valley, centre-of-pressure travel, and
heel/toe timing). Feeding that sequence to a state-space model lets it observe
how the walk changes from step to step, and the dispersion statistics in the
pooling head then measure exactly the variability that separates the groups.

Feature families follow Khera & Kumar, Eng. Sci. Technol. Int. J. 27 (2022).
Channel layout is the enriched 13-channel per-foot representation from data.py:
    0-7 sensors, 8 total force, 9 COP_x, 10 COP_y, 11 heel, 12 toe.
"""
from __future__ import annotations

import numpy as np

from . import config as C

TOTAL, COPX, COPY, HEEL, TOE = 8, 9, 10, 11, 12

FEATURE_NAMES = (
    "stride_time", "stance_time", "swing_time", "stance_frac",
    "peak1", "peak2", "valley", "peak_ratio",
    "force_mean", "force_std", "impulse",
    "cop_x_range", "cop_y_range", "cop_path",
    "heel_peak_frac", "toe_peak_frac", "heel_toe_ratio",
)
N_FEATURES = len(FEATURE_NAMES)


def _stance_onsets(total: np.ndarray, fs: int):
    """Indices where the foot loads (swing -> stance), plus the unload indices."""
    # Force below ~20 N is sensor noise during swing (the usual threshold for
    # this dataset); scale it up for recordings with a very different gain.
    thr = max(20.0, 0.05 * float(np.percentile(total, 99)))
    stance = total > thr
    if stance.sum() < fs:                       # essentially no contact detected
        return np.array([], dtype=int), np.array([], dtype=int)

    d = np.diff(stance.astype(np.int8))
    onsets = np.flatnonzero(d == 1) + 1
    offsets = np.flatnonzero(d == -1) + 1
    if stance[0]:                               # recording starts mid-stance
        onsets = np.r_[0, onsets]
    if stance[-1]:
        offsets = np.r_[offsets, len(stance)]
    return onsets, offsets


def stride_features(sig: np.ndarray, fs: int | None = None) -> np.ndarray:
    """
    (13, T) signal for one foot -> (n_strides, N_FEATURES), in time order.

    A stride runs from one foot-strike to the next strike of the same foot.
    """
    fs = fs or C.SAMPLING_RATE
    total = np.asarray(sig[TOTAL], dtype=float)
    onsets, offsets = _stance_onsets(total, fs)
    if len(onsets) < 2:
        return np.zeros((0, N_FEATURES), dtype=np.float32)

    rows = []
    for i in range(len(onsets) - 1):
        s, e = int(onsets[i]), int(onsets[i + 1])
        stride_n = e - s
        # Reject implausible cycles: human stride time is roughly 0.5-2.5 s.
        if not (0.5 * fs <= stride_n <= 2.5 * fs):
            continue

        later = offsets[offsets > s]
        toe_off = int(min(later[0], e)) if later.size else e
        stance_n = max(toe_off - s, 1)
        swing_n = max(stride_n - stance_n, 0)

        f = total[s:toe_off]
        if f.size < 3:
            continue
        mid = f.size // 2
        peak1 = float(f[:mid].max()) if mid > 0 else float(f.max())
        peak2 = float(f[mid:].max()) if mid < f.size else float(f.max())
        valley = float(f[max(mid - f.size // 6, 1):mid + f.size // 6].min()) if f.size > 6 else float(f.min())
        denom = max(peak1, 1e-6)

        cx, cy = sig[COPX, s:toe_off], sig[COPY, s:toe_off]
        cop_path = float(np.abs(np.diff(cx)).sum() + np.abs(np.diff(cy)).sum())

        heel, toe = sig[HEEL, s:toe_off], sig[TOE, s:toe_off]
        heel_frac = float(np.argmax(heel) / stance_n) if heel.size else 0.0
        toe_frac = float(np.argmax(toe) / stance_n) if toe.size else 0.0

        rows.append([
            stride_n / fs, stance_n / fs, swing_n / fs, stance_n / stride_n,
            peak1, peak2, valley, peak2 / denom,
            float(f.mean()), float(f.std()), float(f.sum() / fs),
            float(cx.max() - cx.min()), float(cy.max() - cy.min()), cop_path,
            heel_frac, toe_frac,
            float(heel.max() / max(toe.max(), 1e-6)) if toe.size else 0.0,
        ])

    if not rows:
        return np.zeros((0, N_FEATURES), dtype=np.float32)
    out = np.asarray(rows, dtype=np.float32)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def stride_waveforms(sig: np.ndarray, out_len: int = 128,
                     fs: int | None = None) -> np.ndarray:
    """
    (13, T) -> (n_strides, 13, out_len): each stride resampled to a common length.

    Time-normalising the gait cycle aligns the phases across strides and across
    subjects — heel-strike at 0 %, toe-off around 60 %, swing to 100 % — so a
    convolutional filter always sees the same phase at the same position. In a
    fixed 5 s window the cycle starts at an arbitrary offset and the network has
    to waste capacity being invariant to that.
    """
    fs = fs or C.SAMPLING_RATE
    total = np.asarray(sig[TOTAL], dtype=float)
    onsets, _ = _stance_onsets(total, fs)
    if len(onsets) < 2:
        return np.zeros((0, sig.shape[0], out_len), dtype=np.float32)

    grid = np.linspace(0.0, 1.0, out_len)
    out = []
    for i in range(len(onsets) - 1):
        s, e = int(onsets[i]), int(onsets[i + 1])
        n = e - s
        if not (0.5 * fs <= n <= 2.5 * fs):
            continue
        src = np.linspace(0.0, 1.0, n)
        seg = sig[:, s:e]
        out.append(np.stack([np.interp(grid, src, seg[c]) for c in range(seg.shape[0])]))
    if not out:
        return np.zeros((0, sig.shape[0], out_len), dtype=np.float32)
    return np.nan_to_num(np.asarray(out, dtype=np.float32))


def build_stride_waveform_bags(lefts, rights, foot: str, out_len: int = 128):
    """Per-subject (n_strides, C, out_len) bags of time-normalised strides."""
    bags = []
    for left, right in zip(lefts, rights):
        if foot == "Left":
            w = stride_waveforms(left, out_len)
        elif foot == "Right":
            w = stride_waveforms(right, out_len)
        elif foot == "Combined":
            a, b = stride_waveforms(left, out_len), stride_waveforms(right, out_len)
            n = min(len(a), len(b))
            w = (np.concatenate([a[:n], b[:n]], axis=1) if n > 0
                 else np.zeros((0, 2 * left.shape[0], out_len), dtype=np.float32))
        else:
            raise ValueError(f"Unknown foot: {foot!r}")
        if w.shape[0] == 0:                      # keep one empty stride for shape
            n_ch = 2 * left.shape[0] if foot == "Combined" else left.shape[0]
            w = np.zeros((1, n_ch, out_len), dtype=np.float32)
        bags.append(w)
    return bags


def build_stride_sequences(subjects, lefts, rights, foot: str) -> list[np.ndarray]:
    """Per-subject (n_strides, F) stride sequences for one foot representation."""
    out = []
    for left, right in zip(lefts, rights):
        if foot == "Left":
            mat = stride_features(left)
        elif foot == "Right":
            mat = stride_features(right)
        elif foot == "Combined":
            a, b = stride_features(left), stride_features(right)
            n = min(len(a), len(b))
            mat = (np.hstack([a[:n], b[:n]]) if n > 0
                   else np.zeros((0, 2 * N_FEATURES), dtype=np.float32))
        else:
            raise ValueError(f"Unknown foot: {foot!r}")
        if mat.shape[0] == 0:                    # keep one zero row so shapes align
            width = 2 * N_FEATURES if foot == "Combined" else N_FEATURES
            mat = np.zeros((1, width), dtype=np.float32)
        out.append(mat)
    return out
