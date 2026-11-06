"""
File loading, enriched 13-channel-per-foot construction, and windowing.

A raw file has 19 columns (format.txt): time, 8 left sensors, 8 right sensors,
total-left force, total-right force, sampled at 100 Hz. For each foot we build a
13-channel representation (see config.py for the layout).
"""
from __future__ import annotations

import glob
import os
from functools import lru_cache

import numpy as np

from . import config as C


def label_from_name(name: str) -> int:
    """PD patient ('Pt') -> 1, control ('Co') -> 0."""
    if "Pt" in name:
        return 1
    if "Co" in name:
        return 0
    raise ValueError(f"Cannot infer label from file name: {name!r}")


@lru_cache(maxsize=256)
def _load_raw(path: str) -> np.ndarray:
    """Load and cache a raw (T, 19) recording."""
    return np.loadtxt(path)


def _enrich(sensors: np.ndarray, total: np.ndarray, coords: np.ndarray) -> np.ndarray:
    """(T, 8) sensor forces -> (13, T) enriched channels for one foot."""
    denom = sensors.sum(axis=1)
    safe = np.where(denom > 1e-6, denom, 1.0)
    cop_x = (sensors * coords[:, 0]).sum(axis=1) / safe
    cop_y = (sensors * coords[:, 1]).sum(axis=1) / safe
    off = denom <= 1e-6            # swing phase / no contact -> COP undefined
    cop_x[off] = 0.0
    cop_y[off] = 0.0

    heel = sensors[:, coords[:, 1] < 0].sum(axis=1)   # posterior sensors
    toe = sensors[:, coords[:, 1] > 0].sum(axis=1)    # anterior sensors

    chans = np.column_stack([sensors, total, cop_x, cop_y, heel, toe])  # (T, 13)
    return chans.T.astype(np.float32)                                   # (13, T)


def load_signal(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (left_13xT, right_13xT) for a recording, with standing trimmed."""
    raw = _load_raw(path)
    s = int(C.TRIM_START_S * C.SAMPLING_RATE)
    e = int(C.TRIM_END_S * C.SAMPLING_RATE)
    if raw.shape[0] - s - e >= C.WIN_TS:              # keep >= one TSFresh window
        raw = raw[s: raw.shape[0] - e]

    left_sensors, right_sensors = raw[:, 1:9], raw[:, 9:17]
    total_left, total_right = raw[:, 17], raw[:, 18]
    left = _enrich(left_sensors, total_left, C.LEFT_COORDS)
    right = _enrich(right_sensors, total_right, C.RIGHT_COORDS)
    return left, right


def _split_files(fold_id: int, folder: str) -> list[str]:
    pattern = os.path.join(C.SPLITS_DIR, f"Fold_{fold_id}", folder, "*.txt")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No .txt files found under {pattern}")
    return files


def load_fold(fold_id: int) -> dict:
    """
    Load one fold. Returns a dict with keys 'train'/'val'/'test', each mapping to
    a tuple ``(subjects, labels, left_signals, right_signals)`` where
    ``*_signals`` are lists of (13, T) float32 arrays.
    """
    split_dirs = {"train": "training", "val": "validation", "test": "test"}
    out: dict = {}
    for split, folder in split_dirs.items():
        subjects, labels, lefts, rights = [], [], [], []
        for f in _split_files(fold_id, folder):
            name = os.path.basename(f)[:-4]
            left, right = load_signal(f)
            subjects.append(name)
            labels.append(label_from_name(name))
            lefts.append(left)
            rights.append(right)
        out[split] = (subjects, np.asarray(labels, dtype=int), lefts, rights)
    return out


def foot_signal(left: np.ndarray, right: np.ndarray, foot: str) -> np.ndarray:
    """Select the channel block for a foot representation."""
    if foot == "Left":
        return left
    if foot == "Right":
        return right
    if foot == "Combined":
        return np.vstack([left, right])          # (26, T)
    raise ValueError(f"Unknown foot: {foot!r}")


def make_windows(sig: np.ndarray, win: int, step: int) -> np.ndarray:
    """(C, T) -> (n_windows, C, win). Short signals are zero-padded to one window."""
    n_ch, T = sig.shape
    if T < win:
        pad = np.zeros((n_ch, win), dtype=sig.dtype)
        pad[:, :T] = sig
        return pad[None]
    starts = range(0, T - win + 1, step)
    return np.stack([sig[:, i:i + win] for i in starts]).astype(np.float32)
