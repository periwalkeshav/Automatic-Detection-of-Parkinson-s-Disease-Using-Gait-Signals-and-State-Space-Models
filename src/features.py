"""
TSFresh feature extraction for the classical branch.

For each subject and foot we:
  1. cut the 13-channel signal into 5 s / 2.5 s windows,
  2. run TSFresh on every window (one feature vector per window),
  3. aggregate the per-window vectors over the recording with
     mean / std / skewness / kurtosis.

Because the same 100 subjects recur across all 5 folds and the features are
fold-independent, each subject's aggregated vector is cached on disk and reused.
"""
from __future__ import annotations

import hashlib
import os
import warnings

import numpy as np
import pandas as pd

from . import config as C
from .data import make_windows

# TSFresh calculators that return all-NaN on short gait windows (nonlinear
# Langevin/Friedrich fits) or require an argument that is never supplied
# (query_similarity_count). They are filled with zeros -> become constant
# columns -> dropped before training, so they add nothing but cost. Skipping
# them removes the "did not have any finite values" warning and speeds up
# extraction.
_DROP_CALCULATORS = (
    "friedrich_coefficients",
    "max_langevin_fixed_point",
    "query_similarity_count",
)

_FC_CACHE: dict = {}


# --------------------------------------------------------------------------- #
# TSFresh setup
# --------------------------------------------------------------------------- #
def _fc_parameters():
    key = C.TSFRESH_FC_PARAMETERS
    if key not in _FC_CACHE:
        from tsfresh.feature_extraction import (
            ComprehensiveFCParameters,
            EfficientFCParameters,
            MinimalFCParameters,
        )
        params = {
            "minimal": MinimalFCParameters,
            "efficient": EfficientFCParameters,
            "comprehensive": ComprehensiveFCParameters,
        }[key]()
        for name in _DROP_CALCULATORS:
            params.pop(name, None)
        _FC_CACHE[key] = params
    return _FC_CACHE[key]


def _config_tag() -> str:
    # Include a hash of the active feature set so that changing which
    # calculators are used invalidates the on-disk cache automatically
    # (avoids mixing feature vectors of different lengths across runs).
    sig = hashlib.md5(",".join(sorted(_fc_parameters())).encode()).hexdigest()[:8]
    return f"{C.TSFRESH_FC_PARAMETERS}_{sig}_w{C.WIN_TS}_s{C.STEP_TS}"


def _cache_path(subject: str, foot: str) -> str:
    return os.path.join(C.CACHE_DIR, f"{subject}__{foot}__{_config_tag()}.npy")


# --------------------------------------------------------------------------- #
# Per-subject extraction
# --------------------------------------------------------------------------- #
def _window_features(sig: np.ndarray) -> pd.DataFrame:
    """(C, T) -> DataFrame (n_windows, n_features) via TSFresh."""
    from tsfresh import extract_features
    from tsfresh.utilities.dataframe_functions import impute

    windows = make_windows(sig, C.WIN_TS, C.STEP_TS)   # (W, C, win)
    n_win, n_ch, win = windows.shape

    # Long-format frame: one block of rows per window, one column per channel.
    records = {"id": np.repeat(np.arange(n_win), win),
               "time": np.tile(np.arange(win), n_win)}
    for ch in range(n_ch):
        records[f"ch{ch:02d}"] = windows[:, ch, :].reshape(-1)
    long_df = pd.DataFrame(records)

    feats = extract_features(
        long_df,
        column_id="id",
        column_sort="time",
        default_fc_parameters=_fc_parameters(),
        n_jobs=C.TSFRESH_N_JOBS,
        disable_progressbar=True,
    )
    feats = feats.sort_index(axis=1)          # deterministic column order
    with warnings.catch_warnings():
        # Any all-NaN column is zero-filled here and later dropped as constant.
        warnings.filterwarnings("ignore", message=".*did not have any finite values.*",
                                category=RuntimeWarning)
        impute(feats)                          # NaN/inf -> finite, in place
    return feats


def _agg_tag() -> str:
    """Cache key for the aggregated view (windows are cached separately)."""
    return hashlib.md5(",".join(C.AGG_FUNCS).encode()).hexdigest()[:6]


def _aggregate(v) -> np.ndarray:
    """
    (W, F) window features -> flat vector of the statistics in C.AGG_FUNCS.

    Beyond the mean and the classical spread statistics this includes quantiles
    and the coefficient of variation. CV = std / |mean| is the scale-free
    dispersion measure used clinically to quantify stride-to-stride variability,
    which is the property that separates Parkinsonian from healthy gait; it is
    not recoverable from the mean and standard deviation once features are
    standardised, so it has to be computed before scaling.
    """
    from scipy.stats import kurtosis, skew

    v = np.asarray(v, dtype=float)
    if v.ndim == 1:
        v = v.reshape(1, -1)
    n_feat = v.shape[1]

    if v.shape[0] < 2:                         # single window: no spread to speak of
        out = []
        for f in C.AGG_FUNCS:
            out.append(v.reshape(-1) if f in ("mean", "q50") else np.zeros(n_feat))
        return np.nan_to_num(np.concatenate(out), nan=0.0, posinf=0.0, neginf=0.0)

    with warnings.catch_warnings():
        # Near-constant columns make the higher moments numerically noisy; the
        # result is sanitised below, so the warnings carry no information.
        warnings.simplefilter("ignore", RuntimeWarning)
        stats = {}
        mean = np.nanmean(v, axis=0)
        std = np.nanstd(v, axis=0)
        stats["mean"] = mean
        stats["std"] = std
        stats["skewness"] = skew(v, axis=0, nan_policy="omit")
        stats["kurtosis"] = kurtosis(v, axis=0, nan_policy="omit")
        stats["cv"] = std / np.maximum(np.abs(mean), 1e-8)
        for q in (10, 25, 50, 75, 90):
            stats[f"q{q}"] = np.nanpercentile(v, q, axis=0)
        stats["iqr"] = stats["q75"] - stats["q25"]
        stats["range"] = np.nanmax(v, axis=0) - np.nanmin(v, axis=0)

    missing = [f for f in C.AGG_FUNCS if f not in stats]
    if missing:
        raise ValueError(f"Unknown AGG_FUNCS entries: {missing}")
    out = np.concatenate([stats[f] for f in C.AGG_FUNCS])
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _window_cache_path(subject: str, foot: str) -> str:
    return os.path.join(C.CACHE_DIR, f"{subject}__{foot}__win__{_config_tag()}.npy")


def subject_window_features(subject: str, sig: np.ndarray, foot: str) -> np.ndarray:
    """
    Cached (n_windows, n_features) TSFresh matrix for one subject/foot.

    This is the *un-aggregated* view: one feature vector per 5 s window, kept in
    time order, so a sequence model can see how the gait descriptors evolve
    across the recording instead of only their summary statistics.
    """
    path = _window_cache_path(subject, foot)
    if os.path.exists(path):
        return np.load(path)
    with warnings.catch_warnings():
        # All-NaN columns are zero-filled here and dropped later as constants.
        warnings.filterwarnings("ignore", message=".*did not have any finite values.*",
                                category=RuntimeWarning)
        mat = _window_features(sig).to_numpy(dtype=np.float32)
    mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    np.save(path, mat)
    return mat


def build_window_sequences(subjects: list[str], lefts: list[np.ndarray],
                           rights: list[np.ndarray], foot: str) -> list[np.ndarray]:
    """Per-subject (n_windows, n_features) sequences for one foot representation."""
    out = []
    for subj, left, right in zip(subjects, lefts, rights):
        if foot == "Left":
            mat = subject_window_features(subj, left, "Left")
        elif foot == "Right":
            mat = subject_window_features(subj, right, "Right")
        elif foot == "Combined":
            a = subject_window_features(subj, left, "Left")
            b = subject_window_features(subj, right, "Right")
            n = min(len(a), len(b))          # align: same windows from both feet
            mat = np.hstack([a[:n], b[:n]])
        else:
            raise ValueError(f"Unknown foot: {foot!r}")
        out.append(mat)
    return out


def subject_features(subject: str, sig: np.ndarray, foot: str) -> np.ndarray:
    """
    Cached aggregated TSFresh vector for one subject/foot.

    Built from the cached per-window matrix, so changing C.AGG_FUNCS only
    re-runs the (cheap) aggregation rather than the (expensive) TSFresh
    extraction. The aggregate cache carries its own key.
    """
    path = os.path.join(
        C.CACHE_DIR, f"{subject}__{foot}__agg{_agg_tag()}__{_config_tag()}.npy")
    if os.path.exists(path):
        return np.load(path)
    windows = subject_window_features(subject, sig, foot)
    vec = _aggregate(windows).astype(np.float32)
    np.save(path, vec)
    return vec


# --------------------------------------------------------------------------- #
# Fold-level matrix assembly
# --------------------------------------------------------------------------- #
def build_feature_matrix(subjects: list[str],
                         lefts: list[np.ndarray],
                         rights: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Assemble (X_left, X_right, X_combined) feature matrices for a subject list.
    Combined features are the concatenation of the left and right vectors.
    """
    xl, xr = [], []
    for subj, left, right in zip(subjects, lefts, rights):
        xl.append(subject_features(subj, left, "Left"))
        xr.append(subject_features(subj, right, "Right"))
    x_left = np.vstack(xl)
    x_right = np.vstack(xr)
    x_comb = np.hstack([x_left, x_right])
    return x_left, x_right, x_comb
