"""
Deliverables #1 & #2 — Parkinson's Disease Gait Classification
===============================================================
Dataset   : PhysioNet GaitPaDB  — 5 pre-defined folds
            Data/splits/Fold_<1..5>/{training, validation, test}
Features  : TSFresh EfficientFCParameters per 5s/2.5s window,
            aggregated per subject via mean/std/skewness/kurtosis.
Models    : Random Forest, SVM-Linear, SVM-RBF, Logistic Regression
            (ElasticNet).
            Optional (behind RUN_DL): 1D-ResNet, LSTM, GRU, PatchTST.
Foot      : Left, Right, Combined (concatenation)
CV        : 5 pre-defined folds (no StratifiedKFold — folds are on disk)
Metrics   : Accuracy, Sensitivity, Specificity, AUC

RUN-MODE FLAGS
--------------
  RUN_D1 = False   Skip Deliverable #1 plots.
  RUN_CLASSICAL = False  Skip classical ML models.
  RUN_DL = True    Train 1D-ResNet / LSTM / GRU / PatchTST per fold.

CHANGES IN THIS REVISION — DL-focused improvements
-----------------------------------------------------
  1. WARMUP-AWARE CHECKPOINTING (bug fix). _train_model previously tracked
     the "best" validation-loss checkpoint starting from epoch 0, while
     warmup_epochs=10 ramps the LR linearly from 10%->100% over that span.
     A spuriously-low loss at e.g. epoch 1 (model still close to random
     init) could get locked in as "best," and the downstream refit would
     then train for only that many epochs on combined data — observed
     directly in a real run: Fold 1 / 1D-ResNet / Right foot printed
     "refit_epochs=1, test_AUC=0.71" off a val_AUC=0.81 HP combo, and
     4/15 fold×foot combinations in that run had test_AUC < 0.5 (worse
     than random). Fix: checkpoints are only considered "best" once
     ep >= warmup_epochs.
  2. CONV-STEM BEFORE LSTM/GRU, mode-aware. ResNet1D already downsamples
     long sequences via its strided stem; LSTM/GRU previously went
     straight from a per-timestep Linear into the recurrent layer,
     scanning up to T=2000 raw samples (DL_INPUT_MODE="raw_full") with no
     downsampling at all. A `downsample` constructor flag adds two
     stride-2 convs (T -> T/4) BEFORE the recurrent layer, but only when
     DL_INPUT_MODE="raw_full" (set via the factory in main()) — for
     tsfresh/raw_windowed, where each timestep is already a meaningful
     aggregated window (T~30-50), the original direct-to-RNN path is
     preserved unchanged.
  3. MIXUP (Zhang et al. 2018). Each training batch is blended with a
     randomly permuted copy of itself (and its label) via a
     Beta(MIXUP_ALPHA, MIXUP_ALPHA) coefficient. Generates genuinely new
     interpolated training signal rather than perturbing existing points
     — particularly valuable in this dataset's small-N regime (as few as
     60 training subjects). Applied in both HP-search and refit training;
     never at val/test.
  4. TOP-K DL ENSEMBLE. train_eval_dl_fold now keeps the TOP_K_DL_ENSEMBLE
     (default 3) best HP combos by validation AUC instead of only the
     single best, refits each on train+val, and averages their test-set
     probabilities — the DL analogue of the existing classical soft-
     voting ensemble. Directly mitigates picking one spuriously-high-
     val-AUC combo (the same noisy-small-val-set problem documented in
     fix #1's evidence).
  5. LABEL SMOOTHING (LABEL_SMOOTHING=0.1) on CrossEntropyLoss — cheap
     regularization against label noise.

Deliverable #1 additions (only if RUN_D1=True)
-------------------------------------------------
  - Dataset demographics table parsed from GaitPaDB filename conventions.
  - Raw signal examples: one PD and one Control subject per channel.
  - PCA plots (PC1 vs PC2, coloured by diagnosis) for Left/Right/Combined.

Fold workflow (per fold, per foot, per classical model)
---------------------------------------------------------
1. TRAINING   → fit VarianceThreshold + FRESH relevance filter + StandardScaler
                (+ PCA for SVM-RBF only) on TRAINING data only.
2. VALIDATION → transform with the fitted preprocessor,
                sweep hyperparameter grid → pick best combo by AUC.
3. REFIT      → refit best classifier on train + val combined.
4. TEST       → evaluate refitted model → record metrics.

Key design decisions
--------------------
CHANNELS (13 per foot)
  8 FSR sensors + total force + COP_x + COP_y + heel_mean + toe_mean.

SIGNAL PREPROCESSING
  1. Trim TRIM_SEC=20s from each end — removes turning/acceleration artefacts.
  2. Median filter (kernel 11) per channel — removes sensor impulse noise.
  3. Compute derived channels (COP, heel, toe) from the 8 FSR sensors.

FEATURE SELECTION (classical ML)
  VarianceThreshold -> FRESH relevance filter (Mann-Whitney U + FDR) ->
  StandardScaler [-> PCA for SVM-RBF only]. Fit on TRAINING data only.

SUBJECT-LEVEL FEATURE CACHE
  Keyed by the FULL FILE STEM (e.g. "GaCo02_01"), not the bare subject id
  — two trials of the same subject are different recordings and must
  never share a cache entry. Cache key also encodes WIN_SEC, STEP_SEC,
  TRIM_SEC, MEDIAN_FILTER_LEN, N_CHANNELS, FC_PARAMS class name.
"""

import warnings
warnings.filterwarnings("ignore")

import hashlib
import inspect
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import skew, kurtosis as kurt
from scipy.signal import medfilt
from sklearn.decomposition import PCA

from tsfresh import extract_features, select_features
from tsfresh.feature_extraction import EfficientFCParameters
from tsfresh.utilities.dataframe_functions import impute

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold, f_classif
from sklearn.model_selection import ParameterGrid
from sklearn.manifold import TSNE
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             roc_auc_score, recall_score)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

np.random.seed(42)
torch.manual_seed(42)


# ─────────────────────────────────────────────────────────────────
# 0.  RUN-MODE FLAGS & PATHS & CONSTANTS
# ─────────────────────────────────────────────────────────────────
RUN_D1         = False
RUN_CLASSICAL  = False
RUN_DL         = True

DL_INPUT_MODE = "raw_full"   # "tsfresh" | "raw_windowed" | "raw_full"
RAW_FULL_SECONDS = 20.0

DATA_ROOT   = Path("Data/splits")
N_FOLDS     = 5
OUT_DIR     = Path("results");        OUT_DIR.mkdir(exist_ok=True)
FIG_DIR     = OUT_DIR / "figures";    FIG_DIR.mkdir(exist_ok=True)
CACHE_DIR   = OUT_DIR / "feature_cache"; CACHE_DIR.mkdir(exist_ok=True)

FS           = 100
WIN_SEC      = 5.0
STEP_SEC     = 2.5
WIN_SAMPLES  = int(WIN_SEC  * FS)
STEP_SAMPLES = int(STEP_SEC * FS)

RAW_FULL_T = int(RAW_FULL_SECONDS * FS)   # 2000 samples @ FS=100Hz, 20s

L_COLS = list(range(1, 9)) + [17]
R_COLS = list(range(9, 17)) + [18]

TRIM_SEC          = 0
TRIM_SAMPLES      = int(TRIM_SEC * FS)
MEDIAN_FILTER_LEN = 11

SENSOR_X = np.array([-500, -700, -300, -700, -300, -700, -300, -500],
                    dtype=np.float32)
SENSOR_Y = np.array([-800, -400, -400,    0,    0,  400,  400,  800],
                    dtype=np.float32)

N_CHANNELS = 13

L_CHANNEL_NAMES = (
    [f"L{i+1}" for i in range(8)] +
    ["L_total", "L_cop_x", "L_cop_y", "L_heel", "L_toe"]
)
R_CHANNEL_NAMES = (
    [f"R{i+1}" for i in range(8)] +
    ["R_total", "R_cop_x", "R_cop_y", "R_heel", "R_toe"]
)
assert len(L_CHANNEL_NAMES) == N_CHANNELS, "L_CHANNEL_NAMES length mismatch"
assert len(R_CHANNEL_NAMES) == N_CHANNELS, "R_CHANNEL_NAMES length mismatch"

N_PCA_COMPONENTS = 30

FRESH_FDR_LEVEL          = 0.05
MIN_SELECTED_FEATURES    = 30
MAX_SELECTED_FEATURES    = 300
FEATURE_SELECTION_N_JOBS = 10

D_MODEL_SMALL    = 32
D_MODEL_LARGE    = 64
D_MODEL          = D_MODEL_SMALL
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"

# ── DL training-improvement constants (this revision) ────────────
TOP_K_DL_ENSEMBLE = 3     # refit + average this many top-val-AUC HP combos
MIXUP_ALPHA       = 0.2   # Beta(alpha,alpha) mixing coeff; 0.0 disables mixup
LABEL_SMOOTHING   = 0.1   # CrossEntropyLoss label_smoothing factor

print(f"Device: {DEVICE}")
print(f"RUN_D1={RUN_D1}  RUN_CLASSICAL={RUN_CLASSICAL}  RUN_DL={RUN_DL}  "
      f"DL_INPUT_MODE={DL_INPUT_MODE!r}  TRIM_SEC={TRIM_SEC}")
print(f"TOP_K_DL_ENSEMBLE={TOP_K_DL_ENSEMBLE}  MIXUP_ALPHA={MIXUP_ALPHA}  "
      f"LABEL_SMOOTHING={LABEL_SMOOTHING}")

FAU_BLUE = "#003865"
FAU_TEAL = "#00b1eb"
FAU_RED  = "#c8102e"
FAU_GRAY = "#98a4ae"

FC_PARAMS = EfficientFCParameters()


# ─────────────────────────────────────────────────────────────────
# 1.  SUBJECT-LEVEL FEATURE CACHE
# ─────────────────────────────────────────────────────────────────
_MEMORY_CACHE: dict = {}


def _subject_cache_key(subject_id: str) -> str:
    fingerprint = "|".join([
        subject_id,
        f"win={WIN_SEC}",
        f"step={STEP_SEC}",
        f"trim={TRIM_SEC}",
        f"medfilt={MEDIAN_FILTER_LEN}",
        f"channels={N_CHANNELS}",
        type(FC_PARAMS).__name__,
    ])
    return hashlib.sha256(fingerprint.encode()).hexdigest()[:12]


def _subject_cache_path(subject_id: str) -> Path:
    return CACHE_DIR / f"subject_{subject_id}_{_subject_cache_key(subject_id)}.npz"


def load_subject_window_features(subject_id: str):
    path = _subject_cache_path(subject_id)
    if not path.exists():
        return None
    try:
        data = np.load(str(path), allow_pickle=False)
        return data["L_features"], data["R_features"]
    except Exception as exc:
        print(f"  [CACHE WARN] {path.name}: {exc}. Re-extracting …")
        return None


def save_subject_window_features(subject_id: str,
                                  L_features: np.ndarray,
                                  R_features: np.ndarray) -> None:
    path = _subject_cache_path(subject_id)
    try:
        np.savez_compressed(str(path),
                            L_features=L_features, R_features=R_features)
    except Exception as exc:
        print(f"  [CACHE WARN] Could not save cache for {subject_id}: {exc}")


def get_subject_window_features(subject_id: str,
                                 L_raw: np.ndarray, R_raw: np.ndarray):
    cached = _MEMORY_CACHE.get(subject_id)
    if cached is not None:
        return cached

    cached = load_subject_window_features(subject_id)
    if cached is not None:
        _MEMORY_CACHE[subject_id] = cached
        return cached

    L_feat = _window_features(L_raw, L_CHANNEL_NAMES)
    R_feat = _window_features(R_raw, R_CHANNEL_NAMES)
    save_subject_window_features(subject_id, L_feat, R_feat)
    _MEMORY_CACHE[subject_id] = (L_feat, R_feat)
    return L_feat, R_feat


# ─────────────────────────────────────────────────────────────────
# 2.  DATA LOADING
# ─────────────────────────────────────────────────────────────────
def _compute_derived_channels(fsr: np.ndarray) -> np.ndarray:
    eps       = 1e-6
    total     = fsr.sum(axis=1, keepdims=True) + eps
    cop_x     = (fsr * SENSOR_X).sum(axis=1, keepdims=True) / total
    cop_y     = (fsr * SENSOR_Y).sum(axis=1, keepdims=True) / total
    heel_mean = fsr[:, [0]].mean(axis=1, keepdims=True)
    toe_mean  = fsr[:, [7]].mean(axis=1, keepdims=True)
    return np.concatenate([cop_x, cop_y, heel_mean, toe_mean], axis=1)


def load_subject(path: Path):
    data = np.loadtxt(str(path))
    if data.ndim == 1:
        data = data.reshape(1, -1)

    if TRIM_SAMPLES > 0 and data.shape[0] > 2 * TRIM_SAMPLES + WIN_SAMPLES:
        data = data[TRIM_SAMPLES : -TRIM_SAMPLES]

    if data.shape[0] < WIN_SAMPLES:
        return None, None

    if MEDIAN_FILTER_LEN > 1:
        data = np.apply_along_axis(
            medfilt, 0, data, kernel_size=MEDIAN_FILTER_LEN)

    L_fsr     = data[:, list(range(1, 9))].astype(np.float32)
    R_fsr     = data[:, list(range(9, 17))].astype(np.float32)
    L_tot     = data[:, [17]].astype(np.float32)
    R_tot     = data[:, [18]].astype(np.float32)
    L_derived = _compute_derived_channels(L_fsr)
    R_derived = _compute_derived_channels(R_fsr)

    L = np.concatenate([L_fsr, L_tot, L_derived], axis=1)
    R = np.concatenate([R_fsr, R_tot, R_derived], axis=1)
    return L, R


def parse_label(filename: str) -> int:
    return 1 if "Pt" in filename else 0


def load_split(directory: Path):
    """Full file stem used as subject id (unique per trial)."""
    subjects, labels, left_raw, right_raw = [], [], [], []
    for path in sorted(directory.glob("*.txt")):
        L, R = load_subject(path)
        if L is None:
            print(f"  [SKIP] {path.name}  (empty or too short after trimming)")
            continue
        name = path.stem
        subjects.append(name)
        labels.append(parse_label(name))
        left_raw.append(L)
        right_raw.append(R)
    return subjects, np.array(labels), left_raw, right_raw


# ─────────────────────────────────────────────────────────────────
# 3.  TSFresh FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────────────
def _build_tsfresh_df(signal_2d: np.ndarray, channel_names: list) -> pd.DataFrame:
    T, C = signal_2d.shape
    assert C == len(channel_names)
    blocks, win_id, start = [], 0, 0
    while start + WIN_SAMPLES <= T:
        w  = signal_2d[start : start + WIN_SAMPLES]
        df = pd.DataFrame(w, columns=channel_names)
        df.insert(0, "time", np.arange(WIN_SAMPLES))
        df.insert(0, "id",   win_id)
        blocks.append(df)
        win_id += 1
        start  += STEP_SAMPLES
    if not blocks:
        w  = np.zeros((WIN_SAMPLES, C))
        df = pd.DataFrame(w, columns=channel_names)
        df.insert(0, "time", np.arange(WIN_SAMPLES))
        df.insert(0, "id",   0)
        blocks.append(df)
    return pd.concat(blocks, ignore_index=True)


def _window_features(signal_2d: np.ndarray, channel_names: list) -> np.ndarray:
    df    = _build_tsfresh_df(signal_2d, channel_names)
    feats = extract_features(
                df, column_id="id", column_sort="time",
                default_fc_parameters=FC_PARAMS,
                disable_progressbar=True, n_jobs=10)
    impute(feats)
    return np.nan_to_num(feats.values.astype(np.float64),
                         nan=0.0, posinf=0.0, neginf=0.0)


def _aggregate_window_features(wf: np.ndarray) -> np.ndarray:
    agg = np.concatenate([wf.mean(0), wf.std(0), skew(wf, 0), kurt(wf, 0)])
    return np.nan_to_num(agg, nan=0.0, posinf=0.0, neginf=0.0)


def build_feature_matrix(subjects: list, left_raw: list, right_raw: list):
    X_L, X_R = [], []
    n = len(subjects)
    for i, (sid, L, R) in enumerate(zip(subjects, left_raw, right_raw)):
        print(f"    Subject {i+1}/{n} ({sid}) …", end="\r", flush=True)
        L_feat, R_feat = get_subject_window_features(sid, L, R)
        X_L.append(_aggregate_window_features(L_feat))
        X_R.append(_aggregate_window_features(R_feat))
    print()
    X_L = np.array(X_L); X_R = np.array(X_R)
    X_B = np.concatenate([X_L, X_R], axis=1)
    return X_L, X_R, X_B


def build_dl_dataset_tsfresh(subjects: list, left_raw: list, right_raw: list,
                             labels: np.ndarray):
    seqs_L, seqs_R = [], []
    n = len(subjects)
    for i, (sid, L, R) in enumerate(zip(subjects, left_raw, right_raw)):
        print(f"    DL subject {i+1}/{n} ({sid}) …", end="\r", flush=True)
        L_feat, R_feat = get_subject_window_features(sid, L, R)
        seqs_L.append(L_feat.T.astype(np.float32))
        seqs_R.append(R_feat.T.astype(np.float32))
    print()

    T_MAX = max(s.shape[1] for s in seqs_L + seqs_R)

    def _pad(seqs, T):
        out = []
        for s in seqs:
            if s.shape[1] < T:
                pad = np.zeros((s.shape[0], T - s.shape[1]), dtype=s.dtype)
                out.append(np.concatenate([s, pad], axis=1))
            else:
                out.append(s[:, :T])
        return np.stack(out).astype(np.float32)

    X_L = _pad(seqs_L, T_MAX)
    X_R = _pad(seqs_R, T_MAX)
    X_B = np.concatenate([X_L, X_R], axis=1)
    y   = np.array(labels, dtype=np.int64)
    return X_L, X_R, X_B, y


def _raw_window_features(signal_2d: np.ndarray) -> np.ndarray:
    T, C = signal_2d.shape
    rows, start = [], 0
    while start + WIN_SAMPLES <= T:
        w = signal_2d[start : start + WIN_SAMPLES]
        rows.append(w.T.reshape(-1))
        start += STEP_SAMPLES
    if not rows:
        rows.append(np.zeros(C * WIN_SAMPLES, dtype=np.float32))
    return np.array(rows, dtype=np.float32)


def build_dl_dataset_raw_windowed(subjects: list, left_raw: list, right_raw: list,
                                  labels: np.ndarray):
    seqs_L, seqs_R = [], []
    n = len(subjects)
    for i, (sid, L, R) in enumerate(zip(subjects, left_raw, right_raw)):
        print(f"    Raw-windowed subject {i+1}/{n} ({sid}) …", end="\r", flush=True)
        seqs_L.append(_raw_window_features(L).T.astype(np.float32))
        seqs_R.append(_raw_window_features(R).T.astype(np.float32))
    print()

    T_MAX = max(s.shape[1] for s in seqs_L + seqs_R)

    def _pad(seqs, T):
        out = []
        for s in seqs:
            if s.shape[1] < T:
                pad = np.zeros((s.shape[0], T - s.shape[1]), dtype=s.dtype)
                out.append(np.concatenate([s, pad], axis=1))
            else:
                out.append(s[:, :T])
        return np.stack(out).astype(np.float32)

    X_L = _pad(seqs_L, T_MAX)
    X_R = _pad(seqs_R, T_MAX)
    X_B = np.concatenate([X_L, X_R], axis=1)
    y   = np.array(labels, dtype=np.int64)
    return X_L, X_R, X_B, y


def build_dl_dataset_raw_full(subjects: list, left_raw: list, right_raw: list,
                              labels: np.ndarray):
    def _crop_or_pad(sig: np.ndarray) -> np.ndarray:
        sig_T = sig.T
        T     = sig_T.shape[1]
        if T >= RAW_FULL_T:
            return sig_T[:, :RAW_FULL_T]
        pad = np.zeros((sig_T.shape[0], RAW_FULL_T - T), dtype=sig_T.dtype)
        return np.concatenate([sig_T, pad], axis=1)

    X_L = np.stack([_crop_or_pad(L) for L in left_raw]).astype(np.float32)
    X_R = np.stack([_crop_or_pad(R) for R in right_raw]).astype(np.float32)
    X_B = np.concatenate([X_L, X_R], axis=1)
    y   = np.array(labels, dtype=np.int64)
    return X_L, X_R, X_B, y


def build_dl_dataset(subjects: list, left_raw: list, right_raw: list,
                     labels: np.ndarray):
    if DL_INPUT_MODE == "tsfresh":
        return build_dl_dataset_tsfresh(subjects, left_raw, right_raw, labels)
    elif DL_INPUT_MODE == "raw_windowed":
        return build_dl_dataset_raw_windowed(subjects, left_raw, right_raw, labels)
    elif DL_INPUT_MODE == "raw_full":
        return build_dl_dataset_raw_full(subjects, left_raw, right_raw, labels)
    else:
        raise ValueError(f"Unknown DL_INPUT_MODE={DL_INPUT_MODE!r}")


# ─────────────────────────────────────────────────────────────────
# 4.  DELIVERABLE #1 — DATA ANALYSIS  (only used if RUN_D1=True)
# ─────────────────────────────────────────────────────────────────
def _parse_demographics(subjects: list, labels: np.ndarray) -> pd.DataFrame:
    rows = []
    for sid, lbl in zip(subjects, labels):
        base      = sid.split("_")[0]
        diagnosis = "Patient" if lbl == 1 else "Control"
        try:
            sex = "Male" if base[6].upper() == "M" else "Female"
            age = int(base[8:10])
        except (IndexError, ValueError):
            sex = "Unknown"
            age = float("nan")
        rows.append(dict(SubjectID=base, Diagnosis=diagnosis, Sex=sex, Age=age))
    return pd.DataFrame(rows)


def report_demographics(all_dirs: list):
    all_subjects, all_labels = [], []
    seen = set()
    for d in all_dirs:
        subs, labs, _, _ = load_split(d)
        for s, l in zip(subs, labs):
            base_s = s.split("_")[0]
            if base_s not in seen:
                seen.add(base_s)
                all_subjects.append(base_s)
                all_labels.append(l)

    df = _parse_demographics(all_subjects, np.array(all_labels))

    print("\n" + "=" * 55)
    print("DATASET DEMOGRAPHICS")
    print("=" * 55)
    for diag in ["Patient", "Control"]:
        sub      = df[df["Diagnosis"] == diag]
        n        = len(sub)
        age_vals = sub["Age"].dropna()
        age_str  = (f"{age_vals.mean():.1f} ± {age_vals.std():.1f}"
                    if len(age_vals) else "N/A")
        n_male   = (sub["Sex"] == "Male").sum()
        n_female = (sub["Sex"] == "Female").sum()
        print(f"  {diag:8s}: N={n:3d}  Age={age_str}  "
              f"Male={n_male}  Female={n_female}")
    print(f"  {'Total':8s}: N={len(df)}")
    print("=" * 55)

    df.to_csv(OUT_DIR / "demographics.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, diag in zip(axes, ["Patient", "Control"]):
        sub    = df[df["Diagnosis"] == diag]
        counts = sub["Sex"].value_counts()
        ax.bar(counts.index, counts.values,
               color=[FAU_BLUE, FAU_TEAL, FAU_GRAY], alpha=0.9)
        ax.set_title(f"{diag}  (N={len(sub)})", color=FAU_BLUE, fontweight="bold")
        ax.set_ylabel("Count"); ax.set_ylim(0, max(counts.values) + 2)
        for bar in ax.patches:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.3,
                    str(int(bar.get_height())),
                    ha="center", fontsize=11, fontweight="bold")
    fig.suptitle("Sex distribution by diagnosis group",
                 fontsize=12, color=FAU_BLUE, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "demographics_sex.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for diag, color in [("Patient", FAU_RED), ("Control", FAU_BLUE)]:
        ages = df[df["Diagnosis"] == diag]["Age"].dropna()
        if len(ages):
            ax.hist(ages, bins=10, alpha=0.6, color=color, label=diag)
    ax.set_xlabel("Age (years)"); ax.set_ylabel("Count")
    ax.set_title("Age distribution by diagnosis group",
                 color=FAU_BLUE, fontweight="bold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "demographics_age.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    return df


def plot_signal_examples(left_raw: list, right_raw: list,
                          labels: np.ndarray, subjects: list):
    pd_idx  = np.where(labels == 1)[0]
    ctl_idx = np.where(labels == 0)[0]
    if len(pd_idx) == 0 or len(ctl_idx) == 0:
        print("  [SKIP] Signal examples: need at least one PD and one Control.")
        return

    pd_i,  ctl_i  = pd_idx[0], ctl_idx[0]

    def _ch_labels(prefix):
        return ([f"{prefix}{i+1}" for i in range(8)] +
                ["Total", "COP_x", "COP_y", "Heel", "Toe"])

    for foot, arrays, prefix in [("Left", left_raw, "L"), ("Right", right_raw, "R")]:
        ch_labels = _ch_labels(prefix)
        fig, axes = plt.subplots(N_CHANNELS, 2,
                                  figsize=(14, N_CHANNELS * 1.4), sharex=False)
        for ch in range(N_CHANNELS):
            for col, (idx, diag, color) in enumerate([
                    (pd_i,  "Patient", FAU_RED),
                    (ctl_i, "Control", FAU_BLUE)]):
                sig  = arrays[idx]
                t    = np.arange(sig.shape[0]) / FS
                mask = t <= 10.0
                axes[ch, col].plot(t[mask], sig[mask, ch],
                                   color=color, linewidth=0.8)
                axes[ch, col].set_ylabel(ch_labels[ch], fontsize=8)
                axes[ch, col].tick_params(labelsize=7)
                if ch == 0:
                    axes[ch, col].set_title(
                        f"{diag}  ({subjects[idx]})",
                        color=color, fontweight="bold", fontsize=10)
                if ch == N_CHANNELS - 1:
                    axes[ch, col].set_xlabel("Time (s)", fontsize=8)

        fig.suptitle(f"{foot} foot — VGRF signals (first 10 s)",
                     fontsize=12, color=FAU_BLUE, fontweight="bold", y=1.01)
        fig.tight_layout()
        fname = f"signal_examples_{foot.lower()}.png"
        fig.savefig(FIG_DIR / fname, dpi=130, bbox_inches="tight")
        plt.close(fig)


def plot_pca_tsne(X_left: np.ndarray, X_right: np.ndarray,
                  X_both: np.ndarray, labels: np.ndarray,
                  method: str = "pca"):
    method = method.lower()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    titles     = ["Left foot", "Right foot", "Combined"]
    colors     = {0: FAU_BLUE, 1: FAU_RED}
    diag_names = {0: "Control", 1: "Patient"}

    for ax, X, title in zip(axes, [X_left, X_right, X_both], titles):
        scaler = StandardScaler()
        Xs     = scaler.fit_transform(X)
        Xs     = np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0)

        if method == "pca":
            proj   = PCA(n_components=2, random_state=42)
            coords = proj.fit_transform(Xs)
            var    = proj.explained_variance_ratio_ * 100
            xlabel = f"PC1 ({var[0]:.1f}%)"
            ylabel = f"PC2 ({var[1]:.1f}%)"
        else:
            n_pre  = min(50, Xs.shape[1], Xs.shape[0] - 1)
            Xpre   = PCA(n_components=n_pre, random_state=42).fit_transform(Xs)
            coords = TSNE(n_components=2, random_state=42,
                          perplexity=min(30, len(labels) // 2)
                         ).fit_transform(Xpre)
            xlabel = "t-SNE dim 1"; ylabel = "t-SNE dim 2"

        for lbl in np.unique(labels):
            mask = labels == lbl
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       c=colors[lbl], label=diag_names[lbl],
                       alpha=0.75, edgecolors="white", linewidths=0.4, s=60)

        ax.set_xlabel(xlabel, fontsize=10); ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11, color=FAU_BLUE, fontweight="bold")
        ax.axhline(0, color="grey", linewidth=0.5, alpha=0.4)
        ax.axvline(0, color="grey", linewidth=0.5, alpha=0.4)
        ax.legend(fontsize=9)

    fig.suptitle(f"{method.upper()} of TSFresh features — Fold 1 training set",
                 fontsize=13, color=FAU_BLUE, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{method}_plot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────
# 5.  METRICS
# ─────────────────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred, y_prob=None) -> dict:
    acc  = accuracy_score(y_true, y_pred)
    sens = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    spec = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    auc  = roc_auc_score(y_true, y_prob) if y_prob is not None else float("nan")
    return dict(accuracy=acc, sensitivity=sens, specificity=spec, auc=auc)


def _safe_prob(clf, X):
    if hasattr(clf, "predict_proba"):
        p = clf.predict_proba(X)
        return p[:, 1] if p.shape[1] == 2 else p[:, 0]
    return clf.decision_function(X)


# ─────────────────────────────────────────────────────────────────
# 6.  CLASSICAL ML
# ─────────────────────────────────────────────────────────────────
class CalibratedSVC(BaseEstimator, ClassifierMixin):
    def __init__(self, C=1.0, gamma="scale", kernel="rbf",
                 class_weight="balanced", cv=3, method="isotonic"):
        self.C            = C
        self.gamma        = gamma
        self.kernel       = kernel
        self.class_weight = class_weight
        self.cv           = cv
        self.method       = method

    def fit(self, X, y):
        base = SVC(C=self.C, gamma=self.gamma, kernel=self.kernel,
                   class_weight=self.class_weight, probability=False)
        try:
            self.calibrated_ = CalibratedClassifierCV(
                estimator=base, method=self.method, cv=self.cv)
        except TypeError:
            self.calibrated_ = CalibratedClassifierCV(
                base_estimator=base, method=self.method, cv=self.cv)
        self.calibrated_.fit(X, y)
        self.classes_ = self.calibrated_.classes_
        return self

    def predict(self, X):
        return self.calibrated_.predict(X)

    def predict_proba(self, X):
        return self.calibrated_.predict_proba(X)

    def decision_function(self, X):
        if hasattr(self.calibrated_, "decision_function"):
            return self.calibrated_.decision_function(X)
        return self.predict_proba(X)[:, 1]


def _fresh_select_indices(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    n_features = X.shape[1]
    col_names  = [f"f{i}" for i in range(n_features)]
    X_df = pd.DataFrame(X, columns=col_names)
    y_s  = pd.Series(y, index=X_df.index)

    try:
        X_sel = select_features(
            X_df, y_s, ml_task="classification",
            fdr_level=FRESH_FDR_LEVEL, n_jobs=FEATURE_SELECTION_N_JOBS,
            show_warnings=False)
        sel_idx = np.array(sorted(int(c[1:]) for c in X_sel.columns), dtype=int)
    except Exception as exc:
        print(f"    [FRESH WARN] selection failed ({exc}); falling back to F-test ranking")
        sel_idx = np.array([], dtype=int)

    if len(sel_idx) < MIN_SELECTED_FEATURES or len(sel_idx) > MAX_SELECTED_FEATURES:
        f_vals, p_vals = f_classif(X, y)
        p_vals = np.nan_to_num(p_vals, nan=1.0)
        order  = np.argsort(p_vals)

        if len(sel_idx) < MIN_SELECTED_FEATURES:
            k = min(MIN_SELECTED_FEATURES, n_features)
            sel_idx = np.sort(order[:k])
        else:
            capped = set(sel_idx.tolist())
            ranked = [i for i in order if i in capped]
            sel_idx = np.sort(np.array(ranked[:MAX_SELECTED_FEATURES], dtype=int))

    return sel_idx


def _prepare_features_for_fold(X_tr, y_tr, X_val, X_te, fold_id, foot_name):
    n_raw = X_tr.shape[1]

    vt      = VarianceThreshold(threshold=1e-10)
    X_tr_vt = vt.fit_transform(X_tr)

    sel_idx      = _fresh_select_indices(X_tr_vt, y_tr)
    X_tr_sel_raw = X_tr_vt[:, sel_idx]

    scaler   = StandardScaler()
    X_tr_sel = scaler.fit_transform(X_tr_sel_raw)

    n_comp   = min(N_PCA_COMPONENTS, X_tr_sel.shape[0] - 1, X_tr_sel.shape[1])
    n_comp   = max(n_comp, 1)
    pca      = PCA(n_components=n_comp, random_state=42)
    X_tr_pca = pca.fit_transform(X_tr_sel)

    def _transform(X):
        Xv = vt.transform(X)
        Xs = Xv[:, sel_idx]
        return scaler.transform(Xs)

    X_val_sel = _transform(X_val)
    X_te_sel  = _transform(X_te)
    X_val_pca = pca.transform(X_val_sel)
    X_te_pca  = pca.transform(X_te_sel)

    print(f"    [Fold {fold_id} | {foot_name}] raw={n_raw}  VT→{X_tr_vt.shape[1]}  "
          f"FRESH→{len(sel_idx)}  PCA→{n_comp} "
          f"({pca.explained_variance_ratio_.sum()*100:.1f}% var)")

    return dict(sel=(X_tr_sel, X_val_sel, X_te_sel),
                pca=(X_tr_pca, X_val_pca, X_te_pca),
                n_selected=len(sel_idx), n_pca=n_comp)


def train_eval_classical_fold(clf_class, param_grid,
                               X_tr_p, y_tr, X_val_p, y_val, X_te_p, y_te,
                               model_name, foot_name, fold_id):
    best_auc    = float("-inf")
    best_params = list(ParameterGrid(param_grid))[0]

    for params in ParameterGrid(param_grid):
        try:
            clf = clf_class(**params)
        except TypeError:
            clf = clf_class(**{k: v for k, v in params.items() if k != "random_state"})
        clf.fit(X_tr_p, y_tr)
        y_prob_val = _safe_prob(clf, X_val_p)

        if len(np.unique(y_val)) < 2:
            val_auc = float("nan")
        else:
            try:
                val_auc = roc_auc_score(y_val, y_prob_val)
            except Exception:
                val_auc = float("nan")

        if val_auc > best_auc:
            best_auc    = val_auc
            best_params = params

    X_tv_p = np.vstack([X_tr_p, X_val_p])
    y_tv   = np.concatenate([y_tr, y_val])
    try:
        best_clf = clf_class(**best_params)
    except TypeError:
        best_clf = clf_class(**{k: v for k, v in best_params.items() if k != "random_state"})
    best_clf.fit(X_tv_p, y_tv)

    y_pred_te = best_clf.predict(X_te_p)
    y_prob_te = _safe_prob(best_clf, X_te_p)
    test_m    = compute_metrics(y_te, y_pred_te, y_prob_te)
    cm        = confusion_matrix(y_te, y_pred_te)

    print(f"  Fold {fold_id} | [{model_name}] foot={foot_name} "
          f"best_params={best_params}  val_AUC={best_auc:.3f}  test_AUC={test_m['auc']:.3f}")

    res = dict(model=model_name, foot=foot_name, fold=fold_id,
              best_params=best_params, val_auc=best_auc,
              test_metrics=test_m, confusion_matrix=cm)
    return res, best_clf


# ─────────────────────────────────────────────────────────────────
# 6b.  DL — PCA / channel-norm preprocessing (leakage-free, per fold)
# ─────────────────────────────────────────────────────────────────
def _fit_pca(X_tr, n_components=N_PCA_COMPONENTS):
    N_tr, F, T = X_tr.shape
    n_comp      = min(n_components, F, N_tr * T)
    W_tr        = X_tr.transpose(0, 2, 1).reshape(-1, F)
    scaler      = StandardScaler()
    W_tr_s      = scaler.fit_transform(W_tr)
    pca         = PCA(n_components=n_comp, random_state=42)
    pca.fit(W_tr_s)
    return scaler, pca, n_comp


def _apply_pca(scaler, pca, X):
    N, F, T = X.shape
    W       = X.transpose(0, 2, 1).reshape(-1, F)
    W       = np.nan_to_num(scaler.transform(W), nan=0.0, posinf=0.0, neginf=0.0)
    P       = pca.transform(W).reshape(N, T, -1).transpose(0, 2, 1)
    return P.astype(np.float32)


def _align_T(X: np.ndarray, T_target: int) -> np.ndarray:
    T = X.shape[2]
    if T == T_target:
        return X
    if T > T_target:
        return X[:, :, :T_target]
    pad = np.zeros((X.shape[0], X.shape[1], T_target - T), dtype=X.dtype)
    return np.concatenate([X, pad], axis=2)


def _fit_channel_norm(X_tr: np.ndarray):
    mean = X_tr.mean(axis=(0, 2), keepdims=True)
    std  = X_tr.std(axis=(0, 2),  keepdims=True) + 1e-8
    return mean, std


def _apply_channel_norm(mean: np.ndarray, std: np.ndarray, X: np.ndarray) -> np.ndarray:
    return ((X - mean) / std).astype(np.float32)


# ─────────────────────────────────────────────────────────────────
# 7.  DEEP LEARNING MODELS  (only used when RUN_DL=True)
# ─────────────────────────────────────────────────────────────────
class ResidualBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch,  out_ch, 7, stride=stride, padding=3, bias=False)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 7, stride=1,      padding=3, bias=False)
        self.bn2   = nn.BatchNorm1d(out_ch)
        self.act   = nn.GELU()
        self.shortcut = (
            nn.Sequential(nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                          nn.BatchNorm1d(out_ch))
            if stride != 1 or in_ch != out_ch else nn.Identity())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r   = self.shortcut(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + r)


class ResNet1D(nn.Module):
    """
    Stem -> 3 stages (×1, ×2, ×4 channels) -> global pool -> head.
    d_model=32 -> base C=8 (final 32 features); d_model=64 -> C=16 (final 64).
    """
    def __init__(self, f_in: int, num_classes: int = 2,
                 dropout: float = 0.3, d_model: int = D_MODEL_SMALL) -> None:
        super().__init__()
        C = max(d_model // 4, 4)
        self.stem = nn.Sequential(
            nn.Conv1d(f_in, C, kernel_size=9, stride=2, padding=4, bias=False),
            nn.BatchNorm1d(C), nn.GELU())
        self.layer1 = nn.Sequential(ResidualBlock1D(C,   C,   stride=1),
                                     ResidualBlock1D(C,   C,   stride=1))
        self.layer2 = nn.Sequential(ResidualBlock1D(C,   C*2, stride=2),
                                     ResidualBlock1D(C*2, C*2, stride=1))
        self.layer3 = nn.Sequential(ResidualBlock1D(C*2, C*4, stride=2),
                                     ResidualBlock1D(C*4, C*4, stride=1))
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Dropout(dropout), nn.Linear(C * 4, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.head(x)


class LSTMClassifier(nn.Module):
    """
    `downsample=True` (set via main()'s factory only when
    DL_INPUT_MODE="raw_full") inserts two stride-2 convs before the LSTM,
    reducing T by 4x — without this, the LSTM had to scan up to T=2000 raw
    samples with no downsampling, relying on the hidden state to survive
    that entire span before reading the final timestep. For
    tsfresh/raw_windowed (downsample=False, the default), behaviour is
    unchanged from before: a direct per-timestep Linear projection into
    the LSTM, appropriate since T is already short (~30-50) and each
    timestep already represents a meaningful aggregated window.
    """
    def __init__(self, f_in, num_classes=2, dropout=0.3, d_model=D_MODEL_SMALL,
                 downsample: bool = False):
        super().__init__()
        if downsample:
            self.stem = nn.Sequential(
                nn.Conv1d(f_in, d_model, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm1d(d_model), nn.GELU(),
                nn.Conv1d(d_model, d_model, kernel_size=5, stride=2, padding=2, bias=False),
                nn.BatchNorm1d(d_model), nn.GELU())
            self.proj = None
        else:
            self.stem = None
            self.proj = nn.Linear(f_in, d_model)
        self.lstm = nn.LSTM(d_model, d_model, 2, batch_first=True,
                            dropout=dropout, bidirectional=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model*2, num_classes))

    def forward(self, x):
        if self.stem is not None:
            x = self.stem(x).permute(0, 2, 1)
        else:
            x = self.proj(x.permute(0, 2, 1))
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


class GRUClassifier(nn.Module):
    """Same downsample logic as LSTMClassifier — see its docstring."""
    def __init__(self, f_in, num_classes=2, dropout=0.3, d_model=D_MODEL_SMALL,
                 downsample: bool = False):
        super().__init__()
        if downsample:
            self.stem = nn.Sequential(
                nn.Conv1d(f_in, d_model, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm1d(d_model), nn.GELU(),
                nn.Conv1d(d_model, d_model, kernel_size=5, stride=2, padding=2, bias=False),
                nn.BatchNorm1d(d_model), nn.GELU())
            self.proj = None
        else:
            self.stem = None
            self.proj = nn.Linear(f_in, d_model)
        self.gru  = nn.GRU(d_model, d_model, 2, batch_first=True,
                           dropout=dropout, bidirectional=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model*2, num_classes))

    def forward(self, x):
        if self.stem is not None:
            x = self.stem(x).permute(0, 2, 1)
        else:
            x = self.proj(x.permute(0, 2, 1))
        out, _ = self.gru(x)
        return self.head(out[:, -1, :])


class PatchTST(nn.Module):
    """
    Learnable positional encoding (nn.Parameter). patch_len/stride scaled
    up by main()'s factory specifically for DL_INPUT_MODE="raw_full"
    (50, 25 instead of the default 4, 2) — the defaults were tuned for
    tsfresh/raw_windowed's ~30-50-step sequences; applied to raw_full's
    T~2000 they would produce ~999 tiny 0.04s patches.
    """
    def __init__(self, f_in, num_classes=2, patch_len=4, stride=2,
                 n_heads=4, n_layers=2, dropout=0.1, max_seq_len=512,
                 d_model=D_MODEL_SMALL):
        super().__init__()
        self.patch_len  = patch_len
        self.stride     = stride
        self.d_model    = d_model
        self.input_proj = nn.Linear(f_in, d_model)
        self.patch_proj = nn.Linear(d_model * patch_len, d_model)
        self.pos_drop   = nn.Dropout(dropout)

        max_patches = (max_seq_len - patch_len) // max(1, stride) + 2
        self.pe = nn.Parameter(torch.empty(1, max_patches, d_model))
        nn.init.trunc_normal_(self.pe, std=0.02)

        n_heads_actual = n_heads if d_model % n_heads == 0 else 1
        enc_layer = nn.TransformerEncoderLayer(
            d_model, n_heads_actual, d_model * 4, dropout, "gelu",
            batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, n_layers)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, num_classes))

    def forward(self, x):
        B, F, T = x.shape
        x = self.input_proj(x.permute(0, 2, 1)).permute(0, 2, 1)

        patches, start = [], 0
        while start + self.patch_len <= T:
            patches.append(x[:, :, start:start + self.patch_len])
            start += self.stride
        if not patches:
            pad = torch.zeros(B, self.d_model, self.patch_len - T, device=x.device)
            patches.append(torch.cat([x, pad], dim=2))

        NP     = len(patches)
        tokens = self.patch_proj(torch.stack(patches, 1).reshape(B, NP, -1))
        tokens = self.pos_drop(tokens + self.pe[:, :NP, :])
        enc    = self.transformer(tokens)
        return self.head(enc.mean(1))


# ─────────────────────────────────────────────────────────────────
# 8.  DL TRAINING HELPERS  (only used when RUN_DL=True)
# ─────────────────────────────────────────────────────────────────
def _make_loader(X, y, batch_size, shuffle):
    ds = TensorDataset(torch.tensor(X), torch.tensor(y))
    return DataLoader(ds, batch_size=min(batch_size, len(X)), shuffle=shuffle)


def _class_weights(y: np.ndarray) -> torch.Tensor:
    classes, counts = np.unique(y, return_counts=True)
    n_total         = len(y)
    n_classes       = len(classes)
    weights         = n_total / (n_classes * counts.astype(np.float32))
    w_tensor        = torch.zeros(n_classes, dtype=torch.float32)
    for cls, w in zip(classes, weights):
        w_tensor[int(cls)] = float(w)
    return w_tensor.to(DEVICE)


def _augment_batch(x: torch.Tensor) -> torch.Tensor:
    """
    Gaussian noise + amplitude jitter, applied during training only — runs
    on both the PCA-projected branch (tsfresh/raw_windowed) and the
    per-channel z-scored branch (raw_full), both roughly zero-mean /
    unit-scale so the same noise magnitude is reasonable for either.
    """
    x     = x + 0.1 * torch.randn_like(x)
    scale = 0.9 + 0.2 * torch.rand(x.shape[0], 1, 1, device=x.device)
    return x * scale


def _train_model(model, train_loader, val_X_t, val_y,
                 n_epochs, lr, warmup_epochs, patience,
                 class_weights: torch.Tensor = None,
                 weight_decay: float = 1e-2,
                 augment: bool = False,
                 mixup_alpha: float = 0.0,
                 label_smoothing: float = 0.0):
    """
    Train model and return (model, best_epoch).

    WARMUP-AWARE CHECKPOINTING (bug fix this revision): checkpoints are
    only considered "best" once ep >= warmup_epochs. During warmup, the LR
    ramps 0->1 linearly (lr_lambda), so early-epoch loss values reflect a
    barely-updated model, not genuine learning progress. Previously, a
    spuriously-low loss at e.g. epoch 1 could get locked in as "best,"
    causing the downstream refit to train for as little as 1 epoch on a
    near-random-init model — observed directly: Fold 1 / 1D-ResNet / Right
    foot printed "refit_epochs=1, test_AUC=0.71" despite val_AUC=0.81, and
    4/15 fold×foot combinations in that run scored test_AUC < 0.5.

    HP-search mode (val_X_t is not None): early stopping on post-warmup
    validation loss.
    Refit mode (val_X_t is None): no early stopping, trains for exactly
    n_epochs, saving state every epoch.

    mixup_alpha > 0 enables mixup (Zhang et al. 2018): each TRAINING batch
    is blended with a randomly permuted copy of itself (and its label)
    via a Beta(alpha, alpha) coefficient. Never applied at val/test.
    label_smoothing is passed straight to CrossEntropyLoss.
    """
    opt  = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    crit = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)

    def lr_lambda(ep):
        if ep < warmup_epochs:
            return (ep + 1) / warmup_epochs
        prog = (ep - warmup_epochs) / max(1, n_epochs - warmup_epochs)
        return 0.5 * (1.0 + np.cos(np.pi * prog))

    sched      = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    best_loss  = float("inf")
    best_state = None
    no_improve = 0
    best_epoch = 0

    for ep in range(n_epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            if augment:
                xb = _augment_batch(xb)
            opt.zero_grad()
            if mixup_alpha > 0:
                lam = float(np.random.beta(mixup_alpha, mixup_alpha))
                idx = torch.randperm(xb.size(0), device=xb.device)
                xb_mixed = lam * xb + (1 - lam) * xb[idx]
                out  = model(xb_mixed)
                loss = lam * crit(out, yb) + (1 - lam) * crit(out, yb[idx])
            else:
                loss = crit(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        if val_X_t is not None:
            model.eval()
            with torch.no_grad():
                val_loss = crit(model(val_X_t.to(DEVICE)),
                                torch.tensor(val_y, device=DEVICE)).item()
            if ep >= warmup_epochs:                       # ← warmup-aware fix
                if val_loss < best_loss:
                    best_loss  = val_loss
                    best_state = {k: v.cpu().clone()
                                  for k, v in model.state_dict().items()}
                    no_improve = 0
                    best_epoch = ep + 1
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        break
        else:
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = ep + 1

    if best_state is None:
        # n_epochs <= warmup_epochs: never reached a post-warmup checkpoint.
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        best_epoch = n_epochs

    model.load_state_dict(best_state)
    return model, best_epoch


def _eval_dl(model, X_np):
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X_np).to(DEVICE))
        probs  = torch.softmax(logits, 1)[:, 1].cpu().numpy()
        preds  = logits.argmax(1).cpu().numpy()
    return preds, probs


def train_eval_dl_fold(model_fn_factory, X_tr, y_tr, X_val, y_val, X_te, y_te,
                        model_name, foot_name, fold_id,
                        n_epochs=200, batch_size=16,
                        warmup_epochs=10, patience=25,
                        use_pca: bool = True) -> dict:
    """
    One fold of the DL pipeline.

    TOP-K DL ENSEMBLE (this revision): keeps the TOP_K_DL_ENSEMBLE best HP
    combos by validation AUC (instead of only the single best), refits
    each on train+val, and averages their test-set probabilities. With
    ~20 validation subjects, the val-AUC ranking of 24 HP combos is itself
    noisy (observed: mean val_AUC=0.91 vs mean test_AUC=0.62 across a
    1D-ResNet/tsfresh run, with 4/15 folds scoring below-random test AUC).
    Averaging several plausible combos reduces the chance that one
    spuriously-high-val-AUC combo dominates the fold's reported result —
    the DL analogue of the classical soft-voting ensemble.
    """
    if use_pca:
        scaler_pca, pca, n_comp = _fit_pca(X_tr)
        X_tr_p  = _apply_pca(scaler_pca, pca, X_tr)
        T_tr    = X_tr_p.shape[2]
        X_val_p = _align_T(_apply_pca(scaler_pca, pca, X_val), T_tr)
        X_te_p  = _align_T(_apply_pca(scaler_pca, pca, X_te),  T_tr)
    else:
        mean, std = _fit_channel_norm(X_tr)
        X_tr_p    = _apply_channel_norm(mean, std, X_tr)
        n_comp    = X_tr_p.shape[1]
        T_tr      = X_tr_p.shape[2]
        X_val_p   = _align_T(_apply_channel_norm(mean, std, X_val), T_tr)
        X_te_p    = _align_T(_apply_channel_norm(mean, std, X_te),  T_tr)

    cw = _class_weights(y_tr)

    lr_choices      = [1e-4, 3e-4, 1e-3]
    dropout_choices = [0.2, 0.4]
    wd_choices      = [1e-3, 1e-2]
    dm_choices      = [D_MODEL_SMALL, D_MODEL_LARGE]

    combo_results = []   # list of {hp, val_auc, epoch}

    for lr in lr_choices:
        for dropout in dropout_choices:
            for wd in wd_choices:
                for dm in dm_choices:
                    model  = model_fn_factory(n_comp, dropout, dm, T_tr).to(DEVICE)
                    loader = _make_loader(X_tr_p, y_tr, batch_size, shuffle=True)
                    model, epoch = _train_model(
                        model, loader, torch.tensor(X_val_p), y_val,
                        n_epochs, lr, warmup_epochs, patience,
                        class_weights=cw, weight_decay=wd, augment=True,
                        mixup_alpha=MIXUP_ALPHA, label_smoothing=LABEL_SMOOTHING)
                    _, probs_val = _eval_dl(model, X_val_p)

                    val_auc = (roc_auc_score(y_val, probs_val)
                               if len(np.unique(y_val)) == 2 else float("nan"))

                    if not np.isnan(val_auc):
                        combo_results.append(dict(
                            hp={"lr": lr, "dropout": dropout,
                               "weight_decay": wd, "d_model": dm},
                            val_auc=val_auc,
                            epoch=max(epoch, warmup_epochs)))   # epoch floor

    if not combo_results:
        combo_results = [dict(
            hp={"lr": 3e-4, "dropout": 0.2, "weight_decay": 1e-2, "d_model": D_MODEL_SMALL},
            val_auc=float("nan"), epoch=warmup_epochs)]

    combo_results.sort(key=lambda c: c["val_auc"], reverse=True)
    top_k = combo_results[:TOP_K_DL_ENSEMBLE]

    X_tv_p = np.vstack([X_tr_p, X_val_p])
    y_tv   = np.concatenate([y_tr, y_val])
    cw_tv  = _class_weights(y_tv)

    probs_te_list = []
    for combo in top_k:
        hp = combo["hp"]
        m  = model_fn_factory(n_comp, hp["dropout"], hp["d_model"], T_tr).to(DEVICE)
        tv_loader = _make_loader(X_tv_p, y_tv, batch_size, shuffle=True)
        m, _ = _train_model(
            m, tv_loader,
            val_X_t       = None,
            val_y         = None,
            n_epochs      = combo["epoch"],
            lr            = hp["lr"],
            warmup_epochs = min(warmup_epochs, combo["epoch"]),
            patience      = combo["epoch"] + 1,
            class_weights = cw_tv,
            weight_decay  = hp["weight_decay"],
            augment       = True,
            mixup_alpha   = MIXUP_ALPHA,
            label_smoothing = LABEL_SMOOTHING)
        _, probs_te = _eval_dl(m, X_te_p)
        probs_te_list.append(probs_te)

    probs_te_avg = np.mean(probs_te_list, axis=0)
    y_pred_te    = (probs_te_avg >= 0.5).astype(int)
    test_m = compute_metrics(y_te, y_pred_te, probs_te_avg)
    cm     = confusion_matrix(y_te, y_pred_te)

    best_val_auc = top_k[0]["val_auc"]
    print(f"  Fold {fold_id} | [{model_name}] foot={foot_name}  "
          f"ensembled_top_{len(top_k)}_HPs  "
          f"best_val_AUC={best_val_auc:.3f}  test_AUC={test_m['auc']:.3f}")

    return dict(model=model_name, foot=foot_name, fold=fold_id,
                best_params={"ensemble_top_k": [c["hp"] for c in top_k],
                             "epochs": [c["epoch"] for c in top_k]},
                val_auc=best_val_auc,
                test_metrics=test_m, confusion_matrix=cm)


# ─────────────────────────────────────────────────────────────────
# 9.  AGGREGATE FOLD RESULTS
# ─────────────────────────────────────────────────────────────────
def aggregate_folds(fold_results: list) -> dict:
    keys = ["accuracy", "sensitivity", "specificity", "auc"]
    agg  = {}
    for k in keys:
        vals          = [r["test_metrics"][k] for r in fold_results]
        agg[k]        = round(float(np.nanmean(vals)), 4)
        agg[k+"_std"] = round(float(np.nanstd(vals)),  4)
    return agg


# ─────────────────────────────────────────────────────────────────
# 10.  PLOTTING
# ─────────────────────────────────────────────────────────────────
def plot_confusion_matrix(cm, model_name, foot_name, metrics, tag=""):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                cbar=False, linewidths=0.5, linecolor="white",
                annot_kws={"size": 16, "weight": "bold"})
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True",      fontsize=12)
    ax.set_xticklabels(["Control", "PD"], fontsize=11)
    ax.set_yticklabels(["Control", "PD"], fontsize=11, rotation=0)
    title = (f"{model_name}  |  {foot_name} foot  {tag}\n"
             f"Acc={metrics['accuracy']:.2f}  Sens={metrics['sensitivity']:.2f}  "
             f"Spec={metrics['specificity']:.2f}  AUC={metrics['auc']:.2f}")
    ax.set_title(title, fontsize=9, pad=8, color=FAU_BLUE, fontweight="bold")
    fig.tight_layout()
    fname = f"cm_{model_name.replace(' ', '_')}_{foot_name}{tag}.png"
    fig.savefig(FIG_DIR / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_bar_comparison(summary_rows, foot_name):
    subset = [r for r in summary_rows if r["Foot"] == foot_name]
    if not subset: return
    models = [r["Model"] for r in subset]
    x, w   = np.arange(len(models)), 0.20
    keys   = ["Acc", "Sens", "Spec", "AUC"]
    colors = [FAU_BLUE, FAU_TEAL, FAU_RED, FAU_GRAY]
    fig, ax = plt.subplots(figsize=(max(10, len(models)*1.8), 5))
    for i, (k, c) in enumerate(zip(keys, colors)):
        ax.bar(x + i*w, [r[k] for r in subset], w, label=k, color=c, alpha=0.9)
        ax.errorbar(x + i*w, [r[k] for r in subset],
                    yerr=[r[k+"_std"] for r in subset],
                    fmt="none", color="black", capsize=3, linewidth=1)
    ax.set_xticks(x + 1.5*w)
    ax.set_xticklabels(models, rotation=25, ha="right", fontsize=10)
    ax.set_ylim(0, 1.12); ax.set_ylabel("Score")
    ax.set_title(f"Model Comparison — {foot_name} foot (mean ± std, 5 folds)",
                 fontsize=12, color=FAU_BLUE, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, alpha=0.5)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"comparison_{foot_name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_fold_auc(all_fold_results, foot_name):
    subset = {m: [] for m in dict.fromkeys(r["model"] for r in all_fold_results)}
    for r in all_fold_results:
        if r["foot"] == foot_name:
            subset[r["model"]].append(r["test_metrics"]["auc"])
    models = [m for m, v in subset.items() if v]
    data   = [subset[m] for m in models]
    fig, ax = plt.subplots(figsize=(max(8, len(models)*1.5), 4))
    bp = ax.boxplot(data, patch_artist=True,
                    medianprops=dict(color=FAU_RED, linewidth=2))
    for p in bp["boxes"]:
        p.set_facecolor(FAU_TEAL); p.set_alpha(0.7)
    ax.set_xticks(range(1, len(models)+1))
    ax.set_xticklabels(models, rotation=20, ha="right", fontsize=10)
    ax.set_ylabel("Test AUC"); ax.set_ylim(0, 1.05)
    ax.set_title(f"Per-Fold Test AUC — {foot_name} foot",
                 color=FAU_BLUE, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"fold_auc_{foot_name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────
# 11.  MAIN
# ─────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)

    if RUN_D1:
        print("\n── D1: Demographics ──")
        all_split_dirs = [
            DATA_ROOT / f"Fold_{fid}" / split
            for fid in range(1, N_FOLDS + 1)
            for split in ("training", "validation", "test")
        ]
        report_demographics(all_split_dirs)

        print("\n── D1: Signal examples ──")
        fold1_train = DATA_ROOT / "Fold_1" / "training"
        subs_f1, y_f1, L_f1, R_f1 = load_split(fold1_train)
        plot_signal_examples(L_f1, R_f1, y_f1, subs_f1)

        print("\n── D1: PCA / t-SNE ──")
        XL_f1, XR_f1, XB_f1 = build_feature_matrix(subs_f1, L_f1, R_f1)
        plot_pca_tsne(XL_f1, XR_f1, XB_f1, y_f1, method="pca")
        plot_pca_tsne(XL_f1, XR_f1, XB_f1, y_f1, method="tsne")
    else:
        print("\n[RUN_D1=False] Skipping Deliverable #1 plots.")

    rf_grid = {
        "n_estimators": [200, 500, 1000], "max_depth": [None, 10, 20],
        "max_features": ["sqrt", "log2", 0.3], "min_samples_leaf": [1, 2],
        "class_weight": ["balanced"], "random_state": [42],
    }
    svm_l_grid = {
        "C": [0.001, 0.01, 0.1, 1, 10, 100], "kernel": ["linear"],
        "probability": [True], "class_weight": ["balanced"],
    }
    svm_r_grid = {
        "C": [0.01, 0.1, 1, 10, 100], "gamma": ["scale", "auto", 0.1, 0.01, 0.001],
        "kernel": ["rbf"], "class_weight": ["balanced"], "cv": [3], "method": ["isotonic"],
    }
    logreg_grid = {
        "C": [0.001, 0.01, 0.1, 1, 10, 100], "penalty": ["elasticnet"],
        "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9], "solver": ["saga"],
        "max_iter": [5000], "class_weight": ["balanced"], "random_state": [42],
    }

    classical_models = [
        ("Random Forest",           RandomForestClassifier, rf_grid,     False),
        ("SVM-Linear",              SVC,                    svm_l_grid,  False),
        ("SVM-RBF",                 CalibratedSVC,          svm_r_grid,  True),
        ("Logistic Regression",     LogisticRegression,     logreg_grid, False),
        ("Ensemble (RF+SVM-L+LR)", "ENSEMBLE",
            {"members": ["Random Forest", "SVM-Linear", "Logistic Regression"]}, False),
    ]

    dl_model_fns = {
        "1D-ResNet": ResNet1D,
        # "LSTM":      LSTMClassifier,
        # "GRU":       GRUClassifier,
        # "PatchTST":  PatchTST,
    }
    feet = ["Left", "Right", "Combined"]

    all_fold_results = []

    for fold_id in range(1, N_FOLDS + 1):
        fold_dir  = DATA_ROOT / f"Fold_{fold_id}"
        train_dir = fold_dir / "training"
        val_dir   = fold_dir / "validation"
        test_dir  = fold_dir / "test"

        print(f"\n{'='*65}")
        print(f"FOLD {fold_id}  —  loading splits …")

        subs_tr,  y_tr,  L_tr,  R_tr  = load_split(train_dir)
        subs_val, y_val, L_val, R_val = load_split(val_dir)
        subs_te,  y_te,  L_te,  R_te  = load_split(test_dir)
        print(f"  Train={len(y_tr)}  Val={len(y_val)}  Test={len(y_te)}")

        if RUN_CLASSICAL:
            print(f"  Extracting classical features — training …")
            XL_tr, XR_tr, XB_tr = build_feature_matrix(subs_tr, L_tr, R_tr)
            print(f"  Extracting classical features — validation …")
            XL_val, XR_val, XB_val = build_feature_matrix(subs_val, L_val, R_val)
            print(f"  Extracting classical features — test …")
            XL_te, XR_te, XB_te = build_feature_matrix(subs_te, L_te, R_te)

            cl_splits = {
                "Left":     (XL_tr, XL_val, XL_te),
                "Right":    (XR_tr, XR_val, XR_te),
                "Combined": (XB_tr, XB_val, XB_te),
            }

            print(f"\n  ── Feature selection (FRESH) ──")
            feature_preps = {}
            for foot in feet:
                Xtr, Xval, Xte = cl_splits[foot]
                feature_preps[foot] = _prepare_features_for_fold(
                    Xtr, y_tr, Xval, Xte, fold_id, foot)

            fitted_models = {foot: {} for foot in feet}

            for model_name, clf_class, param_grid, use_pca in classical_models:
                print(f"\n  ── {model_name} ──")

                if clf_class == "ENSEMBLE":
                    members = param_grid["members"]
                    for foot in feet:
                        prep      = feature_preps[foot]
                        X_te_sel  = prep["sel"][2]

                        probs_list = []
                        for member_name in members:
                            if member_name not in fitted_models[foot]:
                                raise RuntimeError(
                                    f"Ensemble member '{member_name}' not fitted "
                                    f"yet for foot={foot}.")
                            clf_m = fitted_models[foot][member_name]
                            probs_list.append(_safe_prob(clf_m, X_te_sel))

                        probs_te = np.mean(probs_list, axis=0)
                        preds_te = (probs_te >= 0.5).astype(int)
                        test_m   = compute_metrics(y_te, preds_te, probs_te)
                        cm       = confusion_matrix(y_te, preds_te)

                        print(f"  Fold {fold_id} | [{model_name}] foot={foot}  "
                              f"members={members}  test_AUC={test_m['auc']:.3f}")

                        all_fold_results.append(dict(
                            model=model_name, foot=foot, fold=fold_id,
                            best_params={"members": members, "weights": "uniform"},
                            val_auc=float("nan"),
                            test_metrics=test_m, confusion_matrix=cm))
                    continue

                for foot in feet:
                    prep = feature_preps[foot]
                    Xtr_p, Xval_p, Xte_p = prep["pca"] if use_pca else prep["sel"]
                    res, best_clf = train_eval_classical_fold(
                        clf_class, param_grid,
                        Xtr_p, y_tr, Xval_p, y_val, Xte_p, y_te,
                        model_name, foot, fold_id)
                    all_fold_results.append(res)
                    fitted_models[foot][model_name] = best_clf
        else:
            print(f"\n  [RUN_CLASSICAL=False] Skipping classical models.")

        if RUN_DL:
            print(f"\n  Extracting DL features (mode={DL_INPUT_MODE!r}) — training …")
            DL_XL_tr, DL_XR_tr, DL_XB_tr, DL_y_tr = build_dl_dataset(subs_tr, L_tr, R_tr, y_tr)
            print(f"  Extracting DL features — validation …")
            DL_XL_val, DL_XR_val, DL_XB_val, DL_y_val = build_dl_dataset(subs_val, L_val, R_val, y_val)
            print(f"  Extracting DL features — test …")
            DL_XL_te, DL_XR_te, DL_XB_te, DL_y_te = build_dl_dataset(subs_te, L_te, R_te, y_te)
            print(f"  DL input shapes (mode={DL_INPUT_MODE!r}): "
                  f"Left={DL_XL_tr.shape}  Right={DL_XR_tr.shape}  Both={DL_XB_tr.shape}")

            dl_splits = {
                "Left":     (DL_XL_tr, DL_XL_val, DL_XL_te),
                "Right":    (DL_XR_tr, DL_XR_val, DL_XR_te),
                "Combined": (DL_XB_tr, DL_XB_val, DL_XB_te),
            }
            dl_y_splits = (DL_y_tr, DL_y_val, DL_y_te)

            use_pca_for_mode = (DL_INPUT_MODE != "raw_full")

            for model_name, model_cls in dl_model_fns.items():
                print(f"\n  ── {model_name} ──")
                for foot in feet:
                    Xtr, Xval, Xte              = dl_splits[foot]
                    y_tr_dl, y_val_dl, y_te_dl  = dl_y_splits

                    def factory(nc, do, dm, T, mc=model_cls):
                        kwargs = dict(f_in=nc, dropout=do, d_model=dm)
                        ctor_params = inspect.signature(mc.__init__).parameters
                        if "max_seq_len" in ctor_params:
                            kwargs["max_seq_len"] = T
                        if (mc is PatchTST and DL_INPUT_MODE == "raw_full"
                                and "patch_len" in ctor_params):
                            kwargs["patch_len"] = 50
                            kwargs["stride"]    = 25
                        if "downsample" in ctor_params:
                            kwargs["downsample"] = (DL_INPUT_MODE == "raw_full")
                        return mc(**kwargs)

                    res = train_eval_dl_fold(
                        factory, Xtr, y_tr_dl, Xval, y_val_dl, Xte, y_te_dl,
                        model_name, foot, fold_id,
                        use_pca=use_pca_for_mode)
                    all_fold_results.append(res)

    print(f"\n{'='*65}")
    print("Aggregating results across 5 folds …")

    summary_rows = []
    groups = {}
    for r in all_fold_results:
        key = (r["model"], r["foot"])
        groups.setdefault(key, []).append(r)

    for (model_name, foot), fold_list in groups.items():
        agg = aggregate_folds(fold_list)
        summary_rows.append(dict(
            Model=model_name, Foot=foot,
            Acc=agg["accuracy"],     Acc_std=agg["accuracy_std"],
            Sens=agg["sensitivity"], Sens_std=agg["sensitivity_std"],
            Spec=agg["specificity"], Spec_std=agg["specificity_std"],
            AUC=agg["auc"],          AUC_std=agg["auc_std"],
        ))

    print("Plotting confusion matrices …")

    for foot in feet:
        foot_groups = {(m, f): flist for (m, f), flist in groups.items() if f == foot}
        best_key = max(foot_groups,
                       key=lambda k: np.nanmean([r["test_metrics"]["auc"] for r in foot_groups[k]]))
        best_fold_res = max(foot_groups[best_key], key=lambda r: r["test_metrics"]["auc"])
        plot_confusion_matrix(
            best_fold_res["confusion_matrix"], best_key[0], foot,
            best_fold_res["test_metrics"], tag=f"_BEST_fold{best_fold_res['fold']}")
        print(f"  Best for {foot}: {best_key[0]}  (fold {best_fold_res['fold']}, "
              f"AUC={best_fold_res['test_metrics']['auc']:.3f})")

    CLASSICAL_NAMES = ("Random Forest", "Extra Trees", "SVM-Linear", "SVM-RBF",
                       "Logistic Regression", "Ensemble (RF+SVM-L+LR)")
    for (model_name, foot), fold_list in groups.items():
        if model_name in CLASSICAL_NAMES:
            best_fold_res = max(fold_list, key=lambda r: r["test_metrics"]["auc"])
            plot_confusion_matrix(
                best_fold_res["confusion_matrix"], model_name, foot,
                best_fold_res["test_metrics"], tag=f"_fold{best_fold_res['fold']}")

    for foot in feet:
        plot_bar_comparison(summary_rows, foot)
        plot_fold_auc(all_fold_results, foot)

    df = pd.DataFrame(summary_rows)
    filename_summary = "resnet_dl_improved_results_table_" + str(TRIM_SEC) + "_" + DL_INPUT_MODE + "_v1.csv"
    df.to_csv(OUT_DIR / filename_summary, index=False)

    print(f"\n{'='*65}")
    print(df.to_string(index=False))
    print(f"{'='*65}")
    print(f"\nResults → {OUT_DIR / filename_summary}")
    print(f"Figures → {FIG_DIR}/")

    fold_rows = []
    for r in all_fold_results:
        tm = r["test_metrics"]
        fold_rows.append(dict(
            Fold=r["fold"], Model=r["model"], Foot=r["foot"],
            BestParams=str(r.get("best_params", "")),
            Val_AUC=round(r.get("val_auc", float("nan")), 4),
            Test_Acc=round(tm["accuracy"],    4),
            Test_Sens=round(tm["sensitivity"], 4),
            Test_Spec=round(tm["specificity"], 4),
            Test_AUC=round(tm["auc"],         4),
        ))
    filename = "resnet_dl_improved_results_per_fold_" + str(TRIM_SEC) + "_" + DL_INPUT_MODE + "_v1.csv"
    pd.DataFrame(fold_rows).to_csv(OUT_DIR / filename, index=False)
    print(f"Per-fold detail → {filename}")


if __name__ == "__main__":
    main()