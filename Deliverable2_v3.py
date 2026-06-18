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

RUN-MODE FLAGS  (this revision)
--------------------------------
  RUN_D1 = False   Skip Deliverable #1 plots (demographics, signal
                   examples, PCA/t-SNE). Set True to regenerate them.
  RUN_DL = False   Skip the deep-learning models entirely. Set True to
                   also train 1D-ResNet / LSTM / GRU / PatchTST per fold.

  Flags are used instead of literally deleting/commenting hundreds of
  lines: every DL/D1 function definition stays intact (zero runtime cost
  if unused) and either path is a one-line toggle. This run focuses
  exclusively on the four CLASSICAL models and on fixing their accuracy.

Deliverable #1 additions (only if RUN_D1=True)
-------------------------------------------------
  - Dataset demographics table (N patients/controls, age mean±std, sex)
    parsed from GaitPaDB filename conventions.
  - Raw signal examples: one PD and one Control subject overlaid per channel.
  - PCA plots (PC1 vs PC2, coloured by diagnosis) for Left/Right/Combined,
    using Fold_1/training features (z-score normalised before PCA).

Fold workflow (per fold, per foot, per classical model)
---------------------------------------------------------
1. TRAINING   → fit VarianceThreshold + FRESH relevance filter + StandardScaler
                (+ PCA for SVM-RBF only) on TRAINING data only.
2. VALIDATION → transform with the fitted preprocessor,
                sweep hyperparameter grid → pick best combo by AUC.
3. REFIT      → refit best classifier on train + val combined.
4. TEST       → evaluate refitted model → record metrics.

Final reported metrics = mean ± std across the 5 test sets.

Key design decisions
--------------------
CHANNELS (13 per foot)
  8 FSR sensors + total force + COP_x + COP_y + heel_mean + toe_mean.
  Derived channels capture postural instability and heel/toe loading
  asymmetry — top-ranked PD biomarkers (Navita et al. 2025, Fig. 5).

SIGNAL PREPROCESSING
  1. Trim TRIM_SEC=20s from each end — removes turning/acceleration artefacts.
  2. Median filter (kernel 11) per channel — removes sensor impulse noise.
  3. Compute derived channels (COP, heel, toe) from the 8 FSR sensors.

FEATURE SELECTION  (the main fix in this revision)
----------------------------------------------------
  PROBLEM: the aggregated TSFresh feature matrix has tens of thousands of
  columns (EfficientFCParameters × 13 channels × 4 aggregation stats ×
  1-2 feet), while each fold has only ~60 training subjects. Fitting PCA
  directly on this space — as in the previous revision — lets PCA's
  *unsupervised* variance-maximisation latch onto nuisance variance (e.g.
  which GaitPaDB sub-study — Ga/Ju/Si — a subject came from, which uses
  different sensor hardware/sampling) rather than PD-related signal. This
  was the main driver of the mediocre AUCs (~0.65-0.79) in the previous run.

  NEW PIPELINE  (fit on TRAINING data only, per fold per foot, in
  _prepare_features_for_fold — computed ONCE and reused by all 4 models):
    1. VarianceThreshold(threshold=1e-10)
         Drop exact-constant columns (numerical hygiene only).
    2. FRESH relevance filter  (tsfresh.select_features)
         Per-feature Mann-Whitney U test vs. the PD/Control label, with
         Benjamini-Hochberg FDR control (fdr_level=FRESH_FDR_LEVEL). This
         is the *supervised* step that was missing: it keeps only features
         that are statistically associated with diagnosis, collapsing
         tens of thousands of columns down to tens/hundreds. Bounded to
         [MIN_SELECTED_FEATURES, MAX_SELECTED_FEATURES] via an F-test
         fallback/cap (see _fresh_select_indices).
    3. StandardScaler — fit on the FRESH-selected training features.
    4. PCA(N_PCA_COMPONENTS) — fit on top of (2)+(3); used ONLY for
         SVM-RBF, where a compact well-conditioned input space makes the
         RBF kernel's distance metric meaningful again. Random Forest,
         SVM-Linear, and Logistic Regression use the FRESH-selected +
         scaled features directly (no PCA) — they handle the residual
         dimensionality via per-split feature subsampling / L1-L2
         regularisation, and benefit from the extra (non-PCA-mixed)
         features for interpretability.

  Expected cost: FRESH selection on ~40k-80k columns with n_jobs=1 takes
  roughly 30-90s per (fold, foot) → ~10-25 min total across 5 folds × 3
  feet. Increase FEATURE_SELECTION_N_JOBS on multi-core machines.

SIGNAL PREPROCESSING / SUBJECT-LEVEL FEATURE CACHE
  The 5 GaitPaDB folds reuse the SAME underlying subject recordings across
  training/validation/test splits — e.g. subject "GaCo02" might appear in
  Fold_1/training, Fold_3/validation, and Fold_5/test. Since the raw
  recording for a given subject is identical wherever it appears, and
  TSFresh extraction is deterministic given the extraction configuration,
  caching is keyed by SUBJECT ID (+ extraction config) rather than by
  (fold, split).

  This means each subject's expensive TSFresh extraction runs AT MOST ONCE
  across the entire pipeline. Cache key encodes: subject_id, WIN_SEC,
  STEP_SEC, TRIM_SEC, MEDIAN_FILTER_LEN, N_CHANNELS, FC_PARAMS class name.
  Any change to these parameters busts the cache automatically.
  An in-memory dict additionally avoids re-reading the same .npz from disk
  when a subject appears in multiple splits within a single run.
"""

import warnings
warnings.filterwarnings("ignore")

import hashlib
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
# See module docstring "RUN-MODE FLAGS".
RUN_D1         = False   # Deliverable #1 plots (demographics, signal examples, PCA/t-SNE)
RUN_CLASSICAL  = False   # Classical ML models (RF, SVM-L, SVM-RBF, LR, Ensemble)
RUN_DL         = True    # Deep learning models (1D-ResNet, LSTM, GRU, PatchTST)

DATA_ROOT   = Path("Data/splits")
N_FOLDS     = 5
OUT_DIR     = Path("results");        OUT_DIR.mkdir(exist_ok=True)
FIG_DIR     = OUT_DIR / "figures";    FIG_DIR.mkdir(exist_ok=True)
CACHE_DIR   = OUT_DIR / "feature_cache"; CACHE_DIR.mkdir(exist_ok=True)

FS           = 100          # Hz
WIN_SEC      = 5.0
STEP_SEC     = 2.5
WIN_SAMPLES  = int(WIN_SEC  * FS)     # 500 samples / window
STEP_SAMPLES = int(STEP_SEC * FS)     # 250 samples step

# GaitPaDB raw column layout (0-indexed after np.loadtxt):
#   col  0     : time
#   cols 1–8   : Left  VGRF  L1..L8  (8 individual pressure sensors)
#   cols 9–16  : Right VGRF  R1..R8  (8 individual pressure sensors)
#   col 17     : Total force under left foot  [= sum(L1..L8) ± float rounding]
#   col 18     : Total force under right foot [= sum(R1..R8) ± float rounding]
#
# L_COLS / R_COLS are kept as documentation; load_subject extracts columns
# explicitly to also compute the 4 derived channels.
L_COLS = list(range(1, 9)) + [17]   # L1–L8 + left total  (reference)
R_COLS = list(range(9, 17)) + [18]  # R1–R8 + right total (reference)

# ── Signal preprocessing constants ───────────────────────────────
TRIM_SEC          = 0.0
TRIM_SAMPLES      = int(TRIM_SEC * FS)
MEDIAN_FILTER_LEN = 11               # must be odd

# ── Sensor layout for Centre-of-Pressure computation ─────────────
SENSOR_X = np.array([-500, -700, -300, -700, -300, -700, -300, -500],
                    dtype=np.float32)
SENSOR_Y = np.array([-800, -400, -400,    0,    0,  400,  400,  800],
                    dtype=np.float32)

# ── Channel count after derived features ─────────────────────────
# Per foot: 8 FSR + 1 total + COP_x + COP_y + heel_mean + toe_mean = 13
N_CHANNELS = 13

# ── Channel names for TSFresh (must align with N_CHANNELS = 13) ──
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

# PCA components — used only for SVM-RBF (after FRESH selection) and for
# DL (if RUN_DL=True). Random Forest / SVM-Linear / Logistic Regression use
# the FRESH-selected + scaled features directly (no PCA).
N_PCA_COMPONENTS = 30

# ── FRESH supervised feature-selection constants ──────────────────
# See module docstring "FEATURE SELECTION" for rationale.
FRESH_FDR_LEVEL          = 0.05   # Benjamini-Hochberg FDR level. Loosened
                                   # from 0.05: with ~60 balanced training
                                   # subjects/fold, Mann-Whitney U has
                                   # reasonable power at FDR=0.10, admitting
                                   # more borderline-relevant features for
                                   # RF / Extra Trees to exploit. If the
                                   # looser threshold pushes the selected
                                   # count above MAX_SELECTED_FEATURES, the
                                   # existing p-value cap still applies.
MIN_SELECTED_FEATURES    = 30     # floor: fall back to top-K by F-test p-value
MAX_SELECTED_FEATURES    = 300    # ceiling: cap to top-K by p-value if exceeded
FEATURE_SELECTION_N_JOBS = 1      # increase on multi-core machines

D_MODEL_SMALL    = 32
D_MODEL_LARGE    = 64
D_MODEL          = D_MODEL_SMALL    # default; swept per-model in DL HP grid
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")
print(f"RUN_D1={RUN_D1}  RUN_CLASSICAL={RUN_CLASSICAL}  RUN_DL={RUN_DL}")

FAU_BLUE = "#003865"
FAU_TEAL = "#00b1eb"
FAU_RED  = "#c8102e"
FAU_GRAY = "#98a4ae"

FC_PARAMS = EfficientFCParameters()


# ─────────────────────────────────────────────────────────────────
# 1.  SUBJECT-LEVEL FEATURE CACHE
# ─────────────────────────────────────────────────────────────────
_MEMORY_CACHE: dict = {}   # subject_id -> (L_features, R_features)


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
    """
    Return (L_features, R_features) — each (N_windows, F_tsfresh) — from the
    on-disk cache for this subject under the current extraction config,
    or None on a cache miss / invalid cache file.
    """
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
    """
    Return (L_features, R_features) — each (N_windows, F_tsfresh) — for one
    subject, using the in-memory cache first, then the on-disk cache, and
    only running TSFresh extraction on a full cache miss. The result is
    written to both caches so subsequent lookups are instant.
    """
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
    """
    Compute 4 derived channels from 8 FSR sensors (T, 8) → (T, 4).

    COP_x, COP_y  — Centre of Pressure (weighted sensor positions).
                    Captures postural instability — top PD biomarker.
    heel_mean     — Mean force under heel sensor (index 0, Y=−800).
    toe_mean      — Mean force under toe sensor (index 7, Y=+800).
    """
    eps       = 1e-6
    total     = fsr.sum(axis=1, keepdims=True) + eps
    cop_x     = (fsr * SENSOR_X).sum(axis=1, keepdims=True) / total
    cop_y     = (fsr * SENSOR_Y).sum(axis=1, keepdims=True) / total
    heel_mean = fsr[:, [0]].mean(axis=1, keepdims=True)
    toe_mean  = fsr[:, [7]].mean(axis=1, keepdims=True)
    return np.concatenate([cop_x, cop_y, heel_mean, toe_mean], axis=1)


def load_subject(path: Path):
    """
    Load one GaitPaDB recording and return (left, right) arrays of shape (T, 13).

    Preprocessing:
      1. Trim TRIM_SEC from each end (removes turn/acceleration artefacts).
      2. Median filter per channel (removes sensor impulse noise).
      3. Compute COP_x, COP_y, heel_mean, toe_mean → 13 channels per foot.

    Returns (None, None) if the recording is too short after trimming.
    """
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

    L = np.concatenate([L_fsr, L_tot, L_derived], axis=1)   # (T, 13)
    R = np.concatenate([R_fsr, R_tot, R_derived], axis=1)   # (T, 13)
    return L, R


def parse_label(filename: str) -> int:
    """1 = Parkinson's patient, 0 = healthy control."""
    return 1 if "Pt" in filename else 0


def load_split(directory: Path):
    """Load all subjects from a split directory (training/validation/test)."""
    subjects, labels, left_raw, right_raw = [], [], [], []
    for path in sorted(directory.glob("*.txt")):
        L, R = load_subject(path)
        if L is None:
            print(f"  [SKIP] {path.name}  (empty or too short after trimming)")
            continue
        name = path.stem.split("_")[0]
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
    assert C == len(channel_names), (
        f"Signal has {C} channels but channel_names has {len(channel_names)} entries")
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
    """(N_windows, F_tsfresh) array for one subject. NOT cached itself —
    callers should go through get_subject_window_features() for caching."""
    df    = _build_tsfresh_df(signal_2d, channel_names)
    feats = extract_features(
                df,
                column_id             = "id",
                column_sort           = "time",
                default_fc_parameters = FC_PARAMS,
                disable_progressbar   = True,
                n_jobs                = 1,
            )
    impute(feats)
    return np.nan_to_num(feats.values.astype(np.float64),
                         nan=0.0, posinf=0.0, neginf=0.0)


def _aggregate_window_features(wf: np.ndarray) -> np.ndarray:
    """mean/std/skew/kurt aggregation → 1-D vector of length 4 × F_tsfresh."""
    agg = np.concatenate([wf.mean(0), wf.std(0), skew(wf, 0), kurt(wf, 0)])
    return np.nan_to_num(agg, nan=0.0, posinf=0.0, neginf=0.0)


def build_feature_matrix(subjects: list, left_raw: list, right_raw: list):
    """
    Per-subject aggregated feature matrices — shape (N, 4 × F_tsfresh).
    Returns X_left, X_right, X_both.

    Uses the subject-level feature cache (get_subject_window_features):
    TSFresh extraction for a given subject runs at most once across the
    whole pipeline, regardless of how many folds/splits it appears in.
    """
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


def build_dl_dataset(subjects: list, left_raw: list, right_raw: list,
                     labels: np.ndarray):
    """
    Raw window-feature sequences — shape (N, F_tsfresh, T_MAX).
    Returns X_left, X_right, X_both, y.

    Only used when RUN_DL=True. Uses the subject-level feature cache, so
    subjects already extracted for the classical-ML pass are retrieved
    instantly instead of re-running TSFresh.
    T_MAX = max N_windows across subjects in THIS split; shorter sequences
    are zero-padded.
    """
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


# ─────────────────────────────────────────────────────────────────
# 4.  DELIVERABLE #1 — DATA ANALYSIS  (only used if RUN_D1=True)
# ─────────────────────────────────────────────────────────────────
def _parse_demographics(subjects: list, labels: np.ndarray) -> pd.DataFrame:
    rows = []
    for sid, lbl in zip(subjects, labels):
        diagnosis = "Patient" if lbl == 1 else "Control"
        try:
            sex = "Male" if sid[6].upper() == "M" else "Female"
            age = int(sid[8:10])
        except (IndexError, ValueError):
            sex = "Unknown"
            age = float("nan")
        rows.append(dict(SubjectID=sid, Diagnosis=diagnosis, Sex=sex, Age=age))
    return pd.DataFrame(rows)


def report_demographics(all_dirs: list):
    all_subjects, all_labels = [], []
    seen = set()
    for d in all_dirs:
        subs, labs, _, _ = load_split(d)
        for s, l in zip(subs, labs):
            if s not in seen:
                seen.add(s)
                all_subjects.append(s)
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
    print(f"  Saved → {OUT_DIR / 'demographics.csv'}")

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
    print("  Saved → demographics_sex.png, demographics_age.png")

    return df


def plot_signal_examples(left_raw: list, right_raw: list,
                          labels: np.ndarray, subjects: list):
    """
    Plot raw VGRF signals for one PD and one Control subject side-by-side.
    Shows all N_CHANNELS=13 channels (8 FSR + total + COP_x/y + heel/toe).
    """
    pd_idx  = np.where(labels == 1)[0]
    ctl_idx = np.where(labels == 0)[0]
    if len(pd_idx) == 0 or len(ctl_idx) == 0:
        print("  [SKIP] Signal examples: need at least one PD and one Control.")
        return

    pd_i,  ctl_i  = pd_idx[0], ctl_idx[0]

    def _ch_labels(prefix):
        return (
            [f"{prefix}{i+1}" for i in range(8)] +
            ["Total", "COP_x", "COP_y", "Heel", "Toe"]
        )

    for foot, arrays, prefix in [
            ("Left",  left_raw,  "L"),
            ("Right", right_raw, "R")]:
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
        print(f"  Saved → {fname}")


def plot_pca_tsne(X_left: np.ndarray, X_right: np.ndarray,
                  X_both: np.ndarray, labels: np.ndarray,
                  method: str = "pca"):
    method = method.lower()
    assert method in ("pca", "tsne")

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
            xlabel = "t-SNE dim 1"
            ylabel = "t-SNE dim 2"

        for lbl in np.unique(labels):
            mask = labels == lbl
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       c=colors[lbl], label=diag_names[lbl],
                       alpha=0.75, edgecolors="white", linewidths=0.4, s=60)

        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11, color=FAU_BLUE, fontweight="bold")
        ax.axhline(0, color="grey", linewidth=0.5, alpha=0.4)
        ax.axvline(0, color="grey", linewidth=0.5, alpha=0.4)
        ax.legend(fontsize=9)

    fig.suptitle(f"{method.upper()} of TSFresh features — Fold 1 training set",
                 fontsize=13, color=FAU_BLUE, fontweight="bold")
    fig.tight_layout()
    fname = f"{method}_plot.png"
    fig.savefig(FIG_DIR / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {fname}")


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
# 6.  CLASSICAL ML — supervised feature selection + per-fold pipeline
#
#     See module docstring "FEATURE SELECTION" for full rationale.
#     _prepare_features_for_fold() is called ONCE per (fold, foot) in
#     main() and its outputs are reused across all classical models —
#     avoiding redundant FRESH computation.
# ─────────────────────────────────────────────────────────────────
class CalibratedSVC(BaseEstimator, ClassifierMixin):
    """
    SVM-RBF wrapped in CalibratedClassifierCV (default: isotonic, cv=3).

    Why: SVC(probability=True) estimates probabilities via an internal
    5-fold CV + Platt (sigmoid) scaling. With ~60 training subjects this
    internal CV is itself noisy, and Platt scaling assumes a sigmoid-shaped
    score distribution that the RBF decision function often doesn't have —
    a likely cause of SVM-RBF's weaker AUC relative to SVM-Linear (e.g.
    Right foot: 0.776 vs 0.872), whose probabilities come from a much
    better-behaved (near-linear) decision surface.

    Fix: fit the base SVC with probability=False (raw decision_function,
    no internal Platt CV), then calibrate with CalibratedClassifierCV using
    `method` (default "isotonic", non-parametric — makes no assumption
    about the score distribution's shape) and `cv` folds (default 3).
    This is the standard remedy for poorly-calibrated SVM probabilities on
    small datasets.

    Accepts the same hyperparameters as SVC (C, gamma, kernel, class_weight)
    plus `cv` and `method` for the calibration wrapper, so it can be used
    as a drop-in clf_class with the existing ParameterGrid / refit logic.
    """
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
            # sklearn < 1.2 used `base_estimator` instead of `estimator`
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
    """
    Supervised relevance filtering via TSFresh's FRESH algorithm:
    per-feature Mann-Whitney U tests (binary target, real-valued features)
    with Benjamini-Hochberg FDR control at FRESH_FDR_LEVEL. Operates on
    TRAINING data only — must be called with (X_tr, y_tr).

    Bounds the result to [MIN_SELECTED_FEATURES, MAX_SELECTED_FEATURES]:
      - too FEW survive FDR control (common with 40k-80k candidates and
        ~60 samples) → fall back to the MIN_SELECTED_FEATURES most
        significant features by ANOVA F-test p-value.
      - too MANY survive → cap to the MAX_SELECTED_FEATURES most
        significant (by p-value) among the FRESH-selected set.

    Returns: sorted np.ndarray of selected column indices into X.
    """
    n_features = X.shape[1]
    col_names  = [f"f{i}" for i in range(n_features)]
    X_df = pd.DataFrame(X, columns=col_names)
    y_s  = pd.Series(y, index=X_df.index)

    try:
        X_sel = select_features(
            X_df, y_s,
            ml_task   = "classification",
            fdr_level = FRESH_FDR_LEVEL,
            n_jobs    = FEATURE_SELECTION_N_JOBS,
            show_warnings = False,
        )
        sel_idx = np.array(sorted(int(c[1:]) for c in X_sel.columns), dtype=int)
    except Exception as exc:
        print(f"    [FRESH WARN] selection failed ({exc}); "
              f"falling back to F-test ranking")
        sel_idx = np.array([], dtype=int)

    if len(sel_idx) < MIN_SELECTED_FEATURES or len(sel_idx) > MAX_SELECTED_FEATURES:
        f_vals, p_vals = f_classif(X, y)
        p_vals = np.nan_to_num(p_vals, nan=1.0)
        order  = np.argsort(p_vals)   # most significant first

        if len(sel_idx) < MIN_SELECTED_FEATURES:
            k = min(MIN_SELECTED_FEATURES, n_features)
            sel_idx = np.sort(order[:k])
        else:
            capped = set(sel_idx.tolist())
            ranked = [i for i in order if i in capped]
            sel_idx = np.sort(np.array(ranked[:MAX_SELECTED_FEATURES], dtype=int))

    return sel_idx


def _prepare_features_for_fold(X_tr, y_tr, X_val, X_te, fold_id, foot_name):
    """
    Fit  VarianceThreshold → FRESH select → StandardScaler  on training
    data only, producing a compact 'sel' feature space shared by Random
    Forest / SVM-Linear / Logistic Regression. Additionally fit PCA on top
    of that space for SVM-RBF ('pca' variant).

    All fitting uses TRAINING data only; val/test are transformed with the
    fitted transformers (no leakage).

    Returns:
      {
        "sel": (X_tr_sel, X_val_sel, X_te_sel),
        "pca": (X_tr_pca, X_val_pca, X_te_pca),
        "n_selected": int, "n_pca": int,
      }
    """
    n_raw = X_tr.shape[1]

    # Step 1: drop exact-constant columns
    vt      = VarianceThreshold(threshold=1e-10)
    X_tr_vt = vt.fit_transform(X_tr)

    # Step 2: FRESH supervised relevance filter
    sel_idx      = _fresh_select_indices(X_tr_vt, y_tr)
    X_tr_sel_raw = X_tr_vt[:, sel_idx]

    # Step 3: scale (fit on training only)
    scaler   = StandardScaler()
    X_tr_sel = scaler.fit_transform(X_tr_sel_raw)

    # Step 4: PCA on top, for SVM-RBF only
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

    print(f"    [Fold {fold_id} | {foot_name}] "
          f"raw={n_raw}  VT→{X_tr_vt.shape[1]}  "
          f"FRESH→{len(sel_idx)}  PCA→{n_comp} "
          f"({pca.explained_variance_ratio_.sum()*100:.1f}% var)")

    return dict(
        sel=(X_tr_sel, X_val_sel, X_te_sel),
        pca=(X_tr_pca, X_val_pca, X_te_pca),
        n_selected=len(sel_idx),
        n_pca=n_comp,
    )


def train_eval_classical_fold(clf_class, param_grid,
                               X_tr_p, y_tr,
                               X_val_p, y_val,
                               X_te_p, y_te,
                               model_name, foot_name, fold_id):
    """
    One fold of the classical ML pipeline, given ALREADY-PREPROCESSED
    feature matrices (see _prepare_features_for_fold — VT → FRESH →
    Scaler [→ PCA for SVM-RBF], fitted on training data only).

    Steps:
      1. HP search on validation AUC — classifier trained on training set.
      2. Refit best classifier on train + val combined (the upstream
         preprocessor is NOT refitted — it stays fitted on training data).
      3. Evaluate once on test set.

    Returns (result_dict, best_clf):
      best_clf is the refit-on-train+val estimator, fitted in the SAME
      feature space as X_te_p. Returned so soft-voting ensembles can reuse
      it (predict_proba on the same preprocessed test features) without
      retraining — see the "ENSEMBLE" handling in main().
    """
    best_auc    = float("-inf")
    best_params = list(ParameterGrid(param_grid))[0]

    for params in ParameterGrid(param_grid):
        try:
            clf = clf_class(**params)
        except TypeError:
            clf = clf_class(**{k: v for k, v in params.items()
                               if k != "random_state"})
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

    # ── Refit best classifier on train + val combined ──────────────
    X_tv_p = np.vstack([X_tr_p, X_val_p])
    y_tv   = np.concatenate([y_tr, y_val])
    try:
        best_clf = clf_class(**best_params)
    except TypeError:
        best_clf = clf_class(**{k: v for k, v in best_params.items()
                                if k != "random_state"})
    best_clf.fit(X_tv_p, y_tv)

    # ── Evaluate once on test set ───────────────────────────────────
    y_pred_te = best_clf.predict(X_te_p)
    y_prob_te = _safe_prob(best_clf, X_te_p)
    test_m    = compute_metrics(y_te, y_pred_te, y_prob_te)
    cm        = confusion_matrix(y_te, y_pred_te)

    print(f"  Fold {fold_id} | [{model_name}] foot={foot_name} "
          f"best_params={best_params}  val_AUC={best_auc:.3f}  "
          f"test_AUC={test_m['auc']:.3f}")

    res = dict(model=model_name, foot=foot_name, fold=fold_id,
              best_params=best_params, val_auc=best_auc,
              test_metrics=test_m, confusion_matrix=cm)
    return res, best_clf


# ─────────────────────────────────────────────────────────────────
# 6b.  DL — PCA preprocessing (leakage-free, per fold)
#       Only used when RUN_DL=True.
# ─────────────────────────────────────────────────────────────────
def _fit_pca(X_tr, n_components=N_PCA_COMPONENTS):
    """
    Fit StandardScaler + PCA on training-subject windows.
    Input  : (N_subj, F, T)
    Returns: (scaler, pca, n_comp)
    """
    N_tr, F, T = X_tr.shape
    n_comp      = min(n_components, F, N_tr * T)
    W_tr        = X_tr.transpose(0, 2, 1).reshape(-1, F)
    scaler      = StandardScaler()
    W_tr_s      = scaler.fit_transform(W_tr)
    pca         = PCA(n_components=n_comp, random_state=42)
    pca.fit(W_tr_s)
    return scaler, pca, n_comp


def _apply_pca(scaler, pca, X):
    """Project (N, F, T) with fitted scaler + PCA → (N, n_comp, T)."""
    N, F, T = X.shape
    W       = X.transpose(0, 2, 1).reshape(-1, F)
    W       = np.nan_to_num(scaler.transform(W), nan=0.0, posinf=0.0, neginf=0.0)
    P       = pca.transform(W).reshape(N, T, -1).transpose(0, 2, 1)
    return P.astype(np.float32)


# ─────────────────────────────────────────────────────────────────
# 7.  DEEP LEARNING MODELS  (only used when RUN_DL=True)
# ─────────────────────────────────────────────────────────────────
class ResBlock1D(nn.Module):
    def __init__(self, channels, kernel=3, dropout=0.3):
        super().__init__()
        pad = kernel // 2
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel, padding=pad, bias=False),
            nn.BatchNorm1d(channels), nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel, padding=pad, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.act = nn.GELU()

    def forward(self, x): return self.act(x + self.block(x))


class ResNet1D(nn.Module):
    def __init__(self, f_in, num_classes=2, dropout=0.3, d_model=D_MODEL_SMALL):
        super().__init__()
        self.stem   = nn.Sequential(nn.Conv1d(f_in, d_model, 1, bias=False),
                                    nn.BatchNorm1d(d_model), nn.GELU())
        self.blocks = nn.Sequential(ResBlock1D(d_model, 5, dropout),
                                    ResBlock1D(d_model, 3, dropout),
                                    ResBlock1D(d_model, 3, dropout))
        self.head   = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(),
                                    nn.Dropout(dropout), nn.Linear(d_model, num_classes))

    def forward(self, x): return self.head(self.blocks(self.stem(x)))


class LSTMClassifier(nn.Module):
    def __init__(self, f_in, num_classes=2, dropout=0.3, d_model=D_MODEL_SMALL):
        super().__init__()
        self.proj = nn.Linear(f_in, d_model)
        self.lstm = nn.LSTM(d_model, d_model, 2, batch_first=True,
                            dropout=dropout, bidirectional=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model*2, num_classes))

    def forward(self, x):
        out, _ = self.lstm(self.proj(x.permute(0, 2, 1)))
        return self.head(out[:, -1, :])


class GRUClassifier(nn.Module):
    def __init__(self, f_in, num_classes=2, dropout=0.3, d_model=D_MODEL_SMALL):
        super().__init__()
        self.proj = nn.Linear(f_in, d_model)
        self.gru  = nn.GRU(d_model, d_model, 2, batch_first=True,
                           dropout=dropout, bidirectional=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model*2, num_classes))

    def forward(self, x):
        out, _ = self.gru(self.proj(x.permute(0, 2, 1)))
        return self.head(out[:, -1, :])


class PatchTST(nn.Module):
    """
    PatchTST with:
      - Learnable positional encoding (nn.Parameter, fixed bug of per-call reinit).
      - d_model as a constructor argument for the HP sweep.
      - n_heads gracefully reduced to 1 when d_model is not divisible by n_heads.
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
        self.head = nn.Sequential(nn.Dropout(dropout),
                                  nn.Linear(d_model, num_classes))

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
    """
    Compute inverse-frequency class weights from training labels.
    Both cls and w are cast to native Python scalars to avoid the
    TypeError raised by PyTorch ≥ 2.0 when assigning numpy scalars
    into a tensor via a numpy integer index.
    """
    classes, counts = np.unique(y, return_counts=True)
    n_total         = len(y)
    n_classes       = len(classes)
    weights         = n_total / (n_classes * counts.astype(np.float32))
    w_tensor        = torch.zeros(n_classes, dtype=torch.float32)
    for cls, w in zip(classes, weights):
        w_tensor[int(cls)] = float(w)      # ← explicit cast avoids TypeError
    return w_tensor.to(DEVICE)


def _augment_batch(x: torch.Tensor) -> torch.Tensor:
    """
    Augmentation applied to PCA-projected feature sequences during training only —
    never at val/test. After StandardScaler + PCA the scores are roughly zero-mean
    with values in roughly [-3, 3] per component, so:
      Gaussian noise   : σ=0.1  (~3% of typical range — perceptible but not destructive)
      Amplitude jitter : per-sample uniform scale U[0.9, 1.1]
    Both are applied in PCA space, which is mathematically equivalent to applying a
    small random rotation + scaling in the original feature space.
    """
    x     = x + 0.1 * torch.randn_like(x)
    scale = 0.9 + 0.2 * torch.rand(x.shape[0], 1, 1, device=x.device)
    return x * scale


def _train_model(model, train_loader, val_X_t, val_y,
                 n_epochs, lr, warmup_epochs, patience,
                 class_weights: torch.Tensor = None,
                 weight_decay: float = 1e-2,
                 augment: bool = False):
    """
    Train model and return (model, best_epoch).

    Two modes depending on val_X_t:

    HP-search mode (val_X_t is not None):
      Uses validation loss for early stopping. Saves the best checkpoint.
      Returns the epoch number at which the best validation loss was found.

    Refit mode (val_X_t is None):
      No early stopping. Trains for exactly n_epochs epochs.
      Saves state every epoch so the final state is returned.
      best_epoch equals n_epochs after the loop.
    """
    opt  = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    crit = nn.CrossEntropyLoss(weight=class_weights)

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
            best_state = {k: v.cpu().clone()
                          for k, v in model.state_dict().items()}
            best_epoch = ep + 1

    if best_state:
        model.load_state_dict(best_state)
    return model, best_epoch


def _eval_dl(model, X_np):
    """Run inference, return (preds, probs)."""
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X_np).to(DEVICE))
        probs  = torch.softmax(logits, 1)[:, 1].cpu().numpy()
        preds  = logits.argmax(1).cpu().numpy()
    return preds, probs


def train_eval_dl_fold(model_fn_factory, X_tr, y_tr, X_val, y_val, X_te, y_te,
                        model_name, foot_name, fold_id,
                        n_epochs=200, batch_size=16,
                        warmup_epochs=10, patience=25) -> dict:
    """
    One fold of the DL pipeline. Only called when RUN_DL=True.

    Steps:
      1. Fit StandardScaler + PCA on training windows only; project all
         three splits. Val/test are CROPPED OR ZERO-PADDED along the time
         axis to match the training T, so np.vstack(train+val) is safe.
      2. Compute class weights from training labels.
      3. HP grid search (lr × dropout × weight_decay × d_model = 24 combos)
         evaluated by validation AUC. Record the stopping epoch of the best
         combo.  If no combo yields a finite AUC (e.g. degenerate val split),
         fall back to a safe minimum of 10 epochs for the refit.
      4. Refit the best model on train + val combined for exactly
         best_epoch_for_refit epochs, no early stopping.
         Class weights recomputed from combined labels. Augmentation active.
      5. Evaluate once on test set.

    Bug fix — T-dimension alignment:
      _apply_pca projects (N, F, T) → (N, n_comp, T) where T is taken from
      the input, not from the training split.  If val/test subjects have
      fewer or more windows than training subjects, X_tr_p and X_val_p have
      different T, making np.vstack crash.  _align_T resolves this by
      cropping (when val/test T > train T) or zero-padding (when shorter).
    """

    def _align_T(X: np.ndarray, T_target: int) -> np.ndarray:
        """Crop or zero-pad axis-2 (T) of (N, C, T) array to T_target."""
        T = X.shape[2]
        if T == T_target:
            return X
        if T > T_target:
            return X[:, :, :T_target]
        pad = np.zeros((X.shape[0], X.shape[1], T_target - T), dtype=X.dtype)
        return np.concatenate([X, pad], axis=2)

    scaler_pca, pca, n_comp = _fit_pca(X_tr)
    X_tr_p  = _apply_pca(scaler_pca, pca, X_tr)
    T_tr    = X_tr_p.shape[2]            # canonical T — set by training split
    X_val_p = _align_T(_apply_pca(scaler_pca, pca, X_val), T_tr)
    X_te_p  = _align_T(_apply_pca(scaler_pca, pca, X_te),  T_tr)

    cw = _class_weights(y_tr)

    lr_choices      = [1e-4, 3e-4, 1e-3]
    dropout_choices = [0.2, 0.4]
    wd_choices      = [1e-3, 1e-2]
    dm_choices      = [D_MODEL_SMALL, D_MODEL_LARGE]

    best_val_auc         = float("-inf")
    best_hp              = {"lr": 3e-4, "dropout": 0.2,
                            "weight_decay": 1e-2, "d_model": D_MODEL_SMALL}
    best_epoch_for_refit = 10    # safe fallback if all val AUCs are NaN

    for lr in lr_choices:
        for dropout in dropout_choices:
            for wd in wd_choices:
                for dm in dm_choices:
                    model  = model_fn_factory(n_comp, dropout, dm).to(DEVICE)
                    loader = _make_loader(X_tr_p, y_tr, batch_size, shuffle=True)
                    model, epoch = _train_model(
                        model, loader, torch.tensor(X_val_p), y_val,
                        n_epochs, lr, warmup_epochs, patience,
                        class_weights=cw, weight_decay=wd, augment=True)
                    _, probs_val = _eval_dl(model, X_val_p)

                    val_auc = (roc_auc_score(y_val, probs_val)
                               if len(np.unique(y_val)) == 2 else float("nan"))

                    if not np.isnan(val_auc) and val_auc > best_val_auc:
                        best_val_auc         = val_auc
                        best_hp              = {"lr": lr, "dropout": dropout,
                                                "weight_decay": wd, "d_model": dm}
                        best_epoch_for_refit = max(epoch, 1)

    # Refit on train + val combined with best HP
    X_tv_p = np.vstack([X_tr_p, X_val_p])          # (N_tr+N_val, n_comp, T_tr)
    y_tv   = np.concatenate([y_tr, y_val])
    cw_tv  = _class_weights(y_tv)

    best_model = model_fn_factory(
        n_comp, best_hp["dropout"], best_hp["d_model"]).to(DEVICE)
    tv_loader  = _make_loader(X_tv_p, y_tv, batch_size, shuffle=True)

    best_model, _ = _train_model(
        best_model, tv_loader,
        val_X_t       = None,
        val_y         = None,
        n_epochs      = best_epoch_for_refit,
        lr            = best_hp["lr"],
        warmup_epochs = min(warmup_epochs, best_epoch_for_refit),
        patience      = best_epoch_for_refit + 1,   # irrelevant in refit mode
        class_weights = cw_tv,
        weight_decay  = best_hp["weight_decay"],
        augment       = True)

    y_pred_te, y_prob_te = _eval_dl(best_model, X_te_p)
    test_m = compute_metrics(y_te, y_pred_te, y_prob_te)
    cm     = confusion_matrix(y_te, y_pred_te)

    print(f"  Fold {fold_id} | [{model_name}] foot={foot_name}  "
          f"best={best_hp}  refit_epochs={best_epoch_for_refit}  "
          f"val_AUC={best_val_auc:.3f}  test_AUC={test_m['auc']:.3f}")

    return dict(model=model_name, foot=foot_name, fold=fold_id,
                best_params=best_hp, val_auc=best_val_auc,
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
             f"Acc={metrics['accuracy']:.2f}  "
             f"Sens={metrics['sensitivity']:.2f}  "
             f"Spec={metrics['specificity']:.2f}  "
             f"AUC={metrics['auc']:.2f}")
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

    # ══════════════════════════════════════════════════════════════
    # DELIVERABLE #1 — Data Analysis  (gated by RUN_D1)
    # ══════════════════════════════════════════════════════════════
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

    # ══════════════════════════════════════════════════════════════
    # DELIVERABLE #2 — Baseline Models
    # ══════════════════════════════════════════════════════════════

    rf_grid = {
        "n_estimators":     [200, 500, 1000],
        "max_depth":        [None, 10, 20],
        "max_features":     ["sqrt", "log2", 0.3],
        "min_samples_leaf": [1, 2],
        "class_weight":     ["balanced"],
        "random_state":     [42],
    }

    svm_l_grid = {
        "C":            [0.001, 0.01, 0.1, 1, 10, 100],
        "kernel":       ["linear"],
        "probability":  [True],
        "class_weight": ["balanced"],
    }
    # SVM-RBF — now wrapped in CalibratedSVC (CalibratedClassifierCV +
    # isotonic, cv=3) instead of probability=True. See CalibratedSVC's
    # docstring for why: the previous internal-Platt-CV probabilities were
    # poorly calibrated on ~60 training subjects, directly suppressing AUC
    # (e.g. SVM-RBF Right=0.776 vs SVM-Linear Right=0.872). No "probability"
    # key here — CalibratedSVC always fits the base SVC with
    # probability=False internally.
    svm_r_grid = {
        "C":            [0.01, 0.1, 1, 10, 100],
        "gamma":        ["scale", "auto", 0.1, 0.01, 0.001],
        "kernel":       ["rbf"],
        "class_weight": ["balanced"],
        "cv":           [3],
        "method":       ["isotonic"],
    }
    # Logistic Regression with ElasticNet — pairs well with FRESH-
    # selected features (≤MAX_SELECTED_FEATURES dims): the l1_ratio mixes
    # L1 (further sparsifies / does its own selection) and L2 (stability)
    # penalties, often a very strong baseline for small-N / high-D
    # biomedical tabular data.
    logreg_grid = {
        "C":            [0.001, 0.01, 0.1, 1, 10, 100],
        "penalty":      ["elasticnet"],
        "l1_ratio":     [0.1, 0.3, 0.5, 0.7, 0.9],
        "solver":       ["saga"],
        "max_iter":     [5000],
        "class_weight": ["balanced"],
        "random_state": [42],
    }

    # (clf_class, param_grid, use_pca)
    #   use_pca=True  → trained on PCA(N_PCA_COMPONENTS) of FRESH-selected features
    #   use_pca=False → trained directly on FRESH-selected + scaled features
    #
    # "ENSEMBLE" is a sentinel string handled separately in the fold loop.
    classical_models = [
        ("Random Forest",           RandomForestClassifier, rf_grid,     False),
        ("SVM-Linear",              SVC,                    svm_l_grid,  False),
        ("SVM-RBF",                 CalibratedSVC,          svm_r_grid,  True),
        ("Logistic Regression",     LogisticRegression,     logreg_grid, False),
        ("Ensemble (RF+SVM-L+LR)", "ENSEMBLE",
            {"members": ["Random Forest", "SVM-Linear", "Logistic Regression"]},
            False),
    ]

    dl_model_fns = {
        "1D-ResNet": ResNet1D,
        "LSTM":      LSTMClassifier,
        "GRU":       GRUClassifier,
        "PatchTST":  PatchTST,
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

        # ── Classical ML — gated by RUN_CLASSICAL ────────────────────
        # build_feature_matrix uses the subject-level cache so subjects
        # extracted here are instantly reused by the DL path below (both
        # paths call get_subject_window_features which reads from
        # _MEMORY_CACHE or disk, never re-runs TSFresh).
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

            # Feature selection (FRESH) — once per (fold, foot), reused by all models
            print(f"\n  ── Feature selection (FRESH) ──")
            feature_preps = {}
            for foot in feet:
                Xtr, Xval, Xte = cl_splits[foot]
                feature_preps[foot] = _prepare_features_for_fold(
                    Xtr, y_tr, Xval, Xte, fold_id, foot)

            # Classical models — reuse feature_preps across all models
            # fitted_models[foot][model_name] holds each model's refit-on-
            # train+val estimator returned by train_eval_classical_fold,
            # fitted in the SAME preprocessed space as the test features.
            # The "Ensemble" entry (last in classical_models) reuses these
            # directly — no retraining.
            fitted_models = {foot: {} for foot in feet}

            for model_name, clf_class, param_grid, use_pca in classical_models:
                print(f"\n  ── {model_name} ──")

                if clf_class == "ENSEMBLE":
                    members = param_grid["members"]
                    for foot in feet:
                        prep      = feature_preps[foot]
                        X_te_sel  = prep["sel"][2]   # all members use_pca=False

                        probs_list = []
                        for member_name in members:
                            if member_name not in fitted_models[foot]:
                                raise RuntimeError(
                                    f"Ensemble member '{member_name}' has not "
                                    f"been fitted yet for foot={foot}. Ensure "
                                    f"it appears earlier in classical_models.")
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

        # ── DL features + models — gated by RUN_DL ──────────────────
        # Feature extraction reuses the subject-level cache (populated by
        # build_feature_matrix above when RUN_CLASSICAL=True, or loaded
        # directly from disk/memory if this is the first pass).
        if RUN_DL:
            print(f"\n  Extracting DL features — training …")
            DL_XL_tr, DL_XR_tr, DL_XB_tr, DL_y_tr = build_dl_dataset(
                subs_tr, L_tr, R_tr, y_tr)
            print(f"  Extracting DL features — validation …")
            DL_XL_val, DL_XR_val, DL_XB_val, DL_y_val = build_dl_dataset(
                subs_val, L_val, R_val, y_val)
            print(f"  Extracting DL features — test …")
            DL_XL_te, DL_XR_te, DL_XB_te, DL_y_te = build_dl_dataset(
                subs_te, L_te, R_te, y_te)

            dl_splits = {
                "Left":     (DL_XL_tr, DL_XL_val, DL_XL_te),
                "Right":    (DL_XR_tr, DL_XR_val, DL_XR_te),
                "Combined": (DL_XB_tr, DL_XB_val, DL_XB_te),
            }
            # BUG FIX: dl_labels unpacking was previously inside the model
            # loop but outside the foot loop, meaning all three foot splits
            # used DL_y_tr/val/te from the last iteration — no actual bug
            # since labels are the same per-fold regardless of foot, but
            # moved here for clarity and to avoid confusion.
            dl_y_splits = (DL_y_tr, DL_y_val, DL_y_te)

            for model_name, model_cls in dl_model_fns.items():
                print(f"\n  ── {model_name} ──")
                for foot in feet:
                    Xtr, Xval, Xte       = dl_splits[foot]
                    y_tr_dl, y_val_dl, y_te_dl = dl_y_splits
                    factory = lambda nc, do, dm, mc=model_cls: mc(
                        f_in=nc, dropout=do, d_model=dm)
                    res = train_eval_dl_fold(
                        factory, Xtr, y_tr_dl, Xval, y_val_dl, Xte, y_te_dl,
                        model_name, foot, fold_id)
                    all_fold_results.append(res)

    # ── Aggregate across folds ─────────────────────────────────────
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

    # ── Confusion matrices ─────────────────────────────────────────
    print("Plotting confusion matrices …")

    for foot in feet:
        foot_groups = {(m, f): flist
                       for (m, f), flist in groups.items() if f == foot}
        best_key = max(foot_groups,
                       key=lambda k: np.nanmean(
                           [r["test_metrics"]["auc"] for r in foot_groups[k]]))
        best_fold_res = max(foot_groups[best_key],
                            key=lambda r: r["test_metrics"]["auc"])
        plot_confusion_matrix(
            best_fold_res["confusion_matrix"], best_key[0], foot,
            best_fold_res["test_metrics"],
            tag=f"_BEST_fold{best_fold_res['fold']}")
        print(f"  Best for {foot}: {best_key[0]}  "
              f"(fold {best_fold_res['fold']}, "
              f"AUC={best_fold_res['test_metrics']['auc']:.3f})")

    CLASSICAL_NAMES = ("Random Forest", "Extra Trees", "SVM-Linear", "SVM-RBF",
                       "Logistic Regression", "Ensemble (RF+SVM-L+LR)")
    for (model_name, foot), fold_list in groups.items():
        if model_name in CLASSICAL_NAMES:
            best_fold_res = max(fold_list,
                                key=lambda r: r["test_metrics"]["auc"])
            plot_confusion_matrix(
                best_fold_res["confusion_matrix"], model_name, foot,
                best_fold_res["test_metrics"],
                tag=f"_fold{best_fold_res['fold']}")

    for foot in feet:
        plot_bar_comparison(summary_rows, foot)
        plot_fold_auc(all_fold_results, foot)

    # ── Results CSV ───────────────────────────────────────────────
    df = pd.DataFrame(summary_rows)
    filename_summary = "deep_results_table_" + str(TRIM_SEC) + "_" + str(N_PCA_COMPONENTS) + ".csv"
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
    filename ="deep_results_per_fold_" + str(TRIM_SEC) + "_" + str(N_PCA_COMPONENTS) + ".csv"
    pd.DataFrame(fold_rows).to_csv(OUT_DIR /filename, index=False)
    print(f"Per-fold detail → {filename}")


if __name__ == "__main__":
    main()