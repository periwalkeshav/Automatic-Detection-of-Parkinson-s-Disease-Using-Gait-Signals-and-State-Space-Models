"""
Deliverables #1 & #2 — Parkinson's Disease Gait Classification
===============================================================
Dataset   : PhysioNet GaitPaDB  — 5 pre-defined folds
            Data/splits/Fold_<1..5>/{training, validation, test}
Features  : TSFresh ComprehensiveFCParameters per 5s/2.5s window,
            aggregated per subject via mean/std/skewness/kurtosis.
            Feature selection: VarianceThreshold (no label use, no leakage).
Models    : Random Forest, SVM-Linear, SVM-RBF,
            1D-ResNet, LSTM, GRU, PatchTST
Foot      : Left, Right, Combined (concatenation)
CV        : 5 pre-defined folds (no StratifiedKFold — folds are on disk)
Metrics   : Accuracy, Sensitivity (PD recall), Specificity, AUC

Deliverable #1 additions
-------------------------
  - Dataset demographics table (N patients/controls, age mean±std, sex)
    parsed from GaitPaDB filename conventions.
  - Raw signal examples: one PD and one Control subject overlaid per channel.
  - PCA plots (PC1 vs PC2, coloured by diagnosis) for Left/Right/Combined,
    using Fold_1/training features (z-score normalised before PCA).

Fold workflow (per fold, per model, per foot)
---------------------------------------------
1. TRAINING   → fit VarianceThreshold (raw) + StandardScaler + PCA + classifier
2. VALIDATION → transform with fitted scaler/selector,
                sweep hyperparameter grid → pick best combo by AUC
3. TEST       → transform with fitted scaler/selector,
                evaluate best model → record metrics

Final reported metrics = mean ± std across the 5 test sets.

This matches the sklearn cross-validation / grid-search workflow:
  https://scikit-learn.org/stable/modules/cross_validation.html
  https://scikit-learn.org/stable/_images/grid_search_workflow.png

Key design decisions
--------------------
FEATURE SET
  ComprehensiveFCParameters (~7056 features/window for 9 channels) includes
  sample_entropy and approximate_entropy — established biomarkers for PD —
  alongside spectral, nonlinear, autocorrelation, and distributional descriptors.

FEATURE SELECTION
  Pipeline per fold: VarianceThreshold (raw features) → StandardScaler →
  PCA(n_components = min(20, n_train-1)).
  Order is critical: StandardScaler sets every non-constant feature to
  variance 1.0, so running VT after scaling removes only perfectly constant
  features (a near-no-op on TSFresh output). VT must run first on raw features.
  PCA caps at n_train-1 to avoid rank deficiency and reduces ~25,000 dims to
  a tractable space for the classifiers.

CLASSICAL ML PIPELINE (per fold)
  a. Fit scaler + selector on training subjects.
  b. Transform train / val / test.
  c. For each param combo: train on training set, score on validation set.
  d. Refit best combo on training set (already done in step c — keep that model).
  e. Evaluate on test set.

DEEP LEARNING PIPELINE (per fold)
  Same scaler + PCA (replaces SelectKBest for DL — PCA on training windows only)
  applied to val/test. Hyperparameter sweep (lr, dropout) on validation loss.

FEATURE CACHE
  Extracted TSFresh features are expensive. Cache key = SHA-256 of
  (data_dir, fold_id, split_tag, WIN_SEC, STEP_SEC, FC_PARAMS class name).
  Stored under OUT_DIR/feature_cache/<key>.npz.

DEMOGRAPHICS PARSING
  GaitPaDB filenames follow the convention:
    <ID><Type><Gender><Age>_<trial>.txt
  e.g. "GaCo01MF57_01.txt" → Control, Male, age 57
       "GaPt02FF65_01.txt" → Patient, Female, age 65
  Type : "Co" = Control, "Pt" = Patient
  Gender: character at index 6 — 'M' = Male, 'F' = Female
  Age  : two-digit integer at the end of the ID field (characters 7–8)
  Adjust _parse_demographics() if your dataset uses different conventions.
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
from sklearn.decomposition import PCA

from tsfresh import extract_features
from tsfresh.feature_extraction import EfficientFCParameters
from tsfresh.utilities.dataframe_functions import impute

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
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
# 0.  PATHS & CONSTANTS
# ─────────────────────────────────────────────────────────────────
DATA_ROOT   = Path("Data/splits")
N_FOLDS     = 5                       # Fold_1 … Fold_5 on disk
OUT_DIR     = Path("results");        OUT_DIR.mkdir(exist_ok=True)
FIG_DIR     = OUT_DIR / "figures";    FIG_DIR.mkdir(exist_ok=True)
CACHE_DIR   = OUT_DIR / "feature_cache"; CACHE_DIR.mkdir(exist_ok=True)

FS           = 100          # Hz
WIN_SEC      = 5.0
STEP_SEC     = 2.5
WIN_SAMPLES  = int(WIN_SEC  * FS)     # 500 samples / window
STEP_SAMPLES = int(STEP_SEC * FS)     # 250 samples step

# GaitPaDB column layout (0-indexed after np.loadtxt):
#   col  0     : time
#   cols 1–8   : Left  VGRF  L1..L8  (8 individual sensors)
#   cols 9–16  : Right VGRF  R1..R8  (8 individual sensors)
#   col 17     : Total force under left foot
#   col 18     : Total force under right foot
#
# We use all 9 channels per foot: 8 individual sensors + total force.
# The total force provides a strong aggregate gait rhythm signal and captures
# stance/swing phase transitions clearly — a well-established PD biomarker.
L_COLS = list(range(1, 9)) + [17]   # L1–L8 + left total  → 9 channels
R_COLS = list(range(9, 17)) + [18]  # R1–R8 + right total → 9 channels
N_CHANNELS = 9   # channels per foot

K_BEST_FEATURES  = 500    # kept for reference — SelectKBest removed (see below)
N_PCA_COMPONENTS = 20     # PCA components for DL input (reduced from 50;
                           # with ~15 training subjects, 50 components causes
                           # the PCA to explain very little signal — 20 is safer)
D_MODEL          = 32     # hidden dim for all DL models
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

# FAU brand colours
FAU_BLUE = "#003865"
FAU_TEAL = "#00b1eb"
FAU_RED  = "#c8102e"
FAU_GRAY = "#98a4ae"

FC_PARAMS = EfficientFCParameters()


# ─────────────────────────────────────────────────────────────────
# 1.  FEATURE CACHE
# ─────────────────────────────────────────────────────────────────
def _cache_key(data_dir: Path, fold_id: int, split_tag: str) -> str:
    fingerprint = "|".join([
        str(data_dir.resolve()),
        f"fold={fold_id}",
        split_tag,
        f"win={WIN_SEC}",
        f"step={STEP_SEC}",
        f"channels={N_CHANNELS}",   # busts cache if channel set changes
        type(FC_PARAMS).__name__,
    ])
    return hashlib.sha256(fingerprint.encode()).hexdigest()[:12]


def _cache_path(data_dir: Path, fold_id: int, split_tag: str, kind: str) -> Path:
    key = _cache_key(data_dir, fold_id, split_tag)
    return CACHE_DIR / f"{kind}_fold{fold_id}_{split_tag}_{key}.npz"


def load_cache(data_dir: Path, fold_id: int, split_tag: str, kind: str):
    path = _cache_path(data_dir, fold_id, split_tag, kind)
    if not path.exists():
        return None
    try:
        arrays = dict(np.load(str(path), allow_pickle=False))
        print(f"  [CACHE HIT]  {path.name}")
        return arrays
    except Exception as exc:
        print(f"  [CACHE WARN] Could not load {path.name}: {exc}. Re-extracting …")
        return None


def save_cache(data_dir: Path, fold_id: int, split_tag: str, kind: str, **arrays):
    path = _cache_path(data_dir, fold_id, split_tag, kind)
    try:
        np.savez_compressed(str(path), **arrays)
        print(f"  [CACHE SAVE] {path.name}")
    except Exception as exc:
        print(f"  [CACHE WARN] Could not save cache: {exc}")


# ─────────────────────────────────────────────────────────────────
# 2.  DATA LOADING
# ─────────────────────────────────────────────────────────────────
def load_subject(path: Path):
    """Return (left, right) arrays of shape (T, 8), or (None, None)."""
    data = np.loadtxt(str(path))
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[0] < WIN_SAMPLES:
        return None, None
    return data[:, L_COLS].copy(), data[:, R_COLS].copy()


def parse_label(filename: str) -> int:
    """1 = Parkinson's patient, 0 = healthy control."""
    return 1 if "Pt" in filename else 0


def load_split(directory: Path):
    """Load all subjects from a split directory (training/validation/test)."""
    subjects, labels, left_raw, right_raw = [], [], [], []
    for path in sorted(directory.glob("*.txt")):
        L, R = load_subject(path)
        if L is None:
            print(f"  [SKIP] {path.name}  (empty or too short)")
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
    """(N_windows, F_tsfresh) array for one subject."""
    df    = _build_tsfresh_df(signal_2d, channel_names)
    feats = extract_features(
                df,
                column_id             = "id",
                column_sort           = "time",
                default_fc_parameters = FC_PARAMS,
                disable_progressbar   = True,
                n_jobs                = 10,
            )
    impute(feats)
    return np.nan_to_num(feats.values.astype(np.float64),
                         nan=0.0, posinf=0.0, neginf=0.0)


def subject_feature_vector(signal_2d: np.ndarray, channel_names: list) -> np.ndarray:
    """mean/std/skew/kurt aggregation → 1-D vector of length 4 × F_tsfresh."""
    wf  = _window_features(signal_2d, channel_names)
    agg = np.concatenate([wf.mean(0), wf.std(0), skew(wf, 0), kurt(wf, 0)])
    return np.nan_to_num(agg, nan=0.0, posinf=0.0, neginf=0.0)


def build_feature_matrix(left_raw: list, right_raw: list,
                          data_dir: Path, fold_id: int, split_tag: str):
    """
    Per-subject aggregated feature matrices — shape (N, 4 × F_tsfresh).
    Returns X_left, X_right, X_both.
    Results are cached; extraction is skipped on cache hit.
    """
    cached = load_cache(data_dir, fold_id, split_tag, "classical")
    if cached is not None:
        return cached["X_left"], cached["X_right"], cached["X_both"]

    L_names = [f"L{i+1}" for i in range(8)] + ["L_total"]
    R_names = [f"R{i+1}" for i in range(8)] + ["R_total"]
    X_L, X_R = [], []
    n = len(left_raw)
    for i, (L, R) in enumerate(zip(left_raw, right_raw)):
        print(f"    Subject {i+1}/{n} …", end="\r", flush=True)
        X_L.append(subject_feature_vector(L, L_names))
        X_R.append(subject_feature_vector(R, R_names))
    print()
    X_L = np.array(X_L);  X_R = np.array(X_R)
    X_B = np.concatenate([X_L, X_R], axis=1)
    save_cache(data_dir, fold_id, split_tag, "classical",
               X_left=X_L, X_right=X_R, X_both=X_B)
    return X_L, X_R, X_B


def build_dl_dataset(left_raw: list, right_raw: list, labels: np.ndarray,
                     data_dir: Path, fold_id: int, split_tag: str):
    """
    Raw window-feature sequences — shape (N, F_tsfresh, T_MAX).
    Returns X_left, X_right, X_both, y.
    PCA is applied later, inside the fold loop. Results are cached.
    """
    cached = load_cache(data_dir, fold_id, split_tag, "dl")
    if cached is not None:
        return (cached["X_left"], cached["X_right"],
                cached["X_both"], cached["y"])

    L_names = [f"L{i+1}" for i in range(8)] + ["L_total"]
    R_names = [f"R{i+1}" for i in range(8)] + ["R_total"]
    seqs_L, seqs_R = [], []
    n = len(left_raw)
    for i, (L, R) in enumerate(zip(left_raw, right_raw)):
        print(f"    DL subject {i+1}/{n} …", end="\r", flush=True)
        seqs_L.append(_window_features(L, L_names).T.astype(np.float32))
        seqs_R.append(_window_features(R, R_names).T.astype(np.float32))
    print()

    T_MAX = max(s.shape[1] for s in seqs_L + seqs_R)

    def _pad(seqs, T):
        """Zero-pad each sequence to length T along the time axis."""
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
    save_cache(data_dir, fold_id, split_tag, "dl",
               X_left=X_L, X_right=X_R, X_both=X_B, y=y)
    return X_L, X_R, X_B, y


# ─────────────────────────────────────────────────────────────────
# 4.  DELIVERABLE #1 — DATA ANALYSIS
#
#     (a) Demographics  — parse filename, print table, save CSV
#     (b) Signal plots  — one PD vs one Control example per foot
#     (c) PCA plots     — PC1 vs PC2 coloured by diagnosis
#         (run once on Fold_1/training after feature extraction)
# ─────────────────────────────────────────────────────────────────

def _parse_demographics(subjects: list, labels: np.ndarray) -> pd.DataFrame:
    """
    Parse age and sex from GaitPaDB subject IDs.

    GaitPaDB naming convention (example):
        GaCo01MF57   → Ga=dataset, Co=Control, 01=subj#, M=Male, F=?, 57=age
        GaPt02FF65   → Ga=dataset, Pt=Patient, 02=subj#, F=Female, F=?, 65=age

    Index  : 0-1 = "Ga", 2-3 = type ("Co"/"Pt"), 4-5 = subject number,
             6   = gender ('M'/'F'), 7 = extra char, 8-9 = age (2 digits)

    If a subject ID does not match the expected pattern, age is set to NaN
    and sex to "Unknown" so the rest of the analysis still runs.
    """
    rows = []
    for sid, lbl in zip(subjects, labels):
        diagnosis = "Patient" if lbl == 1 else "Control"
        try:
            # Characters 6 = gender, 8:10 = age (two-digit)
            sex = "Male" if sid[6].upper() == "M" else "Female"
            age = int(sid[8:10])
        except (IndexError, ValueError):
            sex = "Unknown"
            age = float("nan")
        rows.append(dict(SubjectID=sid, Diagnosis=diagnosis, Sex=sex, Age=age))
    return pd.DataFrame(rows)


def report_demographics(all_dirs: list):
    """
    Collect subjects from every provided directory, deduplicate by SubjectID,
    then print and save a demographics summary table.

    Parameters
    ----------
    all_dirs : list of Path
        Typically all training + validation + test directories across all folds
        so every subject appears exactly once (duplicates dropped).
    """
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
        sub = df[df["Diagnosis"] == diag]
        n   = len(sub)
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

    # ── Summary bar chart: sex breakdown per group ─────────────────
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, diag in zip(axes, ["Patient", "Control"]):
        sub   = df[df["Diagnosis"] == diag]
        counts = sub["Sex"].value_counts()
        ax.bar(counts.index, counts.values,
               color=[FAU_BLUE, FAU_TEAL, FAU_GRAY], alpha=0.9)
        ax.set_title(f"{diag}  (N={len(sub)})",
                     color=FAU_BLUE, fontweight="bold")
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

    # ── Age distribution per group ─────────────────────────────────
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
    Plot raw VGRF signals for one PD subject and one Control subject side-by-side.
    One figure for the left foot, one for the right.
    Shows all N_CHANNELS channels stacked vertically (8 sensors + total force).
    x-axis in seconds.
    """
    pd_idx  = np.where(labels == 1)[0]
    ctl_idx = np.where(labels == 0)[0]
    if len(pd_idx) == 0 or len(ctl_idx) == 0:
        print("  [SKIP] Signal examples: need at least one PD and one Control subject.")
        return

    pd_i,  ctl_i  = pd_idx[0], ctl_idx[0]
    t_sec_l = np.arange(left_raw[pd_i].shape[0])  / FS
    t_sec_r = np.arange(right_raw[pd_i].shape[0]) / FS

    # Channel labels: L1–L8 / R1–R8 for sensors, "Total" for the last channel
    def _ch_labels(prefix):
        return [f"{prefix}{i+1}" for i in range(8)] + ["Total"]

    for foot, arrays, t_sec, prefix in [
            ("Left",  left_raw,  t_sec_l, "L"),
            ("Right", right_raw, t_sec_r, "R")]:
        ch_labels = _ch_labels(prefix)
        fig, axes = plt.subplots(N_CHANNELS, 2,
                                  figsize=(14, N_CHANNELS * 1.4), sharex=False)
        for ch in range(N_CHANNELS):
            for col, (idx, diag, color) in enumerate([
                    (pd_i,  "Patient", FAU_RED),
                    (ctl_i, "Control", FAU_BLUE)]):
                sig  = arrays[idx]
                t    = np.arange(sig.shape[0]) / FS
                mask = t <= 10.0   # first 10 s for readability
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
    """
    Produce dimensionality-reduction scatter plots (PCA or t-SNE) for
    Left, Right, and Combined features, coloured by diagnosis.

    Parameters
    ----------
    X_left, X_right, X_both : (N_subjects, F) feature matrices
    labels                  : (N_subjects,) binary labels
    method                  : "pca" or "tsne"
    """
    method = method.lower()
    assert method in ("pca", "tsne"), "method must be 'pca' or 'tsne'"

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    titles = ["Left foot", "Right foot", "Combined"]
    colors = {0: FAU_BLUE, 1: FAU_RED}
    diag_names = {0: "Control", 1: "Patient"}

    for ax, X, title in zip(axes,
                             [X_left, X_right, X_both],
                             titles):
        # z-score normalise before projection (required by both D1 and D2)
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
            # t-SNE: first reduce with PCA to 50 dims for speed/stability
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

    tag = method.upper()
    fig.suptitle(f"{tag} of TSFresh features — Fold 1 training set",
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
# 6.  CLASSICAL ML — per-fold pipeline
#
#  Flow (sklearn grid_search_workflow):
#   ┌─────────────┐   fit scaler + selector   ┌──────────────────────┐
#   │  Training   │ ─────────────────────────▶│  transform train/val │
#   └─────────────┘                           └──────────────────────┘
#          │  train clf on each param combo          │
#          ▼                                         ▼
#   ┌─────────────┐   score each combo        ┌──────────────────────┐
#   │ Validation  │ ◀────────────────────────▶│  pick best by AUC    │
#   └─────────────┘                           └──────────────────────┘
#          │  refit best combo on training data      │
#          ▼                                         ▼
#   ┌─────────────┐   evaluate once only      ┌──────────────────────┐
#   │    Test     │ ─────────────────────────▶│  record test metrics │
#   └─────────────┘                           └──────────────────────┘
# ─────────────────────────────────────────────────────────────────
def _fit_preprocessor(X_tr, y_tr):
    """
    Fit VarianceThreshold → StandardScaler → PCA on training data only.
    Returns (selector, scaler, pca, X_tr_processed).

    Order matters:
      1. VarianceThreshold on RAW features — StandardScaler amplifies every
         non-zero-std feature to variance 1.0, so running VT after scaling
         is a near-no-op (only perfectly constant features are removed).
         Running VT first on the raw features correctly discards near-constant
         features before any amplification happens.
      2. StandardScaler after VT — centres/scales the surviving features.
      3. PCA capped at (n_train - 1) — avoids rank deficiency and reduces the
         ~25,000-dim space to a tractable subspace for the classifiers.
         With ~15 training subjects, even 14 components capture the meaningful
         variance while avoiding the curse of dimensionality.
    """
    # Step 1: remove near-constant features on raw unscaled data
    selector = VarianceThreshold(threshold=0.01)
    X_tr_f   = selector.fit_transform(X_tr)
    print(f"    VT  : {X_tr_f.shape[1]}/{X_tr.shape[1]} features retained")

    # Step 2: scale the filtered set
    scaler  = StandardScaler()
    X_tr_s  = scaler.fit_transform(X_tr_f)

    # Step 3: PCA — cap at n_train-1 to avoid rank deficiency
    n_comp  = min(20, X_tr_s.shape[0] - 1, X_tr_s.shape[1])
    pca     = PCA(n_components=n_comp, random_state=42)
    X_tr_p  = pca.fit_transform(X_tr_s)
    print(f"    PCA : {n_comp} components, "
          f"{pca.explained_variance_ratio_.sum()*100:.1f}% variance explained")
    return selector, scaler, pca, X_tr_p


def _apply_preprocessor(selector, scaler, pca, X):
    """Apply already-fitted VT → scaler → PCA to val or test data."""
    return pca.transform(
               scaler.transform(
                   selector.transform(X)))


def train_eval_classical_fold(clf_class, param_grid,
                               X_tr, y_tr,
                               X_val, y_val,
                               X_te, y_te,
                               model_name, foot_name, fold_id) -> dict:
    """
    One fold of the classical ML pipeline.
    Preprocessor is fitted on training only; val used for HP selection;
    test touched exactly once.
    """
    # ── Step 1: fit preprocessor on training set ──────────────────
    selector, scaler, pca, X_tr_p = _fit_preprocessor(X_tr, y_tr)

    # ── Step 2: transform val and test with the fitted preprocessor ─
    X_val_p = _apply_preprocessor(selector, scaler, pca, X_val)
    X_te_p  = _apply_preprocessor(selector, scaler, pca, X_te)

    # ── Step 3: hyperparameter search on validation set ────────────
    best_auc    = float("-inf")
    best_params = list(ParameterGrid(param_grid))[0]
    best_clf    = None

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
            best_clf    = clf   # already trained with best params on training set

    # ── Step 4: evaluate best model on test set (once only) ────────
    y_pred_te = best_clf.predict(X_te_p)
    y_prob_te = _safe_prob(best_clf, X_te_p)
    test_m    = compute_metrics(y_te, y_pred_te, y_prob_te)
    cm        = confusion_matrix(y_te, y_pred_te)

    print(f"  Fold {fold_id} | [{model_name}] foot={foot_name} "
          f"best_params={best_params}  val_AUC={best_auc:.3f}  "
          f"test_AUC={test_m['auc']:.3f}")

    return dict(model=model_name, foot=foot_name, fold=fold_id,
                best_params=best_params, val_auc=best_auc,
                test_metrics=test_m, confusion_matrix=cm)


# ─────────────────────────────────────────────────────────────────
# 6.  DL — PCA preprocessing (leakage-free, per fold)
# ─────────────────────────────────────────────────────────────────
def _fit_pca(X_tr, n_components=N_PCA_COMPONENTS):
    """
    Fit StandardScaler + PCA on training-subject windows.
    Input:  (N_subj, F, T)
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
    """Project (N, F, T) using already-fitted scaler + PCA → (N, n_comp, T)."""
    N, F, T = X.shape
    W       = X.transpose(0, 2, 1).reshape(-1, F)
    W       = np.nan_to_num(scaler.transform(W), nan=0.0, posinf=0.0, neginf=0.0)
    P       = pca.transform(W).reshape(N, T, -1).transpose(0, 2, 1)
    return P.astype(np.float32)


# ─────────────────────────────────────────────────────────────────
# 7.  DEEP LEARNING MODELS
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
    def __init__(self, f_in, num_classes=2, dropout=0.3):
        super().__init__()
        self.stem   = nn.Sequential(nn.Conv1d(f_in, D_MODEL, 1, bias=False),
                                    nn.BatchNorm1d(D_MODEL), nn.GELU())
        self.blocks = nn.Sequential(ResBlock1D(D_MODEL, 5, dropout),
                                    ResBlock1D(D_MODEL, 3, dropout),
                                    ResBlock1D(D_MODEL, 3, dropout))
        self.head   = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(),
                                    nn.Dropout(dropout), nn.Linear(D_MODEL, num_classes))

    def forward(self, x): return self.head(self.blocks(self.stem(x)))


class LSTMClassifier(nn.Module):
    def __init__(self, f_in, num_classes=2, dropout=0.3):
        super().__init__()
        self.proj = nn.Linear(f_in, D_MODEL)
        self.lstm = nn.LSTM(D_MODEL, D_MODEL, 2, batch_first=True,
                            dropout=dropout, bidirectional=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(D_MODEL*2, num_classes))

    def forward(self, x):
        out, _ = self.lstm(self.proj(x.permute(0, 2, 1)))
        return self.head(out[:, -1, :])


class GRUClassifier(nn.Module):
    def __init__(self, f_in, num_classes=2, dropout=0.3):
        super().__init__()
        self.proj = nn.Linear(f_in, D_MODEL)
        self.gru  = nn.GRU(D_MODEL, D_MODEL, 2, batch_first=True,
                           dropout=dropout, bidirectional=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(D_MODEL*2, num_classes))

    def forward(self, x):
        out, _ = self.gru(self.proj(x.permute(0, 2, 1)))
        return self.head(out[:, -1, :])


class PatchTST(nn.Module):
    """
    PatchTST with a learnable positional encoding stored as nn.Parameter.

    Bug fixed: previously `pe` was a local variable re-created and randomly
    re-initialised inside forward() on every call, so the model never learned
    position-dependent patterns (explains AUC collapsing to ~0.35 in fold 5).
    Now pe is initialised once in __init__ and updated by the optimiser.
    """
    def __init__(self, f_in, num_classes=2, patch_len=4, stride=2,
                 n_heads=4, n_layers=2, dropout=0.1, max_seq_len=512):
        super().__init__()
        self.patch_len  = patch_len
        self.stride     = stride
        self.input_proj = nn.Linear(f_in, D_MODEL)
        self.patch_proj = nn.Linear(D_MODEL * patch_len, D_MODEL)
        self.pos_drop   = nn.Dropout(dropout)

        # Maximum number of patches for any input length up to max_seq_len.
        # Sliced to actual NP in forward — avoids shape mismatch.
        max_patches = (max_seq_len - patch_len) // max(1, stride) + 2
        self.pe = nn.Parameter(torch.empty(1, max_patches, D_MODEL))
        nn.init.trunc_normal_(self.pe, std=0.02)   # initialised ONCE, then learned

        enc_layer = nn.TransformerEncoderLayer(
            D_MODEL, n_heads, D_MODEL * 4, dropout, "gelu",
            batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, n_layers)
        self.head = nn.Sequential(nn.Dropout(dropout),
                                  nn.Linear(D_MODEL, num_classes))

    def forward(self, x):
        B, F, T = x.shape
        x = self.input_proj(x.permute(0, 2, 1)).permute(0, 2, 1)  # (B, D, T)

        patches, start = [], 0
        while start + self.patch_len <= T:
            patches.append(x[:, :, start:start + self.patch_len])
            start += self.stride
        if not patches:
            pad = torch.zeros(B, D_MODEL, self.patch_len - T, device=x.device)
            patches.append(torch.cat([x, pad], dim=2))

        NP     = len(patches)
        tokens = self.patch_proj(
            torch.stack(patches, 1).reshape(B, NP, -1))  # (B, NP, D_MODEL)

        # Use the first NP positions of the learnable PE
        tokens = self.pos_drop(tokens + self.pe[:, :NP, :])
        enc    = self.transformer(tokens)
        return self.head(enc.mean(1))


# ─────────────────────────────────────────────────────────────────
# 8.  DL TRAINING HELPERS
# ─────────────────────────────────────────────────────────────────
def _make_loader(X, y, batch_size, shuffle):
    ds = TensorDataset(torch.tensor(X), torch.tensor(y))
    return DataLoader(ds, batch_size=min(batch_size, len(X)), shuffle=shuffle)


def _class_weights(y: np.ndarray) -> torch.Tensor:
    """
    Compute inverse-frequency class weights from training labels.
    Returns a float tensor of shape (n_classes,) on DEVICE.

    With tiny, imbalanced splits a plain CrossEntropyLoss lets the model
    collapse to the majority class.  Weighted CE gives each class equal
    total gradient contribution, mirroring class_weight='balanced' in sklearn.
    """
    classes, counts = np.unique(y, return_counts=True)
    n_total  = len(y)
    n_classes = len(classes)
    weights  = n_total / (n_classes * counts.astype(np.float32))
    w_tensor = torch.zeros(n_classes, dtype=torch.float32)
    for cls, w in zip(classes, weights):
        w_tensor[cls] = w
    return w_tensor.to(DEVICE)


def _train_model(model, train_loader, val_X_t, val_y,
                 n_epochs, lr, warmup_epochs, patience,
                 class_weights: torch.Tensor = None):
    opt  = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    crit = nn.CrossEntropyLoss(weight=class_weights)   # weighted if provided

    def lr_lambda(ep):
        if ep < warmup_epochs:
            return (ep + 1) / warmup_epochs
        prog = (ep - warmup_epochs) / max(1, n_epochs - warmup_epochs)
        return 0.5 * (1.0 + np.cos(np.pi * prog))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    best_loss, best_state, no_improve = float("inf"), None, 0

    for _ in range(n_epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            val_loss = crit(model(val_X_t.to(DEVICE)),
                            torch.tensor(val_y, device=DEVICE)).item()
        if val_loss < best_loss:
            best_loss  = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)
    return model


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
                        n_epochs=100, batch_size=8,
                        warmup_epochs=10, patience=15) -> dict:
    """
    One fold of the DL pipeline.

    HP grid swept on the validation set (lr × dropout):
      lr      : 3e-4, 1e-3
      dropout : 0.2, 0.4

    Class-weighted CrossEntropyLoss is computed from training labels and
    passed to _train_model so the model cannot collapse to the majority class.
    PCA is fitted on training subjects only; test touched exactly once.
    """
    # ── Step 1: fit PCA on training set only ──────────────────────
    scaler, pca, n_comp = _fit_pca(X_tr)

    # ── Step 2: project all three splits ──────────────────────────
    X_tr_p  = _apply_pca(scaler, pca, X_tr)
    X_val_p = _apply_pca(scaler, pca, X_val)
    X_te_p  = _apply_pca(scaler, pca, X_te)

    # ── Step 3: compute class weights from training labels ────────
    cw = _class_weights(y_tr)

    # ── Step 4: HP grid search on validation AUC ──────────────────
    lr_choices      = [3e-4, 1e-3]
    dropout_choices = [0.2, 0.4]

    best_val_auc  = float("-inf")
    best_hp       = {"lr": lr_choices[0], "dropout": dropout_choices[0]}
    best_state    = None

    for lr in lr_choices:
        for dropout in dropout_choices:
            model  = model_fn_factory(n_comp, dropout).to(DEVICE)
            loader = _make_loader(X_tr_p, y_tr, batch_size, shuffle=True)
            model  = _train_model(model, loader, torch.tensor(X_val_p), y_val,
                                  n_epochs, lr, warmup_epochs, patience,
                                  class_weights=cw)
            _, probs_val = _eval_dl(model, X_val_p)

            val_auc = (roc_auc_score(y_val, probs_val)
                       if len(np.unique(y_val)) == 2 else float("nan"))

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_hp      = {"lr": lr, "dropout": dropout}
                best_state   = {k: v.cpu().clone()
                                for k, v in model.state_dict().items()}

    # ── Step 5: evaluate best model on test set (once only) ───────
    best_model = model_fn_factory(n_comp, best_hp["dropout"]).to(DEVICE)
    best_model.load_state_dict(best_state)
    y_pred_te, y_prob_te = _eval_dl(best_model, X_te_p)

    test_m = compute_metrics(y_te, y_pred_te, y_prob_te)
    cm     = confusion_matrix(y_te, y_pred_te)

    print(f"  Fold {fold_id} | [{model_name}] foot={foot_name}  "
          f"best={best_hp}  val_AUC={best_val_auc:.3f}  "
          f"test_AUC={test_m['auc']:.3f}")

    return dict(model=model_name, foot=foot_name, fold=fold_id,
                best_params=best_hp, val_auc=best_val_auc,
                test_metrics=test_m, confusion_matrix=cm)


# ─────────────────────────────────────────────────────────────────
# 9.  AGGREGATE FOLD RESULTS
# ─────────────────────────────────────────────────────────────────
def aggregate_folds(fold_results: list) -> dict:
    """
    Given a list of per-fold result dicts (same model + foot),
    return mean ± std for each metric across folds.
    """
    keys = ["accuracy", "sensitivity", "specificity", "auc"]
    agg  = {}
    for k in keys:
        vals       = [r["test_metrics"][k] for r in fold_results]
        agg[k]     = round(float(np.nanmean(vals)), 4)
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
    """Box plots of per-fold test AUC for each model."""
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
    # DELIVERABLE #1 — Data Analysis
    # ══════════════════════════════════════════════════════════════

    # ── A. Demographics ───────────────────────────────────────────
    print("\n── D1: Demographics ──")
    # Collect every split directory across all folds to get the full cohort
    all_split_dirs = [
        DATA_ROOT / f"Fold_{fid}" / split
        for fid in range(1, N_FOLDS + 1)
        for split in ("training", "validation", "test")
    ]
    report_demographics(all_split_dirs)

    # ── B. Signal examples (Fold 1 training set) ──────────────────
    print("\n── D1: Signal examples ──")
    fold1_train = DATA_ROOT / "Fold_1" / "training"
    subs_f1, y_f1, L_f1, R_f1 = load_split(fold1_train)
    plot_signal_examples(L_f1, R_f1, y_f1, subs_f1)

    # ── C. PCA & t-SNE plots (Fold 1 training features) ──────────
    print("\n── D1: PCA / t-SNE ──")
    XL_f1, XR_f1, XB_f1 = build_feature_matrix(
        L_f1, R_f1, fold1_train.parent, 1, "train")
    plot_pca_tsne(XL_f1, XR_f1, XB_f1, y_f1, method="pca")
    plot_pca_tsne(XL_f1, XR_f1, XB_f1, y_f1, method="tsne")

    # ══════════════════════════════════════════════════════════════
    # DELIVERABLE #2 — Baseline Models
    # ══════════════════════════════════════════════════════════════

    # ── Hyperparameter grids ───────────────────────────────────────
    # class_weight='balanced' is added to all classifiers:
    # With small, potentially unequal class splits per fold, an unweighted
    # classifier collapses to the majority-class prior — causing Sens/Spec
    # swings of 0.9/0.1.  Balanced weighting scales the loss so each class
    # contributes equally regardless of sample count.
    rf_grid = {
        "n_estimators":     [200, 500, 1000],
        "max_depth":        [None, 10, 20],
        "max_features":     ["sqrt", "log2", 0.3],
        "min_samples_leaf": [1, 2],
        "class_weight":     ["balanced"],
        "random_state":     [42],
    }
    svm_l_grid = {"C":           [0.001, 0.01, 0.1, 1, 10, 100],
                  "kernel":      ["linear"],
                  "probability": [True],
                  "class_weight":["balanced"]}
    svm_r_grid = {"C":           [0.01, 0.1, 1, 10, 100],
                  "gamma":       ["scale", "auto", 0.1, 0.01, 0.001],
                  "kernel":      ["rbf"],
                  "probability": [True],
                  "class_weight":["balanced"]}

    classical_models = [
        ("Random Forest", RandomForestClassifier, rf_grid),
        ("SVM-Linear",    SVC,                   svm_l_grid),
        ("SVM-RBF",       SVC,                   svm_r_grid),
    ]
    dl_model_fns = {
        "1D-ResNet": ResNet1D,
        "LSTM":      LSTMClassifier,
        "GRU":       GRUClassifier,
        "PatchTST":  PatchTST,
    }
    feet = ["Left", "Right", "Combined"]

    # all_fold_results: list of per-fold dicts (one entry per fold×model×foot)
    all_fold_results = []

    # ── Loop over the 5 pre-defined folds ─────────────────────────
    for fold_id in range(1, N_FOLDS + 1):
        fold_dir = DATA_ROOT / f"Fold_{fold_id}"
        train_dir = fold_dir / "training"
        val_dir   = fold_dir / "validation"
        test_dir  = fold_dir / "test"

        print(f"\n{'='*65}")
        print(f"FOLD {fold_id}  —  loading splits …")

        _, y_tr,  L_tr,  R_tr  = load_split(train_dir)
        _, y_val, L_val, R_val = load_split(val_dir)
        _, y_te,  L_te,  R_te  = load_split(test_dir)
        print(f"  Train={len(y_tr)}  Val={len(y_val)}  Test={len(y_te)}")

        # ── Classical ML features ─────────────────────────────────
        print(f"  Extracting classical features — training …")
        XL_tr, XR_tr, XB_tr = build_feature_matrix(
            L_tr,  R_tr,  fold_dir, fold_id, "train")
        print(f"  Extracting classical features — validation …")
        XL_val, XR_val, XB_val = build_feature_matrix(
            L_val, R_val, fold_dir, fold_id, "val")
        print(f"  Extracting classical features — test …")
        XL_te, XR_te, XB_te = build_feature_matrix(
            L_te,  R_te,  fold_dir, fold_id, "test")

        cl_splits = {
            "Left":     (XL_tr, XL_val, XL_te),
            "Right":    (XR_tr, XR_val, XR_te),
            "Combined": (XB_tr, XB_val, XB_te),
        }

        # ── Classical ML training/validation/test ─────────────────
        for model_name, clf_class, param_grid in classical_models:
            print(f"\n  ── {model_name} ──")
            for foot in feet:
                Xtr, Xval, Xte = cl_splits[foot]
                res = train_eval_classical_fold(
                    clf_class, param_grid,
                    Xtr, y_tr, Xval, y_val, Xte, y_te,
                    model_name, foot, fold_id)
                all_fold_results.append(res)

        # ── DL features ───────────────────────────────────────────
        print(f"\n  Extracting DL features — training …")
        DL_XL_tr, DL_XR_tr, DL_XB_tr, DL_y_tr = build_dl_dataset(
            L_tr,  R_tr,  y_tr,  fold_dir, fold_id, "train")
        print(f"  Extracting DL features — validation …")
        DL_XL_val, DL_XR_val, DL_XB_val, DL_y_val = build_dl_dataset(
            L_val, R_val, y_val, fold_dir, fold_id, "val")
        print(f"  Extracting DL features — test …")
        DL_XL_te, DL_XR_te, DL_XB_te, DL_y_te = build_dl_dataset(
            L_te,  R_te,  y_te,  fold_dir, fold_id, "test")

        dl_splits = {
            "Left":     (DL_XL_tr, DL_XL_val, DL_XL_te),
            "Right":    (DL_XR_tr, DL_XR_val, DL_XR_te),
            "Combined": (DL_XB_tr, DL_XB_val, DL_XB_te),
        }
        dl_labels = (DL_y_tr, DL_y_val, DL_y_te)

        # ── DL training/validation/test ───────────────────────────
        for model_name, model_cls in dl_model_fns.items():
            print(f"\n  ── {model_name} ──")
            for foot in feet:
                Xtr, Xval, Xte = dl_splits[foot]
                y_tr_dl, y_val_dl, y_te_dl = dl_labels
                # factory takes (n_comp, dropout) — both set by HP sweep
                factory = lambda nc, do, mc=model_cls: mc(f_in=nc, dropout=do)
                res = train_eval_dl_fold(
                    factory, Xtr, y_tr_dl, Xval, y_val_dl, Xte, y_te_dl,
                    model_name, foot, fold_id)
                all_fold_results.append(res)

    # ── Aggregate across folds ─────────────────────────────────────
    print(f"\n{'='*65}")
    print("Aggregating results across 5 folds …")

    summary_rows = []
    # Group by (model, foot)
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
    # Per the deliverable: show the best-performing model for each of
    # Left / Right / Combined.  "Best" = highest mean test AUC across folds.
    print("Plotting confusion matrices …")

    # 1. Best overall model per foot (averaged AUC across folds)
    for foot in feet:
        foot_groups = {(m, f): flist
                       for (m, f), flist in groups.items() if f == foot}
        best_key = max(foot_groups,
                       key=lambda k: np.nanmean(
                           [r["test_metrics"]["auc"] for r in foot_groups[k]]))
        best_model_name = best_key[0]
        # Use the fold with the highest test AUC from that model
        best_fold_res = max(foot_groups[best_key],
                            key=lambda r: r["test_metrics"]["auc"])
        plot_confusion_matrix(
            best_fold_res["confusion_matrix"], best_model_name, foot,
            best_fold_res["test_metrics"],
            tag=f"_BEST_fold{best_fold_res['fold']}")
        print(f"  Best for {foot}: {best_model_name}  "
              f"(fold {best_fold_res['fold']}, "
              f"AUC={best_fold_res['test_metrics']['auc']:.3f})")

    # 2. All classical models (required by deliverable)
    for (model_name, foot), fold_list in groups.items():
        if model_name in ("Random Forest", "SVM-Linear", "SVM-RBF"):
            best_fold_res = max(fold_list,
                                key=lambda r: r["test_metrics"]["auc"])
            plot_confusion_matrix(
                best_fold_res["confusion_matrix"], model_name, foot,
                best_fold_res["test_metrics"],
                tag=f"_fold{best_fold_res['fold']}")

    # ── Summary bar charts & fold AUC box plots ───────────────────
    for foot in feet:
        plot_bar_comparison(summary_rows, foot)
        plot_fold_auc(all_fold_results, foot)

    # ── Results CSV ───────────────────────────────────────────────
    df = pd.DataFrame(summary_rows)
    df.to_csv(OUT_DIR / "results_table.csv", index=False)

    print(f"\n{'='*65}")
    print(df.to_string(index=False))
    print(f"{'='*65}")
    print(f"\nResults → {OUT_DIR}/results_table.csv")
    print(f"Figures → {FIG_DIR}/")

    # Also save per-fold detail
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
    pd.DataFrame(fold_rows).to_csv(OUT_DIR / "results_per_fold.csv", index=False)
    print(f"Per-fold detail → {OUT_DIR}/results_per_fold.csv")


if __name__ == "__main__":
    main()