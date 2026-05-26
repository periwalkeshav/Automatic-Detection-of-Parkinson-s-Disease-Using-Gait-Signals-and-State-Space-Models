"""
=============================================================================
DELIVERABLE 1: Data Analysis — Parkinson's Disease Gait Detection
=============================================================================
Project : Mamba/SSM for PD Detection via Gait Signals
Dataset : PhysioNet "Gait in Parkinson's Disease" (gaitpdb)
          https://physionet.org/content/gaitpdb/1.0.0/

HOW TO RUN
----------
1. Place Data.zip in the same folder (or set ZIP_PATH below)
2. pip install tsfresh scikit-learn scipy matplotlib seaborn pandas numpy xlrd
3. python deliverable1_analysis.py

Outputs (saved to ./deliverable1_outputs/):
  fig1_raw_signals.png              — Raw VGRF traces PD vs HC
  fig2_variability_comparison.png   — Force pattern variability
  feature_distributions_*.png       — KDE plots of key features
  feature_heatmap.png               — Heatmap of top discriminative features
  pca_left_foot.png                 — PCA: left foot (label + cohort + H&Y)
  pca_right_foot.png                — PCA: right foot
  pca_combined.png                  — PCA: combined feet
  tsne_left_foot.png                — t-SNE: left foot
  tsne_right_foot.png               — t-SNE: right foot
  tsne_combined.png                 — t-SNE: combined feet
  demographics_table.csv            — Per-cohort demographics
  features_left/right/combined.csv  — Per-subject feature matrices
=============================================================================
"""

# ── CONFIGURATION ──────────────────────────────────────────────────────────
import os, warnings, zipfile
warnings.filterwarnings("ignore")

ZIP_PATH   = None          # path to the zip; set None if already extracted
DATA_DIR   = "./Data"  # where to extract / already extracted
OUTPUT_DIR = "./deliverable1_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Window settings (deliverable spec: 5s windows, 2.5s step)
WINDOW_SEC     = 5.0
STEP_SEC       = 2.5
SAMPLE_RATE    = 100   # Hz
WINDOW_SAMPLES = int(WINDOW_SEC  * SAMPLE_RATE)   # 500
STEP_SAMPLES   = int(STEP_SEC    * SAMPLE_RATE)   # 250

# Column names (PhysioNet gaitpdb format)
LEFT_COLS   = [f"L{i}" for i in range(1, 9)]
RIGHT_COLS  = [f"R{i}" for i in range(1, 9)]
TOTAL_LEFT  = "TotalLeft"
TOTAL_RIGHT = "TotalRight"
ALL_SIGNAL_COLS = LEFT_COLS + RIGHT_COLS + [TOTAL_LEFT, TOTAL_RIGHT]

# ── IMPORTS ─────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import EfficientFCParameters

# ── COLOUR PALETTES ──────────────────────────────────────────────────────────
PALETTE    = {"PD": "#E63946", "HC": "#457B9D"}
COHORT_PAL = {"Ga": "#2A9D8F", "Ju": "#E9C46A", "Si": "#F4A261"}
HY_PAL     = {2.0: "#fee090", 2.5: "#fc8d59", 3.0: "#d73027"}


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1 — DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════

def extract_zip_if_needed(zip_path, dest):
    if not os.path.exists(os.path.join(dest, "demographics.xls")):
        print(f"[INFO] Extracting {zip_path} → {dest}")
        os.makedirs(dest, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(dest)
    else:
        print(f"[INFO] Data already extracted at: {dest}")


def load_demographics(base_dir):
    """Load demographics.xls. Returns clean DataFrame."""
    path = os.path.join(base_dir, "demographics.xls")
    df = pd.read_excel(path, engine="xlrd")
    df["label"]  = df["Group"].map({"PD": "PD", "CO": "HC"})
    df["cohort"] = df["Study"]
    # Ju study stored heights in cm (>10); convert to metres
    mask_cm = df["Height (meters)"] > 10
    df.loc[mask_cm, "Height (meters)"] /= 100
    return df


def collect_all_files(base_dir):
    """
    Collect unique _01.txt (normal walk) files from Fold_1 training+test.
    Returns DataFrame: subject_id | label | cohort | filepath
    """
    search_dirs = [
        os.path.join(base_dir, "splits", "Fold_1", "training"),
        os.path.join(base_dir, "splits", "Fold_1", "test"),
    ]
    seen, records = set(), []
    for d in search_dirs:
        for fname in sorted(os.listdir(d)):
            if not fname.endswith("_01.txt"):
                continue
            sid = fname.replace("_01.txt", "")
            if sid in seen:
                continue
            seen.add(sid)
            records.append({
                "subject_id": sid,
                "label":  "PD" if "Pt" in sid else "HC",
                "cohort": sid[:2],
                "filepath": os.path.join(d, fname),
            })
    return pd.DataFrame(records)


def load_signal(filepath):
    """Load a single VGRF .txt → DataFrame with named columns."""
    cols = ["Time"] + LEFT_COLS + RIGHT_COLS + [TOTAL_LEFT, TOTAL_RIGHT]
    return pd.read_csv(filepath, sep=r"\s+", header=None, names=cols)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2 — DEMOGRAPHICS SUMMARY
# ═══════════════════════════════════════════════════════════════════════════

def print_demographics(demo_df, file_df):
    """Print and save demographics table. Returns merged DataFrame."""
    # NOTE: don't include 'label' from demo_df — file_df already has it
    merged = file_df.merge(
        demo_df[["ID", "Gender", "Age", "Height (meters)",
                 "Weight (kg)", "HoehnYahr", "UPDRS", "UPDRSM"]],
        left_on="subject_id", right_on="ID", how="left"
    )

    print("\n" + "=" * 70)
    print("DATASET DEMOGRAPHICS")
    print("=" * 70)

    for grp_name, sub in [
        ("Overall", merged),
        ("PD Only", merged[merged["label"] == "PD"]),
        ("HC Only", merged[merged["label"] == "HC"]),
    ]:
        n   = len(sub)
        m   = (sub["Gender"] == "male").sum()
        f   = (sub["Gender"] == "female").sum()
        age = sub["Age"].dropna()
        ht  = sub["Height (meters)"].dropna()
        wt  = sub["Weight (kg)"].dropna()
        print(f"\n  {'─'*38}")
        print(f"  {grp_name}  (N={n})")
        print(f"  {'─'*38}")
        print(f"  Male / Female     : {m} / {f}")
        print(f"  Age  mean±SD      : {age.mean():.1f} ± {age.std():.1f}  "
              f"[{age.min():.0f}–{age.max():.0f}]")
        if len(ht): print(f"  Height mean±SD    : {ht.mean():.2f} ± {ht.std():.2f} m")
        if len(wt): print(f"  Weight mean±SD    : {wt.mean():.1f} ± {wt.std():.1f} kg")
        if grp_name == "PD Only":
            hy = sub["HoehnYahr"].dropna()
            up = sub["UPDRSM"].dropna()
            print(f"  H&Y  mean±SD      : {hy.mean():.2f} ± {hy.std():.2f}")
            if len(up):
                print(f"  UPDRS-M mean±SD   : {up.mean():.1f} ± {up.std():.1f}")
            print("  H&Y distribution  :")
            for v, c in hy.value_counts().sort_index().items():
                print(f"      H&Y {v}: {c} subjects")

    print(f"\n  {'─'*38}")
    print("  Per-Cohort Breakdown")
    print(f"  {'─'*38}")
    for coh in ["Ga", "Ju", "Si"]:
        sub = merged[merged["cohort"] == coh]
        pd_ = (sub["label"] == "PD").sum()
        hc_ = (sub["label"] == "HC").sum()
        age = sub["Age"].dropna()
        print(f"  {coh}: PD={pd_}, HC={hc_}, Total={len(sub)}, "
              f"Age={age.mean():.0f}±{age.std():.0f}")

    # Save
    tbl = merged.groupby(["cohort", "label"]).agg(
        N=("subject_id", "count"),
        Male=("Gender", lambda x: (x == "male").sum()),
        Female=("Gender", lambda x: (x == "female").sum()),
        Age_mean=("Age", "mean"),
        Age_std=("Age", "std"),
        Height_mean=("Height (meters)", "mean"),
        Weight_mean=("Weight (kg)", "mean"),
    ).round(2)
    tbl.to_csv(os.path.join(OUTPUT_DIR, "demographics_table.csv"))
    print(f"\n  [Saved] demographics_table.csv")
    return merged


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3 — RAW SIGNAL PLOTS
# ═══════════════════════════════════════════════════════════════════════════

def plot_raw_signals(file_df, n_examples=3):
    """
    Fig 1: Total force traces (left & right foot) for n PD + n HC subjects.
    Shows 15 seconds — enough to see 10–15 gait cycles.
    """
    rng = np.random.default_rng(42)
    pd_files = file_df[file_df["label"] == "PD"].sample(n=n_examples, random_state=42)
    hc_files = file_df[file_df["label"] == "HC"].sample(n=n_examples, random_state=42)
    all_files = pd.concat([pd_files, hc_files])

    SHOW = 15 * SAMPLE_RATE   # 15 seconds

    fig, axes = plt.subplots(n_examples * 2, 2, figsize=(16, n_examples * 4.5))
    fig.suptitle("Raw VGRF Signals — Total Foot Force\nPD Patients vs Healthy Controls",
                 fontsize=14, fontweight="bold", y=1.01)

    for i, (_, row) in enumerate(all_files.iterrows()):
        sig   = load_signal(row["filepath"])
        t     = sig["Time"].values[:SHOW]
        color = PALETTE[row["label"]]

        for j, (col, foot) in enumerate(
                [(TOTAL_LEFT, "Left Foot"), (TOTAL_RIGHT, "Right Foot")]):
            ax = axes[i, j]
            y  = sig[col].values[:SHOW]
            ax.fill_between(t, y, alpha=0.25, color=color)
            ax.plot(t, y, color=color, lw=1.0)
            ax.set_title(f"{row['subject_id']} ({row['label']}, {row['cohort']}) — {foot}",
                         fontsize=8.5)
            ax.set_xlabel("Time (s)", fontsize=8)
            ax.set_ylabel("Force (N)", fontsize=8)
            ax.tick_params(labelsize=7)
            cv = np.std(y) / (np.mean(y) + 1e-8) * 100
            ax.text(0.97, 0.88, f"CV={cv:.1f}%", transform=ax.transAxes,
                    ha="right", fontsize=8, color=color, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                              edgecolor=color, alpha=0.7))

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "fig1_raw_signals.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Saved] fig1_raw_signals.png")


def plot_variability_comparison(file_df):
    """
    Fig 2: Overlay raw signal + running mean±SD band for one PD and one HC.
    Directly comparable to Navita et al. Fig 3 and Yogev et al. Fig 1.
    """
    pd_row = file_df[file_df["label"] == "PD"].iloc[1]
    hc_row = file_df[file_df["label"] == "HC"].iloc[1]

    SHOW = 20 * SAMPLE_RATE  # 20 seconds

    fig, axes = plt.subplots(2, 2, figsize=(14, 7))
    fig.suptitle("Gait Force Variability: PD vs Healthy Control\n"
                 "(Blue band = rolling Mean±SD showing irregular force patterns in PD)",
                 fontsize=12, fontweight="bold")

    for row_i, (row, lbl) in enumerate([(hc_row, "HC"), (pd_row, "PD")]):
        sig   = load_signal(row["filepath"])
        color = PALETTE[lbl]
        PLOT  = min(len(sig), SHOW)
        t     = sig["Time"].values[:PLOT]

        for col_i, (col, foot) in enumerate(
                [(TOTAL_LEFT, "Left Foot"), (TOTAL_RIGHT, "Right Foot")]):
            ax = axes[row_i, col_i]
            y  = sig[col].values[:PLOT]

            # Rolling stats
            wins, mns, sds = [], [], []
            for s in range(0, len(y) - WINDOW_SAMPLES + 1, STEP_SAMPLES):
                seg = y[s:s + WINDOW_SAMPLES]
                mns.append(np.mean(seg))
                sds.append(np.std(seg))
                wins.append(t[s + WINDOW_SAMPLES // 2])
            mns, sds, wins = np.array(mns), np.array(sds), np.array(wins)

            ax.plot(t, y, color=color, lw=0.7, alpha=0.6, label="Raw")
            ax.plot(wins, mns, color="navy", lw=1.8, label="Rolling mean")
            ax.fill_between(wins, mns - sds, mns + sds,
                            color="steelblue", alpha=0.30, label="Mean±SD")

            overall_cv = np.std(y) / (np.mean(y) + 1e-8) * 100
            ax.set_title(
                f"{lbl} ({row['subject_id']}) — {foot}   |   CV={overall_cv:.1f}%",
                fontsize=9)
            ax.set_xlabel("Time (s)", fontsize=8)
            ax.set_ylabel("Force (N)", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.legend(fontsize=7, loc="upper right")

            # Colour label badge
            ax.text(0.01, 0.92, lbl, transform=ax.transAxes, fontsize=10,
                    fontweight="bold", color=color,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                              edgecolor=color, alpha=0.8))

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "fig2_variability_comparison.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Saved] fig2_variability_comparison.png")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4 — TSFRESH FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════

def _build_window_df(sig_df, subject_id, feat_cols):
    """Convert signal DataFrame to long-format windows for TSFresh."""
    y_all = sig_df[feat_cols].values
    rows, win_id = [], 0
    for start in range(0, len(y_all) - WINDOW_SAMPLES + 1, STEP_SAMPLES):
        seg = y_all[start:start + WINDOW_SAMPLES]
        for t_i in range(WINDOW_SAMPLES):
            r = {"window_id": f"{subject_id}_w{win_id}", "time": t_i}
            for ci, col in enumerate(feat_cols):
                r[col] = seg[t_i, ci]
            rows.append(r)
        win_id += 1
    return pd.DataFrame(rows), win_id


def extract_tsfresh_features(file_df, foot="left"):
    """
    Extract TSFresh (EfficientFCParameters) features for all subjects.
    foot: 'left' | 'right' | 'combined'
    Returns (features_df, meta_df).
    """
    feat_map = {
        "left":     LEFT_COLS + [TOTAL_LEFT],
        "right":    RIGHT_COLS + [TOTAL_RIGHT],
        "combined": ALL_SIGNAL_COLS,
    }
    feat_cols = feat_map[foot]
    print(f"\n  Extracting {foot} foot features "
          f"({len(file_df)} subjects × {len(feat_cols)} channels) ...")

    all_wins, meta_rows = [], []
    for _, row in file_df.iterrows():
        sig = load_signal(row["filepath"])
        win_df, n_wins = _build_window_df(sig, row["subject_id"], feat_cols)
        all_wins.append(win_df)
        for wi in range(n_wins):
            meta_rows.append({
                "window_id":  f"{row['subject_id']}_w{wi}",
                "subject_id": row["subject_id"],
                "label":      row["label"],
                "cohort":     row["cohort"],
            })

    all_wins_df = pd.concat(all_wins, ignore_index=True)
    meta_df     = pd.DataFrame(meta_rows)

    features = extract_features(
        all_wins_df,
        column_id="window_id",
        column_sort="time",
        default_fc_parameters=EfficientFCParameters(),
        n_jobs=1,
        show_warnings=False,
        disable_progressbar=False,
    )
    impute(features)
    print(f"  → {features.shape[0]} windows × {features.shape[1]} features")
    return features, meta_df


def compute_per_subject_statistics(features, meta_df):
    """
    Aggregate window-level features to subject level using
    mean, std, skewness, kurtosis across windows.
    Returns subject-indexed DataFrame with label and cohort columns.
    """
    features = features.copy()
    features.index.name = "window_id"
    feat_reset = features.reset_index()
    merged = feat_reset.merge(
        meta_df[["window_id", "subject_id", "label", "cohort"]],
        on="window_id"
    )

    f_cols = [c for c in features.columns]
    grp = merged.groupby("subject_id")

    mean_df = grp[f_cols].mean().add_suffix("__mean")
    std_df  = grp[f_cols].std().add_suffix("__std")
    skew_df = grp[f_cols].apply(lambda x: x.apply(stats.skew)).add_suffix("__skew")
    kurt_df = grp[f_cols].apply(lambda x: x.apply(stats.kurtosis)).add_suffix("__kurt")

    subject_df = pd.concat([mean_df, std_df, skew_df, kurt_df], axis=1).fillna(0)

    label_map = (meta_df.drop_duplicates("subject_id")
                  .set_index("subject_id")[["label", "cohort"]])
    subject_df = subject_df.join(label_map)
    print(f"  → Subject-level: {len(subject_df)} subjects × "
          f"{subject_df.shape[1] - 2} features")
    return subject_df


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5 — PCA & t-SNE PLOTS
# ═══════════════════════════════════════════════════════════════════════════

def _get_scaled_coords(subject_df, method="pca", n_comp=2):
    """
    Z-score normalise features, then apply PCA or t-SNE.
    Returns: coords (N×2), explained_var (for PCA), labels, cohorts.
    """
    meta_cols = {"label", "cohort", "HoehnYahr"}
    f_cols    = [c for c in subject_df.columns if c not in meta_cols]
    X         = subject_df[f_cols].values
    labels    = subject_df["label"].values
    cohorts   = subject_df["cohort"].values

    X_z = StandardScaler().fit_transform(X)

    if method == "pca":
        reducer = PCA(n_components=n_comp, random_state=42)
        coords  = reducer.fit_transform(X_z)
        return coords, reducer.explained_variance_ratio_, labels, cohorts, X_z, reducer
    else:  # tsne
        # Pre-compress with PCA to speed up t-SNE
        n_pca = min(50, X_z.shape[1], X_z.shape[0] - 1)
        X_pca = PCA(n_components=n_pca, random_state=42).fit_transform(X_z)
        perp  = min(30, max(5, len(labels) // 4))
        reducer = TSNE(n_components=2, random_state=42,
                       perplexity=perp, max_iter=1000,
                       learning_rate="auto", init="pca")
        coords = reducer.fit_transform(X_pca)
        return coords, None, labels, cohorts, X_z, reducer


def _scatter(ax, coords, mask, color, label, marker="o"):
    ax.scatter(coords[mask, 0], coords[mask, 1],
               c=color, label=label, alpha=0.80,
               edgecolors="white", linewidths=0.4,
               s=65, marker=marker)


def plot_pca(subject_df, foot_label, demo_merged=None):
    """
    Three-panel PCA: (A) PD/HC, (B) Cohort, (C) H&Y severity.
    """
    coords, ev, labels, cohorts, X_z, pca_model = \
        _get_scaled_coords(subject_df, method="pca")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        f"PCA Score Plot — {foot_label} Features\n"
        f"PC1={ev[0]*100:.1f}%  |  PC2={ev[1]*100:.1f}%  |  "
        f"Cumulative={sum(ev)*100:.1f}%",
        fontsize=12, fontweight="bold")

    # A — Diagnosis
    ax = axes[0]
    for lbl, col in PALETTE.items():
        _scatter(ax, coords, labels == lbl, col, lbl)
    ax.set_title("A) Diagnosis (PD vs HC)", fontsize=10)
    ax.set_xlabel(f"PC1 ({ev[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({ev[1]*100:.1f}%)")
    ax.legend(fontsize=9)
    ax.axhline(0, c="grey", lw=0.5, ls="--")
    ax.axvline(0, c="grey", lw=0.5, ls="--")

    # B — Cohort
    ax = axes[1]
    for coh, col in COHORT_PAL.items():
        _scatter(ax, coords, cohorts == coh, col, coh, marker="D")
    ax.set_title("B) Cohort (potential confound)", fontsize=10)
    ax.set_xlabel(f"PC1 ({ev[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({ev[1]*100:.1f}%)")
    ax.legend(fontsize=9)
    ax.axhline(0, c="grey", lw=0.5, ls="--")
    ax.axvline(0, c="grey", lw=0.5, ls="--")

    # C — H&Y severity (PD only)
    ax = axes[2]
    hc_mask = labels == "HC"
    _scatter(ax, coords, hc_mask, "lightgrey", "HC")
    if demo_merged is not None:
        hy_lookup = (demo_merged.drop_duplicates("subject_id")
                     .set_index("subject_id")["HoehnYahr"])
        for i, sid in enumerate(subject_df.index):
            hy = hy_lookup.get(sid, np.nan)
            if np.isnan(hy):
                continue
            col = HY_PAL.get(hy, "black")
            ax.scatter(coords[i, 0], coords[i, 1], c=col, s=70, alpha=0.85,
                       edgecolors="white", linewidths=0.4)
        # Legend
        for hy_v, col in HY_PAL.items():
            ax.scatter([], [], c=col, s=60, label=f"H&Y {hy_v}")
    ax.set_title("C) PD Severity (H&Y scale)", fontsize=10)
    ax.set_xlabel(f"PC1 ({ev[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({ev[1]*100:.1f}%)")
    ax.legend(fontsize=9)
    ax.axhline(0, c="grey", lw=0.5, ls="--")
    ax.axvline(0, c="grey", lw=0.5, ls="--")

    plt.tight_layout()
    tag  = foot_label.lower().replace(" ", "_")
    path = os.path.join(OUTPUT_DIR, f"pca_{tag}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Saved] pca_{tag}.png")

    # Print top PC1 loadings
    meta_cols = {"label", "cohort", "HoehnYahr"}
    f_cols    = [c for c in subject_df.columns if c not in meta_cols]
    loadings  = pd.Series(np.abs(pca_model.components_[0]), index=f_cols)
    print(f"  Top 5 PC1 loadings ({foot_label}):")
    for feat, val in loadings.nlargest(5).items():
        print(f"    {feat[:65]:<65} {val:.4f}")


def plot_tsne(subject_df, foot_label):
    """Two-panel t-SNE: (A) PD/HC, (B) Cohort."""
    coords, _, labels, cohorts, _, _ = \
        _get_scaled_coords(subject_df, method="tsne")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"t-SNE Embedding — {foot_label} Features",
                 fontsize=12, fontweight="bold")

    ax = axes[0]
    for lbl, col in PALETTE.items():
        _scatter(ax, coords, labels == lbl, col, lbl)
    ax.set_title("A) Diagnosis (PD vs HC)", fontsize=10)
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
    ax.legend(fontsize=9)

    ax = axes[1]
    for coh, col in COHORT_PAL.items():
        _scatter(ax, coords, cohorts == coh, col, coh, marker="D")
    ax.set_title("B) Cohort (potential confound)", fontsize=10)
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
    ax.legend(fontsize=9)

    plt.tight_layout()
    tag  = foot_label.lower().replace(" ", "_")
    path = os.path.join(OUTPUT_DIR, f"tsne_{tag}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Saved] tsne_{tag}.png")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 6 — FEATURE DISTRIBUTION & HEATMAP PLOTS
# ═══════════════════════════════════════════════════════════════════════════

def plot_feature_distributions(subject_df, foot_label, n_features=6):
    """KDE plots of top discriminative feature means, PD vs HC."""
    meta_cols = {"label", "cohort", "HoehnYahr"}
    mean_cols = [c for c in subject_df.columns
                 if c.endswith("__mean") and c not in meta_cols]

    # Rank by |PD_mean − HC_mean| / pooled_std  (approx effect size)
    pd_df = subject_df[subject_df["label"] == "PD"][mean_cols]
    hc_df = subject_df[subject_df["label"] == "HC"][mean_cols]
    diffs = (pd_df.mean() - hc_df.mean()).abs()
    pool_std = (pd_df.std() + hc_df.std()) / 2 + 1e-9
    effect   = (diffs / pool_std).nlargest(n_features)
    top_cols = effect.index.tolist()

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(f"Top {n_features} Discriminative Feature Distributions — {foot_label}\n"
                 f"(ranked by |PD−HC| / pooled SD)",
                 fontsize=11, fontweight="bold")

    for ax, col in zip(axes.flat, top_cols):
        for lbl, col_color in PALETTE.items():
            vals = subject_df.loc[subject_df["label"] == lbl, col].dropna()
            vals.plot.kde(ax=ax, color=col_color, label=lbl, linewidth=2)
            ax.axvline(vals.mean(), color=col_color, ls="--", lw=1.0)
        short = col.replace("__mean", "")[-45:]
        ax.set_title(short, fontsize=7.5)
        ax.legend(fontsize=7.5)
        ax.set_yticks([])
        ax.set_xlabel("Feature value", fontsize=7)

    plt.tight_layout()
    tag  = foot_label.lower().replace(" ", "_")
    path = os.path.join(OUTPUT_DIR, f"feature_distributions_{tag}.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Saved] feature_distributions_{tag}.png")


def plot_summary_heatmap(left_sf, right_sf, combined_sf, n_top=20):
    """
    Heatmap comparing mean feature values (z-scored) between PD and HC
    for the top most-discriminative features in each foot condition.
    """
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    fig.suptitle(
        f"Top {n_top} Discriminative Features: Mean Z-Score by Group",
        fontsize=13, fontweight="bold")

    for ax, sf, title in zip(
            axes,
            [left_sf, right_sf, combined_sf],
            ["Left Foot", "Right Foot", "Combined"]):

        meta_cols = {"label", "cohort", "HoehnYahr"}
        m_cols = [c for c in sf.columns
                  if c.endswith("__mean") and c not in meta_cols]

        pd_df = sf[sf["label"] == "PD"][m_cols]
        hc_df = sf[sf["label"] == "HC"][m_cols]
        diffs = (pd_df.mean() - hc_df.mean()).abs()
        pool  = (pd_df.std() + hc_df.std()) / 2 + 1e-9
        top   = (diffs / pool).nlargest(n_top).index.tolist()

        sub = sf[top + ["label"]]
        scaler = StandardScaler()
        sub_z  = sub[top].copy()
        sub_z[top] = scaler.fit_transform(sub_z.values)
        sub_z["label"] = sub["label"].values
        grp = sub_z.groupby("label")[top].mean()
        grp.columns = [c.replace("__mean", "")[-22:] for c in top]

        sns.heatmap(grp, ax=ax, cmap="RdBu_r", center=0,
                    linewidths=0.4, cbar_kws={"shrink": 0.6},
                    yticklabels=["HC", "PD"])
        ax.set_title(title, fontsize=10)
        ax.set_xticklabels(ax.get_xticklabels(),
                           rotation=45, ha="right", fontsize=6.5)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "feature_heatmap.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Saved] feature_heatmap.png")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("DELIVERABLE 1 — PD Gait Data Analysis")
    print("=" * 70)

    # Step 0: Extract ──────────────────────────────────────────────────────
    if ZIP_PATH and os.path.exists(ZIP_PATH):
        extract_zip_if_needed(ZIP_PATH, DATA_DIR)
    assert os.path.isdir(DATA_DIR), \
        f"\n❌ DATA_DIR not found: {DATA_DIR}\nCheck ZIP_PATH or DATA_DIR at top of script."

    # Step 1: Load metadata ────────────────────────────────────────────────
    print("\n[1/7] Loading file list and demographics ...")
    demo_df   = load_demographics(DATA_DIR)
    file_df   = collect_all_files(DATA_DIR)
    n_pd = (file_df["label"] == "PD").sum()
    n_hc = (file_df["label"] == "HC").sum()
    print(f"  Subjects: {len(file_df)} total  (PD={n_pd}, HC={n_hc})")
    print(f"  Cohorts : {file_df['cohort'].value_counts().to_dict()}")
    demo_merged = print_demographics(demo_df, file_df)

    # Step 2: Raw signal plots ─────────────────────────────────────────────
    print("\n[2/7] Generating raw signal plots ...")
    plot_raw_signals(file_df, n_examples=3)
    plot_variability_comparison(file_df)

    # Step 3: TSFresh extraction ───────────────────────────────────────────
    print("\n[3/7] Running TSFresh feature extraction ...")
    print("  ⏳ This takes ~5–20 minutes depending on CPU. Get a coffee ☕")
    left_feat,     left_meta     = extract_tsfresh_features(file_df, foot="left")
    right_feat,    right_meta    = extract_tsfresh_features(file_df, foot="right")
    combined_feat, combined_meta = extract_tsfresh_features(file_df, foot="combined")

    # Step 4: Per-subject statistics ───────────────────────────────────────
    print("\n[4/7] Computing per-subject statistics (mean, std, skew, kurt) ...")
    left_sf     = compute_per_subject_statistics(left_feat,     left_meta)
    right_sf    = compute_per_subject_statistics(right_feat,    right_meta)
    combined_sf = compute_per_subject_statistics(combined_feat, combined_meta)

    # Attach H&Y severity
    hy_map = (demo_merged.drop_duplicates("subject_id")
               .set_index("subject_id")["HoehnYahr"])
    for sf in [left_sf, right_sf, combined_sf]:
        sf["HoehnYahr"] = sf.index.map(hy_map)

    # Save to CSV
    for name, sf in [("left", left_sf), ("right", right_sf), ("combined", combined_sf)]:
        sf.to_csv(os.path.join(OUTPUT_DIR, f"features_{name}.csv"))
    print("  [Saved] features_left.csv, features_right.csv, features_combined.csv")

    # Step 5: Distribution plots ───────────────────────────────────────────
    print("\n[5/7] Plotting feature distributions and heatmap ...")
    plot_feature_distributions(left_sf,     "Left Foot")
    plot_feature_distributions(right_sf,    "Right Foot")
    plot_feature_distributions(combined_sf, "Combined")
    plot_summary_heatmap(left_sf, right_sf, combined_sf)

    # Step 6: PCA ──────────────────────────────────────────────────────────
    print("\n[6/7] Running PCA ...")
    plot_pca(left_sf,     "Left Foot",  demo_merged)
    plot_pca(right_sf,    "Right Foot", demo_merged)
    plot_pca(combined_sf, "Combined",   demo_merged)

    # Step 7: t-SNE ────────────────────────────────────────────────────────
    print("\n[7/7] Running t-SNE ...")
    plot_tsne(left_sf,     "Left Foot")
    plot_tsne(right_sf,    "Right Foot")
    plot_tsne(combined_sf, "Combined")

    # Final summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("FEATURE STATISTICS SUMMARY")
    print("=" * 70)
    for name, sf in [("Left", left_sf), ("Right", right_sf), ("Combined", combined_sf)]:
        meta_cols = {"label", "cohort", "HoehnYahr"}
        f_cols = [c for c in sf.columns if c not in meta_cols]
        pd_v = sf[sf["label"] == "PD"][f_cols].values
        hc_v = sf[sf["label"] == "HC"][f_cols].values
        print(f"\n  {name} foot  ({len(f_cols)} features)")
        print(f"  PD — mean: {pd_v.mean():.4f} | std: {pd_v.std():.4f} | "
              f"skew: {stats.skew(pd_v.flatten()):.3f} | "
              f"kurt: {stats.kurtosis(pd_v.flatten()):.3f}")
        print(f"  HC — mean: {hc_v.mean():.4f} | std: {hc_v.std():.4f} | "
              f"skew: {stats.skew(hc_v.flatten()):.3f} | "
              f"kurt: {stats.kurtosis(hc_v.flatten()):.3f}")

    print(f"\n{'='*70}")
    print(f"✅  All outputs saved to: {os.path.abspath(OUTPUT_DIR)}/")
    print(f"{'='*70}")

    return {
        "file_df": file_df, "demo_merged": demo_merged,
        "left_sf": left_sf, "right_sf": right_sf, "combined_sf": combined_sf,
    }


if __name__ == "__main__":
    main()