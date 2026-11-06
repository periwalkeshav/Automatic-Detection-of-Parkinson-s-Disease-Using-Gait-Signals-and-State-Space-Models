"""
Private sanity check against reserved recordings.

This file is a self-contained harness: it imports
main.py and src/ but changes nothing in them, and every byte it writes — TSFresh
feature cache, model scores, CSVs, figures — goes into its own sandbox
directory. Delete this one file and the project is exactly as it was.

Why it exists
-------------
The 100 subjects under splits/ were used to build and cross-validate every
model, so their test folds are an *internal* estimate. The reserved subjects are
external: they may be scored once, by a model that has never seen them, with a
decision threshold fixed beforehand.

What it does, per model and foot
--------------------------------
  1. rebuilds the model once per cross-validation fold using the unchanged
     training protocol (preprocessing fit on train, hyper-parameters selected on
     validation, refit, threshold taken from validation only);
  2. scores the hold-out subjects with each fold model;
  3. averages those scores into one prediction per subject (a fold ensemble) and
     applies the same native threshold the reported results use;
  4. reports accuracy / sensitivity / specificity / AUC with confidence
     intervals — a few dozen subjects cannot pin accuracy down to better than
     roughly +-12 points, and a point estimate alone would overstate the case.

The hold-out labels are read only in step 4. Using them to pick a model, a foot,
a hyper-parameter or a threshold turns the external test back into a validation
set and voids the result.

Usage
-----
  python check_holdout.py --dir /path/to/reserved_recordings
  python check_holdout.py --dir DIR --run both --dl-models MambaFeat
  python check_holdout.py --dir DIR --fusion RandomForest SVM-Linear Mamba
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

from src import config as C
from src.data import label_from_name, load_signal
from src.metrics import class_mean_threshold, compute_metrics

# Hold-out subjects get a prefixed id so their cached features can never collide
# with a development subject that happens to share a file name.
HOLD_PREFIX = "HOLD__"

DEV_CACHE_DIR = C.CACHE_DIR          # captured before the sandbox redirect
DEV_RESULTS_DIR = C.RESULTS_DIR

# src.features looks its cache path up as a module global, so the reuse
# wrapper is installed by rebinding it; the originals are kept here.
_ORIGINAL_CACHE_PATHS: dict = {}


# --------------------------------------------------------------------------- #
# Sandbox — nothing this script runs may touch the real results
# --------------------------------------------------------------------------- #
def install_sandbox(out_dir: str, fresh_cache: bool) -> None:
    """
    Redirect every output path in the project at the sandbox.

    ``C.RESULTS_DIR``/``C.FIG_DIR``/``C.CACHE_DIR`` are read at call time by
    every writer in src/, so re-pointing them here is enough to guarantee the
    real results/ directory is never opened for writing.

    The existing feature cache is still *read* (the TSFresh vector of a
    development subject is fold-independent and identical either way — 100
    subjects would otherwise be re-extracted for nothing), while anything new,
    which means every hold-out subject, is written into the sandbox. Pass
    ``fresh_cache`` to re-extract everything from scratch instead.
    """
    C.RESULTS_DIR = out_dir
    C.FIG_DIR = os.path.join(out_dir, "figures")
    C.CACHE_DIR = os.path.join(out_dir, "feature_cache")
    C.ensure_dirs()

    from src import features as F
    if not _ORIGINAL_CACHE_PATHS:
        _ORIGINAL_CACHE_PATHS.update(agg=F._cache_path, win=F._window_cache_path)
    # always start from the unpatched functions, so the call is idempotent
    F._cache_path = _ORIGINAL_CACHE_PATHS["agg"]
    F._window_cache_path = _ORIGINAL_CACHE_PATHS["win"]
    if fresh_cache:
        return

    def reuse(builder):
        def path(subject, foot):
            name = os.path.basename(builder(subject, foot))
            hit = os.path.join(DEV_CACHE_DIR, name)
            return hit if os.path.exists(hit) else os.path.join(C.CACHE_DIR, name)
        return path

    F._cache_path = reuse(_ORIGINAL_CACHE_PATHS["agg"])
    F._window_cache_path = reuse(_ORIGINAL_CACHE_PATHS["win"])


# --------------------------------------------------------------------------- #
# Loading and auditing the reserved recordings
# --------------------------------------------------------------------------- #
def _walk_id(name: str) -> str:
    """'GaPt22_01' -> 'GaPt22'. PhysioNet appends a walk index per subject."""
    return re.sub(r"_\d+$", "", name)


def development_subjects() -> set[str]:
    """Every recording name used anywhere in splits/ (the 100 known subjects)."""
    pattern = os.path.join(C.SPLITS_DIR, "Fold_*", "*", "*.txt")
    return {os.path.basename(p)[:-4] for p in glob.glob(pattern)}


def _read_label_file(path: str) -> dict[str, int]:
    """Optional label sheet: a subject column and a label column (PD/CO or 1/0)."""
    df = pd.read_csv(path)
    cols = {c.lower().strip(): c for c in df.columns}
    subj_col = next((cols[c] for c in ("subject", "name", "file", "id") if c in cols),
                    df.columns[0])
    lab_col = next((cols[c] for c in ("label", "class", "target", "y") if c in cols),
                   df.columns[-1])
    mapping = {}
    for subj, lab in zip(df[subj_col], df[lab_col]):
        key = os.path.basename(str(subj)).replace(".txt", "").strip()
        s = str(lab).strip().lower()
        if s in ("1", "pd", "patient", "parkinson", "true"):
            mapping[key] = 1
        elif s in ("0", "co", "control", "healthy", "false"):
            mapping[key] = 0
        else:
            raise ValueError(f"Cannot read label {lab!r} for subject {subj!r}")
    return mapping


def load_holdout(directory: str, label_file: str | None = None) -> dict:
    """Load every .txt recording under ``directory`` (searched recursively)."""
    files = sorted(glob.glob(os.path.join(directory, "**", "*.txt"), recursive=True))
    if not files:
        raise FileNotFoundError(f"No .txt recordings found under {directory!r}")

    label_map = _read_label_file(label_file) if label_file else {}
    known = development_subjects()
    known_walks = {_walk_id(n) for n in known}

    subjects, raw_names, labels, lefts, rights, audit = [], [], [], [], [], []
    labelled = True
    for path in files:
        name = os.path.basename(path)[:-4]
        raw = np.loadtxt(path)
        if raw.ndim != 2 or raw.shape[1] != 19:
            raise ValueError(
                f"{name}: expected 19 columns (time + 8 left + 8 right + 2 totals), "
                f"got shape {raw.shape}. The recordings must be in the same "
                f"PhysioNet format as splits/ — see format.txt.")
        dt = float(np.median(np.diff(raw[:, 0]))) if raw.shape[0] > 1 else float("nan")

        if name in label_map:
            lab = label_map[name]
        else:
            try:
                lab = label_from_name(name)
            except ValueError:
                lab, labelled = -1, False

        left, right = load_signal(path)
        subjects.append(HOLD_PREFIX + name)
        raw_names.append(name)
        labels.append(lab)
        lefts.append(left)
        rights.append(right)
        audit.append({
            "subject": name,
            "label": {1: "PD", 0: "CO"}.get(lab, "?"),
            "seconds": round(raw.shape[0] / C.SAMPLING_RATE, 1),
            "used_s": round(left.shape[1] / C.SAMPLING_RATE, 1),
            "dt_s": round(dt, 4),
            "in_dev_set": name in known,
            "same_person_in_dev": _walk_id(name) in known_walks,
        })

    y = np.asarray(labels, dtype=int)
    return {
        "subjects": subjects,
        "raw_names": raw_names,
        "labels": y if labelled else np.zeros(len(y), dtype=int),
        "true_labels": y if labelled else None,
        "lefts": lefts,
        "rights": rights,
        "labelled": labelled,
        "audit": pd.DataFrame(audit),
        "directory": directory,
    }


def audit_report(hold: dict) -> list[str]:
    """Print the pre-flight checks; return the problems that should stop the run."""
    df = hold["audit"]
    problems: list[str] = []

    print("=" * 96)
    print(f"HOLD-OUT SET — {hold['directory']}")
    print("=" * 96)
    print(df.to_string(index=False))

    n = len(df)
    if hold["labelled"]:
        n_pd = int((df["label"] == "PD").sum())
        print(f"\n  {n} recordings — {n_pd} PD / {n - n_pd} CO "
              f"({n_pd / n:.0%} prevalence)")
        if n_pd == 0 or n_pd == n:
            problems.append("Only one class is present — AUC is undefined and "
                            "accuracy is not interpretable.")
    else:
        print(f"\n  {n} recordings — LABELS UNKNOWN. Predictions will be written, "
              f"but no metrics can be computed. Pass --labels <csv> if the "
              f"a label sheet is available.")

    bad_dt = df[(df["dt_s"] - 1.0 / C.SAMPLING_RATE).abs() > 1e-3]
    if len(bad_dt):
        problems.append(f"{len(bad_dt)} recording(s) are not sampled at "
                        f"{C.SAMPLING_RATE} Hz: {list(bad_dt['subject'])}")

    short = df[df["used_s"] < C.WIN_TS / C.SAMPLING_RATE]
    if len(short):
        problems.append(f"{len(short)} recording(s) are shorter than one "
                        f"{C.WIN_TS / C.SAMPLING_RATE:.0f} s window after trimming: "
                        f"{list(short['subject'])}")

    exact = df[df["in_dev_set"]]
    if len(exact):
        problems.append(f"{len(exact)} recording(s) are ALREADY in splits/ and were "
                        f"trained on: {list(exact['subject'])} — remove them, the "
                        f"result would be meaningless.")
    same_person = df[df["same_person_in_dev"] & ~df["in_dev_set"]]
    if len(same_person):
        print(f"\n  WARNING: {len(same_person)} recording(s) are a different walk of "
              f"a subject already in splits/: {list(same_person['subject'])}. That "
              f"is subject-level leakage — score them separately.")

    if problems:
        print("\n  PROBLEMS:")
        for p in problems:
            print(f"    - {p}")
    else:
        print("\n  All pre-flight checks passed.")
    return problems


# --------------------------------------------------------------------------- #
# Turning a CV fold into a hold-out evaluation fold
# --------------------------------------------------------------------------- #
def pseudo_fold(fold: dict, hold: dict, fit: str = "full") -> dict:
    """
    Replace a fold's test split with the hold-out recordings.

    ``fit='cv'``   reproduces the published protocol exactly: train on the
                   60-subject training split, select on validation, refit on
                   train+val (80 subjects).
    ``fit='full'`` folds the fold's own test subjects back into training, so the
                   final model is fitted on all 100 development subjects while
                   the validation split still does hyper-parameter selection and
                   threshold calibration honestly.
    """
    tr_s, tr_y, tr_l, tr_r = fold["train"]
    te_s, te_y, te_l, te_r = fold["test"]

    if fit == "full":
        train = (list(tr_s) + list(te_s),
                 np.concatenate([tr_y, te_y]),
                 list(tr_l) + list(te_l),
                 list(tr_r) + list(te_r))
    elif fit == "cv":
        train = fold["train"]
    else:
        raise ValueError(f"Unknown fit mode: {fit!r} (use 'cv' or 'full')")

    return {"train": train, "val": fold["val"],
            "test": (hold["subjects"], hold["labels"], hold["lefts"], hold["rights"])}


# --------------------------------------------------------------------------- #
# Confidence intervals — a few dozen subjects is a very wide estimate
# --------------------------------------------------------------------------- #
def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval — valid at small n, unlike p +- 1.96*se."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1.0 + z * z / n
    centre = p + z * z / (2 * n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return float((centre - half) / denom), float((centre + half) / denom)


def bootstrap_auc_ci(y, s, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype=float)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, y.size, y.size)
        if np.unique(y[idx]).size < 2:
            continue
        vals.append(roc_auc_score(y[idx], s[idx]))
    if not vals:
        return (float("nan"), float("nan"))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def bootstrap_balacc_ci(y, p, n_boot: int = 2000, seed: int = 0):
    """Percentile CI for balanced accuracy, resampling within each class."""
    y = np.asarray(y).astype(int); p = np.asarray(p).astype(int)
    pos = np.flatnonzero(y == 1); neg = np.flatnonzero(y == 0)
    if pos.size == 0 or neg.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        i = rng.choice(pos, pos.size, replace=True)
        j = rng.choice(neg, neg.size, replace=True)
        vals.append(0.5 * ((p[i] == 1).mean() + (p[j] == 0).mean()))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def with_intervals(y_true, y_pred, y_score) -> dict:
    m = compute_metrics(y_true, y_pred, y_score)
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    n = y_true.size
    n_pos = int((y_true == 1).sum())
    m["accuracy_ci"] = wilson_ci(int((y_pred == y_true).sum()), n)
    m["sensitivity_ci"] = wilson_ci(int(((y_pred == 1) & (y_true == 1)).sum()), n_pos)
    m["specificity_ci"] = wilson_ci(int(((y_pred == 0) & (y_true == 0)).sum()),
                                    n - n_pos)
    m["auc_ci"] = bootstrap_auc_ci(y_true, y_score, seed=C.SEED)
    # The reserved set need not be balanced; on an imbalanced set plain accuracy
    # rewards predicting the majority class, so report balanced accuracy too.
    m["balanced_accuracy"] = 0.5 * (m["sensitivity"] + m["specificity"])
    m["balanced_accuracy_ci"] = bootstrap_balacc_ci(y_true, y_pred, seed=C.SEED)
    m["prevalence"] = n_pos / n if n else float("nan")
    m["majority_baseline"] = max(n_pos, n - n_pos) / n if n else float("nan")
    m["n"] = n
    return m


# --------------------------------------------------------------------------- #
# Fold ensembling of the saved hold-out scores
# --------------------------------------------------------------------------- #
def load_sandbox_scores() -> dict:
    """{(model, foot, fold): arrays} from the sandbox's *_scores directories."""
    out = {}
    for sub in ("classical_scores", "dl_scores"):
        for path in sorted(glob.glob(os.path.join(C.RESULTS_DIR, sub, "*.npz"))):
            name = os.path.basename(path)[:-4]
            try:
                model, foot, fold = name.split("__")
                fold_id = int(fold.replace("fold", ""))
            except ValueError:
                continue
            d = np.load(path)
            out[(model, foot, fold_id)] = {k: d[k] for k in d.files}
    return out


def ensemble_predictors(scores: dict) -> list[dict]:
    """
    Average each model's hold-out scores over the folds it was trained in.

    Probabilities and SVM margins are averaged on their own scale, so the native
    boundary (0.5 / 0.0) still applies and nothing is assumed about the
    prevalence of the hold-out set. A majority vote of the fold models is kept
    alongside as a scale-free cross-check.
    """
    out = []
    for model, foot in sorted({(m, f) for m, f, _ in scores}):
        folds = sorted(f for m, ft, f in scores if (m, ft) == (model, foot))
        mat = np.vstack([scores[(model, foot, f)]["test_scores"] for f in folds])
        is_prob = bool(scores[(model, foot, folds[0])].get("is_probability", True))
        thr = 0.5 if is_prob else 0.0
        mean_score = mat.mean(axis=0)
        votes = (mat >= thr).astype(int).sum(axis=0)
        out.append({
            "model": model, "foot": foot, "folds": folds, "fold_scores": mat,
            "score": mean_score,
            "pred": (mean_score >= thr).astype(int),
            "vote_pred": (votes > len(folds) / 2.0).astype(int),
            "threshold": thr, "is_probability": is_prob,
            "y_test": scores[(model, foot, folds[0])]["y_test"],
        })
    return out


def ensemble_fusion(scores: dict, models: list[str], foot: str) -> dict | None:
    """
    Fold ensemble of a committed model combination, using the same rank-average
    fusion as the cross-validated results.

    Ranks carry no absolute scale, so each fold's cut comes from that fold's own
    validation split and the fold models are combined by majority vote. The
    averaged rank is kept for the AUC, which needs only an ordering.
    """
    from src.fusion import _rank01

    preds, ranks, y_test = [], [], None
    folds = sorted({f for _, ft, f in scores if ft == foot})
    for fold_id in folds:
        keys = [(m, foot, fold_id) for m in models]
        if any(k not in scores for k in keys):
            continue
        ref = scores[keys[0]]
        val = np.mean([_rank01(scores[k]["val_scores"]) for k in keys], axis=0)
        te = np.mean([_rank01(scores[k]["test_scores"]) for k in keys], axis=0)
        preds.append((te >= class_mean_threshold(ref["y_val"], val)).astype(int))
        ranks.append(te)
        y_test = ref["y_test"]
    if not preds:
        return None

    votes = np.vstack(preds).sum(axis=0)
    decision = (votes > len(preds) / 2.0).astype(int)
    return {"model": " + ".join(models), "foot": foot, "folds": folds,
            "fold_scores": np.vstack(ranks), "score": np.vstack(ranks).mean(axis=0),
            "pred": decision, "vote_pred": decision, "threshold": float("nan"),
            "is_probability": False, "y_test": y_test}


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _fmt(m: dict, key: str) -> str:
    lo, hi = m[f"{key}_ci"]
    return f"{m[key]:.3f} [{lo:.2f}-{hi:.2f}]"


def summarise(ens: list[dict], labelled: bool) -> pd.DataFrame:
    rows = []
    for e in ens:
        row = {"model": e["model"], "foot": e["foot"],
               "n_fold_models": len(e["folds"])}
        if labelled:
            m = with_intervals(e["y_test"], e["pred"], e["score"])
            mv = compute_metrics(e["y_test"], e["vote_pred"], e["score"])
            row.update({
                "n": m["n"],
                "accuracy": m["accuracy"], "accuracy_ci": m["accuracy_ci"],
                "balanced_accuracy": m["balanced_accuracy"],
                "balanced_accuracy_ci": m["balanced_accuracy_ci"],
                "sensitivity": m["sensitivity"], "sensitivity_ci": m["sensitivity_ci"],
                "specificity": m["specificity"], "specificity_ci": m["specificity_ci"],
                "auc": m["auc"], "auc_ci": m["auc_ci"],
                "accuracy_majority_vote": mv["accuracy"],
                "prevalence": m["prevalence"],
                "majority_baseline": m["majority_baseline"],
                "cm": m["cm"],
            })
        row["mean_score"] = float(np.mean(e["score"]))
        row["pred_pd"] = int(e["pred"].sum())
        rows.append(row)
    return pd.DataFrame(rows)


def print_summary(df: pd.DataFrame, labelled: bool) -> None:
    print("\n" + "=" * 96)
    print("HOLD-OUT RESULT — fold-ensemble prediction, 95% confidence intervals")
    print("=" * 96)
    if not labelled:
        print(f"{'Model':<28} {'Foot':<10} {'predicted PD':>14}")
        print("-" * 56)
        for _, r in df.iterrows():
            print(f"{r['model']:<28} {r['foot']:<10} {r['pred_pd']:>14d}")
        return
    first = df.iloc[0].to_dict()
    prev = first.get("prevalence")
    if prev is not None and prev == prev:
        print(f"Set is {prev:.0%} PD — always predicting the majority class "
              f"would score {first['majority_baseline']:.3f} accuracy.")
        print("Balanced accuracy = (sens + spec) / 2 is the prevalence-free "
              "figure; read that one first.\n")
    head = (f"{'Model':<26} {'Foot':<10} {'Bal.Acc':>18} {'Accuracy':>18} "
            f"{'Sens':>18} {'Spec':>18} {'AUC':>18}")
    print(head)
    print("-" * len(head))
    for _, r in df.iterrows():
        m = r.to_dict()
        print(f"{r['model']:<26} {r['foot']:<10} "
              f"{_fmt(m, 'balanced_accuracy'):>18} {_fmt(m, 'accuracy'):>18} "
              f"{_fmt(m, 'sensitivity'):>18} {_fmt(m, 'specificity'):>18} "
              f"{_fmt(m, 'auc'):>18}")


def write_markdown(path, df, hold, args, per_fold, labelled) -> None:
    lines = [
        "# Hold-out check — reserved subjects",
        "",
        "*Internal sanity check.*",
        "",
        f"- Recordings: `{args.dir}` — {len(hold['audit'])} subjects",
        f"- Fit mode: `{args.fit}` "
        f"({'all 100 development subjects' if args.fit == 'full' else 'published CV protocol, 80 subjects'})",
        f"- Prediction: mean score over {args.folds[0]}–{args.folds[-1]} fold models, "
        f"native threshold (0.5 probability / 0.0 margin), fixed before scoring",
        f"- Sandbox: `{os.path.relpath(C.RESULTS_DIR, C.BASE_DIR)}`",
        "",
    ]
    if labelled:
        n_pd = int((hold["audit"]["label"] == "PD").sum())
        n_co = len(hold["audit"]) - n_pd
        base = max(n_pd, n_co) / max(len(hold["audit"]), 1)
        lines += [f"- Class balance: {n_pd} PD / {n_co} CO "
                  f"({n_pd / max(n_pd + n_co, 1):.0%} PD)",
                  f"- Majority-class baseline: {base:.3f} accuracy — read "
                  f"**balanced accuracy** rather than accuracy if this is not 0.50",
                  ""]
        lines += ["## Fold-ensemble result (95% CI)", "",
                  "| Model | Foot | n | Bal. accuracy | Accuracy | Sensitivity | "
                  "Specificity | AUC | Acc (majority vote) |",
                  "|---|---|---:|---|---|---|---|---|---:|"]
        for _, r in df.iterrows():
            m = r.to_dict()
            lines.append(
                f"| {r['model']} | {r['foot']} | {int(r['n'])} | "
                f"{_fmt(m,'balanced_accuracy')} | {_fmt(m,'accuracy')} | "
                f"{_fmt(m,'sensitivity')} | {_fmt(m,'specificity')} | {_fmt(m,'auc')} | "
                f"{r['accuracy_majority_vote']:.3f} |")
    else:
        lines += ["## Predictions only (labels unknown)", "",
                  "| Model | Foot | predicted PD |", "|---|---|---:|"]
        for _, r in df.iterrows():
            lines.append(f"| {r['model']} | {r['foot']} | {int(r['pred_pd'])} |")

    lines += ["", "## Per-fold model on the same hold-out set", "",
              "Spread across fold models shows how much of the result is the "
              "training subset rather than the method.", "",
              "```", per_fold.to_string(index=False), "```", "",
              "## Pre-flight audit", "", "```",
              hold["audit"].to_string(index=False), "```", ""]
    with open(path, "w") as fh:
        fh.write("\n".join(lines))


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Score reserved recordings (sandboxed).")
    p.add_argument("--dir", required=True,
                   help="Directory holding the reserved .txt recordings.")
    p.add_argument("--labels", default=None,
                   help="Optional CSV of subject,label — only needed if the file "
                        "names do not carry Pt/Co.")
    p.add_argument("--out", default=os.path.join(C.BASE_DIR, "holdout_check"),
                   help="Sandbox directory for every output of this script.")
    p.add_argument("--fit", choices=["full", "cv"], default="full",
                   help="'full' (default) trains the final model on all 100 "
                        "development subjects; 'cv' reproduces the published "
                        "80-subject protocol exactly.")
    p.add_argument("--run", choices=["classical", "dl", "both"], default="classical",
                   help="Which branch to score (default: classical — it is the "
                        "best single model and takes minutes, not hours).")
    p.add_argument("--folds", nargs="+", type=int,
                   default=list(range(1, C.N_FOLDS + 1)))
    p.add_argument("--feet", nargs="+", default=list(C.FEET), choices=list(C.FEET))
    p.add_argument("--classical-models", nargs="+", default=list(C.CLASSICAL_MODELS),
                   choices=list(C.CLASSICAL_MODELS))
    p.add_argument("--dl-models", nargs="+", default=["MLPFeat"],
                   choices=list(C.DL_MODELS))
    p.add_argument("--fusion", nargs="+", default=None,
                   help="Also score a committed fusion, e.g. "
                        "--fusion RandomForest SVM-Linear Mamba")
    p.add_argument("--fusion-foot", default="Combined", choices=list(C.FEET))
    p.add_argument("--fresh-cache", action="store_true",
                   help="Re-extract TSFresh features for the development subjects "
                        "too, instead of reading the existing cache.")
    p.add_argument("--fc-params", choices=["minimal", "efficient", "comprehensive"],
                   default=C.TSFRESH_FC_PARAMETERS)
    p.add_argument("--tsfresh-jobs", type=int, default=C.TSFRESH_N_JOBS)
    p.add_argument("--dl-quick", action="store_true")
    p.add_argument("--force", action="store_true",
                   help="Run even if the pre-flight audit found problems.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    C.TSFRESH_FC_PARAMETERS = args.fc_params
    C.TSFRESH_N_JOBS = args.tsfresh_jobs
    install_sandbox(os.path.abspath(args.out), args.fresh_cache)
    np.random.seed(C.SEED)

    print(f"Sandbox      : {C.RESULTS_DIR}")
    print(f"Feature cache: writes -> {C.CACHE_DIR}")
    print(f"               reads  -> {DEV_CACHE_DIR}"
          f"{' (disabled)' if args.fresh_cache else ' (existing, read-only)'}")
    print(f"Real results : {DEV_RESULTS_DIR} — untouched\n")

    hold = load_holdout(args.dir, args.labels)
    problems = audit_report(hold)
    if problems and not args.force:
        print("\nStopping. Fix the above, or re-run with --force to override.")
        sys.exit(1)

    from main import run_classical, run_dl
    from src.data import load_fold
    from src.report import build_report

    results: list[dict] = []
    for fold_id in args.folds:
        print(f"\n{'#' * 96}\n# FOLD {fold_id} — model trained here, scored on the "
              f"hold-out set\n{'#' * 96}")
        fold = pseudo_fold(load_fold(fold_id), hold, args.fit)
        n_tr = len(fold["train"][0])
        print(f"  train {n_tr} subjects | val {len(fold['val'][0])} | "
              f"hold-out {len(fold['test'][0])}")
        if args.run in ("classical", "both"):
            run_classical(fold_id, fold, args.feet, args.classical_models, results)
        if args.run in ("dl", "both"):
            run_dl(fold_id, fold, args.feet, args.dl_models, args.dl_quick, results)

    if not results:
        print("\nNothing was scored.")
        return

    build_report(results)          # per-fold CSVs + figures, inside the sandbox
    per_fold = pd.DataFrame(results)[
        ["branch", "model", "foot", "fold", "accuracy", "sensitivity",
         "specificity", "auc"]].round(3)

    scores = load_sandbox_scores()
    ens = ensemble_predictors(scores)
    if args.fusion:
        fused = ensemble_fusion(scores, args.fusion, args.fusion_foot)
        if fused is not None:
            ens.append(fused)
        else:
            print(f"\nNo saved scores for the fusion {args.fusion} on "
                  f"{args.fusion_foot} — run those models first.")

    labelled = hold["labelled"]
    summary = summarise(ens, labelled)
    print_summary(summary, labelled)

    # per-subject predictions, so a disagreement can be traced back to a file
    rows = []
    for e in ens:
        for i, subject in enumerate(hold["raw_names"]):
            row = {"model": e["model"], "foot": e["foot"], "subject": subject,
                   "true": ({1: "PD", 0: "CO"}[int(hold["true_labels"][i])]
                            if labelled else "?"),
                   "score": round(float(e["score"][i]), 4),
                   "pred": "PD" if e["pred"][i] == 1 else "CO"}
            for j, f in enumerate(e["folds"]):
                row[f"fold{f}"] = round(float(e["fold_scores"][j, i]), 4)
            rows.append(row)
    pred_path = os.path.join(C.RESULTS_DIR, "holdout_predictions.csv")
    pd.DataFrame(rows).to_csv(pred_path, index=False)

    summary_path = os.path.join(C.RESULTS_DIR, "holdout_summary.csv")
    summary.to_csv(summary_path, index=False)
    md_path = os.path.join(C.RESULTS_DIR, "HOLDOUT_CHECK.md")
    write_markdown(md_path, summary, hold, args, per_fold, labelled)

    print("\nPer-fold models on the same hold-out set:")
    print(per_fold.to_string(index=False))
    print(f"\nWrote (all inside the sandbox):\n  {md_path}\n  {summary_path}"
          f"\n  {pred_path}")
    if labelled:
        print("\nRead the interval, not the point estimate: with this many "
              "subjects the accuracy is only pinned to roughly +-12 points.")


if __name__ == "__main__":
    main()
