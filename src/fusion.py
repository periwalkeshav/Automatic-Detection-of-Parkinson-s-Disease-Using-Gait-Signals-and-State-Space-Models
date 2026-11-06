"""
Late fusion of saved model scores.

The classical and deep branches see the data very differently — engineered
TSFresh statistics versus the raw signal — so they make different mistakes.
Averaging their per-subject scores usually ranks subjects better than either
branch alone, which is what the accuracy is ultimately limited by.

This module works purely from the ``.npz`` score files written during training
(``results/classical_scores/``, ``results/dl_scores/``), so exploring fusion
combinations costs nothing: no model is re-trained.

Scores are combined after a per-fold rank transform, which puts probabilities
and SVM margins on the same [0, 1] scale without assuming anything about their
calibration. The decision threshold comes from the validation split only.
"""
from __future__ import annotations

import glob
import os

import numpy as np

from . import config as C
from .metrics import class_mean_threshold, compute_metrics

SCORE_DIRS = ("classical_scores", "dl_scores")


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_scores() -> dict:
    """{(model, foot, fold): {y_val, val_scores, y_test, test_scores}}."""
    out = {}
    for sub in SCORE_DIRS:
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


def available(scores: dict) -> tuple[list[str], list[str], list[int]]:
    models = sorted({k[0] for k in scores})
    feet = [f for f in C.FEET if any(k[1] == f for k in scores)]
    folds = sorted({k[2] for k in scores})
    return models, feet, folds


# --------------------------------------------------------------------------- #
# Fusion
# --------------------------------------------------------------------------- #
def _rank01(x: np.ndarray) -> np.ndarray:
    """Map scores to evenly spaced ranks in [0, 1] (ties averaged)."""
    x = np.asarray(x, dtype=float)
    n = x.size
    if n <= 1:
        return np.zeros(n)
    order = x.argsort()
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(n, dtype=float)
    # average ties so equal scores get equal ranks
    for v in np.unique(x):
        m = x == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    return ranks / (n - 1)


def _entry_key(entry, default_foot):
    """Accept either 'Model' (uses default_foot) or ('Model', 'Foot')."""
    if isinstance(entry, (tuple, list)):
        return entry[0], entry[1]
    if "@" in entry:
        m, f = entry.split("@", 1)
        return m, f
    return entry, default_foot


def evaluate_fusion_entries(scores: dict, entries, folds, label=None,
                            default_foot="Combined"):
    """
    Fuse an arbitrary set of (model, foot) predictors.

    Left and Right are different views of the same subject and make different
    mistakes, so combining a model across feet — or different models on
    different feet — adds diversity that same-foot fusion cannot.
    """
    pairs = [_entry_key(e, default_foot) for e in entries]
    name = label or " + ".join(f"{m}@{f}" for m, f in pairs)
    rows = []
    for fold_id in folds:
        keys = [(m, f, fold_id) for m, f in pairs]
        if any(k not in scores for k in keys):
            continue
        ref = scores[keys[0]]
        val = np.mean([_rank01(scores[k]["val_scores"]) for k in keys], axis=0)
        te = np.mean([_rank01(scores[k]["test_scores"]) for k in keys], axis=0)
        thr = class_mean_threshold(ref["y_val"], val)
        metrics = compute_metrics(ref["y_test"], (te >= thr).astype(int), te)
        rows.append({"branch": "fusion", "model": name, "foot": "cross-foot",
                     "fold": fold_id,
                     "params": "rank-avg(" + ",".join(f"{m}@{f}" for m, f in pairs) + ")",
                     **metrics})
    return rows


def search_cross_foot(scores: dict, folds: list[int], max_size: int = 3,
                      models: list[str] | None = None):
    """Rank combinations drawn from every (model, foot) pair, by mean AUC."""
    import itertools
    models = models or sorted({k[0] for k in scores})
    feet_avail = sorted({k[1] for k in scores})
    entries = [(m, f) for m in models for f in feet_avail
               if any((m, f, fold) in scores for fold in folds)]
    out = []
    for size in range(1, max_size + 1):
        for combo in itertools.combinations(entries, size):
            rows = evaluate_fusion_entries(scores, list(combo), folds)
            if len(rows) == len(folds):
                out.append((summarise(rows), list(combo), rows))
    out.sort(key=lambda r: (r[0]["auc"], r[0]["accuracy"]), reverse=True)
    return out


def report_cross_foot(max_size: int = 3, top: int = 15,
                      only: list[str] | None = None):
    """Leaderboard over model x foot combinations."""
    scores = load_scores()
    if not scores:
        print("No saved scores found.")
        return []
    folds = sorted({k[2] for k in scores})
    pool = None
    if only:
        pool = [m for m in only if any(k[0] == m for k in scores)]
        print(f"restricted to: {', '.join(pool)}")
    ranked = search_cross_foot(scores, folds, max_size=max_size, models=pool)
    print(f"\n{'='*92}\nCROSS-FOOT FUSION LEADERBOARD "
          f"(mean over {len(folds)} folds, ranked by AUC)\n{'='*92}")
    print(f"{'Models @ feet':<58} {'Acc':>7} {'Sens':>7} {'Spec':>7} {'AUC':>7}")
    print("-" * 92)
    for summary, combo, _ in ranked[:top]:
        label = " + ".join(f"{m}@{f[:4]}" for m, f in combo)
        print(f"{label:<58} {summary['accuracy']:>7.3f} {summary['sensitivity']:>7.3f} "
              f"{summary['specificity']:>7.3f} {summary['auc']:>7.3f}")
    print("\nRanked on the test folds — commit to a combination on principle "
          "(e.g. one model across all three feet) rather than taking the top row.")
    return ranked


def fuse_fold(scores: dict, models: list[str], foot: str, fold_id: int,
              weights=None):
    """Rank-average the given models for one fold. Returns (y_val, val, y_te, te)."""
    keys = [(m, foot, fold_id) for m in models]
    if any(k not in scores for k in keys):
        return None
    w = np.ones(len(models)) if weights is None else np.asarray(weights, dtype=float)
    w = w / w.sum()

    ref = scores[keys[0]]
    val = sum(wi * _rank01(scores[k]["val_scores"]) for wi, k in zip(w, keys))
    te = sum(wi * _rank01(scores[k]["test_scores"]) for wi, k in zip(w, keys))
    return ref["y_val"], val, ref["y_test"], te


def evaluate_fusion(scores: dict, models: list[str], foot: str, folds: list[int],
                    weights=None, label: str | None = None) -> list[dict]:
    """Per-fold metrics for one fused combination."""
    name = label or " + ".join(models)
    rows = []
    for fold_id in folds:
        fused = fuse_fold(scores, models, foot, fold_id, weights)
        if fused is None:
            continue
        y_val, val, y_te, te = fused
        # Ranks are uniform in [0, 1]; the validation class-mean midpoint is the
        # natural cut and needs no calibration assumption.
        thr = class_mean_threshold(y_val, val)
        metrics = compute_metrics(y_te, (te >= thr).astype(int), te)
        rows.append({"branch": "fusion", "model": name, "foot": foot,
                     "fold": fold_id, "params": f"rank-avg({','.join(models)})",
                     **metrics})
    return rows


def summarise(rows: list[dict]) -> dict:
    """Mean of the reported metrics over folds."""
    keys = ("accuracy", "sensitivity", "specificity", "auc")
    return {k: float(np.mean([r[k] for r in rows])) for k in keys} if rows else {}


def search_fusions(scores: dict, foot: str, folds: list[int],
                   base_models: list[str] | None = None,
                   max_size: int = 3) -> list[tuple]:
    """
    Score every combination of up to `max_size` models, ranked by mean AUC.

    Selection over combinations is itself a choice made on test data, so treat
    the ranking as exploratory: report the combination you commit to, not the
    best of many.
    """
    import itertools
    models = base_models or sorted({k[0] for k in scores if k[1] == foot})
    results = []
    for size in range(1, max_size + 1):
        for combo in itertools.combinations(models, size):
            rows = evaluate_fusion(scores, list(combo), foot, folds)
            if len(rows) == len(folds):
                results.append((summarise(rows), list(combo), rows))
    results.sort(key=lambda r: (r[0]["auc"], r[0]["accuracy"]), reverse=True)
    return results


def report_fusion(foot: str = "Combined", max_size: int = 3, top: int = 12,
                  only: list[str] | None = None):
    """Print the fusion leaderboard for one foot and return all rows.

    `only` restricts the candidate pool, e.g. to the deep models, so the best
    achievable deep result can be reported without classical models carrying it.
    """
    scores = load_scores()
    if not scores:
        print("No saved scores found. Run the classical and DL branches first "
              f"(they write into {C.RESULTS_DIR}/*_scores/).")
        return []
    models, feet, folds = available(scores)
    print(f"Loaded scores: {len(models)} models, feet={feet}, folds={folds}")
    if foot not in feet:
        print(f"No scores for foot={foot!r}.")
        return []

    pool = None
    if only:
        pool = [m for m in only if any(k[0] == m for k in scores)]
        print(f"restricted to: {', '.join(pool)}")
    ranked = search_fusions(scores, foot, folds, base_models=pool, max_size=max_size)
    print(f"\n{'='*84}\nFUSION LEADERBOARD — {foot} "
          f"(mean over {len(folds)} folds, ranked by AUC)\n{'='*84}")
    print(f"{'Models':<44} {'Acc':>7} {'Sens':>7} {'Spec':>7} {'AUC':>7}")
    print("-" * 84)
    for summary, combo, _ in ranked[:top]:
        print(f"{' + '.join(combo):<44} {summary['accuracy']:>7.3f} "
              f"{summary['sensitivity']:>7.3f} {summary['specificity']:>7.3f} "
              f"{summary['auc']:>7.3f}")
    return ranked
