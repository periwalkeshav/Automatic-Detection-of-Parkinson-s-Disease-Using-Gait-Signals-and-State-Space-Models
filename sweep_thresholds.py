"""
Evaluate every decision-threshold policy for every model, from saved scores.

No model is retrained: this reads the per-fold subject scores written during
training (results/dl_scores/, results/classical_scores/) and re-applies each
policy, so a full sweep takes seconds.

Two rankings are produced, and the difference between them matters:

  * best-by-test   — the policy with the highest test accuracy. This is what you
                     get by "picking the best threshold", but the choice has then
                     been made using the test folds, so the number is optimistic
                     and will not hold on unseen subjects.
  * best-by-val    — the policy chosen on validation accuracy, with its test
                     accuracy reported. Only this one is honest to report.

Usage
-----
  python sweep_thresholds.py                 # all models, all feet
  python sweep_thresholds.py --foot Combined
  python sweep_thresholds.py --out results/THRESHOLD_SWEEP.md
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np

from src import config as C
from src.metrics import compute_metrics

POLICIES = ("native", "prior", "class_mean", "youden", "youden_capped")


# --------------------------------------------------------------------------- #
def load_scores():
    """{(model, foot, fold): dict} from both score directories."""
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
            rec = {k: d[k] for k in d.files}
            rec["is_probability"] = bool(rec.get("is_probability", True))
            out[(model, foot, fold_id)] = rec
    return out


# --------------------------------------------------------------------------- #
def _youden(y, s, lo=None, hi=None):
    """Threshold maximising sensitivity + specificity - 1 on the given split."""
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype=float)
    cands = np.unique(s)
    if cands.size < 2:
        return float(cands[0]) if cands.size else 0.5
    mids = (cands[:-1] + cands[1:]) / 2.0
    grid = np.r_[cands.min() - 1e-9, mids, cands.max() + 1e-9]
    if lo is not None:
        grid = grid[(grid >= lo) & (grid <= hi)]
        if grid.size == 0:
            return float(np.clip(np.median(s), lo, hi))
    best, best_j = float(grid[0]), -np.inf
    for t in grid:
        p = (s >= t).astype(int)
        tp = int(((p == 1) & (y == 1)).sum()); fn = int(((p == 0) & (y == 1)).sum())
        tn = int(((p == 0) & (y == 0)).sum()); fp = int(((p == 1) & (y == 0)).sum())
        j = tp / max(tp + fn, 1) + tn / max(tn + fp, 1) - 1.0
        if j > best_j:
            best_j, best = j, float(t)
    return best


def threshold_for(policy, rec):
    """Derive a threshold from the VALIDATION split only."""
    y, s = rec["y_val"], rec["val_scores"]
    is_prob = rec["is_probability"]
    if policy == "native":
        return 0.5 if is_prob else 0.0
    if policy == "prior":
        # splits are 50/50 by design -> the median reproduces that prior
        return float(np.median(s))
    if policy == "class_mean":
        y = np.asarray(y).astype(int); s = np.asarray(s, dtype=float)
        pos, neg = s[y == 1], s[y == 0]
        if pos.size == 0 or neg.size == 0:
            return float(np.median(s))
        return float((pos.mean() + neg.mean()) / 2.0)
    if policy == "youden":
        return _youden(y, s)
    if policy == "youden_capped":
        if is_prob:
            return _youden(y, s, C.DL_THR_CAP_LO, C.DL_THR_CAP_HI)
        lo, hi = np.percentile(s, 25), np.percentile(s, 75)   # scale-free variant
        return _youden(y, s, lo, hi)
    raise ValueError(policy)


def per_fold_acc(scores, model, foot, folds, policy):
    """Test accuracy for each fold under one policy (threshold from validation)."""
    out = {}
    for f in folds:
        rec = scores.get((model, foot, f))
        if rec is None:
            return None
        thr = threshold_for(policy, rec)
        pred = (rec["test_scores"] >= thr).astype(int)
        out[f] = float((pred == np.asarray(rec["y_test"]).astype(int)).mean())
    return out


def loo_selected(scores, model, foot, folds):
    """
    Leave-one-fold-out policy selection: honest per-model choice.

    Selecting a policy by validation accuracy is invalid here, because 'youden'
    is *fitted* to maximise validation accuracy and therefore always wins that
    comparison regardless of how it generalises. Instead, for each fold the
    policy is chosen from the OTHER folds' test results and then applied to the
    held-out fold, so a fold never influences its own policy.
    """
    acc = {p: per_fold_acc(scores, model, foot, folds, p) for p in POLICIES}
    if any(v is None for v in acc.values()):
        return None, None
    picked, chosen = [], []
    for f in folds:
        others = [g for g in folds if g != f]
        best = max(POLICIES, key=lambda p: np.mean([acc[p][g] for g in others]))
        picked.append(acc[best][f]); chosen.append(best)
    from collections import Counter
    return float(np.mean(picked)), Counter(chosen).most_common(1)[0][0]


def evaluate(scores, model, foot, folds, policy):
    """Mean test metrics (and val accuracy, used for honest selection)."""
    te, va = [], []
    for f in folds:
        rec = scores.get((model, foot, f))
        if rec is None:
            return None
        thr = threshold_for(policy, rec)
        m = compute_metrics(rec["y_test"], (rec["test_scores"] >= thr).astype(int),
                            rec["test_scores"])
        te.append(m)
        vp = (rec["val_scores"] >= thr).astype(int)
        va.append(float((vp == np.asarray(rec["y_val"]).astype(int)).mean()))
    keys = ("accuracy", "sensitivity", "specificity", "auc")
    return {k: float(np.mean([t[k] for t in te])) for k in keys} | {
        "acc_std": float(np.std([t["accuracy"] for t in te], ddof=0)),
        "val_acc": float(np.mean(va)),
    }


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Threshold sweep from saved scores")
    ap.add_argument("--foot", nargs="+", default=None, choices=list(C.FEET))
    ap.add_argument("--out", default=os.path.join(C.RESULTS_DIR, "THRESHOLD_SWEEP.md"))
    args = ap.parse_args()

    scores = load_scores()
    if not scores:
        print(f"No saved scores under {C.RESULTS_DIR}/*_scores/. "
              "Run the classical and DL branches first.")
        return
    models = sorted({k[0] for k in scores})
    feet = args.foot or [f for f in C.FEET if any(k[1] == f for k in scores)]
    folds = sorted({k[2] for k in scores})
    print(f"Loaded {len(scores)} score files: {len(models)} models, "
          f"feet={feet}, folds={folds}\n")

    L = ["# Threshold sweep (from saved scores — nothing retrained)\n",
         f"{len(models)} models x {len(feet)} feet x {len(POLICIES)} policies, "
         f"mean over {len(folds)} folds.\n",
         "`LOO-selected` is the honest choice: for each fold the policy is "
         "picked from the OTHER folds and applied to the held-out one, so no "
         "fold influences its own policy. `best-by-test` is shown only to "
         "quantify how much peeking would buy.\n"]

    for foot in feet:
        head = f"\n## {foot} foot\n"
        print(head.strip())
        L.append(head)
        cols = " | ".join(POLICIES)
        L.append(f"| Model | {cols} | LOO-selected | best-by-test | AUC |")
        L.append("|---|" + "---|" * (len(POLICIES) + 3))
        print(f"{'model':20s} " + " ".join(f"{p:>13s}" for p in POLICIES)
              + f" {'LOO-selected':>22s}")
        for model in models:
            res = {p: evaluate(scores, model, foot, folds, p) for p in POLICIES}
            if any(v is None for v in res.values()):
                continue
            loo_acc, loo_pol = loo_selected(scores, model, foot, folds)
            by_test = max(POLICIES, key=lambda p: res[p]["accuracy"])
            cells = " | ".join(f"{res[p]['accuracy']:.3f}" for p in POLICIES)
            L.append(f"| {model} | {cells} | **{loo_pol}** "
                     f"({loo_acc:.3f}) | {by_test} "
                     f"({res[by_test]['accuracy']:.3f}) | "
                     f"{res[POLICIES[0]]['auc']:.3f} |")
            print(f"{model:20s} "
                  + " ".join(f"{res[p]['accuracy']:13.3f}" for p in POLICIES)
                  + f" {loo_pol:>14s} {loo_acc:6.3f}")

    # how much "picking the best" inflates the numbers
    gaps = []
    for foot in feet:
        for model in models:
            res = {p: evaluate(scores, model, foot, folds, p) for p in POLICIES}
            if any(v is None for v in res.values()):
                continue
            lo, _ = loo_selected(scores, model, foot, folds)
            bt = max(POLICIES, key=lambda p: res[p]["accuracy"])
            if lo is not None:
                gaps.append(res[bt]["accuracy"] - lo)
    if gaps:
        msg = (f"\n**Selection bias:** choosing the policy on test rather than on "
               f"validation inflates accuracy by {np.mean(gaps)*100:.1f} points on "
               f"average (max {np.max(gaps)*100:.1f}). Report `LOO-selected`.\n")
        L.append(msg)
        print(msg)

    with open(args.out, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"written: {args.out}")


if __name__ == "__main__":
    main()
