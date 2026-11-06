#!/usr/bin/env python3
"""
Score a folder of recordings with a bundle written by save_model.py.
Same family as check_holdout.py.

    python predict.py --bundle saved_models/RandomForest__Combined__full.joblib \
                      --dir /path/to/recordings

No training happens here. If the file names carry Pt/Co the true labels are
inferred and metrics are reported; otherwise predictions are written and that
is all. The decision threshold is the one stored in the bundle — it is never
tuned on the folder being scored.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src import config as C                                    # noqa: E402

PRED_PREFIX = "PRED__"


def install_cache(cache_dir: str) -> None:
    """Keep TSFresh features for new recordings out of the project cache."""
    os.makedirs(cache_dir, exist_ok=True)
    C.CACHE_DIR = cache_dir


def load_recordings(directory: str):
    from src.data import load_signal, label_from_name
    files = sorted(glob.glob(os.path.join(directory, "**", "*.txt"), recursive=True))
    if not files:
        raise FileNotFoundError(f"No .txt recordings under {directory!r}")
    subjects, names, lefts, rights, labels = [], [], [], [], []
    labelled = True
    for path in files:
        name = os.path.basename(path)[:-4]
        raw = np.loadtxt(path)
        if raw.ndim != 2 or raw.shape[1] != 19:
            raise ValueError(f"{name}: expected 19 columns, got {raw.shape}")
        left, right = load_signal(path)
        subjects.append(PRED_PREFIX + name)     # never collide with a dev subject
        names.append(name)
        lefts.append(left); rights.append(right)
        try:
            labels.append(label_from_name(name))
        except ValueError:
            labels.append(-1); labelled = False
    return names, subjects, lefts, rights, np.asarray(labels, int), labelled


def score_members(bundle, X_raw) -> np.ndarray:
    """Mean probability across bundle members (fold models and/or MLP seeds)."""
    rep = bundle["representation"]
    per_member = []
    for m in bundle["members"]:
        Xt = m["prep"].transform(X_raw, rep)
        if bundle["kind"] == "sklearn":
            est = m["est"]
            if hasattr(est, "predict_proba"):
                per_member.append(est.predict_proba(Xt)[:, 1])
            else:                                  # SVM margin -> squashed
                d = est.decision_function(Xt)
                per_member.append(1.0 / (1.0 + np.exp(-d)))
        else:
            per_member.append(_mlp_score(m["est"], Xt))
    return np.mean(per_member, axis=0)


def _mlp_score(est, X) -> np.ndarray:
    import torch
    from src.models_dl import MLPFeatClassifier
    cfg = est["cfg"]
    out = []
    for state in est["states"]:
        net = MLPFeatClassifier(est["in_features"], hidden=tuple(cfg["hidden"]),
                                dropout=cfg["dropout"])
        net.load_state_dict({k: torch.from_numpy(v) for k, v in state.items()})
        net.eval()
        with torch.no_grad():
            logits = net(torch.from_numpy(np.ascontiguousarray(X, np.float32)))
            out.append(torch.softmax(logits, 1)[:, 1].numpy())
    return np.mean(out, axis=0)


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", required=True, nargs="+",
                    help="One or more .joblib bundles from save_model.py.")
    ap.add_argument("--dir", required=True, help="Folder of .txt recordings.")
    ap.add_argument("--out", default=None,
                    help="Where to write predictions (default: alongside the bundle).")
    ap.add_argument("--cache-dir", default=None,
                    help="TSFresh cache for the new recordings "
                         "(default: <out>/feature_cache, never the project cache).")
    args = ap.parse_args()

    out_dir = args.out or os.path.join(os.path.dirname(os.path.abspath(args.bundle[0])),
                                       "predictions")
    os.makedirs(out_dir, exist_ok=True)
    install_cache(args.cache_dir or os.path.join(out_dir, "feature_cache"))
    print(f"Recordings : {args.dir}")
    print(f"Output     : {out_dir}")
    print(f"TSFresh cache: {C.CACHE_DIR}  (project cache untouched)\n")

    names, subjects, lefts, rights, y_true, labelled = load_recordings(args.dir)
    n_pd = int((y_true == 1).sum()); n_co = int((y_true == 0).sum())
    print(f"{len(names)} recordings" +
          (f"  —  {n_pd} PD / {n_co} CO ({n_pd / len(names):.0%} PD)"
           if labelled else "  —  labels unknown"))
    if labelled and n_pd and n_co and abs(n_pd - n_co) > 0.1 * len(names):
        base = max(n_pd, n_co) / len(names)
        print(f"Imbalanced: predicting the majority class alone scores "
              f"{base:.3f} accuracy — read balanced accuracy, not accuracy.")
    print()

    from src.features import build_feature_matrix
    feats = build_feature_matrix(subjects, lefts, rights)
    idx = {"Left": 0, "Right": 1, "Combined": 2}

    rows, preds = [], {}
    for bpath in args.bundle:
        b = joblib.load(bpath)
        X = feats[idx[b["foot"]]]
        s = score_members(b, X)
        p = (s > 0.5).astype(int)
        preds[f"{b['model']}@{b['foot']}"] = (s, p)
        row = {"model": b["model"], "foot": b["foot"],
               "fit": b["provenance"]["fit_mode"],
               "members": len(b["members"]), "predicted_PD": int(p.sum())}
        if labelled:
            sens = float((p[y_true == 1] == 1).mean()) if n_pd else float("nan")
            spec = float((p[y_true == 0] == 0).mean()) if n_co else float("nan")
            acc = float((p == y_true).mean())
            from sklearn.metrics import roc_auc_score
            try:
                auc = float(roc_auc_score(y_true, s))
            except ValueError:
                auc = float("nan")
            lo, hi = wilson(int((p == y_true).sum()), len(y_true))
            row.update({"balanced_accuracy": 0.5 * (sens + spec), "accuracy": acc,
                        "accuracy_ci": f"[{lo:.2f}-{hi:.2f}]",
                        "sensitivity": sens, "specificity": spec, "auc": auc})
        rows.append(row)

    df = pd.DataFrame(rows)
    cols = ["model", "foot", "fit", "members"]
    if labelled:
        cols += ["balanced_accuracy", "accuracy", "accuracy_ci",
                 "sensitivity", "specificity", "auc"]
    cols += ["predicted_PD"]
    print(df[cols].to_string(index=False,
                             float_format=lambda v: f"{v:.3f}"))

    per_subject = pd.DataFrame({"subject": names})
    if labelled:
        per_subject["true"] = np.where(y_true == 1, "PD", "CO")
    for key, (s, p) in preds.items():
        per_subject[f"{key}__score"] = np.round(s, 4)
        per_subject[f"{key}__pred"] = np.where(p == 1, "PD", "CO")
    per_subject.to_csv(os.path.join(out_dir, "per_subject.csv"), index=False)
    df.to_csv(os.path.join(out_dir, "summary.csv"), index=False)
    print(f"\nper-subject -> {os.path.join(out_dir, 'per_subject.csv')}")
    print(f"summary     -> {os.path.join(out_dir, 'summary.csv')}")


if __name__ == "__main__":
    main()
