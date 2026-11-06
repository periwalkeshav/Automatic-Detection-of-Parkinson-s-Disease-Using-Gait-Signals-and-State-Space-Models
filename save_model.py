#!/usr/bin/env python3
"""
Persist trained models so new recordings can be scored without retraining.
Same family as check_holdout.py.

The project pipeline is pure cross-validation: it fits, scores and throws the
model away. That is correct for reporting, but it means every hold-out check
retrains from scratch. This script fits once and writes a self-contained bundle
(fitted preprocessing + estimator + threshold + provenance) that predict.py can
load and apply to any folder of recordings.

Covers the five classical models and MLPFeat, which all consume the same
aggregated TSFresh vector. The sequence models (1D-ResNet, Mamba, MambaFeat)
train on windows/crops with attention-MIL over five seeds; they are not covered
here — use check_holdout.py for those.

    python save_model.py --models RandomForest SVM-Linear MLPFeat --feet Combined

Writes to saved_models/. Nothing in src/, results/ or splits/ is touched.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src import config as C                                    # noqa: E402
from src.data import load_fold                                 # noqa: E402
from src.metrics import compute_metrics                        # noqa: E402
from src.models_classical import (model_specs, _param_combos, _scores,  # noqa: E402
                                  _native_threshold)
from src.preprocessing import _clean_fit, _clean_apply, prepare_classical_fold  # noqa: E402

BUNDLE_VERSION = 1
MLP_NAME = "MLPFeat"


# --------------------------------------------------------------------------- #
# Preprocessing, but keeping the fitted objects
# --------------------------------------------------------------------------- #
@dataclass
class FittedPrep:
    """The four fitted transforms of prepare_classical_fold, kept for reuse."""
    col_mean: np.ndarray
    keep: np.ndarray
    scaler: StandardScaler
    selector: SelectKBest
    pca: PCA

    def transform(self, X: np.ndarray, representation: str) -> np.ndarray:
        Xs = self.scaler.transform(_clean_apply(X, self.col_mean, self.keep))
        Xsel = self.selector.transform(Xs)
        return Xsel if representation == "sel" else self.pca.transform(Xsel)


def fit_prep(Xtr: np.ndarray, ytr: np.ndarray) -> FittedPrep:
    """Mirrors preprocessing.prepare_classical_fold step for step."""
    col_mean, keep = _clean_fit(Xtr)
    Xtr_c = _clean_apply(Xtr, col_mean, keep)
    scaler = StandardScaler().fit(Xtr_c)
    Xtr_s = scaler.transform(Xtr_c)
    k = min(C.SELECT_K, Xtr_s.shape[1])
    selector = SelectKBest(f_classif, k=k).fit(Xtr_s, ytr)
    Xtr_sel = selector.transform(Xtr_s)
    n_comp = min(C.PCA_MAX, Xtr_sel.shape[1], max(Xtr_sel.shape[0] - 1, 1))
    pca = PCA(n_components=n_comp, random_state=C.SEED).fit(Xtr_sel)
    return FittedPrep(col_mean, keep, scaler, selector, pca)


def assert_matches_pipeline(prep: FittedPrep, Xtr, ytr, Xother) -> None:
    """
    Guard against drift: the saved transforms must reproduce the pipeline's
    own output exactly. If src/preprocessing.py ever changes, this fails loudly
    instead of silently exporting a model that behaves differently.
    """
    ref = prepare_classical_fold(Xtr, ytr, Xother, Xother)
    for rep, (mine_tr, ref_tr, mine_o, ref_o) in {
        "sel": (prep.transform(Xtr, "sel"), ref.Xtr_sel,
                prep.transform(Xother, "sel"), ref.Xval_sel),
        "pca": (prep.transform(Xtr, "pca"), ref.Xtr_pca,
                prep.transform(Xother, "pca"), ref.Xval_pca),
    }.items():
        for label, a, b in (("train", mine_tr, ref_tr), ("held", mine_o, ref_o)):
            if a.shape != b.shape or not np.allclose(a, b, rtol=0, atol=1e-10):
                raise AssertionError(
                    f"saved preprocessing diverges from prepare_classical_fold "
                    f"({rep}/{label}): max |diff| = "
                    f"{np.abs(a - b).max() if a.shape == b.shape else 'shape mismatch'}. "
                    f"src/preprocessing.py has changed — update fit_prep().")


# --------------------------------------------------------------------------- #
# Hyper-parameters: the pipeline picks them per fold on validation
# --------------------------------------------------------------------------- #
def select_params(spec, folds_data) -> tuple[dict, list[dict]]:
    """
    Score every grid point on each fold's validation split, exactly as the
    pipeline does, and keep the configuration that wins most often. Ties break
    on mean validation AUC, then on the grid order.
    """
    wins, per_fold, auc_sum = Counter(), [], {}
    for fd in folds_data:
        prep = fit_prep(fd["Xtr"], fd["ytr"])
        Xtr = prep.transform(fd["Xtr"], spec.representation)
        Xva = prep.transform(fd["Xva"], spec.representation)
        best, best_auc = None, -np.inf
        for params in _param_combos(spec.grid):
            model = spec.factory(**params).fit(Xtr, fd["ytr"])
            try:
                auc = compute_metrics(fd["yva"],
                                      (_scores(model, Xva) > _native_threshold(model)).astype(int),
                                      _scores(model, Xva))["auc"]
            except ValueError:
                auc = 0.0
            key = json.dumps(params, sort_keys=True)
            auc_sum[key] = auc_sum.get(key, 0.0) + auc
            if auc > best_auc:
                best, best_auc = params, auc
        wins[json.dumps(best, sort_keys=True)] += 1
        per_fold.append({"params": best, "val_auc": round(float(best_auc), 4)})
    top = max(wins.items(), key=lambda kv: (kv[1], auc_sum.get(kv[0], 0.0)))[0]
    return json.loads(top), per_fold


# --------------------------------------------------------------------------- #
def build_matrices(fold_ids, foot):
    """Feature matrices per fold, plus the union of all development subjects."""
    from src.features import build_feature_matrix
    idx = {"Left": 0, "Right": 1, "Combined": 2}[foot]
    folds_data, seen, X_all, y_all, subj_all = [], set(), [], [], []
    for fid in fold_ids:
        fold = load_fold(fid)
        parts = {}
        for split in ("train", "val", "test"):
            s, y, l, r = fold[split]
            parts[split] = (build_feature_matrix(s, l, r)[idx], np.asarray(y, int), s)
        folds_data.append({"fold": fid,
                           "Xtr": parts["train"][0], "ytr": parts["train"][1],
                           "Xva": parts["val"][0], "yva": parts["val"][1],
                           "Xte": parts["test"][0], "yte": parts["test"][1]})
        for split in ("train", "val", "test"):
            X, y, s = parts[split]
            for row, lab, name in zip(X, y, s):
                if name not in seen:
                    seen.add(name); X_all.append(row); y_all.append(lab)
                    subj_all.append(name)
    return folds_data, np.vstack(X_all), np.asarray(y_all, int), subj_all


# --------------------------------------------------------------------------- #
# MLPFeat
# --------------------------------------------------------------------------- #
def fit_mlpfeat(Xtr, ytr, Xva, yva, seeds):
    """Deep ensemble of MLPFeat, same config and early stopping as the pipeline."""
    import torch
    from src.train_dl import _fit_mlpfeat
    cfg = C.DL_MLPFEAT_CFG
    states = []
    for seed in seeds:
        model, _ = _fit_mlpfeat(Xtr.astype(np.float32), ytr, Xtr.shape[1], cfg,
                                seed, cfg["epochs"], Xva.astype(np.float32), yva,
                                cfg["patience"])
        states.append({k: v.detach().cpu().numpy()
                       for k, v in model.state_dict().items()})
    return {"states": states, "cfg": dict(cfg), "in_features": int(Xtr.shape[1]),
            "seeds": list(seeds)}


# --------------------------------------------------------------------------- #
def provenance(args, subjects) -> dict:
    keys = ["SEED", "N_FOLDS", "FEET", "SAMPLING_RATE", "TRIM_SECONDS",
            "WIN_TS", "STEP_TS", "TSFRESH_FC_PARAMETERS", "AGG_FUNCS",
            "SELECT_K", "PCA_MAX", "CLASSICAL_THRESHOLD", "DL_THRESHOLD",
            "RF_GRID", "SVM_LINEAR_GRID", "SVM_RBF_GRID", "LOGREG_GRID",
            "EXTRATREES_GRID", "DL_MLPFEAT_CFG", "DL_ENSEMBLE_SEEDS"]
    snap = {}
    for k in keys:
        v = getattr(C, k, None)
        snap[k] = list(v) if isinstance(v, tuple) else v
    return {
        "bundle_version": BUNDLE_VERSION,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "fit_mode": args.fit,
        "folds": args.folds,
        "n_development_subjects": len(subjects),
        "development_subjects": sorted(subjects),
        "config": snap,
        "note": "Internal artefact.",
    }


def save_bundle(path, payload) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(payload, path, compress=3)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+",
                    default=["RandomForest", "SVM-Linear", "SVM-RBF",
                             "LogisticRegression", "ExtraTrees", MLP_NAME])
    ap.add_argument("--feet", nargs="+", default=["Combined"], choices=list(C.FEET))
    ap.add_argument("--folds", nargs="+", type=int,
                    default=list(range(1, C.N_FOLDS + 1)))
    ap.add_argument("--fit", choices=["full", "cv"], default="full",
                    help="'full': one model on all 100 development subjects, "
                         "hyper-parameters chosen by per-fold validation. "
                         "'cv': keep the five fold models and ensemble at "
                         "predict time (reproduces the reported protocol).")
    ap.add_argument("--out", default=os.path.join(C.BASE_DIR, "saved_models"))
    ap.add_argument("--mlp-seeds", nargs="+", type=int,
                    default=list(C.DL_ENSEMBLE_SEEDS))
    args = ap.parse_args()

    specs = {s.name: s for s in model_specs()}
    unknown = [m for m in args.models if m not in specs and m != MLP_NAME]
    if unknown:
        sys.exit(f"Not supported by this script: {unknown}. "
                 f"Sequence models train on windows with attention-MIL — "
                 f"use check_holdout.py for those.")

    print(f"Output    : {args.out}")
    print(f"Fit mode  : {args.fit}")
    print(f"Feet      : {', '.join(args.feet)}")
    print(f"Models    : {', '.join(args.models)}\n")

    for foot in args.feet:
        print(f"=== {foot} " + "=" * 60)
        folds_data, X_all, y_all, subjects = build_matrices(args.folds, foot)
        print(f"development subjects: {len(subjects)}  "
              f"({int(y_all.sum())} PD / {int((y_all == 0).sum())} CO)  "
              f"raw features: {X_all.shape[1]}")

        # one integrity check per foot, on real data
        assert_matches_pipeline(fit_prep(folds_data[0]["Xtr"], folds_data[0]["ytr"]),
                                folds_data[0]["Xtr"], folds_data[0]["ytr"],
                                folds_data[0]["Xva"])
        print("preprocessing matches prepare_classical_fold exactly ✓")

        for name in args.models:
            t0 = time.time()
            if args.fit == "full":
                prep = fit_prep(X_all, y_all)
                if name == MLP_NAME:
                    # MLPFeat early-stops on validation, so it needs one held
                    # out split even in 'full' mode: use fold 1's validation.
                    fd = folds_data[0]
                    p1 = fit_prep(fd["Xtr"], fd["ytr"])
                    est = fit_mlpfeat(p1.transform(fd["Xtr"], "sel"), fd["ytr"],
                                      p1.transform(fd["Xva"], "sel"), fd["yva"],
                                      args.mlp_seeds)
                    prep, rep, params = p1, "sel", None
                    kind = "mlpfeat"
                else:
                    spec = specs[name]
                    params, per_fold = select_params(spec, folds_data)
                    Xt = prep.transform(X_all, spec.representation)
                    est = spec.factory(**params).fit(Xt, y_all)
                    rep, kind = spec.representation, "sklearn"
                members = [{"prep": prep, "est": est, "fold": None}]
            else:
                members, params, rep = [], None, None
                for fd in folds_data:
                    p = fit_prep(fd["Xtr"], fd["ytr"])
                    if name == MLP_NAME:
                        e = fit_mlpfeat(p.transform(fd["Xtr"], "sel"), fd["ytr"],
                                        p.transform(fd["Xva"], "sel"), fd["yva"],
                                        args.mlp_seeds)
                        rep, kind = "sel", "mlpfeat"
                    else:
                        spec = specs[name]
                        pr, _ = select_params(spec, [fd])
                        e = spec.factory(**pr).fit(
                            p.transform(fd["Xtr"], spec.representation), fd["ytr"])
                        rep, kind, params = spec.representation, "sklearn", pr
                    members.append({"prep": p, "est": e, "fold": fd["fold"]})

            bundle = {
                "model": name, "foot": foot, "kind": kind,
                "representation": rep, "params": params,
                "members": members,
                "threshold_policy": (C.DL_THRESHOLD if name == MLP_NAME
                                     else C.CLASSICAL_THRESHOLD),
                "provenance": provenance(args, subjects),
            }
            path = os.path.join(args.out, f"{name}__{foot}__{args.fit}.joblib")
            save_bundle(path, bundle)
            size = os.path.getsize(path) / 1e6
            print(f"  {name:<20} {len(members)} member(s)  "
                  f"{size:6.1f} MB  {time.time() - t0:6.1f}s  -> "
                  f"{os.path.relpath(path, C.BASE_DIR)}")

    man = os.path.join(args.out, "manifest.json")
    with open(man, "w") as fh:
        json.dump(provenance(args, subjects), fh, indent=2)
    print(f"\nmanifest -> {os.path.relpath(man, C.BASE_DIR)}")
    print("Score new recordings with:  python predict.py --bundle <file> --dir <folder>")


if __name__ == "__main__":
    main()
