#!/usr/bin/env python3
"""
Final model for the held-out evaluation set.

    python final_model.py                 # on the GPU machine

Model: MLPFeat on the combined-foot aggregated TSFresh vector, chosen from the
five-fold cross-validation results: the highest AUC of all
models (0.92) with the lowest fold-to-fold spread (+-0.04), and balanced errors
(sensitivity 0.82, specificity 0.80). AUC does not depend on the threshold,
which matters because the class balance of the evaluation set is unknown.

Training uses only the 100 development subjects in splits/. The model is five
fold-members: member k trains on fold k's training and test subjects (80) and
uses fold k's 20 validation subjects for early stopping and for its class-mean
decision threshold, exactly as in the reported protocol; every member is itself
a five-seed ensemble. The members are averaged, each score centred on its own
threshold, and a recording is predicted PD when the average exceeds 0.5.

The 44 recordings in heldout_eval/ are only scored: no label,
identity or file comparison is used, and nothing is tuned on them. Output goes
to final_model/, including a filled copy of prediction_template.xlsx.

Other models, for comparison only (the primary model stays the MLPFeat above):

    python final_model.py --models RandomForest ExtraTrees LogisticRegression SVM-Linear SVM-RBF MLPFeat

Each model is trained the same way (five fold-members; hyper-parameters chosen on
each fold's validation subjects by AUC, classical models at their native
boundary) and written to final_model/models/<model>/. models/comparison.csv lists
every model's prediction per recording and models/agreement.csv how often each
pair of models agrees. No accuracy is computed: the 44 labels are unknown.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time

import joblib
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from src import config as C                                            # noqa: E402
from src.models_classical import model_specs, _scores, _native_threshold  # noqa: E402

import predict as P                                                    # noqa: E402
import save_model as SM                                                # noqa: E402

MODEL = "MLPFeat"
FOOT = "Combined"
EVAL_DIR = os.path.join(ROOT, "heldout_eval")
TEMPLATE = os.path.join(EVAL_DIR, "prediction_template.xlsx")
OUT = os.path.join(ROOT, "final_model")
SPECS = {s.name: s for s in model_specs()}
ALL_MODELS = list(SPECS) + [MODEL]


def member_sets(folds_data):
    """Fold train + fold test for training (80), fold validation held out (20)."""
    return [{"fold": fd["fold"],
             "Xtr": np.vstack([fd["Xtr"], fd["Xte"]]),
             "ytr": np.concatenate([fd["ytr"], fd["yte"]]),
             "Xva": fd["Xva"], "yva": fd["yva"]} for fd in folds_data]


def train_members(name, sets, seeds):
    from sklearn.metrics import roc_auc_score
    from src.train_dl import _pick_dl_threshold
    members = []
    for fd in sets:
        prep = SM.fit_prep(fd["Xtr"], fd["ytr"])
        if name == MODEL:
            Xtr, Xva = prep.transform(fd["Xtr"], "sel"), prep.transform(fd["Xva"], "sel")
            est = SM.fit_mlpfeat(Xtr, fd["ytr"], Xva, fd["yva"], seeds)
            val = P._mlp_score(est, Xva)
            m = {"kind": "mlp", "rep": "sel", "est": est,
                 "thr": float(_pick_dl_threshold(fd["yva"], val)), "params": None}
        else:
            spec = SPECS[name]
            params, _ = SM.select_params(spec, [fd])
            est = spec.factory(**params).fit(prep.transform(fd["Xtr"], spec.representation),
                                             fd["ytr"])
            m = {"kind": "sk", "rep": spec.representation, "est": est,
                 "thr": _native_threshold(est), "params": params}
        m.update(fold=fd["fold"], prep=prep)
        m["val_auc"] = float(roc_auc_score(fd["yva"], raw_score(m, fd["Xva"])))
        members.append(m)
        print(f"  {name} member fold {fd['fold']}: {len(fd['ytr'])} train / "
              f"{len(fd['yva'])} val, threshold {m['thr']:.3f}"
              + (f", params {m['params']}" if m["params"] else ""))
    return members


def raw_score(m, X):
    """The member's own score: probability, or SVM margin."""
    Xt = m["prep"].transform(X, m["rep"])
    return P._mlp_score(m["est"], Xt) if m["kind"] == "mlp" else _scores(m["est"], Xt)


def centred(m, X):
    """Score on a common scale where 0.5 is the member's own decision boundary."""
    s = raw_score(m, X)
    if m["kind"] == "mlp":
        return s - m["thr"] + 0.5                 # class-mean threshold
    if m["thr"] == 0.0:
        return 1.0 / (1.0 + np.exp(-s))           # SVM margin, boundary 0 -> 0.5
    return s                                      # probability, native 0.5


def score(members, X):
    """Mean over members of the score centred on each member's own threshold."""
    return np.mean([centred(m, X) for m in members], axis=0)


def eval_features(cache_dir):
    """Aggregated TSFresh vectors for the evaluation recordings, cached apart."""
    from src.features import build_feature_matrix
    names, subjects, lefts, rights, _, _ = P.load_recordings(EVAL_DIR)
    project_cache = C.CACHE_DIR
    P.install_cache(cache_dir)
    try:
        X = build_feature_matrix(subjects, lefts, rights)[2]          # Combined
    finally:
        C.CACHE_DIR = project_cache
    return names, X


def fill_template(pred: dict, path: str) -> None:
    import openpyxl
    shutil.copyfile(TEMPLATE, path)
    wb = openpyxl.load_workbook(path)
    ws = wb.active
    assert [c.value for c in ws[1]][:2] == ["AnonID", "Prediction"], "unexpected template header"
    ids = [row[0].value for row in ws.iter_rows(min_row=2) if row[0].value]
    assert sorted(ids) == sorted(pred), "template AnonIDs and scored recordings differ"
    for row in ws.iter_rows(min_row=2):
        if row[0].value:
            row[1].value = int(pred[row[0].value])
    wb.save(path)
    # read back: same rows in the same order, only 0/1, nothing blank
    back = [r[:2] for r in openpyxl.load_workbook(path).active.iter_rows(
        min_row=2, values_only=True) if r[0]]
    assert [a for a, _ in back] == ids and all(b in (0, 1) for _, b in back)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--seeds", nargs="+", type=int, default=list(C.DL_ENSEMBLE_SEEDS),
                    help="MLPFeat seeds per member (default: the reported five)")
    ap.add_argument("--models", nargs="+", choices=ALL_MODELS, default=None,
                    help="compare these models instead (written to <out>/models/; "
                         "the primary output files are not touched)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()

    folds_data, _, y_dev, dev_subjects = SM.build_matrices(range(1, C.N_FOLDS + 1), FOOT)
    SM.assert_matches_pipeline(SM.fit_prep(folds_data[0]["Xtr"], folds_data[0]["ytr"]),
                               folds_data[0]["Xtr"], folds_data[0]["ytr"],
                               folds_data[0]["Xva"])
    names, X_ev = eval_features(os.path.join(args.out, "feature_cache"))
    assert len(names) == 44, f"expected 44 evaluation recordings, found {len(names)}"
    assert X_ev.shape[1] == folds_data[0]["Xtr"].shape[1], "feature vectors differ in length"
    print(f"{', '.join(args.models or [MODEL])} ({FOOT}) on {len(dev_subjects)} development subjects "
          f"({int(y_dev.sum())} PD / {int((y_dev == 0).sum())} CO); "
          f"{len(names)} evaluation recordings, {X_ev.shape[1]} raw features\n")

    sets = member_sets(folds_data)
    if not args.models:                                   # the primary run
        run_model(MODEL, sets, X_ev, names, dev_subjects, args.seeds, args.out, t0)
        return
    root = os.path.join(args.out, "models")
    for name in args.models:
        run_model(name, sets, X_ev, names, dev_subjects, args.seeds,
                  os.path.join(root, name), t0)
    write_comparison(root, names)


def write_comparison(root, names):
    """comparison.csv and agreement.csv over every model in <root>/ so far."""
    preds = {}
    for name in sorted(os.listdir(root)):
        f = os.path.join(root, name, "predictions.csv")
        if os.path.isfile(f):
            df = pd.read_csv(f)
            assert list(df.AnonID) == list(names), f"{name}: recordings differ"
            preds[name] = list(df.Prediction)
    comp = pd.DataFrame({"AnonID": names, **preds})
    comp["models_saying_PD"] = comp[list(preds)].sum(axis=1)
    comp.to_csv(os.path.join(root, "comparison.csv"), index=False)
    agree = pd.DataFrame([[int((comp[a] == comp[b]).sum()) for b in preds] for a in preds],
                         index=list(preds), columns=list(preds))
    agree.to_csv(os.path.join(root, "agreement.csv"))
    print("\nPredicted PD per model: " +
          ", ".join(f"{m} {int(comp[m].sum())}" for m in preds))
    unanimous = int(comp.models_saying_PD.isin([0, len(preds)]).sum())
    print(f"All {len(preds)} models agree on {unanimous}/{len(names)} recordings.")
    print("Pairwise agreement (recordings out of 44):\n" + agree.to_string())
    print(f"\nComparison -> {os.path.relpath(root, ROOT)}/  (primary output files untouched)")


def write_outputs(name, s, names, out, summary):
    """Scores, predictions, a filled template and a summary for one model."""
    os.makedirs(out, exist_ok=True)
    pred = {n: int(v > 0.5) for n, v in zip(names, s)}
    pd.DataFrame({"AnonID": names, "score": np.round(s, 4),
                  "Prediction": [pred[n] for n in names]}
                 ).to_csv(os.path.join(out, "eval_scores.csv"), index=False)
    pd.DataFrame({"AnonID": names, "Prediction": [pred[n] for n in names]}
                 ).to_csv(os.path.join(out, "predictions.csv"), index=False)
    fill_template(pred, os.path.join(out, "prediction_template_filled.xlsx"))
    n_pd = sum(pred.values())
    near = int((np.abs(s - 0.5) < 0.05).sum())
    summary = {"model": name, "foot": FOOT, **summary,
               "eval_predicted_PD": n_pd, "eval_predicted_CO": len(pred) - n_pd,
               "eval_scores_within_0.05_of_boundary": near}
    with open(os.path.join(out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    print(f"  {name}: {n_pd} PD / {len(pred) - n_pd} CO predicted "
          f"({near} recordings within 0.05 of the boundary) -> {os.path.relpath(out, ROOT)}/")
    return pred


def run_model(name, sets, X_ev, names, dev_subjects, seeds, out, t0):
    os.makedirs(out, exist_ok=True)
    print(f"\n=== {name} " + "=" * 50)
    members = train_members(name, sets, seeds)
    s = score(members, X_ev)
    joblib.dump({"model": name, "foot": FOOT, "members": members, "seeds": seeds,
                 "trained_on": dev_subjects, "tsfresh": C.TSFRESH_FC_PARAMETERS},
                os.path.join(out, "final_model.joblib"), compress=3)
    return write_outputs(name, s, names, out, {
        "training_subjects": len(dev_subjects),
        "members": [{"fold": m["fold"], "thr": m["thr"], "val_auc": m["val_auc"],
                     "params": m["params"]} for m in members],
        "minutes": round((time.time() - t0) / 60, 1)})


if __name__ == "__main__":
    main()
