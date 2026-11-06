#!/usr/bin/env python3
"""
Deep sequence models on the held-out evaluation set, for comparison only.

    python final_model_dl.py                        # ResNetSeq, Mamba, MambaFeat
    python final_model_dl.py --models Mamba         # a subset

Run final_model.py first: MambaFeat reuses the TSFresh window features it cached
for the 44 recordings in final_model/feature_cache.

Each model is trained with the standard protocol, through main.run_dl unchanged,
on one pseudo-fold per development fold: training = the fold's training and test
subjects (80), validation = its 20 validation subjects (early stopping and the
class-mean threshold), test = the 44 evaluation recordings. Five seeds per fold.
The evaluation labels are unknown, so the pipeline is given placeholder labels
for them: the test accuracy/AUC lines it prints are meaningless, ignore them.

Per model, each fold's scores are centred on that fold's own class-mean threshold
(recomputed from the saved validation scores with the pipeline's own function)
and averaged over the five folds; a recording is PD when the average exceeds 0.5.
That is the same rule as final_model.py, so all predictions are comparable.

Writes final_model/models/<model>/ and refreshes models/comparison.csv and
models/agreement.csv over every model run so far. Training output stays in
final_model/dl/; results/ is never written and the primary output files are not touched.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from src import config as C                                            # noqa: E402

import final_model as FM                                               # noqa: E402
import predict as P                                                    # noqa: E402

DL_MODELS = ["ResNetSeq", "Mamba", "MambaFeat"]
DEV_CACHE_DIR = C.CACHE_DIR          # the development cache, captured before redirecting


def install_sandbox(sandbox: str, eval_cache: str) -> None:
    """
    Send every output of the training code to the sandbox; results/ is never written.

    src/ reads C.RESULTS_DIR / C.FIG_DIR / C.CACHE_DIR at call time, so re-pointing
    them is enough. Feature files are read from the development cache when they
    exist there (every development subject), otherwise from final_model.py's cache
    of the 44 recordings, which is also where anything new would be written.
    """
    C.RESULTS_DIR = sandbox
    C.FIG_DIR = os.path.join(sandbox, "figures")
    C.CACHE_DIR = eval_cache
    C.ensure_dirs()
    from src import features as F

    def reuse(builder):
        def path(subject, foot):
            name = os.path.basename(builder(subject, foot))
            hit = os.path.join(DEV_CACHE_DIR, name)
            return hit if os.path.exists(hit) else os.path.join(C.CACHE_DIR, name)
        return path

    F._cache_path = reuse(F._cache_path)
    F._window_cache_path = reuse(F._window_cache_path)


def pseudo_fold(fold: dict, ev: tuple) -> dict:
    """Fold train + fold test for training (80), fold validation (20), the 44 as test."""
    tr_s, tr_y, tr_l, tr_r = fold["train"]
    te_s, te_y, te_l, te_r = fold["test"]
    train = (list(tr_s) + list(te_s), np.concatenate([tr_y, te_y]),
             list(tr_l) + list(te_l), list(tr_r) + list(te_r))
    return {"train": train, "val": fold["val"], "test": ev}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=FM.OUT)
    ap.add_argument("--models", nargs="+", choices=DL_MODELS, default=DL_MODELS)
    args = ap.parse_args()
    t0 = time.time()

    sandbox = os.path.join(args.out, "dl")
    install_sandbox(sandbox, os.path.join(args.out, "feature_cache"))
    # same subject ids as final_model.py, so its cached features are reused
    names, subjects, lefts, rights, _, labelled = P.load_recordings(FM.EVAL_DIR)
    assert not labelled and len(names) == 44, "expected the 44 unlabelled recordings"
    placeholder = np.zeros(len(names), dtype=int)       # labels unknown; never used
    ev = (subjects, placeholder, lefts, rights)
    print(f"Models : {', '.join(args.models)} ({FM.FOOT})")
    print(f"Sandbox: {os.path.relpath(sandbox, ROOT)}  (results/ untouched)")
    print("Test metrics printed below use placeholder labels and mean nothing.\n")

    from main import run_dl
    from src.data import load_fold
    results = []
    for fold_id in range(1, C.N_FOLDS + 1):
        fold = pseudo_fold(load_fold(fold_id), ev)               # 80 train / 20 val
        run_dl(fold_id, fold, [FM.FOOT], args.models, False, results)

    from sklearn.metrics import roc_auc_score
    from src.train_dl import _pick_dl_threshold
    root = os.path.join(args.out, "models")
    print()
    for model in args.models:
        centred, folds = [], []
        for fold_id in range(1, C.N_FOLDS + 1):
            d = np.load(os.path.join(sandbox, "dl_scores",
                                     f"{model}__{FM.FOOT}__fold{fold_id}.npz"))
            assert len(d["test_scores"]) == len(names)
            thr = float(_pick_dl_threshold(d["y_val"], d["val_scores"]))
            centred.append(np.asarray(d["test_scores"], float) - thr + 0.5)
            folds.append({"fold": fold_id, "thr": thr,
                          "val_auc": float(roc_auc_score(d["y_val"], d["val_scores"]))})
        FM.write_outputs(model, np.mean(centred, axis=0), names,
                         os.path.join(root, model),
                         {"training_subjects": 100, "members": folds,
                          "seeds": list(C.DL_ENSEMBLE_SEEDS),
                          "minutes": round((time.time() - t0) / 60, 1)})
    FM.write_comparison(root, names)


if __name__ == "__main__":
    main()
