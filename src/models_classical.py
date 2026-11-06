"""
Classical baselines: Random Forest, Linear-SVM, RBF-SVM.

Per fold: grid-search the hyper-parameters by fitting on train and scoring on
validation (AUC), tune the decision threshold on validation (balanced accuracy),
refit the best configuration on train+val, then evaluate once on test.
"""
from __future__ import annotations

import itertools
import os
from dataclasses import dataclass
from typing import Callable

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC

from . import config as C
from .metrics import best_threshold, class_mean_threshold, compute_metrics


@dataclass
class ModelSpec:
    name: str
    representation: str            # 'sel' or 'pca'
    factory: Callable[..., object]
    grid: dict


def model_specs() -> list[ModelSpec]:
    def rf(**kw):
        return RandomForestClassifier(random_state=C.SEED, n_jobs=-1,
                                      class_weight="balanced", **kw)

    def svm_linear(**kw):
        return SVC(kernel="linear", class_weight="balanced",
                   random_state=C.SEED, **kw)

    def svm_rbf(**kw):
        return SVC(kernel="rbf", class_weight="balanced",
                   random_state=C.SEED, **kw)

    def logreg(**kw):
        return LogisticRegression(class_weight="balanced", max_iter=2000,
                                  random_state=C.SEED, **kw)

    def extratrees(**kw):
        return ExtraTreesClassifier(random_state=C.SEED, n_jobs=-1,
                                    class_weight="balanced", **kw)

    return [
        ModelSpec("RandomForest", "sel", rf, C.RF_GRID),
        ModelSpec("SVM-Linear", "sel", svm_linear, C.SVM_LINEAR_GRID),
        ModelSpec("SVM-RBF", "pca", svm_rbf, C.SVM_RBF_GRID),
        ModelSpec("LogisticRegression", "sel", logreg, C.LOGREG_GRID),
        ModelSpec("ExtraTrees", "sel", extratrees, C.EXTRATREES_GRID),
    ]


def _param_combos(grid: dict) -> list[dict]:
    keys = list(grid)
    return [dict(zip(keys, vals)) for vals in itertools.product(*(grid[k] for k in keys))]


def _scores(model, X: np.ndarray) -> np.ndarray:
    """Continuous scores for the positive class (probability or margin)."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def _save_scores(model_name, foot, fold_id, y_val, val_scores, y_te, te_scores,
                 is_probability):
    """Persist subject scores so thresholds and model fusion can be explored
    offline, without re-training anything."""
    if not getattr(C, "SAVE_CLASSICAL_SCORES", True):
        return
    out_dir = os.path.join(C.RESULTS_DIR, "classical_scores")
    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, f"{model_name}__{foot}__fold{fold_id}.npz"),
             y_val=np.asarray(y_val), val_scores=np.asarray(val_scores),
             y_test=np.asarray(y_te), test_scores=np.asarray(te_scores),
             is_probability=bool(is_probability))


def _native_threshold(model) -> float:
    """The classifier's own decision boundary, on the scale `_scores` returns."""
    return 0.5 if hasattr(model, "predict_proba") else 0.0


def _pick_threshold(model, y_val, val_scores) -> float:
    """
    Decision threshold according to C.CLASSICAL_THRESHOLD.

    'native' (default) keeps the classifier's own boundary. Every model here is
    fitted with class_weight="balanced" on balanced folds, so that boundary is
    already well placed — and re-tuning it on only 20 validation subjects
    overfits badly (it cost ~7 accuracy points on the Combined representation).
    """
    policy = getattr(C, "CLASSICAL_THRESHOLD", "native")
    if policy == "native":
        return _native_threshold(model)
    if policy == "class_mean":
        return class_mean_threshold(y_val, val_scores)
    if policy == "tuned":
        return best_threshold(y_val, val_scores)
    raise ValueError(f"Unknown CLASSICAL_THRESHOLD={policy!r}")


def train_eval_classical(spec: ModelSpec, prep, y_tr, y_val, y_te,
                         fold_id: int, foot: str) -> dict:
    Xtr, Xval, Xte = prep.get(spec.representation)

    # --- grid search on train -> validation (by AUC) ---
    best_params, best_auc, best_val_scores = None, -np.inf, None
    for params in _param_combos(spec.grid):
        model = spec.factory(**params)
        model.fit(Xtr, y_tr)
        val_scores = _scores(model, Xval)
        try:
            auc = roc_auc_score(y_val, val_scores)
        except ValueError:
            auc = 0.0
        if auc > best_auc:
            best_auc, best_params, best_val_scores = auc, params, val_scores

    # --- refit best config on train+val, evaluate on test ---
    X_trval = np.vstack([Xtr, Xval])
    y_trval = np.concatenate([y_tr, y_val])
    final = spec.factory(**best_params)
    final.fit(X_trval, y_trval)

    thr = _pick_threshold(final, y_val, best_val_scores)
    te_scores = _scores(final, Xte)
    te_pred = (te_scores >= thr).astype(int)
    metrics = compute_metrics(y_te, te_pred, te_scores)
    _save_scores(spec.name, foot, fold_id, y_val, best_val_scores, y_te, te_scores,
                 is_probability=hasattr(final, "predict_proba"))

    print(f"      {spec.name:12s} best={best_params} valAUC={best_auc:.3f} "
          f"thr={thr:.3g} | test acc={metrics['accuracy']:.3f} "
          f"auc={metrics['auc']:.3f}")

    return {"branch": "classical", "model": spec.name, "foot": foot,
            "fold": fold_id, "params": str(best_params), **metrics}
