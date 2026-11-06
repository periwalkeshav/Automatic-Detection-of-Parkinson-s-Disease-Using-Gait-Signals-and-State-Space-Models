"""Classification metrics. Positive class = PD (label 1)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    y_score: np.ndarray) -> dict:
    """
    Return Accuracy, Sensitivity, Specificity, AUC plus
    Precision, Recall, F1 (to match the target table), and the 2x2 confusion
    matrix [[TN, FP], [FN, TP]].
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    sensitivity = tp / max(tp + fn, 1)          # recall of PD
    specificity = tn / max(tn + fp, 1)
    precision = tp / max(tp + fp, 1)
    recall = sensitivity
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)

    try:
        auc = roc_auc_score(y_true, y_score)
    except ValueError:                           # only one class present
        auc = float("nan")

    return {
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "auc": auc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "cm": cm.tolist(),
    }


def youden_capped_threshold(y_true: np.ndarray, y_prob: np.ndarray,
                            lo: float = 0.35, hi: float = 0.65) -> float:
    """
    Threshold on probabilities maximising Youden's J (sensitivity + specificity
    - 1) on a validation set, restricted to the [lo, hi] band. The cap keeps the
    cutoff from running to an extreme on a tiny validation set (the 'youden_capped'
    policy from the reference implementation).
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    cands = np.unique(np.r_[np.linspace(lo, hi, 61), y_prob])
    cands = cands[(cands >= lo) & (cands <= hi)]
    if cands.size == 0:
        return 0.5
    best_thr, best_j = 0.5, -np.inf
    for t in cands:
        pred = (y_prob >= t).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fn = int(((pred == 0) & (y_true == 1)).sum())
        tn = int(((pred == 0) & (y_true == 0)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        j = tp / max(tp + fn, 1) + tn / max(tn + fp, 1) - 1.0
        if j > best_j:
            best_j, best_thr = j, float(t)
    return best_thr


def class_mean_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Robust operating point: the midpoint between the class-conditional mean
    scores on a (validation) set. Uses every point instead of chasing a single
    edge, so it transfers far better than an argmax rule on ~20 subjects, and it
    makes no assumption about the class balance of the held-out test set.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if pos.size == 0 or neg.size == 0:
        return float(np.median(y_score)) if y_score.size else 0.0
    return float((pos.mean() + neg.mean()) / 2.0)


def best_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Threshold on scores maximising balanced accuracy on a validation set."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    candidates = np.unique(y_score)
    if candidates.size == 0:
        return 0.5
    # Evaluate midpoints plus the extremes so every split is reachable.
    mids = (candidates[:-1] + candidates[1:]) / 2.0
    grid = np.concatenate([[candidates.min() - 1e-6], mids, [candidates.max() + 1e-6]])

    best_thr, best_bacc = grid[0], -1.0
    for thr in grid:
        pred = (y_score >= thr).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fn = int(((pred == 0) & (y_true == 1)).sum())
        tn = int(((pred == 0) & (y_true == 0)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        sens = tp / max(tp + fn, 1)
        spec = tn / max(tn + fp, 1)
        bacc = 0.5 * (sens + spec)
        if bacc > best_bacc:
            best_bacc, best_thr = bacc, float(thr)
    return best_thr
