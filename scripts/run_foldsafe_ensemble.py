#!/usr/bin/env python3
"""Validation-tuned ensemble over the strongest fold-safe feature models."""

from __future__ import annotations

import ast
import itertools
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, roc_auc_score
from sklearn.preprocessing import MinMaxScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_deliverable2_foldsafe import (  # noqa: E402
    FIGURES,
    TABLES,
    GROUP_TO_LABEL,
    LABEL_TO_GROUP,
    align_columns,
    base_feature_view,
    classifier_only,
    classical_pipeline,
    fold_manifest,
    fold_split_matrix,
    fresh_feature_views,
    matrix_xy,
    score_from_model,
)


def ensure_dirs() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, score: np.ndarray) -> dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    auc = float("nan")
    if len(np.unique(y_true)) == 2:
        auc = float(roc_auc_score(y_true, score))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else 0.0,
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": auc,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def selection_key(scores: dict[str, float], metric: str) -> tuple[float, float, float, float]:
    auc_key = -1.0 if math.isnan(scores["auc"]) else scores["auc"]
    balanced = (scores["sensitivity"] + scores["specificity"]) / 2.0
    primary = {
        "accuracy": scores["accuracy"],
        "balanced_accuracy": balanced,
        "f1": scores["f1"],
        "auc": auc_key,
    }[metric]
    return primary, scores["accuracy"], auc_key, scores["f1"]


def tune_threshold(y_true: np.ndarray, score: np.ndarray, metric: str) -> tuple[float, dict[str, float]]:
    best_threshold = 0.5
    best_scores: dict[str, float] | None = None
    best_key: tuple[float, float, float, float] | None = None
    thresholds = np.unique(np.r_[np.linspace(0.05, 0.95, 91), score])
    for threshold in thresholds:
        pred = (score >= threshold).astype(int)
        scores = binary_metrics(y_true, pred, score)
        key = selection_key(scores, metric)
        if best_key is None or key > best_key:
            best_key = key
            best_threshold = float(threshold)
            best_scores = scores
    assert best_scores is not None
    return best_threshold, best_scores


def normalize_scores(val_score: np.ndarray, test_score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scaler = MinMaxScaler()
    val_norm = scaler.fit_transform(val_score.reshape(-1, 1)).ravel()
    test_norm = scaler.transform(test_score.reshape(-1, 1)).ravel()
    return np.clip(val_norm, 0.0, 1.0), np.clip(test_norm, 0.0, 1.0)


def load_feature_matrices(manifest: pd.DataFrame, fold: str, feature_set: str, trim_seconds: float):
    return (
        fold_split_matrix(manifest, fold, "training", feature_set, trim_seconds, False),
        fold_split_matrix(manifest, fold, "validation", feature_set, trim_seconds, False),
        fold_split_matrix(manifest, fold, "test", feature_set, trim_seconds, False),
    )


def score_base_model(
    row: pd.Series,
    manifest: pd.DataFrame,
    feature_set: str,
    trim_seconds: float,
) -> dict[str, object]:
    fold = row["fold"]
    scope = row["scope"]
    model_name = row["model"]
    params = ast.literal_eval(row["best_params"])
    train_matrix, val_matrix, test_matrix = load_feature_matrices(manifest, fold, feature_set, trim_seconds)
    columns, train_aligned, val_aligned, test_aligned = align_columns(train_matrix, val_matrix, test_matrix, scope)
    x_train, y_train, _ = matrix_xy(train_aligned, columns)
    x_val, y_val, val_subjects = matrix_xy(val_aligned, columns)
    x_test, y_test, test_subjects = matrix_xy(test_aligned, columns)

    if model_name.endswith("_fresh"):
        base_model_name = model_name.removesuffix("_fresh")
        base = base_feature_view(x_train, y_train, x_val, x_test, columns)
        fdr_level = float(params["fdr_level"])
        views = fresh_feature_views(base, y_train, [fdr_level])
        if fdr_level not in views:
            raise ValueError(f"No TSFresh-selected features for {fold}/{scope}/{model_name}")
        x_train_f, x_val_f, x_test_f, selected_columns = views[fdr_level]
        model = classifier_only(base_model_name, params)
        model.fit(x_train_f, y_train)
        val_score = model.decision_function(x_val_f) if hasattr(model, "decision_function") else model.predict_proba(x_val_f)[:, 1]
        test_score = model.decision_function(x_test_f) if hasattr(model, "decision_function") else model.predict_proba(x_test_f)[:, 1]
        n_selected = len(selected_columns)
    else:
        model = classical_pipeline(model_name, params)
        model.fit(x_train, y_train)
        val_score = score_from_model(model, x_val)
        test_score = score_from_model(model, x_test)
        n_selected = params.get("k_best")

    val_score, test_score = normalize_scores(np.asarray(val_score), np.asarray(test_score))
    return {
        "fold": fold,
        "scope": scope,
        "model": model_name,
        "label": f"{scope}:{model_name}",
        "val_subjects": val_subjects,
        "test_subjects": test_subjects,
        "y_val": y_val,
        "y_test": y_test,
        "val_score": val_score,
        "test_score": test_score,
        "n_selected": n_selected,
    }


def candidate_specs(metrics: pd.DataFrame, top_n: int) -> pd.DataFrame:
    fixed = pd.DataFrame(
        [
            {"family": "feature_classical", "scope": "combined", "model": "linear_svm"},
            {"family": "feature_classical", "scope": "combined", "model": "random_forest"},
            {"family": "feature_classical", "scope": "right", "model": "random_forest"},
            {"family": "feature_classical", "scope": "enhanced", "model": "random_forest"},
            {"family": "feature_classical", "scope": "left", "model": "random_forest"},
        ]
    )
    usable = metrics[metrics["family"].eq("feature_classical")].copy()
    return usable.merge(fixed, on=["family", "scope", "model"], how="inner")


def weight_grid(n_models: int, step: float = 0.25) -> list[np.ndarray]:
    values = np.arange(0.0, 1.0 + 1e-9, step)
    weights = []
    for combo in itertools.product(values, repeat=n_models):
        arr = np.asarray(combo, dtype=float)
        if np.isclose(arr.sum(), 1.0) and np.count_nonzero(arr) >= 1:
            weights.append(arr)
    equal = np.ones(n_models, dtype=float) / n_models
    weights.append(equal)
    return weights


def run_ensemble(feature_set: str = "efficient", trim_seconds: float = 5.0, top_n: int = 8, metric: str = "balanced_accuracy"):
    ensure_dirs()
    manifest = fold_manifest()
    metrics = pd.read_csv(TABLES / "deliverable2_fold_metrics.csv")
    specs = candidate_specs(metrics, top_n)
    rows: list[dict[str, object]] = []
    predictions: list[dict[str, object]] = []
    base_rows: list[dict[str, object]] = []

    for fold in sorted(specs["fold"].unique()):
        fold_specs = specs[specs["fold"] == fold].copy()
        scored = [score_base_model(row, manifest, feature_set, trim_seconds) for _, row in fold_specs.iterrows()]
        labels = [item["label"] for item in scored]
        y_val = scored[0]["y_val"]
        y_test = scored[0]["y_test"]
        val_subjects = scored[0]["val_subjects"]
        test_subjects = scored[0]["test_subjects"]
        for item in scored:
            assert item["val_subjects"] == val_subjects
            assert item["test_subjects"] == test_subjects
            base_threshold, base_val = tune_threshold(y_val, item["val_score"], metric)
            base_test_pred = (item["test_score"] >= base_threshold).astype(int)
            base_test = binary_metrics(y_test, base_test_pred, item["test_score"])
            base_rows.append(
                {
                    "fold": fold,
                    "label": item["label"],
                    "scope": item["scope"],
                    "model": item["model"],
                    "threshold": base_threshold,
                    "n_selected": item["n_selected"],
                    **{f"validation_{key}": value for key, value in base_val.items()},
                    **{f"test_{key}": value for key, value in base_test.items()},
                }
            )

        val_scores = np.column_stack([item["val_score"] for item in scored])
        test_scores = np.column_stack([item["test_score"] for item in scored])
        best = None
        best_key = None
        for weights in weight_grid(len(scored), step=0.25):
            val_score = val_scores @ weights
            threshold, val_metric = tune_threshold(y_val, val_score, metric)
            key = selection_key(val_metric, metric)
            if best_key is None or key > best_key:
                best_key = key
                best = (weights, threshold, val_metric)
        assert best is not None
        weights, threshold, val_metric = best
        test_score = test_scores @ weights
        test_pred = (test_score >= threshold).astype(int)
        test_metric = binary_metrics(y_test, test_pred, test_score)
        nonzero = [
            {"label": label, "weight": float(weight)}
            for label, weight in zip(labels, weights)
            if weight > 1e-9
        ]
        row = {
            "family": "validation_ensemble",
            "scope": "multi_scope",
            "fold": fold,
            "model": f"top{top_n}_weighted_average",
            "n_train": 60,
            "n_validation": 20,
            "n_test": 20,
            "threshold": threshold,
            "weights": repr(nonzero),
        }
        row.update({f"validation_{key}": value for key, value in val_metric.items()})
        row.update({f"test_{key}": value for key, value in test_metric.items()})
        rows.append(row)
        for subject_id, true, pred, score in zip(test_subjects, y_test, test_pred, test_score):
            predictions.append(
                {
                    "family": "validation_ensemble",
                    "scope": "multi_scope",
                    "fold": fold,
                    "model": f"top{top_n}_weighted_average",
                    "subject_id": subject_id,
                    "y_true": int(true),
                    "y_pred": int(pred),
                    "score": float(score),
                    "true_group": LABEL_TO_GROUP[int(true)],
                    "pred_group": LABEL_TO_GROUP[int(pred)],
                }
            )

    ensemble = pd.DataFrame(rows)
    pred_df = pd.DataFrame(predictions)
    base_df = pd.DataFrame(base_rows)
    ensemble.to_csv(TABLES / "deliverable2_validation_ensemble_fold_metrics.csv", index=False)
    pred_df.to_csv(TABLES / "deliverable2_validation_ensemble_predictions.csv", index=False)
    base_df.to_csv(TABLES / "deliverable2_validation_ensemble_base_metrics.csv", index=False)
    summary = (
        ensemble.groupby(["family", "scope", "model"], as_index=False)
        .agg(
            validation_accuracy_mean=("validation_accuracy", "mean"),
            validation_auc_mean=("validation_auc", "mean"),
            test_accuracy_mean=("test_accuracy", "mean"),
            test_sensitivity_mean=("test_sensitivity", "mean"),
            test_specificity_mean=("test_specificity", "mean"),
            test_precision_mean=("test_precision", "mean"),
            test_f1_mean=("test_f1", "mean"),
            test_auc_mean=("test_auc", "mean"),
        )
        .sort_values(["test_accuracy_mean", "test_auc_mean"], ascending=False)
    )
    summary.to_csv(TABLES / "deliverable2_validation_ensemble_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=summary, x="model", y="test_accuracy_mean", ax=ax, color="#2F5F73")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Mean test accuracy")
    ax.set_xlabel("")
    ax.set_title("Validation-tuned model ensemble")
    for patch, value in zip(ax.patches, summary["test_accuracy_mean"]):
        ax.text(patch.get_x() + patch.get_width() / 2, patch.get_height() + 0.02, f"{value*100:.1f}%", ha="center")
    fig.tight_layout()
    fig.savefig(FIGURES / "deliverable2_validation_ensemble_summary.png", dpi=220)
    plt.close(fig)
    return summary, ensemble, base_df


def main() -> None:
    summary, ensemble, base = run_ensemble()
    print("\nValidation-tuned ensemble summary:")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print("\nFold weights:")
    print(ensemble[["fold", "test_accuracy", "test_sensitivity", "test_specificity", "test_auc", "threshold", "weights"]].to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print("\nBest base rows used:")
    print(
        base.sort_values(["fold", "test_accuracy", "test_auc"], ascending=[True, False, False])
        .groupby("fold")
        .head(3)[["fold", "label", "test_accuracy", "test_auc"]]
        .to_string(index=False, float_format=lambda value: f"{value:.3f}")
    )


if __name__ == "__main__":
    main()
