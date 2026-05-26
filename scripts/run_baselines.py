#!/usr/bin/env python3
"""Run non-Mamba 5-fold baselines for PD-vs-control gait classification."""

from __future__ import annotations

import argparse
import itertools
import os
import warnings
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".matplotlib"))

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.feature_selection import VarianceThreshold


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"
RANDOM_STATE = 42


def feature_columns(matrix: pd.DataFrame, scope: str) -> list[str]:
    if scope == "left":
        return [column for column in matrix.columns if column.startswith("left__")]
    if scope == "right":
        return [column for column in matrix.columns if column.startswith("right__")]
    if scope == "combined":
        return [column for column in matrix.columns if column.startswith(("left__", "right__"))]
    if scope == "asymmetry":
        return [column for column in matrix.columns if column.startswith("asym__")]
    if scope == "gait":
        return [column for column in matrix.columns if column.startswith("gait__")]
    if scope == "wavelet":
        return [column for column in matrix.columns if column.startswith("wavelet__")]
    if scope == "enhanced":
        return [
            column
            for column in matrix.columns
            if column.startswith(("left__", "right__", "asym__", "gait__", "wavelet__"))
        ]
    raise ValueError(f"Unknown feature scope: {scope}")


def labels(values: pd.Series) -> np.ndarray:
    return values.map({"Control": 0, "Patient": 1}).to_numpy(dtype=int)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def pipeline_for(model_name: str, params: dict[str, object]) -> Pipeline:
    k_best = params.get("k_best", "all")
    if model_name == "random_forest":
        classifier = RandomForestClassifier(
            n_estimators=int(params["n_estimators"]),
            max_depth=params["max_depth"],
            min_samples_leaf=int(params["min_samples_leaf"]),
            max_features=params["max_features"],
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("variance", VarianceThreshold()),
                ("select", SelectKBest(score_func=f_classif, k=k_best)),
                ("classifier", classifier),
            ]
        )
    if model_name == "extra_trees":
        classifier = ExtraTreesClassifier(
            n_estimators=int(params["n_estimators"]),
            max_depth=params["max_depth"],
            min_samples_leaf=int(params["min_samples_leaf"]),
            max_features=params["max_features"],
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("variance", VarianceThreshold()),
                ("select", SelectKBest(score_func=f_classif, k=k_best)),
                ("classifier", classifier),
            ]
        )
    if model_name == "linear_svm":
        classifier = SVC(
            kernel="linear",
            C=float(params["C"]),
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("variance", VarianceThreshold()),
                ("scaler", StandardScaler()),
                ("select", SelectKBest(score_func=f_classif, k=k_best)),
                ("classifier", classifier),
            ]
        )
    if model_name == "l1_logistic":
        classifier = LogisticRegression(
            C=float(params["C"]),
            l1_ratio=1.0,
            solver="liblinear",
            class_weight="balanced",
            random_state=RANDOM_STATE,
            max_iter=5000,
        )
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("variance", VarianceThreshold()),
                ("scaler", StandardScaler()),
                ("select", SelectKBest(score_func=f_classif, k=k_best)),
                ("classifier", classifier),
            ]
        )
    if model_name == "rbf_svm":
        classifier = SVC(
            kernel="rbf",
            C=float(params["C"]),
            gamma=params["gamma"],
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("variance", VarianceThreshold()),
                ("scaler", StandardScaler()),
                ("select", SelectKBest(score_func=f_classif, k=k_best)),
                ("classifier", classifier),
            ]
        )
    raise ValueError(f"Unknown model: {model_name}")


def k_grid(n_features: int) -> list[int | str]:
    candidates: list[int | str] = [25, 50, 100, 200, 400, "all"]
    return [k for k in candidates if k == "all" or k < n_features]


def l1_logistic_k_grid(n_features: int) -> list[int | str]:
    candidates: list[int | str] = [15, 25, 50, 100, 300, 600, 800, "all"]
    return [k for k in candidates if k == "all" or k < n_features]


def grid(n_features: int) -> dict[str, list[dict[str, object]]]:
    ks = k_grid(n_features)
    logistic_ks = l1_logistic_k_grid(n_features)
    random_forest = [
        dict(zip(("n_estimators", "max_depth", "min_samples_leaf", "max_features", "k_best"), values))
        for values in itertools.product([400], [None, 8, 16], [1, 3], ["sqrt"], ks)
    ]
    extra_trees = [
        dict(zip(("n_estimators", "max_depth", "min_samples_leaf", "max_features", "k_best"), values))
        for values in itertools.product([600], [None, 8, 16], [1, 3], ["sqrt", "log2"], ks)
    ]
    linear_svm = [
        {"C": c, "k_best": k}
        for c, k in itertools.product([0.001, 0.01, 0.1, 1.0, 10.0, 100.0], ks)
    ]
    l1_logistic = [
        {"C": c, "k_best": k}
        for c, k in itertools.product([0.1, 0.3, 1.0, 3.0, 10.0], logistic_ks)
    ]
    rbf_svm = [
        {"C": c, "gamma": gamma, "k_best": k}
        for c, gamma, k in itertools.product([0.1, 1.0, 10.0, 100.0], ["scale", 0.001, 0.01, 0.1], ks)
    ]
    return {
        "random_forest": random_forest,
        "extra_trees": extra_trees,
        "linear_svm": linear_svm,
        "l1_logistic": l1_logistic,
        "rbf_svm": rbf_svm,
    }


def split_subjects(manifest: pd.DataFrame, fold: str, split: str) -> set[str]:
    rows = manifest[(manifest["fold"] == fold) & (manifest["split"] == split)]
    return set(rows["subject_id"])


def matrix_for_subjects(matrix: pd.DataFrame, subject_ids: set[str], columns: list[str]) -> tuple[np.ndarray, np.ndarray]:
    rows = matrix[matrix["subject_id"].isin(subject_ids)].copy().sort_values("subject_id")
    x = rows[columns].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
    y = labels(rows["group"])
    return x, y


def select_model(
    model_name: str,
    candidates: list[dict[str, object]],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
) -> tuple[dict[str, object], dict[str, float]]:
    best_params: dict[str, object] | None = None
    best_scores: dict[str, float] | None = None
    best_key: tuple[float, float, float] | None = None
    for params in candidates:
        model = pipeline_for(model_name, params)
        model.fit(x_train, y_train)
        val_pred = model.predict(x_val)
        val_scores = metrics(y_val, val_pred)
        key = (val_scores["f1"], val_scores["accuracy"], val_scores["recall"])
        if best_key is None or key > best_key:
            best_key = key
            best_params = params
            best_scores = val_scores
    assert best_params is not None and best_scores is not None
    return best_params, best_scores


def run_baselines(scopes: list[str], models: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    matrix = pd.read_csv(TABLES / "subject_feature_matrix.csv")
    manifest = pd.read_csv(TABLES / "fold_manifest.csv")
    folds = sorted(manifest["fold"].unique())
    rows = []

    for scope in scopes:
        columns = feature_columns(matrix, scope)
        grids = grid(len(columns))
        for fold in folds:
            train_ids = split_subjects(manifest, fold, "training")
            val_ids = split_subjects(manifest, fold, "validation")
            test_ids = split_subjects(manifest, fold, "test")
            x_train, y_train = matrix_for_subjects(matrix, train_ids, columns)
            x_val, y_val = matrix_for_subjects(matrix, val_ids, columns)
            x_test, y_test = matrix_for_subjects(matrix, test_ids, columns)
            x_train_val = np.vstack([x_train, x_val])
            y_train_val = np.concatenate([y_train, y_val])

            for model_name, candidates in grids.items():
                if model_name not in models:
                    continue
                print(f"[baseline] {scope} {fold} {model_name}")
                best_params, val_scores = select_model(model_name, candidates, x_train, y_train, x_val, y_val)
                final_model = pipeline_for(model_name, best_params)
                final_model.fit(x_train_val, y_train_val)
                test_scores = metrics(y_test, final_model.predict(x_test))
                row = {
                    "scope": scope,
                    "fold": fold,
                    "model": model_name,
                    "n_train": len(y_train),
                    "n_validation": len(y_val),
                    "n_test": len(y_test),
                    "best_params": repr(best_params),
                }
                row.update({f"validation_{key}": value for key, value in val_scores.items()})
                row.update({f"test_{key}": value for key, value in test_scores.items()})
                rows.append(row)

    fold_metrics = pd.DataFrame(rows)
    fold_metrics.to_csv(TABLES / "baseline_fold_metrics.csv", index=False)
    summary = (
        fold_metrics.groupby(["scope", "model"], as_index=False)
        .agg(
            validation_accuracy_mean=("validation_accuracy", "mean"),
            validation_accuracy_std=("validation_accuracy", "std"),
            validation_f1_mean=("validation_f1", "mean"),
            validation_f1_std=("validation_f1", "std"),
            test_accuracy_mean=("test_accuracy", "mean"),
            test_accuracy_std=("test_accuracy", "std"),
            test_f1_mean=("test_f1", "mean"),
            test_f1_std=("test_f1", "std"),
            test_precision_mean=("test_precision", "mean"),
            test_recall_mean=("test_recall", "mean"),
        )
        .sort_values(["scope", "test_f1_mean"], ascending=[True, False])
    )
    summary.to_csv(TABLES / "baseline_summary.csv", index=False)
    return fold_metrics, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scopes",
        nargs="+",
        default=["left", "right", "combined", "enhanced"],
        choices=["left", "right", "combined", "asymmetry", "gait", "wavelet", "enhanced"],
        help="Feature scopes to evaluate.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["random_forest", "extra_trees", "linear_svm", "l1_logistic", "rbf_svm"],
        choices=["random_forest", "extra_trees", "linear_svm", "l1_logistic", "rbf_svm"],
        help="Baseline model families to evaluate.",
    )
    args = parser.parse_args()
    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.feature_selection")
    warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn.feature_selection")

    required = [TABLES / "subject_feature_matrix.csv", TABLES / "fold_manifest.csv"]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise SystemExit(f"Missing {missing}. Run scripts/analyze_gait.py first.")

    _, summary = run_baselines(args.scopes, args.models)
    print("Wrote baseline metrics to:")
    print(f"  {TABLES / 'baseline_fold_metrics.csv'}")
    print(f"  {TABLES / 'baseline_summary.csv'}")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.3f}"))


if __name__ == "__main__":
    main()
