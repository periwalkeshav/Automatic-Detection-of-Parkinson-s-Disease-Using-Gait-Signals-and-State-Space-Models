#!/usr/bin/env python3
"""Fold-safe Deliverable 2 evaluation.

This runner avoids the earlier global feature-matrix dependency. For every
Fold_i, it extracts feature vectors separately from that fold's training,
validation, and test directories, then fits preprocessing and model selection
inside the fold.
"""

from __future__ import annotations

import argparse
import ast
import math
import os
import random
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tsfresh.feature_selection import select_features

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_gait import (
    DEFAULT_TRIM_SECONDS,
    SubjectFile,
    add_asymmetry_features,
    cross_foot_features,
    fc_parameters_for,
    group_from_subject_id,
    load_signal,
    subject_id_from_path,
    subject_tsfresh_features,
    wavelet_features_for_signal,
)


SPLITS = ROOT / "splits"
RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"
FOLD_FEATURES = RESULTS / "fold_features"
SUBJECT_FEATURE_CACHE = RESULTS / "subject_feature_cache"
RANDOM_STATE = 42
GROUP_TO_LABEL = {"Control": 0, "Patient": 1}
LABEL_TO_GROUP = {0: "Control", 1: "Patient"}


def ensure_dirs() -> None:
    for path in (TABLES, FIGURES, FOLD_FEATURES, SUBJECT_FEATURE_CACHE, ROOT / ".matplotlib"):
        path.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = RANDOM_STATE) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def fold_manifest() -> pd.DataFrame:
    rows = []
    for fold_dir in sorted(SPLITS.glob("Fold_*")):
        fold = fold_dir.name
        for split in ("training", "validation", "test"):
            for path in sorted((fold_dir / split).glob("*.txt")):
                subject_id = subject_id_from_path(path)
                rows.append(
                    {
                        "fold": fold,
                        "split": split,
                        "subject_id": subject_id,
                        "group": group_from_subject_id(subject_id),
                        "file": str(path.relative_to(ROOT)),
                    }
                )
    return pd.DataFrame(rows)


def subjects_for_fold_split(manifest: pd.DataFrame, fold: str, split: str) -> list[SubjectFile]:
    rows = manifest[(manifest["fold"] == fold) & (manifest["split"] == split)].copy()
    rows = rows.sort_values("subject_id")
    return [
        SubjectFile(subject_id=row.subject_id, group=row.group, path=ROOT / row.file)
        for row in rows.itertuples(index=False)
    ]


def add_engineered_features_local(matrix: pd.DataFrame, subjects: list[SubjectFile], trim_seconds: float) -> pd.DataFrame:
    matrix = add_asymmetry_features(matrix.copy())
    gait_rows = []
    wavelet_rows = []
    for subject in subjects:
        signal = load_signal(subject.path, trim_seconds=trim_seconds)
        gait_row = {"subject_id": subject.subject_id}
        gait_row.update(cross_foot_features(signal))
        gait_rows.append(gait_row)

        left_force = signal[:, 17]
        right_force = signal[:, 18]
        symmetry = (left_force - right_force) / (left_force + right_force + 1e-9)
        wavelet_row = {"subject_id": subject.subject_id}
        wavelet_row.update(wavelet_features_for_signal(left_force, "left"))
        wavelet_row.update(wavelet_features_for_signal(right_force, "right"))
        wavelet_row.update(wavelet_features_for_signal(symmetry, "sym"))
        wavelet_rows.append(wavelet_row)

    return (
        matrix.merge(pd.DataFrame(gait_rows), on="subject_id", how="left")
        .merge(pd.DataFrame(wavelet_rows), on="subject_id", how="left")
    )


def trim_tag(trim_seconds: float) -> str:
    return f"trim_{trim_seconds:g}s".replace(".", "p")


def subject_feature_cache_path(subject: SubjectFile, feature_set: str, trim_seconds: float) -> Path:
    return SUBJECT_FEATURE_CACHE / feature_set / trim_tag(trim_seconds) / f"{subject.subject_id}.csv"


def extract_subject_features_cached(
    subject: SubjectFile,
    feature_set: str,
    fc_parameters: dict,
    trim_seconds: float,
    force: bool,
) -> pd.Series:
    cache_path = subject_feature_cache_path(subject, feature_set, trim_seconds)
    if cache_path.exists() and not force:
        cached = pd.read_csv(cache_path)
        if cached["subject_id"].iloc[0] == subject.subject_id:
            return cached.iloc[0]

    features = subject_tsfresh_features(subject, fc_parameters, trim_seconds=trim_seconds)
    row = {"subject_id": subject.subject_id, "group": subject.group}
    row.update(features.to_dict())
    matrix = add_engineered_features_local(pd.DataFrame([row]), [subject], trim_seconds)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(cache_path, index=False)
    return matrix.iloc[0]


def extract_split_features(
    subjects: list[SubjectFile],
    feature_set: str,
    trim_seconds: float,
    output: Path,
    force: bool,
) -> pd.DataFrame:
    expected_ids = [subject.subject_id for subject in subjects]
    if output.exists() and not force:
        cached = pd.read_csv(output)
        if cached["subject_id"].tolist() == expected_ids:
            return cached

    rows = []
    fc_parameters = fc_parameters_for(feature_set)
    for idx, subject in enumerate(subjects, start=1):
        print(f"[features-cache] {feature_set}/{trim_tag(trim_seconds)} {idx:02d}/{len(subjects):02d} {subject.subject_id}", flush=True)
        rows.append(extract_subject_features_cached(subject, feature_set, fc_parameters, trim_seconds, force).to_dict())

    matrix = pd.DataFrame(rows).sort_values("subject_id").reset_index(drop=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(output, index=False)
    return matrix


def fold_split_matrix(
    manifest: pd.DataFrame,
    fold: str,
    split: str,
    feature_set: str,
    trim_seconds: float,
    force_features: bool,
) -> pd.DataFrame:
    subjects = subjects_for_fold_split(manifest, fold, split)
    output = FOLD_FEATURES / fold / f"{split}_feature_matrix_{feature_set}.csv"
    return extract_split_features(subjects, feature_set, trim_seconds, output, force_features)


def write_feature_audit(manifest: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    rows = []
    for fold in sorted(manifest["fold"].unique()):
        for split in ("training", "validation", "test"):
            path = FOLD_FEATURES / fold / f"{split}_feature_matrix_{feature_set}.csv"
            matrix = pd.read_csv(path)
            rows.append(
                {
                    "fold": fold,
                    "split": split,
                    "rows": len(matrix),
                    "feature_columns": len([c for c in matrix.columns if c not in {"subject_id", "group"}]),
                    "path": str(path.relative_to(ROOT)),
                    "feature_set": feature_set,
                    "extraction_protocol": "fixed TSFresh calculators cached per subject; fold-local split matrices; learned preprocessing and selection fit on training split",
                }
            )
    audit = pd.DataFrame(rows)
    audit.to_csv(TABLES / "deliverable2_fold_safe_feature_audit.csv", index=False)
    return audit


def feature_columns(matrix: pd.DataFrame, scope: str) -> list[str]:
    if scope == "left":
        return [column for column in matrix.columns if column.startswith("left__")]
    if scope == "right":
        return [column for column in matrix.columns if column.startswith("right__")]
    if scope == "combined":
        return [column for column in matrix.columns if column.startswith(("left__", "right__"))]
    if scope == "enhanced":
        return [
            column
            for column in matrix.columns
            if column.startswith(("left__", "right__", "asym__", "gait__", "wavelet__"))
        ]
    raise ValueError(f"Unknown scope: {scope}")


def align_columns(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, scope: str) -> tuple[list[str], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    columns = feature_columns(train, scope)
    val = val.reindex(columns=list(val.columns) + [c for c in columns if c not in val.columns])
    test = test.reindex(columns=list(test.columns) + [c for c in columns if c not in test.columns])
    return columns, train, val, test


def labels_for(rows: pd.DataFrame) -> np.ndarray:
    return rows["group"].map(GROUP_TO_LABEL).to_numpy(dtype=np.int64)


def matrix_xy(matrix: pd.DataFrame, columns: list[str]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    x = matrix[columns].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float32)
    y = labels_for(matrix)
    return x, y, matrix["subject_id"].tolist()


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, score: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    auc = float("nan")
    if len(np.unique(y_true)) == 2:
        auc = roc_auc_score(y_true, score)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "sensitivity": tp / (tp + fn) if tp + fn else 0.0,
        "specificity": tn / (tn + fp) if tn + fp else 0.0,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc": auc,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def score_from_model(model: Pipeline, x: np.ndarray) -> np.ndarray:
    classifier = model.named_steps["classifier"]
    if hasattr(classifier, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    if hasattr(classifier, "decision_function"):
        return model.decision_function(x)
    return model.predict(x)


def k_grid(n_features: int) -> list[int | str]:
    return [k for k in [25, 50, 100, 200, 400, "all"] if k == "all" or k < n_features]


def classical_grid(n_features: int) -> dict[str, list[dict[str, object]]]:
    ks = k_grid(n_features)
    return {
        "random_forest": [
            {"n_estimators": 400, "max_depth": depth, "min_samples_leaf": leaf, "max_features": "sqrt", "k_best": k}
            for depth in [None, 8, 16]
            for leaf in [1, 3]
            for k in ks
        ],
        "linear_svm": [{"C": c, "k_best": k} for c in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0] for k in ks],
        "svm_rbf": [
            {"C": c, "gamma": gamma, "k_best": k}
            for c in [0.1, 1.0, 10.0, 100.0]
            for gamma in ["scale", 0.001, 0.01, 0.1]
            for k in ks
        ],
    }


def classical_pipeline(model_name: str, params: dict[str, object]) -> Pipeline:
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
    elif model_name == "linear_svm":
        classifier = SVC(kernel="linear", C=float(params["C"]), class_weight="balanced", random_state=RANDOM_STATE)
    elif model_name == "svm_rbf":
        classifier = SVC(kernel="rbf", C=float(params["C"]), gamma=params["gamma"], class_weight="balanced", random_state=RANDOM_STATE)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("variance", VarianceThreshold()),
            ("scaler", StandardScaler()),
            ("select", SelectKBest(score_func=f_classif, k=params["k_best"])),
            ("classifier", classifier),
        ]
    )


def classifier_only(model_name: str, params: dict[str, object]):
    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=int(params["n_estimators"]),
            max_depth=params["max_depth"],
            min_samples_leaf=int(params["min_samples_leaf"]),
            max_features=params["max_features"],
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    if model_name == "linear_svm":
        return SVC(kernel="linear", C=float(params["C"]), class_weight="balanced", random_state=RANDOM_STATE)
    if model_name == "svm_rbf":
        return SVC(kernel="rbf", C=float(params["C"]), gamma=params["gamma"], class_weight="balanced", random_state=RANDOM_STATE)
    raise ValueError(f"Unknown model: {model_name}")


def fresh_classical_grid() -> dict[str, list[dict[str, object]]]:
    return {
        "random_forest": [
            {"n_estimators": 400, "max_depth": depth, "min_samples_leaf": leaf, "max_features": "sqrt"}
            for depth in [None, 8, 16]
            for leaf in [1, 3]
        ],
        "linear_svm": [{"C": c} for c in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]],
        "svm_rbf": [
            {"C": c, "gamma": gamma}
            for c in [0.1, 1.0, 10.0, 100.0]
            for gamma in ["scale", 0.001, 0.01, 0.1]
        ],
    }


def classifier_score(model, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(x)
    return model.predict(x)


@dataclass
class BaseFeatureView:
    x_train: np.ndarray
    x_val: np.ndarray
    x_test: np.ndarray
    columns: list[str]


def base_feature_view(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
    columns: list[str],
) -> BaseFeatureView:
    imputer = SimpleImputer(strategy="median")
    variance = VarianceThreshold()
    scaler = StandardScaler()
    x_train_imp = imputer.fit_transform(x_train)
    x_val_imp = imputer.transform(x_val)
    x_test_imp = imputer.transform(x_test)
    x_train_var = variance.fit_transform(x_train_imp)
    x_val_var = variance.transform(x_val_imp)
    x_test_var = variance.transform(x_test_imp)
    kept_columns = [column for column, keep in zip(columns, variance.get_support()) if keep]
    x_train_scaled = scaler.fit_transform(x_train_var)
    x_val_scaled = scaler.transform(x_val_var)
    x_test_scaled = scaler.transform(x_test_var)
    return BaseFeatureView(
        x_train=x_train_scaled.astype(np.float32),
        x_val=x_val_scaled.astype(np.float32),
        x_test=x_test_scaled.astype(np.float32),
        columns=kept_columns,
    )


def fresh_feature_views(
    base: BaseFeatureView,
    y_train: np.ndarray,
    fdr_levels: list[float],
) -> dict[float, tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]]:
    views = {}
    train_df = pd.DataFrame(base.x_train, columns=base.columns)
    y_series = pd.Series(y_train)
    for fdr in fdr_levels:
        try:
            selected = select_features(
                train_df,
                y_series,
                ml_task="classification",
                fdr_level=fdr,
                n_jobs=1,
                show_warnings=False,
            )
        except ValueError:
            selected = pd.DataFrame(index=train_df.index)
        selected_columns = selected.columns.tolist()
        if not selected_columns:
            continue
        indices = [base.columns.index(column) for column in selected_columns]
        views[fdr] = (
            base.x_train[:, indices],
            base.x_val[:, indices],
            base.x_test[:, indices],
            selected_columns,
        )
    return views


def select_fresh_classical_model(
    model_name: str,
    candidate_params: list[dict[str, object]],
    fresh_views: dict[float, tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]],
    y_train: np.ndarray,
    y_val: np.ndarray,
    selection_metric: str,
) -> tuple[object, dict[str, object], dict[str, float], tuple[np.ndarray, np.ndarray, list[str]]]:
    best_model = None
    best_params: dict[str, object] | None = None
    best_scores: dict[str, float] | None = None
    best_data = None
    best_key: tuple[float, float, float, float] | None = None
    for fdr, (x_train_f, x_val_f, x_test_f, selected_columns) in fresh_views.items():
        for params in candidate_params:
            model = classifier_only(model_name, params)
            model.fit(x_train_f, y_train)
            val_pred = model.predict(x_val_f)
            val_score = classifier_score(model, x_val_f)
            scores = binary_metrics(y_val, val_pred, val_score)
            key = selection_key(scores, selection_metric)
            if best_key is None or key > best_key:
                best_key = key
                best_model = model
                best_params = params | {
                    "selector": "tsfresh.select_features",
                    "fdr_level": fdr,
                    "n_selected_features": len(selected_columns),
                    "final_fit": "training split only",
                }
                best_scores = scores
                best_data = (x_test_f, x_val_f, selected_columns)
    if best_model is None or best_params is None or best_scores is None or best_data is None:
        raise ValueError("TSFresh select_features did not select any features for this fold/scope.")
    return best_model, best_params, best_scores, best_data


def selection_key(scores: dict[str, float], selection_metric: str) -> tuple[float, float, float, float]:
    auc_key = -1.0 if math.isnan(scores["auc"]) else scores["auc"]
    balanced_accuracy = (scores["sensitivity"] + scores["specificity"]) / 2.0
    primary = {
        "auc": auc_key,
        "accuracy": scores["accuracy"],
        "f1": scores["f1"],
        "balanced_accuracy": balanced_accuracy,
    }[selection_metric]
    return primary, scores["accuracy"], auc_key, scores["f1"]


def select_classical_model(
    model_name: str,
    candidates: list[dict[str, object]],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    selection_metric: str,
) -> tuple[dict[str, object], dict[str, float]]:
    best_params: dict[str, object] | None = None
    best_scores: dict[str, float] | None = None
    best_key: tuple[float, float, float, float] | None = None
    for params in candidates:
        model = classical_pipeline(model_name, params)
        model.fit(x_train, y_train)
        val_pred = model.predict(x_val)
        val_score = score_from_model(model, x_val)
        scores = binary_metrics(y_val, val_pred, val_score)
        key = selection_key(scores, selection_metric)
        if best_key is None or key > best_key:
            best_key = key
            best_params = params
            best_scores = scores
    assert best_params is not None and best_scores is not None
    return best_params, best_scores


class FeatureResNet1D(nn.Module):
    def __init__(self, n_features: int) -> None:
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(1, 32, kernel_size=9, padding=4), nn.BatchNorm1d(32), nn.ReLU())
        self.block1 = nn.Sequential(nn.Conv1d(32, 32, kernel_size=7, padding=3), nn.BatchNorm1d(32), nn.ReLU())
        self.block2 = nn.Sequential(nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2), nn.BatchNorm1d(64), nn.ReLU())
        self.block3 = nn.Sequential(nn.Conv1d(64, 96, kernel_size=5, stride=2, padding=2), nn.BatchNorm1d(96), nn.ReLU())
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(96, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.head(x).squeeze(-1)


class FeatureRecurrent(nn.Module):
    def __init__(self, n_features: int, cell: str) -> None:
        super().__init__()
        recurrent = nn.LSTM if cell == "lstm" else nn.GRU
        self.rnn = recurrent(input_size=1, hidden_size=48, num_layers=1, batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(48), nn.Linear(48, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sequence = x.unsqueeze(-1)
        _, hidden = self.rnn(sequence)
        if isinstance(hidden, tuple):
            hidden = hidden[0]
        return self.head(hidden[-1]).squeeze(-1)


class FeaturePatchTST(nn.Module):
    def __init__(self, n_features: int, patch_len: int = 16, stride: int = 8, d_model: int = 48) -> None:
        super().__init__()
        self.patch_len = min(patch_len, n_features)
        self.stride = min(stride, self.patch_len)
        n_patches = 1 + max(0, (n_features - self.patch_len) // self.stride)
        self.proj = nn.Linear(self.patch_len, d_model)
        self.pos = nn.Parameter(torch.zeros(1, n_patches, d_model))
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, dim_feedforward=96, dropout=0.15, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] < self.patch_len:
            x = nn.functional.pad(x, (0, self.patch_len - x.shape[1]))
        patches = x.unfold(dimension=1, size=self.patch_len, step=self.stride)
        z = self.proj(patches) + self.pos[:, : patches.shape[1], :]
        z = self.encoder(z).mean(dim=1)
        return self.head(z).squeeze(-1)


def make_feature_model(model_name: str, n_features: int) -> nn.Module:
    if model_name == "resnet1d":
        return FeatureResNet1D(n_features)
    if model_name == "lstm":
        return FeatureRecurrent(n_features, "lstm")
    if model_name == "gru":
        return FeatureRecurrent(n_features, "gru")
    if model_name == "patchtst":
        return FeaturePatchTST(n_features)
    raise ValueError(f"Unknown feature neural model: {model_name}")


@dataclass
class TorchResult:
    val_scores: dict[str, float]
    test_scores: dict[str, float]
    test_pred: np.ndarray
    test_score: np.ndarray
    best_epoch: int
    k_features: int | str
    learning_rate: float
    weight_decay: float


def preprocess_feature_vectors(
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    k_features: int | str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    imputer = SimpleImputer(strategy="median")
    variance = VarianceThreshold()
    scaler = StandardScaler()
    x_train = imputer.fit_transform(x_train)
    x_val = imputer.transform(x_val)
    x_test = imputer.transform(x_test)
    x_train = variance.fit_transform(x_train)
    x_val = variance.transform(x_val)
    x_test = variance.transform(x_test)
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)
    x_test = scaler.transform(x_test)
    if k_features != "all":
        k = min(int(k_features), x_train.shape[1])
        selector = SelectKBest(score_func=f_classif, k=k)
        x_train = selector.fit_transform(x_train, y_train)
        x_val = selector.transform(x_val)
        x_test = selector.transform(x_test)
    return x_train.astype(np.float32), x_val.astype(np.float32), x_test.astype(np.float32)


def eval_torch(model: nn.Module, x: np.ndarray, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    scores = []
    with torch.no_grad():
        for start in range(0, len(x), 64):
            xb = torch.from_numpy(x[start : start + 64]).to(device)
            scores.append(torch.sigmoid(model(xb)).cpu().numpy())
    score = np.concatenate(scores)
    pred = (score >= 0.5).astype(int)
    return pred, score


def train_feature_torch_model(
    model_name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    max_epochs: int,
    patience: int,
    batch_size: int,
    device: torch.device,
    learning_rate: float,
    weight_decay: float,
    selection_metric: str,
) -> tuple[dict[str, float], dict[str, float], np.ndarray, np.ndarray, int]:
    model = make_feature_model(model_name, x_train.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    pos_weight = torch.tensor([(len(y_train) - y_train.sum()) / max(y_train.sum(), 1)], dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train.astype(np.float32))),
        batch_size=batch_size,
        shuffle=True,
    )
    best_state = None
    best_key: tuple[float, float] | None = None
    best_epoch = 0
    stale = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
        val_pred, val_score = eval_torch(model, x_val, device)
        val_scores = binary_metrics(y_val, val_pred, val_score)
        key = selection_key(val_scores, selection_metric)
        if best_key is None or key > best_key:
            best_key = key
            best_state = {key_: value.detach().cpu().clone() for key_, value in model.state_dict().items()}
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    assert best_state is not None
    model.load_state_dict(best_state)
    val_pred, val_score = eval_torch(model, x_val, device)
    test_pred, test_score = eval_torch(model, x_test, device)
    return (
        binary_metrics(y_val, val_pred, val_score),
        binary_metrics(y_test, test_pred, test_score),
        test_pred,
        test_score,
        best_epoch,
    )


def select_feature_torch_model(
    model_name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    k_candidates: list[int | str],
    max_epochs: int,
    patience: int,
    batch_size: int,
    device: torch.device,
    selection_metric: str,
    learning_rates: list[float],
    weight_decays: list[float],
) -> TorchResult:
    best: TorchResult | None = None
    best_key: tuple[float, float, float, float] | None = None
    for k_features in k_candidates:
        for learning_rate in learning_rates:
            for weight_decay in weight_decays:
                set_seed(RANDOM_STATE)
                xt, xv, xs = preprocess_feature_vectors(x_train, x_val, x_test, y_train, k_features)
                val_scores, test_scores, test_pred, test_score, best_epoch = train_feature_torch_model(
                    model_name,
                    xt,
                    y_train,
                    xv,
                    y_val,
                    xs,
                    y_test,
                    max_epochs,
                    patience,
                    batch_size,
                    device,
                    learning_rate,
                    weight_decay,
                    selection_metric,
                )
                key = selection_key(val_scores, selection_metric)
                if best_key is None or key > best_key:
                    best_key = key
                    best = TorchResult(
                        val_scores,
                        test_scores,
                        test_pred,
                        test_score,
                        best_epoch,
                        k_features,
                        learning_rate,
                        weight_decay,
                    )
    assert best is not None
    return best


def run_fold_safe(
    scopes: list[str],
    classical_models: list[str],
    neural_models: list[str],
    feature_set: str,
    trim_seconds: float,
    force_features: bool,
    selection_metric: str,
    max_epochs: int,
    patience: int,
    batch_size: int,
    deep_k_features: list[int | str],
    deep_learning_rates: list[float],
    deep_weight_decays: list[float],
    run_tsfresh_selection: bool,
    fresh_fdr_levels: list[float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest = fold_manifest()
    manifest.to_csv(TABLES / "fold_manifest.csv", index=False)
    folds = sorted(manifest["fold"].unique())
    rows: list[dict[str, object]] = []
    predictions: list[dict[str, object]] = []
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[feature-neural] device={device}", flush=True)

    for fold in folds:
        train_matrix = fold_split_matrix(manifest, fold, "training", feature_set, trim_seconds, force_features)
        val_matrix = fold_split_matrix(manifest, fold, "validation", feature_set, trim_seconds, force_features)
        test_matrix = fold_split_matrix(manifest, fold, "test", feature_set, trim_seconds, force_features)
        for scope in scopes:
            columns, train_aligned, val_aligned, test_aligned = align_columns(train_matrix, val_matrix, test_matrix, scope)
            x_train, y_train, _ = matrix_xy(train_aligned, columns)
            x_val, y_val, _ = matrix_xy(val_aligned, columns)
            x_test, y_test, test_subjects = matrix_xy(test_aligned, columns)

            grids = classical_grid(len(columns))
            for model_name in classical_models:
                print(f"[classical] {fold} {scope} {model_name}", flush=True)
                best_params, val_scores = select_classical_model(
                    model_name, grids[model_name], x_train, y_train, x_val, y_val, selection_metric
                )
                model = classical_pipeline(model_name, best_params)
                model.fit(x_train, y_train)
                test_pred = model.predict(x_test)
                test_score = score_from_model(model, x_test)
                test_scores = binary_metrics(y_test, test_pred, test_score)
                row = {
                    "family": "feature_classical",
                    "scope": scope,
                    "fold": fold,
                    "model": model_name,
                    "n_train": len(y_train),
                    "n_validation": len(y_val),
                    "n_test": len(y_test),
                    "n_features": len(columns),
                    "best_params": repr(best_params | {"feature_source": "fold-local split matrices", "final_fit": "training split only"}),
                }
                row.update({f"validation_{key}": value for key, value in val_scores.items()})
                row.update({f"test_{key}": value for key, value in test_scores.items()})
                rows.append(row)
                for subject_id, true, pred, score in zip(test_subjects, y_test, test_pred, test_score):
                    predictions.append(
                        {
                            "family": "feature_classical",
                            "scope": scope,
                            "fold": fold,
                            "model": model_name,
                            "subject_id": subject_id,
                            "y_true": int(true),
                            "y_pred": int(pred),
                            "score": float(score),
                            "true_group": LABEL_TO_GROUP[int(true)],
                            "pred_group": LABEL_TO_GROUP[int(pred)],
                        }
                    )

            if run_tsfresh_selection:
                print(f"[tsfresh-select] {fold} {scope}", flush=True)
                base = base_feature_view(x_train, y_train, x_val, x_test, columns)
                fresh_views = fresh_feature_views(base, y_train, fresh_fdr_levels)
                if not fresh_views:
                    print(f"[tsfresh-select] {fold} {scope} selected no features; skipping", flush=True)
                fresh_grids = fresh_classical_grid()
                for model_name in classical_models:
                    try:
                        fresh_model, fresh_params, val_scores, fresh_data = select_fresh_classical_model(
                            model_name,
                            fresh_grids[model_name],
                            fresh_views,
                            y_train,
                            y_val,
                            selection_metric,
                        )
                    except ValueError:
                        continue
                    x_test_f, _, selected_columns = fresh_data
                    test_pred = fresh_model.predict(x_test_f)
                    test_score = classifier_score(fresh_model, x_test_f)
                    test_scores = binary_metrics(y_test, test_pred, test_score)
                    fresh_model_name = f"{model_name}_fresh"
                    row = {
                        "family": "feature_classical_tsfresh",
                        "scope": scope,
                        "fold": fold,
                        "model": fresh_model_name,
                        "n_train": len(y_train),
                        "n_validation": len(y_val),
                        "n_test": len(y_test),
                        "n_features": len(columns),
                        "best_params": repr(fresh_params | {"feature_source": "fold-local split matrices"}),
                    }
                    row.update({f"validation_{key}": value for key, value in val_scores.items()})
                    row.update({f"test_{key}": value for key, value in test_scores.items()})
                    rows.append(row)
                    for subject_id, true, pred, score in zip(test_subjects, y_test, test_pred, test_score):
                        predictions.append(
                            {
                                "family": "feature_classical_tsfresh",
                                "scope": scope,
                                "fold": fold,
                                "model": fresh_model_name,
                                "subject_id": subject_id,
                                "y_true": int(true),
                                "y_pred": int(pred),
                                "score": float(score),
                                "true_group": LABEL_TO_GROUP[int(true)],
                                "pred_group": LABEL_TO_GROUP[int(pred)],
                            }
                        )

            for model_name in neural_models:
                print(f"[feature-neural] {fold} {scope} {model_name}", flush=True)
                result = select_feature_torch_model(
                    model_name,
                    x_train,
                    y_train,
                    x_val,
                    y_val,
                    x_test,
                    y_test,
                    [k for k in deep_k_features if k == "all" or int(k) < len(columns)],
                    max_epochs,
                    patience,
                    batch_size,
                    device,
                    selection_metric,
                    deep_learning_rates,
                    deep_weight_decays,
                )
                params = {
                    "input": "fold-local feature vector",
                    "k_features": result.k_features,
                    "learning_rate": result.learning_rate,
                    "weight_decay": result.weight_decay,
                    "max_epochs": max_epochs,
                    "patience": patience,
                    "best_epoch": result.best_epoch,
                    "preprocessing": "median impute, variance filter, train-fitted z-score, train-fitted SelectKBest",
                    "selection_metric": selection_metric,
                }
                row = {
                    "family": "feature_neural",
                    "scope": scope,
                    "fold": fold,
                    "model": model_name,
                    "n_train": len(y_train),
                    "n_validation": len(y_val),
                    "n_test": len(y_test),
                    "n_features": len(columns),
                    "best_params": repr(params),
                }
                row.update({f"validation_{key}": value for key, value in result.val_scores.items()})
                row.update({f"test_{key}": value for key, value in result.test_scores.items()})
                rows.append(row)
                for subject_id, true, pred, score in zip(test_subjects, y_test, result.test_pred, result.test_score):
                    predictions.append(
                        {
                            "family": "feature_neural",
                            "scope": scope,
                            "fold": fold,
                            "model": model_name,
                            "subject_id": subject_id,
                            "y_true": int(true),
                            "y_pred": int(pred),
                            "score": float(score),
                            "true_group": LABEL_TO_GROUP[int(true)],
                            "pred_group": LABEL_TO_GROUP[int(pred)],
                        }
                    )

    write_feature_audit(manifest, feature_set)
    return pd.DataFrame(rows), pd.DataFrame(predictions)


def summarize(metrics_df: pd.DataFrame) -> pd.DataFrame:
    return (
        metrics_df.groupby(["family", "scope", "model"], as_index=False)
        .agg(
            validation_accuracy_mean=("validation_accuracy", "mean"),
            validation_auc_mean=("validation_auc", "mean"),
            test_accuracy_mean=("test_accuracy", "mean"),
            test_accuracy_std=("test_accuracy", "std"),
            test_sensitivity_mean=("test_sensitivity", "mean"),
            test_specificity_mean=("test_specificity", "mean"),
            test_precision_mean=("test_precision", "mean"),
            test_f1_mean=("test_f1", "mean"),
            test_auc_mean=("test_auc", "mean"),
            test_auc_std=("test_auc", "std"),
        )
        .sort_values(["scope", "test_accuracy_mean", "test_auc_mean"], ascending=[True, False, False])
    )


def model_label(model: str) -> str:
    return {
        "random_forest": "RF",
        "random_forest_fresh": "RF + FRESH",
        "linear_svm": "SVM-L",
        "linear_svm_fresh": "SVM-L + FRESH",
        "svm_rbf": "SVM-RBF",
        "svm_rbf_fresh": "SVM-RBF + FRESH",
        "resnet1d": "1D-ResNet",
        "lstm": "LSTM",
        "gru": "GRU",
        "patchtst": "PatchTST",
    }.get(model, model)


def plot_metric_summary(summary: pd.DataFrame) -> None:
    order = [
        "random_forest",
        "linear_svm",
        "svm_rbf",
        "random_forest_fresh",
        "linear_svm_fresh",
        "svm_rbf_fresh",
        "resnet1d",
        "lstm",
        "gru",
        "patchtst",
    ]
    plot_df = summary.copy()
    plot_df["model_label"] = pd.Categorical(plot_df["model"].map(model_label), categories=[model_label(m) for m in order], ordered=True)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
    for ax, metric, title in [
        (axes[0], "test_accuracy_mean", "Mean Test Accuracy"),
        (axes[1], "test_auc_mean", "Mean Test AUC"),
    ]:
        sns.barplot(data=plot_df, x="scope", y=metric, hue="model_label", ax=ax)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("Fold-local feature view")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.axhline(0.5, color="#777777", linestyle="--", linewidth=1)
        ax.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGURES / "deliverable2_metric_summary.png", dpi=220)
    plt.close(fig)


def best_models_by_scope(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scope, group in summary.groupby("scope"):
        ranked = group.sort_values(["test_accuracy_mean", "test_auc_mean"], ascending=False)
        rows.append(ranked.iloc[0].to_dict())
    return pd.DataFrame(rows)


def plot_confusion_matrices(best: pd.DataFrame, predictions: pd.DataFrame) -> None:
    n = len(best)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.2))
    if n == 1:
        axes = [axes]
    cm_rows = []
    for ax, row in zip(axes, best.itertuples(index=False)):
        subset = predictions[(predictions["scope"] == row.scope) & (predictions["model"] == row.model)]
        cm = confusion_matrix(subset["y_true"], subset["y_pred"], labels=[0, 1])
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=ax,
            xticklabels=["Control", "Patient"],
            yticklabels=["Control", "Patient"],
            annot_kws={"size": 18, "weight": "bold"},
        )
        ax.set_title(f"{row.scope.capitalize()}: {model_label(row.model)}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        tn, fp, fn, tp = cm.ravel()
        cm_rows.append({"scope": row.scope, "model": row.model, "tn": tn, "fp": fp, "fn": fn, "tp": tp})
    fig.tight_layout()
    fig.savefig(FIGURES / "deliverable2_best_confusion_matrices.png", dpi=240)
    plt.close(fig)
    pd.DataFrame(cm_rows).to_csv(TABLES / "deliverable2_best_confusion_matrices.csv", index=False)


def write_outputs(metrics: pd.DataFrame, predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics.to_csv(TABLES / "deliverable2_fold_metrics.csv", index=False)
    predictions.to_csv(TABLES / "deliverable2_predictions.csv", index=False)
    metrics[metrics["family"] == "feature_classical"].to_csv(TABLES / "deliverable2_classical_fold_metrics.csv", index=False)
    predictions[predictions["family"] == "feature_classical"].to_csv(TABLES / "deliverable2_classical_predictions.csv", index=False)
    metrics[metrics["family"] == "feature_classical_tsfresh"].to_csv(TABLES / "deliverable2_tsfresh_selection_fold_metrics.csv", index=False)
    predictions[predictions["family"] == "feature_classical_tsfresh"].to_csv(TABLES / "deliverable2_tsfresh_selection_predictions.csv", index=False)
    metrics[metrics["family"] == "feature_neural"].to_csv(TABLES / "deliverable2_feature_neural_fold_metrics.csv", index=False)
    predictions[predictions["family"] == "feature_neural"].to_csv(TABLES / "deliverable2_feature_neural_predictions.csv", index=False)
    summary = summarize(metrics)
    summary.to_csv(TABLES / "deliverable2_model_summary.csv", index=False)
    selector_summary = summary[summary["family"].isin(["feature_classical", "feature_classical_tsfresh"])].copy()
    if not selector_summary.empty:
        selector_summary["selector"] = np.where(
            selector_summary["family"].eq("feature_classical_tsfresh"),
            "tsfresh.select_features",
            "sklearn.SelectKBest",
        )
        selector_summary.to_csv(TABLES / "deliverable2_feature_selection_comparison.csv", index=False)
    best = best_models_by_scope(summary)
    best.to_csv(TABLES / "deliverable2_best_models_by_scope.csv", index=False)
    plot_metric_summary(summary)
    plot_confusion_matrices(best, predictions)
    return summary, best


def parse_k_features(values: list[str]) -> list[int | str]:
    parsed: list[int | str] = []
    for value in values:
        parsed.append("all" if value == "all" else int(value))
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scopes", nargs="+", default=["left", "right", "combined", "enhanced"], choices=["left", "right", "combined", "enhanced"])
    parser.add_argument("--feature-set", default="efficient", choices=["minimal", "curated", "efficient"])
    parser.add_argument("--trim-seconds", type=float, default=DEFAULT_TRIM_SECONDS)
    parser.add_argument("--force-features", action="store_true")
    parser.add_argument(
        "--classical-models",
        nargs="+",
        default=["random_forest", "linear_svm", "svm_rbf"],
        choices=["random_forest", "linear_svm", "svm_rbf"],
    )
    parser.add_argument(
        "--feature-neural-models",
        nargs="+",
        default=["resnet1d", "lstm", "gru", "patchtst"],
        choices=["resnet1d", "lstm", "gru", "patchtst"],
    )
    parser.add_argument("--skip-feature-neural", action="store_true")
    parser.add_argument("--only-feature-neural", action="store_true")
    parser.add_argument("--skip-tsfresh-selection", action="store_true")
    parser.add_argument("--fresh-fdr-levels", nargs="+", type=float, default=[0.05, 0.1, 0.2, 0.5])
    parser.add_argument("--selection-metric", default="f1", choices=["accuracy", "f1", "balanced_accuracy", "auc"])
    parser.add_argument("--max-epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--deep-k-features", nargs="+", default=["200"])
    parser.add_argument("--deep-learning-rates", nargs="+", type=float, default=[0.0003, 0.0008])
    parser.add_argument("--deep-weight-decays", nargs="+", type=float, default=[0.0001, 0.001])
    return parser.parse_args()


def merge_with_existing_non_neural(metrics: pd.DataFrame, predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics_path = TABLES / "deliverable2_fold_metrics.csv"
    predictions_path = TABLES / "deliverable2_predictions.csv"
    if metrics_path.exists():
        old_metrics = pd.read_csv(metrics_path)
        old_metrics = old_metrics[old_metrics["family"] != "feature_neural"]
        metrics = pd.concat([old_metrics, metrics], ignore_index=True)
    if predictions_path.exists():
        old_predictions = pd.read_csv(predictions_path)
        old_predictions = old_predictions[old_predictions["family"] != "feature_neural"]
        predictions = pd.concat([old_predictions, predictions], ignore_index=True)
    return metrics, predictions


def main() -> None:
    args = parse_args()
    ensure_dirs()
    set_seed(RANDOM_STATE)
    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.feature_selection")
    warnings.filterwarnings("ignore", category=RuntimeWarning, module="tsfresh.utilities.dataframe_functions")
    if args.only_feature_neural and args.skip_feature_neural:
        raise ValueError("--only-feature-neural cannot be combined with --skip-feature-neural")
    neural_models = [] if args.skip_feature_neural else args.feature_neural_models
    classical_models = [] if args.only_feature_neural else args.classical_models
    metrics, predictions = run_fold_safe(
        scopes=args.scopes,
        classical_models=classical_models,
        neural_models=neural_models,
        feature_set=args.feature_set,
        trim_seconds=args.trim_seconds,
        force_features=args.force_features,
        selection_metric=args.selection_metric,
        max_epochs=args.max_epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        deep_k_features=parse_k_features(args.deep_k_features),
        deep_learning_rates=args.deep_learning_rates,
        deep_weight_decays=args.deep_weight_decays,
        run_tsfresh_selection=(not args.skip_tsfresh_selection and not args.only_feature_neural),
        fresh_fdr_levels=args.fresh_fdr_levels,
    )
    if args.only_feature_neural:
        metrics, predictions = merge_with_existing_non_neural(metrics, predictions)
    summary, best = write_outputs(metrics, predictions)
    print("\nFold-safe Deliverable 2 summary:")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print("\nBest models by scope:")
    print(best[["scope", "model", "test_accuracy_mean", "test_sensitivity_mean", "test_specificity_mean", "test_auc_mean"]].to_string(index=False, float_format=lambda value: f"{value:.3f}"))


if __name__ == "__main__":
    main()
