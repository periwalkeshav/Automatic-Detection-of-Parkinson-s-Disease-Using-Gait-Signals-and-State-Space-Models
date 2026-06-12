#!/usr/bin/env python3
"""Fold-safe raw-window GRU/LSTM evaluation.

This complements the feature-vector neural models. It trains on raw gait windows
from training subjects only, chooses architecture/window/threshold on validation
subjects, and evaluates once on held-out test subjects.
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
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_gait import (  # noqa: E402
    DEFAULT_TRIM_SECONDS,
    SAMPLE_RATE,
    SubjectFile,
    group_from_subject_id,
    load_signal,
    subject_id_from_path,
)


SPLITS = ROOT / "splits"
RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"
RANDOM_STATE = 42
GROUP_TO_LABEL = {"Control": 0, "Patient": 1}
LABEL_TO_GROUP = {0: "Control", 1: "Patient"}


def ensure_dirs() -> None:
    for path in (TABLES, FIGURES, ROOT / ".matplotlib"):
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


def channel_indices(channel_set: str) -> list[int]:
    if channel_set == "totals":
        return [17, 18]
    if channel_set == "sensors":
        return list(range(1, 17))
    if channel_set == "all":
        return list(range(1, 19))
    raise ValueError(f"Unknown channel set: {channel_set}")


def windows_for_subject(
    subject: SubjectFile,
    channel_set: str,
    trim_seconds: float,
    window_seconds: float,
    step_seconds: float,
    downsample: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    signal = load_signal(subject.path, trim_seconds=trim_seconds)
    values = signal[:, channel_indices(channel_set)].astype(np.float32)
    window_samples = int(window_seconds * SAMPLE_RATE)
    step_samples = int(step_seconds * SAMPLE_RATE)
    if len(values) < window_samples:
        pad = window_samples - len(values)
        values = np.pad(values, ((0, pad), (0, 0)), mode="edge")

    windows = []
    for start in range(0, len(values) - window_samples + 1, step_samples):
        windows.append(values[start : start + window_samples : downsample])
    if not windows:
        windows.append(values[:window_samples:downsample])
    x = np.stack(windows).astype(np.float32)
    y = np.full(len(x), GROUP_TO_LABEL[subject.group], dtype=np.int64)
    subject_ids = [subject.subject_id] * len(x)
    return x, y, subject_ids


def make_window_matrix(
    subjects: list[SubjectFile],
    channel_set: str,
    trim_seconds: float,
    window_seconds: float,
    step_seconds: float,
    downsample: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    xs = []
    ys = []
    subject_ids: list[str] = []
    for subject in subjects:
        x, y, ids = windows_for_subject(subject, channel_set, trim_seconds, window_seconds, step_seconds, downsample)
        xs.append(x)
        ys.append(y)
        subject_ids.extend(ids)
    return np.concatenate(xs), np.concatenate(ys), subject_ids


def fit_channel_normalizer(x_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=(0, 1), keepdims=True)
    std = x_train.std(axis=(0, 1), keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def normalize(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((x - mean) / std).astype(np.float32)


class WindowDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, augment: bool = False) -> None:
        self.x = x
        self.y = y.astype(np.float32)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.x[idx].copy()
        if self.augment:
            if random.random() < 0.5:
                scale = np.random.normal(1.0, 0.03, size=(1, x.shape[1])).astype(np.float32)
                x *= scale
            if random.random() < 0.5:
                x += np.random.normal(0.0, 0.015, size=x.shape).astype(np.float32)
        return torch.from_numpy(x), torch.tensor(self.y[idx], dtype=torch.float32)


class StemAttentionRNN(nn.Module):
    def __init__(self, n_channels: int, cell: str, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=9, stride=2, padding=4),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(32, 48, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(48),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        recurrent = nn.GRU if cell == "gru" else nn.LSTM
        self.rnn = recurrent(
            input_size=48,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        out_size = hidden_size * 2
        self.attention = nn.Sequential(nn.Linear(out_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, 1))
        self.head = nn.Sequential(nn.LayerNorm(out_size), nn.Dropout(dropout), nn.Linear(out_size, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.stem(x.transpose(1, 2)).transpose(1, 2)
        z, _ = self.rnn(z)
        weights = torch.softmax(self.attention(z).squeeze(-1), dim=1)
        pooled = torch.sum(z * weights.unsqueeze(-1), dim=1)
        return self.head(pooled).squeeze(-1)


def aggregate_subject_scores(
    subject_ids: list[str],
    y_window: np.ndarray,
    score_window: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    rows = pd.DataFrame({"subject_id": subject_ids, "y": y_window.astype(int), "score": score_window.astype(float)})
    grouped = rows.groupby("subject_id", sort=True).agg(y=("y", "first"), score=("score", "mean")).reset_index()
    y_true = grouped["y"].to_numpy(dtype=int)
    score = grouped["score"].to_numpy(dtype=float)
    pred = (score >= threshold).astype(int)
    return y_true, pred, score, grouped["subject_id"].tolist()


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
        "f1": scores["f1"],
        "accuracy": scores["accuracy"],
        "balanced_accuracy": balanced,
        "auc": auc_key,
    }[metric]
    return primary, scores["accuracy"], auc_key, scores["f1"]


def tune_threshold(
    subject_ids: list[str],
    y_window: np.ndarray,
    score_window: np.ndarray,
    metric: str,
) -> tuple[float, dict[str, float]]:
    best_threshold = 0.5
    best_scores: dict[str, float] | None = None
    best_key: tuple[float, float, float, float] | None = None
    for threshold in np.linspace(0.2, 0.8, 31):
        y_true, pred, score, _ = aggregate_subject_scores(subject_ids, y_window, score_window, float(threshold))
        scores = binary_metrics(y_true, pred, score)
        key = selection_key(scores, metric)
        if best_key is None or key > best_key:
            best_key = key
            best_threshold = float(threshold)
            best_scores = scores
    assert best_scores is not None
    return best_threshold, best_scores


def predict_window_scores(model: nn.Module, x: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    loader = DataLoader(WindowDataset(x, np.zeros(len(x))), batch_size=batch_size, shuffle=False)
    model.eval()
    scores = []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            scores.append(torch.sigmoid(model(xb)).cpu().numpy())
    return np.concatenate(scores)


@dataclass(frozen=True)
class Candidate:
    model: str
    channel_set: str
    window_seconds: float
    step_seconds: float
    downsample: int
    hidden_size: int
    dropout: float
    learning_rate: float
    weight_decay: float


@dataclass
class CandidateResult:
    candidate: Candidate
    val_scores: dict[str, float]
    test_scores: dict[str, float]
    test_subject_ids: list[str]
    test_true: np.ndarray
    test_pred: np.ndarray
    test_score: np.ndarray
    threshold: float
    best_epoch: int


def train_candidate(
    candidate: Candidate,
    fold_data: dict[tuple[str, float], tuple[np.ndarray, np.ndarray, list[str], np.ndarray, np.ndarray, list[str], np.ndarray, np.ndarray, list[str]]],
    selection_metric: str,
    max_epochs: int,
    patience: int,
    batch_size: int,
    device: torch.device,
    augment: bool,
) -> CandidateResult:
    key = (candidate.channel_set, candidate.window_seconds)
    x_train, y_train, train_ids, x_val, y_val, val_ids, x_test, y_test, test_ids = fold_data[key]
    set_seed(RANDOM_STATE)
    model = StemAttentionRNN(
        n_channels=x_train.shape[2],
        cell=candidate.model,
        hidden_size=candidate.hidden_size,
        dropout=candidate.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=candidate.learning_rate, weight_decay=candidate.weight_decay)
    positives = max(float(y_train.sum()), 1.0)
    negatives = max(float(len(y_train) - y_train.sum()), 1.0)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([negatives / positives], dtype=torch.float32, device=device))
    loader = DataLoader(WindowDataset(x_train, y_train, augment=augment), batch_size=batch_size, shuffle=True)

    best_state = None
    best_threshold = 0.5
    best_scores: dict[str, float] | None = None
    best_epoch = 0
    best_key: tuple[float, float, float, float] | None = None
    stale = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()

        val_scores_window = predict_window_scores(model, x_val, batch_size, device)
        threshold, val_scores = tune_threshold(val_ids, y_val, val_scores_window, selection_metric)
        epoch_key = selection_key(val_scores, selection_metric)
        if best_key is None or epoch_key > best_key:
            best_key = epoch_key
            best_state = {name: param.detach().cpu().clone() for name, param in model.state_dict().items()}
            best_scores = val_scores
            best_threshold = threshold
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break

    assert best_state is not None and best_scores is not None
    model.load_state_dict(best_state)
    test_scores_window = predict_window_scores(model, x_test, batch_size, device)
    test_true, test_pred, test_score, test_subject_ids = aggregate_subject_scores(test_ids, y_test, test_scores_window, best_threshold)
    test_scores = binary_metrics(test_true, test_pred, test_score)
    return CandidateResult(
        candidate=candidate,
        val_scores=best_scores,
        test_scores=test_scores,
        test_subject_ids=test_subject_ids,
        test_true=test_true,
        test_pred=test_pred,
        test_score=test_score,
        threshold=best_threshold,
        best_epoch=best_epoch,
    )


def prepare_fold_data(
    train_subjects: list[SubjectFile],
    val_subjects: list[SubjectFile],
    test_subjects: list[SubjectFile],
    candidates: list[Candidate],
    trim_seconds: float,
) -> dict[tuple[str, float], tuple[np.ndarray, np.ndarray, list[str], np.ndarray, np.ndarray, list[str], np.ndarray, np.ndarray, list[str]]]:
    fold_data = {}
    configs = sorted({(candidate.channel_set, candidate.window_seconds, candidate.step_seconds, candidate.downsample) for candidate in candidates})
    for channel_set, window_seconds, step_seconds, downsample in configs:
        print(f"[raw-windows] {channel_set} {window_seconds:g}s/{step_seconds:g}s downsample={downsample}", flush=True)
        x_train, y_train, train_ids = make_window_matrix(train_subjects, channel_set, trim_seconds, window_seconds, step_seconds, downsample)
        x_val, y_val, val_ids = make_window_matrix(val_subjects, channel_set, trim_seconds, window_seconds, step_seconds, downsample)
        x_test, y_test, test_ids = make_window_matrix(test_subjects, channel_set, trim_seconds, window_seconds, step_seconds, downsample)
        mean, std = fit_channel_normalizer(x_train)
        fold_data[(channel_set, window_seconds)] = (
            normalize(x_train, mean, std),
            y_train,
            train_ids,
            normalize(x_val, mean, std),
            y_val,
            val_ids,
            normalize(x_test, mean, std),
            y_test,
            test_ids,
        )
    return fold_data


def candidate_grid(args: argparse.Namespace, model_name: str) -> list[Candidate]:
    candidates = []
    for channel_set in args.channel_sets:
        for window_seconds in args.window_seconds:
            step_seconds = args.step_seconds if args.step_seconds else window_seconds / 2.0
            for hidden_size in args.hidden_sizes:
                for dropout in args.dropouts:
                    for learning_rate in args.learning_rates:
                        for weight_decay in args.weight_decays:
                            candidates.append(
                                Candidate(
                                    model=model_name,
                                    channel_set=channel_set,
                                    window_seconds=float(window_seconds),
                                    step_seconds=float(step_seconds),
                                    downsample=args.downsample,
                                    hidden_size=int(hidden_size),
                                    dropout=float(dropout),
                                    learning_rate=float(learning_rate),
                                    weight_decay=float(weight_decay),
                                )
                            )
    return candidates


def run_raw_sequence(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ensure_dirs()
    set_seed(RANDOM_STATE)
    if args.torch_threads:
        torch.set_num_threads(args.torch_threads)
    manifest = fold_manifest()
    folds = args.folds or sorted(manifest["fold"].unique())
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[raw-sequence] device={device}", flush=True)

    rows: list[dict[str, object]] = []
    predictions: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []

    for fold in folds:
        train_subjects = subjects_for_fold_split(manifest, fold, "training")
        val_subjects = subjects_for_fold_split(manifest, fold, "validation")
        test_subjects = subjects_for_fold_split(manifest, fold, "test")
        all_candidates = [candidate for model_name in args.models for candidate in candidate_grid(args, model_name)]
        fold_data = prepare_fold_data(train_subjects, val_subjects, test_subjects, all_candidates, args.trim_seconds)
        for model_name in args.models:
            best: CandidateResult | None = None
            best_key: tuple[float, float, float, float] | None = None
            model_candidates = candidate_grid(args, model_name)
            for idx, candidate in enumerate(model_candidates, start=1):
                print(f"[raw-sequence] {fold} {model_name} candidate {idx:02d}/{len(model_candidates):02d} {candidate}", flush=True)
                result = train_candidate(
                    candidate,
                    fold_data,
                    selection_metric=args.selection_metric,
                    max_epochs=args.max_epochs,
                    patience=args.patience,
                    batch_size=args.batch_size,
                    device=device,
                    augment=not args.no_augment,
                )
                candidate_row = {
                    "fold": fold,
                    "model": model_name,
                    "channel_set": candidate.channel_set,
                    "window_seconds": candidate.window_seconds,
                    "step_seconds": candidate.step_seconds,
                    "downsample": candidate.downsample,
                    "hidden_size": candidate.hidden_size,
                    "dropout": candidate.dropout,
                    "learning_rate": candidate.learning_rate,
                    "weight_decay": candidate.weight_decay,
                    "threshold": result.threshold,
                    "best_epoch": result.best_epoch,
                }
                candidate_row.update({f"validation_{key}": value for key, value in result.val_scores.items()})
                candidate_row.update({f"test_{key}": value for key, value in result.test_scores.items()})
                candidate_rows.append(candidate_row)
                key = selection_key(result.val_scores, args.selection_metric)
                if best_key is None or key > best_key:
                    best_key = key
                    best = result

            assert best is not None
            params = {
                "input": "raw subject windows",
                "architecture": "conv temporal stem + bidirectional attention RNN",
                "channel_set": best.candidate.channel_set,
                "window_seconds": best.candidate.window_seconds,
                "step_seconds": best.candidate.step_seconds,
                "downsample": best.candidate.downsample,
                "hidden_size": best.candidate.hidden_size,
                "dropout": best.candidate.dropout,
                "learning_rate": best.candidate.learning_rate,
                "weight_decay": best.candidate.weight_decay,
                "threshold": best.threshold,
                "best_epoch": best.best_epoch,
                "normalization": "channel z-score fit on training windows only",
                "aggregation": "mean window probability per subject",
                "selection_metric": args.selection_metric,
            }
            row = {
                "family": "raw_sequence",
                "scope": "raw_windows",
                "fold": fold,
                "model": model_name,
                "n_train": len(train_subjects),
                "n_validation": len(val_subjects),
                "n_test": len(test_subjects),
                "best_params": repr(params),
            }
            row.update({f"validation_{key}": value for key, value in best.val_scores.items()})
            row.update({f"test_{key}": value for key, value in best.test_scores.items()})
            rows.append(row)
            for subject_id, true, pred, score in zip(best.test_subject_ids, best.test_true, best.test_pred, best.test_score):
                predictions.append(
                    {
                        "family": "raw_sequence",
                        "scope": "raw_windows",
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

    return pd.DataFrame(rows), pd.DataFrame(predictions), pd.DataFrame(candidate_rows)


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    return (
        metrics.groupby(["family", "scope", "model"], as_index=False)
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
        .sort_values(["test_accuracy_mean", "test_auc_mean"], ascending=False)
    )


def write_outputs(metrics: pd.DataFrame, predictions: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    metrics.to_csv(TABLES / "deliverable2_raw_sequence_fold_metrics.csv", index=False)
    predictions.to_csv(TABLES / "deliverable2_raw_sequence_predictions.csv", index=False)
    candidates.to_csv(TABLES / "deliverable2_raw_sequence_candidates.csv", index=False)
    summary = summarize(metrics)
    summary.to_csv(TABLES / "deliverable2_raw_sequence_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 4))
    plot_df = summary.copy()
    plot_df["accuracy_percent"] = plot_df["test_accuracy_mean"] * 100
    sns.barplot(data=plot_df, x="model", y="accuracy_percent", ax=ax, color="#3D6EA8")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Mean test accuracy (%)")
    ax.set_xlabel("Raw-window model")
    ax.set_title("Raw-window sequence models")
    for patch, value in zip(ax.patches, plot_df["accuracy_percent"]):
        ax.text(patch.get_x() + patch.get_width() / 2, patch.get_height() + 1, f"{value:.1f}%", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(FIGURES / "deliverable2_raw_sequence_summary.png", dpi=220)
    plt.close(fig)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", nargs="+")
    parser.add_argument("--models", nargs="+", default=["gru", "lstm"], choices=["gru", "lstm"])
    parser.add_argument("--channel-sets", nargs="+", default=["totals", "all"], choices=["totals", "sensors", "all"])
    parser.add_argument("--window-seconds", nargs="+", type=float, default=[5.0])
    parser.add_argument("--step-seconds", type=float)
    parser.add_argument("--trim-seconds", type=float, default=DEFAULT_TRIM_SECONDS)
    parser.add_argument("--downsample", type=int, default=2)
    parser.add_argument("--hidden-sizes", nargs="+", type=int, default=[64])
    parser.add_argument("--dropouts", nargs="+", type=float, default=[0.2])
    parser.add_argument("--learning-rates", nargs="+", type=float, default=[0.001])
    parser.add_argument("--weight-decays", nargs="+", type=float, default=[0.0001])
    parser.add_argument("--selection-metric", default="f1", choices=["accuracy", "f1", "balanced_accuracy", "auc"])
    parser.add_argument("--max-epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--no-augment", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    metrics, predictions, candidates = run_raw_sequence(args)
    summary = write_outputs(metrics, predictions, candidates)
    print("\nRaw sequence summary:")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.3f}"))

    print("\nSelected fold params:")
    selected = metrics[["fold", "model", "test_accuracy", "test_sensitivity", "test_specificity", "test_auc", "best_params"]].copy()
    selected["best_params"] = selected["best_params"].map(lambda value: ast.literal_eval(value))
    selected["channel_set"] = selected["best_params"].map(lambda value: value["channel_set"])
    selected["window_seconds"] = selected["best_params"].map(lambda value: value["window_seconds"])
    selected["threshold"] = selected["best_params"].map(lambda value: value["threshold"])
    print(selected.drop(columns=["best_params"]).to_string(index=False, float_format=lambda value: f"{value:.3f}"))


if __name__ == "__main__":
    main()
