#!/usr/bin/env python3
"""Prepare fold-aware data analysis artifacts for the gait project.

This script covers the non-Mamba project foundation:

- demographics and split summaries
- TSFresh feature extraction on 5 s / 2.5 s windows
- subject-level aggregation of window features by mean/std/skew/kurtosis
- z-scored PCA figures for left, right, and combined feet
- representative patient/control signal examples
"""

from __future__ import annotations

import argparse
import os
import warnings
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pywt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tsfresh import extract_features
from tsfresh.feature_extraction import EfficientFCParameters, MinimalFCParameters
from tsfresh.utilities.dataframe_functions import impute


ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "splits"
RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"
CACHE = RESULTS / "cache"

SAMPLE_RATE = 100
WINDOW_SECONDS = 5.0
STEP_SECONDS = 2.5
WINDOW_SAMPLES = int(WINDOW_SECONDS * SAMPLE_RATE)
STEP_SAMPLES = int(STEP_SECONDS * SAMPLE_RATE)
DEFAULT_TRIM_SECONDS = 5.0
TSFRESH_N_JOBS = int(os.environ.get("TSFRESH_N_JOBS", "5"))

LEFT_CHANNELS = [(f"L{i}", i) for i in range(1, 9)] + [("L_total", 17)]
RIGHT_CHANNELS = [(f"R{i}", i) for i in range(1, 9)] + [("R_total", 18)]
GROUP_BY_CODE = {"Co": "Control", "Pt": "Patient"}
SENSOR_COORDS_LEFT = np.array(
    [
        [-500, -800],
        [-700, -400],
        [-300, -400],
        [-700, 0],
        [-300, 0],
        [-700, 400],
        [-300, 400],
        [-500, 800],
    ],
    dtype=float,
)
SENSOR_COORDS_RIGHT = np.array(
    [
        [500, -800],
        [700, -400],
        [300, -400],
        [700, 0],
        [300, 0],
        [700, 400],
        [300, 400],
        [500, 800],
    ],
    dtype=float,
)


@dataclass(frozen=True)
class SubjectFile:
    subject_id: str
    group: str
    path: Path


def ensure_dirs() -> None:
    for path in (TABLES, FIGURES, CACHE, ROOT / ".matplotlib"):
        path.mkdir(parents=True, exist_ok=True)


def group_from_subject_id(subject_id: str) -> str:
    return GROUP_BY_CODE[subject_id[2:4]]


def subject_id_from_path(path: Path) -> str:
    return path.stem.rsplit("_", 1)[0]


def unique_subject_files() -> list[SubjectFile]:
    files: dict[str, SubjectFile] = {}
    for path in sorted(SPLITS.glob("Fold_*/*/*.txt")):
        subject_id = subject_id_from_path(path)
        files.setdefault(
            subject_id,
            SubjectFile(subject_id=subject_id, group=group_from_subject_id(subject_id), path=path),
        )
    return sorted(files.values(), key=lambda item: item.subject_id)


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


def read_demographics(subject_ids: set[str]) -> pd.DataFrame:
    df = pd.read_excel(ROOT / "demographics.xls", engine="xlrd")
    df = df[df["ID"].isin(subject_ids)].copy()
    df["Group"] = df["Group"].replace({"CO": "Control", "PD": "Patient"})
    df["Gender"] = df["Gender"].astype(str).str.strip().str.lower()
    return df


def write_summaries(subjects: list[SubjectFile], manifest: pd.DataFrame) -> None:
    subject_ids = {subject.subject_id for subject in subjects}
    demographics = read_demographics(subject_ids)
    demo_summary = (
        demographics.groupby("Group", as_index=False)
        .agg(
            subjects=("ID", "count"),
            age_mean=("Age", "mean"),
            age_sd=("Age", "std"),
            male=("Gender", lambda s: int((s == "male").sum())),
            female=("Gender", lambda s: int((s == "female").sum())),
        )
        .sort_values("Group")
    )
    demo_summary["age_mean"] = demo_summary["age_mean"].round(2)
    demo_summary["age_sd"] = demo_summary["age_sd"].round(2)
    demo_summary.to_csv(TABLES / "demographics_summary.csv", index=False)

    split_summary = (
        manifest.groupby(["fold", "split", "group"], as_index=False)
        .size()
        .rename(columns={"size": "subjects"})
        .sort_values(["fold", "split", "group"])
    )
    split_summary.to_csv(TABLES / "fold_split_summary.csv", index=False)
    manifest.to_csv(TABLES / "fold_manifest.csv", index=False)


def load_signal(path: Path, trim_seconds: float = DEFAULT_TRIM_SECONDS) -> np.ndarray:
    signal = np.loadtxt(path, dtype=float)
    trim_samples = int(trim_seconds * SAMPLE_RATE)
    if trim_samples <= 0:
        return signal
    if len(signal) <= 2 * trim_samples + WINDOW_SAMPLES:
        return signal
    return signal[trim_samples:-trim_samples]


def make_tsfresh_frame(signal: np.ndarray, channels: list[tuple[str, int]]) -> pd.DataFrame:
    frames = []
    for window_idx, start in enumerate(range(0, len(signal) - WINDOW_SAMPLES + 1, STEP_SAMPLES)):
        block = signal[start : start + WINDOW_SAMPLES]
        time = np.arange(WINDOW_SAMPLES, dtype=np.int32)
        window_id = f"w{window_idx:03d}"
        for channel_name, col_idx in channels:
            frames.append(
                pd.DataFrame(
                    {
                        "id": window_id,
                        "time": time,
                        "kind": channel_name,
                        "value": block[:, col_idx],
                    }
                )
            )
    return pd.concat(frames, ignore_index=True)


def aggregate_window_features(features: pd.DataFrame, prefix: str) -> pd.Series:
    impute(features)
    feature_stats = {
        "window_mean": features.mean(axis=0),
        "window_std": features.std(axis=0, ddof=0),
        "window_skew": features.skew(axis=0),
        "window_kurtosis": features.kurt(axis=0),
    }
    parts = []
    for stat_name, values in feature_stats.items():
        stat_values = values.fillna(0.0)
        stat_values.index = [f"{prefix}__{column}__{stat_name}" for column in stat_values.index]
        parts.append(stat_values)
    return pd.concat(parts)


def subject_tsfresh_features(subject: SubjectFile, fc_parameters: dict, trim_seconds: float) -> pd.Series:
    signal = load_signal(subject.path, trim_seconds=trim_seconds)
    left_frame = make_tsfresh_frame(signal, LEFT_CHANNELS)
    right_frame = make_tsfresh_frame(signal, RIGHT_CHANNELS)

    left_features = extract_features(
        left_frame,
        column_id="id",
        column_sort="time",
        column_kind="kind",
        column_value="value",
        default_fc_parameters=fc_parameters,
        disable_progressbar=True,
        n_jobs=TSFRESH_N_JOBS,
    )
    right_features = extract_features(
        right_frame,
        column_id="id",
        column_sort="time",
        column_kind="kind",
        column_value="value",
        default_fc_parameters=fc_parameters,
        disable_progressbar=True,
        n_jobs=TSFRESH_N_JOBS,
    )
    return pd.concat(
        [
            aggregate_window_features(left_features, "left"),
            aggregate_window_features(right_features, "right"),
        ]
    )


def curated_fc_parameters() -> dict:
    """A stronger TSFresh profile that avoids very slow/unstable calculators."""
    return {
        "sum_values": None,
        "median": None,
        "mean": None,
        "length": None,
        "standard_deviation": None,
        "variance": None,
        "root_mean_square": None,
        "maximum": None,
        "minimum": None,
        "abs_energy": None,
        "skewness": None,
        "kurtosis": None,
        "absolute_sum_of_changes": None,
        "mean_abs_change": None,
        "mean_change": None,
        "cid_ce": [{"normalize": True}, {"normalize": False}],
        "count_above_mean": None,
        "count_below_mean": None,
        "first_location_of_maximum": None,
        "first_location_of_minimum": None,
        "last_location_of_maximum": None,
        "last_location_of_minimum": None,
        "longest_strike_above_mean": None,
        "longest_strike_below_mean": None,
        "quantile": [{"q": q} for q in [0.1, 0.25, 0.5, 0.75, 0.9]],
        "autocorrelation": [{"lag": lag} for lag in range(1, 11)],
        "partial_autocorrelation": [{"lag": lag} for lag in range(1, 6)],
        "agg_autocorrelation": [
            {"f_agg": agg, "maxlag": 10}
            for agg in ["mean", "median", "var"]
        ],
        "c3": [{"lag": lag} for lag in [1, 2, 3]],
        "number_peaks": [{"n": n} for n in [1, 3, 5]],
        "fft_aggregated": [{"aggtype": agg} for agg in ["centroid", "variance", "skew", "kurtosis"]],
        "fft_coefficient": [
            {"coeff": coeff, "attr": attr}
            for coeff in range(1, 6)
            for attr in ["real", "imag", "abs"]
        ],
        "spkt_welch_density": [{"coeff": coeff} for coeff in [2, 5, 8]],
        "linear_trend": [
            {"attr": attr}
            for attr in ["pvalue", "rvalue", "intercept", "slope", "stderr"]
        ],
    }


def fc_parameters_for(feature_set: str) -> dict:
    if feature_set == "minimal":
        return MinimalFCParameters()
    if feature_set == "curated":
        return curated_fc_parameters()
    if feature_set == "efficient":
        return EfficientFCParameters()
    raise ValueError(f"Unknown feature set: {feature_set}")


def finite_stats(prefix: str, values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_std": 0.0,
            f"{prefix}_cv": 0.0,
            f"{prefix}_min": 0.0,
            f"{prefix}_max": 0.0,
            f"{prefix}_median": 0.0,
            f"{prefix}_iqr": 0.0,
            f"{prefix}_p10": 0.0,
            f"{prefix}_p90": 0.0,
        }
    mean = float(np.mean(values))
    std = float(np.std(values))
    q10, q25, q50, q75, q90 = np.percentile(values, [10, 25, 50, 75, 90])
    return {
        f"{prefix}_mean": mean,
        f"{prefix}_std": std,
        f"{prefix}_cv": float(std / (abs(mean) + 1e-9)),
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_max": float(np.max(values)),
        f"{prefix}_median": float(q50),
        f"{prefix}_iqr": float(q75 - q25),
        f"{prefix}_p10": float(q10),
        f"{prefix}_p90": float(q90),
    }


def segments_from_mask(mask: np.ndarray, min_samples: int = 10) -> tuple[np.ndarray, np.ndarray]:
    padded = np.r_[False, mask, False]
    starts = np.flatnonzero(~padded[:-1] & padded[1:])
    ends = np.flatnonzero(padded[:-1] & ~padded[1:])
    keep = (ends - starts) >= min_samples
    return starts[keep], ends[keep]


def cop_features(sensor_values: np.ndarray, coords: np.ndarray, side: str) -> dict[str, float]:
    total = sensor_values.sum(axis=1)
    active = total > max(25.0, 0.08 * float(np.percentile(total, 95)))
    if not np.any(active):
        return finite_stats(f"gait__{side}_cop_x", np.array([])) | finite_stats(f"gait__{side}_cop_y", np.array([]))
    weighted = sensor_values[active] @ coords
    cop = weighted / (total[active, None] + 1e-9)
    speed = np.linalg.norm(np.diff(cop, axis=0), axis=1) * SAMPLE_RATE if len(cop) > 1 else np.array([])
    features = {}
    features.update(finite_stats(f"gait__{side}_cop_x", cop[:, 0]))
    features.update(finite_stats(f"gait__{side}_cop_y", cop[:, 1]))
    features.update(finite_stats(f"gait__{side}_cop_speed", speed))
    return features


def foot_gait_features(force: np.ndarray, side: str) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    threshold = max(25.0, 0.08 * float(np.percentile(force, 95)))
    mask = force > threshold
    starts, ends = segments_from_mask(mask)
    durations = (ends - starts) / SAMPLE_RATE
    intervals = np.diff(starts) / SAMPLE_RATE if len(starts) > 1 else np.array([])
    swings = (starts[1:] - ends[:-1]) / SAMPLE_RATE if len(starts) > 1 else np.array([])
    peaks = np.array([np.max(force[start:end]) for start, end in zip(starts, ends)], dtype=float)
    impulses = np.array([np.sum(force[start:end]) / SAMPLE_RATE for start, end in zip(starts, ends)], dtype=float)
    duration_seconds = len(force) / SAMPLE_RATE
    features = {
        f"gait__{side}_threshold": threshold,
        f"gait__{side}_contact_fraction": float(np.mean(mask)),
        f"gait__{side}_contact_count": float(len(starts)),
        f"gait__{side}_cadence_per_min": float(len(starts) / duration_seconds * 60.0),
    }
    features.update(finite_stats(f"gait__{side}_force", force))
    features.update(finite_stats(f"gait__{side}_active_force", force[mask]))
    features.update(finite_stats(f"gait__{side}_contact_duration", durations))
    features.update(finite_stats(f"gait__{side}_swing_duration", swings))
    features.update(finite_stats(f"gait__{side}_stride_interval", intervals))
    features.update(finite_stats(f"gait__{side}_contact_peak", peaks))
    features.update(finite_stats(f"gait__{side}_contact_impulse", impulses))
    return features, mask, starts


def cross_foot_features(signal: np.ndarray) -> dict[str, float]:
    left_force = signal[:, 17]
    right_force = signal[:, 18]
    left_features, left_mask, left_starts = foot_gait_features(left_force, "left")
    right_features, right_mask, right_starts = foot_gait_features(right_force, "right")
    features = {}
    features.update(left_features)
    features.update(right_features)
    features.update(cop_features(signal[:, 1:9], SENSOR_COORDS_LEFT, "left"))
    features.update(cop_features(signal[:, 9:17], SENSOR_COORDS_RIGHT, "right"))

    total_force = left_force + right_force
    symmetry = (left_force - right_force) / (total_force + 1e-9)
    features.update(finite_stats("gait__force_symmetry_index", symmetry))
    features["gait__double_support_fraction"] = float(np.mean(left_mask & right_mask))
    features["gait__no_support_fraction"] = float(np.mean(~left_mask & ~right_mask))
    features["gait__left_only_fraction"] = float(np.mean(left_mask & ~right_mask))
    features["gait__right_only_fraction"] = float(np.mean(right_mask & ~left_mask))
    features["gait__contact_fraction_absdiff"] = abs(
        features["gait__left_contact_fraction"] - features["gait__right_contact_fraction"]
    )
    features["gait__cadence_absdiff"] = abs(
        features["gait__left_cadence_per_min"] - features["gait__right_cadence_per_min"]
    )
    all_starts = np.sort(np.r_[left_starts, right_starts])
    features.update(finite_stats("gait__step_interval", np.diff(all_starts) / SAMPLE_RATE if len(all_starts) > 1 else np.array([])))
    if np.std(left_force) > 0 and np.std(right_force) > 0:
        left_z = (left_force - left_force.mean()) / left_force.std()
        right_z = (right_force - right_force.mean()) / right_force.std()
        max_lag = SAMPLE_RATE
        lags = np.arange(-max_lag, max_lag + 1)
        corr = []
        for lag in lags:
            if lag < 0:
                corr.append(float(np.mean(left_z[-lag:] * right_z[:lag])))
            elif lag > 0:
                corr.append(float(np.mean(left_z[:-lag] * right_z[lag:])))
            else:
                corr.append(float(np.mean(left_z * right_z)))
        corr = np.asarray(corr)
        best = int(np.argmax(corr))
        features["gait__left_right_corr_max"] = float(corr[best])
        features["gait__left_right_corr_lag_seconds"] = float(lags[best] / SAMPLE_RATE)
    else:
        features["gait__left_right_corr_max"] = 0.0
        features["gait__left_right_corr_lag_seconds"] = 0.0
    return features


def wavelet_features_for_signal(values: np.ndarray, side: str) -> dict[str, float]:
    values = np.asarray(values[::2], dtype=float)
    values = (values - np.mean(values)) / (np.std(values) + 1e-9)
    scales = np.arange(1, 65)
    coefficients, _ = pywt.cwt(values, scales, "morl")
    power = np.abs(coefficients) ** 2
    features = {}
    bands = {
        "s1_8": slice(0, 8),
        "s9_16": slice(8, 16),
        "s17_32": slice(16, 32),
        "s33_64": slice(32, 64),
    }
    for band_name, band_slice in bands.items():
        band_power = power[band_slice].ravel()
        features.update(finite_stats(f"wavelet__{side}_{band_name}_power", band_power))
    scale_energy = power.mean(axis=1)
    for idx, energy in enumerate(scale_energy, start=1):
        features[f"wavelet__{side}_scale_{idx:02d}_energy"] = float(energy)
    return features


def add_wavelet_features(matrix: pd.DataFrame, subjects: list[SubjectFile], trim_seconds: float) -> pd.DataFrame:
    rows = []
    for subject in subjects:
        signal = load_signal(subject.path, trim_seconds=trim_seconds)
        left_force = signal[:, 17]
        right_force = signal[:, 18]
        symmetry = (left_force - right_force) / (left_force + right_force + 1e-9)
        row = {"subject_id": subject.subject_id}
        row.update(wavelet_features_for_signal(left_force, "left"))
        row.update(wavelet_features_for_signal(right_force, "right"))
        row.update(wavelet_features_for_signal(symmetry, "sym"))
        rows.append(row)
    wavelet = pd.DataFrame(rows)
    wavelet.to_csv(TABLES / "wavelet_engineered_features.csv", index=False)
    return matrix.merge(wavelet, on="subject_id", how="left")


def add_asymmetry_features(matrix: pd.DataFrame) -> pd.DataFrame:
    additions = {}
    right_columns = set(column for column in matrix.columns if column.startswith("right__"))
    for left_column in [column for column in matrix.columns if column.startswith("left__")]:
        right_column = left_column.replace("left__L_total__", "right__R_total__", 1)
        if right_column == left_column:
            for sensor_idx in range(1, 9):
                prefix = f"left__L{sensor_idx}__"
                if left_column.startswith(prefix):
                    right_column = left_column.replace(prefix, f"right__R{sensor_idx}__", 1)
                    break
        if right_column not in right_columns:
            continue
        feature_name = "asym__" + left_column.removeprefix("left__").replace("L_total", "total")
        for sensor_idx in range(1, 9):
            feature_name = feature_name.replace(f"L{sensor_idx}__", f"S{sensor_idx}__", 1)
        additions[f"{feature_name}__absdiff"] = (matrix[left_column] - matrix[right_column]).abs()
    if not additions:
        return matrix
    return pd.concat([matrix, pd.DataFrame(additions)], axis=1)


def add_gait_features(matrix: pd.DataFrame, subjects: list[SubjectFile], trim_seconds: float) -> pd.DataFrame:
    rows = []
    for subject in subjects:
        signal = load_signal(subject.path, trim_seconds=trim_seconds)
        row = {"subject_id": subject.subject_id}
        row.update(cross_foot_features(signal))
        rows.append(row)
    gait = pd.DataFrame(rows)
    gait.to_csv(TABLES / "gait_engineered_features.csv", index=False)
    return matrix.merge(gait, on="subject_id", how="left")


def add_engineered_features(matrix: pd.DataFrame, subjects: list[SubjectFile], trim_seconds: float) -> pd.DataFrame:
    base_columns = [column for column in matrix.columns if not column.startswith(("gait__", "asym__", "wavelet__"))]
    matrix = matrix[base_columns].copy()
    matrix = add_asymmetry_features(matrix)
    matrix = add_gait_features(matrix, subjects, trim_seconds)
    matrix = add_wavelet_features(matrix, subjects, trim_seconds)
    return matrix


def compute_feature_matrix(
    subjects: list[SubjectFile],
    feature_set: str,
    trim_seconds: float,
    force: bool = False,
) -> pd.DataFrame:
    output = TABLES / f"subject_feature_matrix_{feature_set}.csv"
    engineered_output = TABLES / f"subject_feature_matrix_{feature_set}_engineered.csv"
    default_output = TABLES / "subject_feature_matrix.csv"
    if engineered_output.exists() and not force:
        matrix = pd.read_csv(engineered_output)
        if not any(column.startswith("wavelet__") for column in matrix.columns):
            matrix = add_engineered_features(matrix, subjects, trim_seconds)
            matrix.to_csv(engineered_output, index=False)
        matrix.to_csv(default_output, index=False)
        return matrix
    if output.exists() and not force:
        matrix = pd.read_csv(output)
        matrix = add_engineered_features(matrix, subjects, trim_seconds)
        matrix.to_csv(engineered_output, index=False)
        matrix.to_csv(default_output, index=False)
        return matrix

    rows = []
    fc_parameters = fc_parameters_for(feature_set)
    for idx, subject in enumerate(subjects, start=1):
        print(f"[features] {idx:03d}/{len(subjects):03d} {subject.subject_id}", flush=True)
        features = subject_tsfresh_features(subject, fc_parameters, trim_seconds=trim_seconds)
        row = {"subject_id": subject.subject_id, "group": subject.group}
        row.update(features.to_dict())
        rows.append(row)

    matrix = pd.DataFrame(rows).sort_values("subject_id").reset_index(drop=True)
    matrix.to_csv(output, index=False)
    matrix = add_engineered_features(matrix, subjects, trim_seconds)
    matrix.to_csv(engineered_output, index=False)
    matrix.to_csv(default_output, index=False)
    return matrix


def signal_summary(subjects: list[SubjectFile], trim_seconds: float) -> pd.DataFrame:
    rows = []
    for subject in subjects:
        raw_signal = load_signal(subject.path, trim_seconds=0.0)
        signal = load_signal(subject.path, trim_seconds=trim_seconds)
        rows.append(
            {
                "subject_id": subject.subject_id,
                "group": subject.group,
                "file": str(subject.path.relative_to(ROOT)),
                "raw_duration_seconds": round(float(raw_signal[-1, 0] - raw_signal[0, 0]), 2),
                "used_duration_seconds": round(float(signal[-1, 0] - signal[0, 0]), 2),
                "trim_seconds_each_side": trim_seconds,
                "n_windows": len(range(0, len(signal) - WINDOW_SAMPLES + 1, STEP_SAMPLES)),
                "left_total_mean_force": round(float(np.mean(signal[:, 17])), 4),
                "right_total_mean_force": round(float(np.mean(signal[:, 18])), 4),
            }
        )
    summary = pd.DataFrame(rows).sort_values("subject_id")
    summary.to_csv(TABLES / "subject_signal_summary.csv", index=False)
    return summary


def feature_columns(matrix: pd.DataFrame, scope: str) -> list[str]:
    if scope == "left":
        return [column for column in matrix.columns if column.startswith("left__")]
    if scope == "right":
        return [column for column in matrix.columns if column.startswith("right__")]
    if scope == "combined":
        return [column for column in matrix.columns if column.startswith(("left__", "right__"))]
    raise ValueError(f"Unknown feature scope: {scope}")


def pca_for_scope(matrix: pd.DataFrame, scope: str) -> pd.DataFrame:
    columns = feature_columns(matrix, scope)
    x = matrix[columns].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
    x = StandardScaler().fit_transform(x)
    pca = PCA(n_components=2, random_state=42)
    scores = pca.fit_transform(x)
    result = pd.DataFrame(
        {
            "subject_id": matrix["subject_id"],
            "group": matrix["group"],
            "pc1": scores[:, 0],
            "pc2": scores[:, 1],
            "pc1_explained": pca.explained_variance_ratio_[0],
            "pc2_explained": pca.explained_variance_ratio_[1],
        }
    )
    slug = {"left": "left_foot", "right": "right_foot", "combined": "combined_feet"}[scope]
    result.to_csv(TABLES / f"pca_{slug}.csv", index=False)
    return result


def plot_pca(pca_df: pd.DataFrame, scope: str) -> None:
    explained = pca_df[["pc1_explained", "pc2_explained"]].iloc[0] * 100
    fig, ax = plt.subplots(figsize=(9.6, 6.4))
    sns.scatterplot(
        data=pca_df,
        x="pc1",
        y="pc2",
        hue="group",
        hue_order=["Patient", "Control"],
        palette={"Patient": "#c44e52", "Control": "#2f6f9f"},
        s=62,
        edgecolor="white",
        linewidth=0.8,
        alpha=0.82,
        ax=ax,
    )
    title = {"left": "PCA: Left Foot Features", "right": "PCA: Right Foot Features", "combined": "PCA: Combined Left + Right Features"}[scope]
    ax.set_title(title, loc="left", fontsize=16, fontweight="bold")
    ax.set_xlabel(f"PC1 ({explained.iloc[0]:.1f}% var.)")
    ax.set_ylabel(f"PC2 ({explained.iloc[1]:.1f}% var.)")
    ax.grid(True, color="#e9e3db", linewidth=0.8)
    ax.set_facecolor("#fbfaf7")
    fig.patch.set_facecolor("#fbfaf7")
    fig.tight_layout()
    slug = {"left": "left_foot", "right": "right_foot", "combined": "combined_feet"}[scope]
    fig.savefig(FIGURES / f"pca_{slug}.png", dpi=180)
    fig.savefig(FIGURES / f"pca_{slug}.svg")
    plt.close(fig)


def example_subjects(subjects: list[SubjectFile]) -> tuple[SubjectFile, SubjectFile]:
    patient = next(subject for subject in subjects if subject.group == "Patient")
    control = next(subject for subject in subjects if subject.group == "Control")
    return patient, control


def plot_examples(subjects: list[SubjectFile]) -> None:
    patient, control = example_subjects(subjects)
    patient_signal = load_signal(patient.path)
    control_signal = load_signal(control.path)
    n = 10 * SAMPLE_RATE
    time = np.arange(n) / SAMPLE_RATE
    for label, col_idx, output in (
        ("Left-Foot Total Force", 17, "example_left_total_force"),
        ("Right-Foot Total Force", 18, "example_right_total_force"),
    ):
        fig, ax = plt.subplots(figsize=(9.6, 5.4))
        ax.plot(time, patient_signal[:n, col_idx], color="#c44e52", linewidth=1.6, label=f"Patient ({patient.subject_id})")
        ax.plot(time, control_signal[:n, col_idx], color="#2f6f9f", linewidth=1.6, label=f"Control ({control.subject_id})")
        ax.set_title(f"Example {label}", loc="left", fontsize=16, fontweight="bold")
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Total vertical force (N)")
        ax.grid(True, color="#e9e3db", linewidth=0.8)
        ax.legend(frameon=False, loc="upper right")
        ax.set_facecolor("#fbfaf7")
        fig.patch.set_facecolor("#fbfaf7")
        fig.tight_layout()
        fig.savefig(FIGURES / f"{output}.png", dpi=180)
        fig.savefig(FIGURES / f"{output}.svg")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-features", action="store_true", help="Recompute TSFresh features even if cached CSV exists.")
    parser.add_argument(
        "--feature-set",
        choices=["minimal", "curated", "efficient"],
        default="curated",
        help="TSFresh feature calculator set. Curated is the stronger baseline-compatible default.",
    )
    parser.add_argument(
        "--trim-seconds",
        type=float,
        default=DEFAULT_TRIM_SECONDS,
        help="Seconds to remove from the start and end of every recording before windowing.",
    )
    args = parser.parse_args()

    ensure_dirs()
    warnings.filterwarnings("ignore", category=RuntimeWarning, module="tsfresh.utilities.dataframe_functions")
    subjects = unique_subject_files()
    manifest = fold_manifest()
    write_summaries(subjects, manifest)
    signal_summary(subjects, trim_seconds=args.trim_seconds)
    feature_matrix = compute_feature_matrix(
        subjects,
        feature_set=args.feature_set,
        trim_seconds=args.trim_seconds,
        force=args.force_features,
    )
    for scope in ("left", "right", "combined"):
        pca_df = pca_for_scope(feature_matrix, scope)
        plot_pca(pca_df, scope)
    plot_examples(subjects)
    print(f"Wrote tables to {TABLES}")
    print(f"Wrote figures to {FIGURES}")
    print(f"Feature mode: TSFresh {args.feature_set}, aggregated across windows")
    print(f"Trim: {args.trim_seconds} seconds removed from each side before windowing")
    print("Fold protocol: 5 folds, each with training/validation/test splits")


if __name__ == "__main__":
    main()
