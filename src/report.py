"""
Aggregate per-fold results into CSVs, printed tables, and confusion-matrix
figures, and compare against the minimum-target table.
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402
import pandas as pd               # noqa: E402

from . import config as C         # noqa: E402

METRIC_COLS = ["accuracy", "sensitivity", "specificity", "auc",
               "precision", "recall", "f1"]

# Minimum-target table (Accuracy %, Precision %, Recall %, F1).
# Only the classical baselines are mapped.
TARGET = {
    "SVM-Linear": (79.4, 79.1, 79.1, 0.791),
    "SVM-RBF": (80.6, 80.3, 80.3, 0.803),
}
DISPLAY_NAME = {"SVM-Linear": "Linear-SVM", "SVM-RBF": "RBF-SVM"}


def _summary(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["branch", "model", "foot"], sort=False)
    mean = g[METRIC_COLS].mean().add_suffix("_mean")
    std = g[METRIC_COLS].std(ddof=0).add_suffix("_std")
    return pd.concat([mean, std], axis=1).reset_index()


def _print_summary_table(summary: pd.DataFrame) -> None:
    print("\n" + "=" * 96)
    print("SUMMARY — mean ± std over folds")
    print("=" * 96)
    header = f"{'Model':13s} {'Foot':9s} {'Acc':>13s} {'Sens':>13s} {'Spec':>13s} {'AUC':>13s}"
    print(header)
    print("-" * len(header))
    for _, r in summary.iterrows():
        def cell(m):
            return f"{r[m+'_mean']*100:5.1f}±{r[m+'_std']*100:4.1f}"
        print(f"{r['model']:13s} {r['foot']:9s} "
              f"{cell('accuracy'):>13s} {cell('sensitivity'):>13s} "
              f"{cell('specificity'):>13s} {cell('auc'):>13s}")


def _print_target_comparison(summary: pd.DataFrame, foot: str = "Combined") -> None:
    print("\n" + "=" * 96)
    print(f"TARGET COMPARISON — {foot} foot (Accuracy %, Precision %, Recall %, F1)")
    print("=" * 96)
    print(f"{'Model':13s} {'Acc':>18s} {'Prec':>18s} {'Recall':>18s} {'F1':>16s}")
    print("-" * 85)
    sub = summary[summary["foot"] == foot]
    for _, r in sub.iterrows():
        acc, prec, rec, f1 = (r["accuracy_mean"] * 100, r["precision_mean"] * 100,
                              r["recall_mean"] * 100, r["f1_mean"])
        tgt = TARGET.get(r["model"])
        name = DISPLAY_NAME.get(r["model"], r["model"])
        if tgt:
            print(f"{name:13s} {acc:7.1f} (t {tgt[0]:4.1f}) {prec:7.1f} (t {tgt[1]:4.1f}) "
                  f"{rec:7.1f} (t {tgt[2]:4.1f}) {f1:6.3f} (t {tgt[3]:.3f})")
        else:
            print(f"{name:13s} {acc:7.1f} {'':9s} {prec:7.1f} {'':9s} "
                  f"{rec:7.1f} {'':9s} {f1:6.3f}")
    print("(t = minimum target)")


def _confusion_figure(df: pd.DataFrame, branch: str) -> None:
    sub = df[df["branch"] == branch]
    if sub.empty:
        return
    models = list(dict.fromkeys(sub["model"]))
    feet = [f for f in C.FEET if f in set(sub["foot"])]
    fig, axes = plt.subplots(len(models), len(feet),
                             figsize=(3.2 * len(feet), 3.0 * len(models)),
                             squeeze=False)
    for i, model in enumerate(models):
        for j, foot in enumerate(feet):
            ax = axes[i][j]
            rows = sub[(sub["model"] == model) & (sub["foot"] == foot)]
            cm = np.sum([np.array(c) for c in rows["cm"]], axis=0)
            ax.imshow(cm, cmap="Blues")
            for (r, c), v in np.ndenumerate(cm):
                ax.text(c, r, str(int(v)), ha="center", va="center",
                        color="white" if v > cm.max() / 2 else "black",
                        fontsize=12, fontweight="bold")
            ax.set_xticks([0, 1], ["CO", "PD"])
            ax.set_yticks([0, 1], ["CO", "PD"])
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            ax.set_title(f"{model} — {foot}")
    fig.suptitle(f"Confusion matrices (summed over folds) — {branch}", y=1.0)
    fig.tight_layout()
    suffix = f"_{C.RUN_TAG}" if getattr(C, "RUN_TAG", "") else ""
    out = os.path.join(C.FIG_DIR, f"confusion_matrices_{branch}{suffix}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {os.path.relpath(out, C.BASE_DIR)}")


def build_report(results: list[dict]) -> None:
    if not results:
        print("No results to report.")
        return

    df = pd.DataFrame(results)
    # Suffix outputs with the branch(es) in this run ("classical", "dl", or
    # "classical_dl") so a parallel classical + dl run never overwrites the
    # other's CSVs. Confusion-matrix figures are already per-branch.
    tag = "_".join(sorted(df["branch"].unique()))
    if getattr(C, "RUN_TAG", ""):
        tag = f"{tag}_{C.RUN_TAG}"
    per_fold_path = os.path.join(C.RESULTS_DIR, f"results_per_fold_{tag}.csv")
    df.drop(columns=["cm"]).to_csv(per_fold_path, index=False)

    summary = _summary(df)
    summary_path = os.path.join(C.RESULTS_DIR, f"results_summary_{tag}.csv")
    summary.to_csv(summary_path, index=False)

    _print_summary_table(summary)
    if "Combined" in set(df["foot"]):
        _print_target_comparison(summary, "Combined")

    print("\nFigures:")
    for branch in ("classical", "dl"):
        _confusion_figure(df, branch)

    print(f"\nWrote:\n  {os.path.relpath(per_fold_path, C.BASE_DIR)}"
          f"\n  {os.path.relpath(summary_path, C.BASE_DIR)}")
