"""
Single-command driver for every remaining experiment.

Runs the classical branch, all deep/SSM models, and both fusion variants in one
pass, then writes ONE consolidated file — results/FINAL_RESULTS.md — containing
every summary table and both fusion leaderboards. That single file is all that
needs to be shared.

Everything is also mirrored to results/run_all_log.txt, and a failure in one
model is caught so it cannot abort the rest of the run.

Examples
--------
  python run_all.py                    # everything (slowest; includes raw-signal Mamba)
  python run_all.py --fast             # skip raw-signal Mamba (~3 h shorter)
  python run_all.py --skip-classical   # classical already done and cached
  python run_all.py --folds 1 --fast   # quick smoke test of the whole driver
"""
from __future__ import annotations

import argparse
import contextlib
import io
import os
import platform
import sys
import time

import numpy as np

from main import run_classical, run_dl
from src import config as C
from src.data import load_fold
from src.report import build_report

# Raw-signal Mamba is by far the slowest (30 s crops); --fast drops it.
SLOW_MODELS = ("Mamba",)


class _Tee:
    """Mirror stdout to a log file so nothing is lost in a long run."""

    def __init__(self, path):
        self.term = sys.stdout
        self.log = open(path, "w", buffering=1)

    def write(self, s):
        self.term.write(s)
        self.log.write(s)

    def flush(self):
        self.term.flush()
        self.log.flush()


def parse_args():
    p = argparse.ArgumentParser(description="Run every pending experiment in one pass")
    p.add_argument("--folds", nargs="+", type=int,
                   default=list(range(1, C.N_FOLDS + 1)))
    p.add_argument("--feet", nargs="+", default=list(C.FEET), choices=list(C.FEET))
    p.add_argument("--fast", action="store_true",
                   help=f"Skip the slowest model(s): {', '.join(SLOW_MODELS)}.")
    p.add_argument("--skip-classical", action="store_true",
                   help="Skip the classical branch (already run / cached).")
    p.add_argument("--skip-dl", action="store_true", help="Skip the deep branch.")
    p.add_argument("--dl-quick", action="store_true",
                   help="One seed instead of the full ensemble (smoke test only).")
    p.add_argument("--fc-params", choices=["minimal", "efficient", "comprehensive"],
                   default=C.TSFRESH_FC_PARAMETERS)
    p.add_argument("--tsfresh-jobs", type=int, default=C.TSFRESH_N_JOBS)
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Consolidated report
# --------------------------------------------------------------------------- #
METRICS = ("accuracy", "sensitivity", "specificity", "auc")


def _summary_rows(results):
    """{(model, foot): {metric: (mean, std)}} plus the model order seen."""
    import pandas as pd
    if not results:
        return {}, []
    df = pd.DataFrame(results)
    order, table = [], {}
    for (model, foot), grp in df.groupby(["model", "foot"], sort=False):
        if model not in order:
            order.append(model)
        table[(model, foot)] = {m: (grp[m].mean(), grp[m].std(ddof=0))
                                for m in METRICS}
    return table, order


def write_final(results, fusion_text, cross_text, elapsed, args, path):
    table, order = _summary_rows(results)
    feet = [f for f in C.FEET if any(k[1] == f for k in table)]

    L = []
    L.append("# FINAL RESULTS — Parkinson's gait\n")
    L.append(f"Generated {time.strftime('%Y-%m-%d %H:%M')} · "
             f"total compute {elapsed/60:.1f} min\n")
    L.append("**Send this whole file.**\n")

    L.append("\n## Configuration\n")
    try:
        import torch
        dev = ("cuda" if torch.cuda.is_available()
               else "mps" if torch.backends.mps.is_available() else "cpu")
    except Exception:
        dev = "unknown"
    from src.mamba_block import mamba_backend
    L.append(f"- device: `{dev}` · python {platform.python_version()}")
    L.append(f"- mamba backend: {mamba_backend()}")
    L.append(f"- folds: {args.folds} · feet: {args.feet} · "
             f"fast={args.fast} · dl_quick={args.dl_quick}")
    L.append(f"- thresholds: classical=`{C.CLASSICAL_THRESHOLD}` "
             f"dl=`{C.DL_THRESHOLD}` · pad=`{C.DL_PAD_MODE}`")
    L.append(f"- mamba refit/mixup/ema: {C.DL_MAMBA_REFIT_TRAINVAL}/"
             f"{C.DL_MAMBA_MIXUP_ALPHA}/{C.DL_MAMBA_EMA_DECAY}")
    L.append(f"- featmamba refit/mixup/ema: {C.DL_FEATMAMBA_REFIT_TRAINVAL}/"
             f"{C.DL_FEATMAMBA_MIXUP_ALPHA}/{C.DL_FEATMAMBA_EMA_DECAY}"
             f" · lr {C.DL_FEATMAMBA_CFG['lr']}"
             f" · multistat {C.DL_FEATMAMBA_CFG['multistat_pool']}")
    L.append(f"- ensemble seeds: {C.DL_ENSEMBLE_SEEDS}\n")

    # one table per foot
    for foot in feet:
        L.append(f"\n## Results — {foot} foot (mean ± std over "
                 f"{len(args.folds)} folds)\n")
        L.append("| Model | Accuracy | Sensitivity | Specificity | AUC |")
        L.append("|---|---|---|---|---|")
        for model in order:
            cell = table.get((model, foot))
            if not cell:
                continue
            vals = " | ".join(f"{cell[m][0]:.3f} ± {cell[m][1]:.3f}" for m in METRICS)
            L.append(f"| {model} | {vals} |")

    # per-fold detail (useful for paired significance tests)
    L.append("\n## Per-fold accuracy (for significance testing)\n")
    L.append("| Model | Foot | " + " | ".join(f"F{f}" for f in args.folds) + " |")
    L.append("|---|---|" + "---|" * len(args.folds))
    by = {}
    for r in results:
        by.setdefault((r["model"], r["foot"]), {})[r["fold"]] = r["accuracy"]
    for (model, foot), d in by.items():
        cells = " | ".join(f"{d[f]:.2f}" if f in d else "—" for f in args.folds)
        L.append(f"| {model} | {foot} | {cells} |")

    L.append("\n## Fusion — same foot\n\n```")
    L.append(fusion_text.strip() or "(not produced)")
    L.append("```")
    L.append("\n## Fusion — cross foot\n\n```")
    L.append(cross_text.strip() or "(not produced)")
    L.append("```")

    missing = [m for m in C.DL_MODELS
               if not any(k[0] == m for k in table)
               and not (args.fast and m in SLOW_MODELS) and not args.skip_dl]
    if missing:
        L.append("\n## Models that produced no result\n")
        L.append(", ".join(missing) + " — see `run_all_log.txt` for the traceback.")

    with open(path, "w") as fh:
        fh.write("\n".join(L) + "\n")


# --------------------------------------------------------------------------- #
def main():
    args = parse_args()
    C.TSFRESH_FC_PARAMETERS = args.fc_params
    C.TSFRESH_N_JOBS = args.tsfresh_jobs
    C.RUN_TAG = ""                     # untagged: this is the definitive run
    C.ensure_dirs()
    np.random.seed(C.SEED)

    sys.stdout = _Tee(os.path.join(C.RESULTS_DIR, "run_all_log.txt"))

    dl_models = [m for m in C.DL_MODELS
                 if not (args.fast and m in SLOW_MODELS)]

    print("=" * 96)
    print("RUN ALL — every pending experiment in one pass")
    print(f"  folds     : {args.folds}")
    print(f"  feet      : {args.feet}")
    print(f"  classical : {'skipped' if args.skip_classical else list(C.CLASSICAL_MODELS)}")
    print(f"  dl        : {'skipped' if args.skip_dl else dl_models}")
    if args.fast:
        print(f"  (--fast: skipping {', '.join(SLOW_MODELS)})")
    print("=" * 96)

    results: list[dict] = []
    t0 = time.time()

    for fold_id in args.folds:
        print(f"\n{'#' * 96}\n# FOLD {fold_id}\n{'#' * 96}")
        fold = load_fold(fold_id)
        for split in ("train", "val", "test"):
            subj, lab, _, _ = fold[split]
            print(f"  {split:5s}: {len(subj):3d} subjects "
                  f"({int((lab == 1).sum())} PD / {int((lab == 0).sum())} CO)")
        if not args.skip_classical:
            try:
                run_classical(fold_id, fold, args.feet,
                              list(C.CLASSICAL_MODELS), results)
            except Exception as exc:
                import traceback
                print(f"  !! classical failed on fold {fold_id}: {exc}")
                traceback.print_exc()
        if not args.skip_dl:
            run_dl(fold_id, fold, args.feet, dl_models, args.dl_quick, results)

    elapsed = time.time() - t0
    print(f"\nTotal model compute: {elapsed/60:.1f} min")

    if results:
        build_report(results)

    # ---- fusion (reads the scores just written) ----
    fusion_text = cross_text = ""
    try:
        from src.fusion import report_cross_foot, report_fusion
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            report_fusion(foot="Combined", max_size=3, top=15)
        fusion_text = buf.getvalue()
        print("\n" + fusion_text)

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            report_cross_foot(max_size=3, top=15)
        cross_text = buf.getvalue()
        print("\n" + cross_text)
    except Exception as exc:
        import traceback
        print(f"!! fusion failed: {exc}")
        traceback.print_exc()

    out = os.path.join(C.RESULTS_DIR, "FINAL_RESULTS.md")
    write_final(results, fusion_text, cross_text, elapsed, args, out)

    print("\n" + "=" * 96)
    print("DONE.  Send this one file:")
    print(f"   {out}")
    print(f"   (full log: {os.path.join(C.RESULTS_DIR, 'run_all_log.txt')})")
    print("=" * 96)


if __name__ == "__main__":
    main()
