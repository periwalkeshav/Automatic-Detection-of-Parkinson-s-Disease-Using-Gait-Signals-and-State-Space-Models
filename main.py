"""
Parkinson's-disease gait classification, baseline models.

Runs the classical branch (Random Forest, SVM-Linear, SVM-RBF, Logistic
Regression, Extra Trees on aggregated TSFresh features) and/or the deep-learning
branch (1D-ResNet, LSTM under gated-attention MIL on raw windowed signals) over
the 5 predefined folds, for the Left / Right / Combined foot representations, and
writes a report (CSVs + confusion matrices) under results/.

Examples
--------
  python main.py --run both                 # everything (default)
  python main.py --run classical            # only RF / SVM-Linear / SVM-RBF
  python main.py --run dl --dl-quick        # only DL, single learning rate (fast)
  python main.py --run classical --fc-params minimal   # fast TSFresh smoke test
  python main.py --run both --folds 1       # a single fold
  python main.py --run dl --feet Combined
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from src import config as C
from src.data import load_fold
from src.report import build_report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Parkinson's gait — baseline models")
    p.add_argument("--run", choices=["classical", "dl", "both", "fusion"],
                   default="both",
                   help="Which model family to run (default: both). 'fusion' "
                        "re-uses saved scores from earlier runs — it trains nothing.")
    p.add_argument("--fusion-foot", default="Combined", choices=list(C.FEET),
                   help="Foot representation for --run fusion (default: Combined).")
    p.add_argument("--fusion-max-size", type=int, default=3,
                   help="Largest model combination to try in --run fusion.")
    p.add_argument("--fusion-dl-only", action="store_true",
                   help="Restrict the fusion leaderboard to deep/SSM models, so "
                        "the best achievable DEEP result can be reported "
                        "separately from the classical models.")
    p.add_argument("--fusion-cross-foot", action="store_true",
                   help="Also fuse ACROSS feet (Left/Right/Combined are different "
                        "views and make different errors). Entries may be given as "
                        "Model@Foot, e.g. --fusion-models RandomForest@Left "
                        "RandomForest@Right RandomForest@Combined")
    p.add_argument("--fusion-models", nargs="+", default=None,
                   help="Report this exact combination instead of the leaderboard "
                        "best. Prefer this: committing to a combination up front "
                        "avoids selecting one on the test folds. "
                        "e.g. --fusion-models RandomForest SVM-Linear Mamba")
    p.add_argument("--folds", nargs="+", type=int,
                   default=list(range(1, C.N_FOLDS + 1)),
                   help="Fold ids to run (default: 1..5).")
    p.add_argument("--feet", nargs="+", default=list(C.FEET), choices=list(C.FEET),
                   help="Foot representations to evaluate (default: Left Right Combined).")
    p.add_argument("--classical-models", nargs="+", default=list(C.CLASSICAL_MODELS),
                   choices=list(C.CLASSICAL_MODELS), help="Subset of classical models.")
    p.add_argument("--dl-models", nargs="+", default=list(C.DL_MODELS),
                   choices=list(C.DL_MODELS) + list(C.DL_CONTROL_MODELS),
                   help="Subset of DL models (DL_CONTROL_MODELS run only when named).")
    p.add_argument("--fc-params", choices=["minimal", "efficient", "comprehensive"],
                   default=C.TSFRESH_FC_PARAMETERS,
                   help="TSFresh feature set (classical branch).")
    p.add_argument("--tsfresh-jobs", type=int, default=C.TSFRESH_N_JOBS,
                   help="TSFresh worker processes (0 = single process).")
    p.add_argument("--dl-quick", action="store_true",
                   help="Single learning rate for DL (skip the LR grid).")
    p.add_argument("--tag", default="",
                   help="Label appended to the output filenames, e.g. --tag mixup "
                        "writes results_summary_dl_mixup.csv. Use it so ablation "
                        "runs do not overwrite each other.")
    return p.parse_args()


def run_classical(fold_id, fold, feet, model_names, results) -> None:
    from src.features import build_feature_matrix
    from src.models_classical import model_specs, train_eval_classical
    from src.preprocessing import prepare_classical_fold

    tr_subj, tr_lab, tr_l, tr_r = fold["train"]
    va_subj, va_lab, va_l, va_r = fold["val"]
    te_subj, te_lab, te_l, te_r = fold["test"]

    print(f"  Building TSFresh feature matrices (fold {fold_id}) …")
    xtr = build_feature_matrix(tr_subj, tr_l, tr_r)
    xva = build_feature_matrix(va_subj, va_l, va_r)
    xte = build_feature_matrix(te_subj, te_l, te_r)
    mats = {"Left": (xtr[0], xva[0], xte[0]),
            "Right": (xtr[1], xva[1], xte[1]),
            "Combined": (xtr[2], xva[2], xte[2])}

    specs = [s for s in model_specs() if s.name in model_names]
    for foot in feet:
        x_tr, x_val, x_te = mats[foot]
        prep = prepare_classical_fold(x_tr, tr_lab, x_val, x_te)
        print(f"    [Fold {fold_id} | {foot:8s}] features {x_tr.shape[1]} "
              f"-> selected {prep.n_selected} -> PCA {prep.n_pca}")
        for spec in specs:
            results.append(
                train_eval_classical(spec, prep, tr_lab, va_lab, te_lab, fold_id, foot))


def run_dl(fold_id, fold, feet, model_names, quick, results) -> None:
    from src.mamba_block import mamba_backend
    from src.train_dl import (DEVICE, feature_source, needs_aggregate_features,
                              needs_feature_sequence, needs_sequence,
                              needs_stride_windows, prepare_dl_aggregate,
                              prepare_dl_feature_sequence, prepare_dl_foot,
                              prepare_dl_sequence, prepare_dl_stride_windows,
                              train_eval_dl)

    print(f"  Deep-learning branch (fold {fold_id}, device={DEVICE}) …")
    if any(needs_sequence(m) or needs_feature_sequence(m) or m == "MambaWin"
           for m in model_names):
        print(f"    Mamba backend: {mamba_backend()}")
    for foot in feet:
        win_data = seq_data = stride_data = agg_data = None  # built lazily
        feat_data = {}                           # one entry per feature source
        for model_name in model_names:
            if needs_aggregate_features(model_name):
                if agg_data is None:
                    agg_data = prepare_dl_aggregate(fold, foot)
                    print(f"    [Fold {fold_id} | {foot:8s}] aggregated features "
                          f"train={len(agg_data['train'][0])} "
                          f"dim={agg_data['in_features']}")
                data = agg_data
            elif needs_feature_sequence(model_name):
                src = feature_source(model_name)
                if src not in feat_data:
                    kind = "stride" if src == "gait" else "TSFresh window"
                    print(f"    [Fold {fold_id} | {foot:8s}] building {kind} "
                          f"sequences …")
                    feat_data[src] = prepare_dl_feature_sequence(fold, foot, src)
                    lens = [m.shape[0] for m in feat_data[src]["train"][0]]
                    print(f"    [Fold {fold_id} | {foot:8s}] {src} sequences "
                          f"train={len(lens)} tokens={min(lens)}–{max(lens)} "
                          f"features={feat_data[src]['in_features']}")
                data = feat_data[src]
            elif needs_stride_windows(model_name):
                if stride_data is None:
                    stride_data = prepare_dl_stride_windows(fold, foot)
                    x_tr = stride_data["train"][0]
                    print(f"    [Fold {fold_id} | {foot:8s}] strides train="
                          f"{x_tr.shape[0]} channels={stride_data['in_channels']} "
                          f"len={x_tr.shape[2]} (time-normalised)")
                data = stride_data
            elif needs_sequence(model_name):
                if seq_data is None:
                    seq_data = prepare_dl_sequence(fold, foot)
                    lens = [s.shape[1] for s in seq_data["train"][0]]
                    print(f"    [Fold {fold_id} | {foot:8s}] sequences train="
                          f"{len(lens)} channels={seq_data['in_channels']} "
                          f"T={min(lens)}–{max(lens)} crop={int(C.DL_MAMBA_CROP_S)}s")
                data = seq_data
            else:
                if win_data is None:
                    win_data = prepare_dl_foot(fold, foot)
                    x_tr = win_data["train"][0]
                    print(f"    [Fold {fold_id} | {foot:8s}] windows train="
                          f"{x_tr.shape[0]} channels={win_data['in_channels']} "
                          f"T={x_tr.shape[2]}")
                data = win_data
            try:
                results.append(
                    train_eval_dl(model_name, data, fold_id, foot, quick=quick))
            except Exception as exc:          # keep a long unattended run alive
                import traceback
                print(f"      !! {model_name} failed on fold {fold_id}/{foot}: "
                      f"{type(exc).__name__}: {exc}")
                traceback.print_exc()


def main() -> None:
    args = parse_args()
    C.TSFRESH_FC_PARAMETERS = args.fc_params
    C.TSFRESH_N_JOBS = args.tsfresh_jobs
    C.RUN_TAG = args.tag
    C.ensure_dirs()
    np.random.seed(C.SEED)

    if args.run == "fusion":
        import contextlib
        import io
        import os

        from src.fusion import (evaluate_fusion, evaluate_fusion_entries,
                                load_scores, report_cross_foot, report_fusion)
        print("=" * 96)
        print("Late fusion of saved model scores (no training)")
        print("=" * 96)
        dl_only = list(C.DL_MODELS) if args.fusion_dl_only else None

        def _capture(fn, **kw):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                out = fn(**kw)
            print(buf.getvalue())
            return out, buf.getvalue()

        ranked, same_txt = _capture(report_fusion, foot=args.fusion_foot,
                                    max_size=args.fusion_max_size, only=dl_only)
        cross_txt = ""
        cross_ranked = None
        if args.fusion_cross_foot:
            cross_ranked, cross_txt = _capture(
                report_cross_foot, max_size=args.fusion_max_size, only=dl_only)

        # Persist the leaderboards, so a fusion run always leaves a file behind.
        suffix = ("_dl" if args.fusion_dl_only else "") + \
                 (f"_{C.RUN_TAG}" if C.RUN_TAG else "")
        out_md = os.path.join(C.RESULTS_DIR, f"FUSION{suffix}.md")
        with open(out_md, "w") as fh:
            fh.write(f"# Fusion results ({'deep models only' if dl_only else 'all models'})\n\n")
            fh.write(f"foot: {args.fusion_foot} · max combination size: "
                     f"{args.fusion_max_size}\n\n## Same-foot\n\n```\n")
            fh.write(same_txt.strip() + "\n```\n")
            if cross_txt:
                fh.write("\n## Cross-foot\n\n```\n" + cross_txt.strip() + "\n```\n")

        # CSVs for whichever combination is being reported.
        scores = load_scores()
        folds = sorted({k[2] for k in scores})
        rows = None
        if args.fusion_models:
            rows = (evaluate_fusion_entries(scores, args.fusion_models, folds,
                                            default_foot=args.fusion_foot)
                    if args.fusion_cross_foot
                    else evaluate_fusion(scores, args.fusion_models,
                                         args.fusion_foot, folds))
            if rows:
                print(f"\nCommitted combination: {' + '.join(args.fusion_models)}")
        else:
            best = (cross_ranked or ranked)
            if best:
                rows = best[0][2]
                print("\nNote: the leaderboard is ranked on the test folds. For "
                      "final results, pick a combination up front with "
                      "--fusion-models rather than taking the top row.")
        if rows:
            if args.fusion_dl_only:
                C.RUN_TAG = f"{C.RUN_TAG}_dl".lstrip("_") if C.RUN_TAG else "dl"
            build_report(rows)
        print(f"\nLeaderboards written to: {out_md}")
        return

    do_classical = args.run in ("classical", "both")
    do_dl = args.run in ("dl", "both")

    print("=" * 96)
    print("Parkinson's gait baseline models")
    print(f"  run={args.run}  folds={args.folds}  feet={args.feet}")
    if do_classical:
        print(f"  classical models = {args.classical_models}  (TSFresh: {args.fc_params})")
    if do_dl:
        print(f"  DL models        = {args.dl_models}  (dl_quick={args.dl_quick})")
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
        if do_classical:
            run_classical(fold_id, fold, args.feet, args.classical_models, results)
        if do_dl:
            run_dl(fold_id, fold, args.feet, args.dl_models, args.dl_quick, results)

    print(f"\nTotal compute time: {(time.time() - t0) / 60.0:.1f} min")
    build_report(results)


if __name__ == "__main__":
    main()
