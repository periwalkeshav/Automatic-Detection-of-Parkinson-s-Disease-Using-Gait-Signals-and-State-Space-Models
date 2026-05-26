# Project Time Series: Automatic Detection of Parkinson’s Disease Using Gait Signals and State Space Models

This folder contains the non-Mamba foundation for the project described in
`../Meeting_1_TSI_Project_SS26.pptx` and the screenshot notes.

## Current Scope

Implemented:

- dataset and demographics summaries
- 5-fold train/validation/test manifest checks
- TSFresh feature extraction on windowed gait signals
- engineered gait timing, COP, left-right asymmetry, and wavelet features
- PCA plots for left foot, right foot, and combined left+right features
- baseline classifiers: Random Forest, Extra Trees, sparse logistic regression, linear SVM, and RBF SVM

Not implemented yet:

- Mamba / SSM models

## Data Protocol

The data is already divided into five folds:

- `training`: 60 subjects per fold, 30 controls and 30 PD patients
- `validation`: 20 subjects per fold, 10 controls and 10 PD patients
- `test`: 20 subjects per fold, 10 controls and 10 PD patients

Baseline model selection uses the validation split. Final metrics are reported
on the test split for each fold.

## Feature Pipeline

Signals are sampled at 100 Hz. By default, 5 seconds are removed from the start
and end of every recording before windowing to avoid gait-initiation and
stopping artifacts. Features are computed from 5-second windows with a
2.5-second step. For each subject and foot, a curated TSFresh feature set is
computed per window, then every resulting feature is summarized across windows
by:

- mean
- standard deviation
- skewness
- kurtosis

The combined-feet representation concatenates left-foot and right-foot feature
vectors. The enhanced representation also adds paired left-right TSFresh
absolute differences plus gait timing, contact, force symmetry, center of
pressure descriptors, and Morlet wavelet power summaries for left, right, and
left-right symmetry signals.

## Outputs

- `results/Deliverable_1_Data_Analysis_Draft.pptx`: updated draft slides
- `results/tables/fold_manifest.csv`: all fold/split assignments
- `results/tables/fold_split_summary.csv`: fold counts by split and group
- `results/tables/demographics_summary.csv`: patient/control counts, age, sex
- `results/tables/subject_feature_matrix.csv`: TSFresh subject vectors
- `results/tables/gait_engineered_features.csv`: contact, COP, and symmetry features
- `results/tables/wavelet_engineered_features.csv`: CWT/Morlet power features
- `results/tables/pca_*.csv`: PCA coordinates and explained variance
- `results/tables/baseline_fold_metrics.csv`: per-fold validation/test metrics
- `results/tables/baseline_summary.csv`: mean/std baseline metrics over folds
- `results/figures/*.png` and `*.svg`: example signals and PCA plots

## Reproduce

Create and populate the virtual environment:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install -r requirements.txt
```

Run the analysis and baselines:

```bash
.venv/bin/python scripts/analyze_gait.py --force-features
.venv/bin/python scripts/run_baselines.py
NODE_PATH=/Users/keshavperiwal/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules /Users/keshavperiwal/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node scripts/build_deck.mjs
```

