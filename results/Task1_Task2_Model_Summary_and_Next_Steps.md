# Task 1 + Task 2 Model Summary and Next Steps

This document summarizes the project work completed so far, how the pipeline was implemented, what each model does, what results were obtained, and what should be done next. It is written as presentation notes, so you can use the structure directly when explaining the work.

## 1. Project Goal

The project is to classify Parkinson's disease status from gait signals. Each subject has force/time-series measurements from foot sensors. The target is binary classification:

- Control
- Parkinson's patient

The important correction made in the latest implementation is that the full machine-learning flow now respects the provided 5-fold split protocol:

1. Training split: fit feature selection, preprocessing, and model.
2. Validation split: choose hyperparameters.
3. Test split: final evaluation only.

This avoids using validation or test subjects while fitting the model. That was the main methodological issue we corrected.

## 2. Data and Fold Structure

The data is already divided into 5 folds:

| Fold | Training | Validation | Test |
|---|---:|---:|---:|
| Fold_1 | 60 | 20 | 20 |
| Fold_2 | 60 | 20 | 20 |
| Fold_3 | 60 | 20 | 20 |
| Fold_4 | 60 | 20 | 20 |
| Fold_5 | 60 | 20 | 20 |

Each fold has the same structure. For every fold, we build that fold's feature matrices independently:

- `training_feature_matrix_efficient.csv`
- `validation_feature_matrix_efficient.csv`
- `test_feature_matrix_efficient.csv`

Each matrix has:

- one row per subject
- 84,430 feature columns
- the subject ID and group label columns

The feature extraction is fixed and deterministic. The learned operations are still fit only on the training split.

## 3. Signal Preprocessing

The raw gait files are loaded using the project signal loader in `scripts/analyze_gait.py`.

Current setting:

- sampling rate: 100 Hz
- window size: 5 seconds
- step size: 2.5 seconds
- trimming: 5 seconds removed from the start and 5 seconds removed from the end before feature extraction

Why trimming is used:

- the start/end of walking trials can contain acceleration, stopping, or setup artifacts
- the central walking segment is usually more stable
- this follows a common gait-analysis assumption: avoid transitional walking phases when estimating steady gait rhythm/features

Important note for presentation:

- Earlier Task 1 analysis also considered the question of no trimming vs trimming.
- The current Task 2 model run uses the trimmed setting by default.
- A useful next ablation is to rerun the exact same fold-safe pipeline with `--trim-seconds 0` and compare results.

## 4. Feature Extraction

The current implementation uses TSFresh `EfficientFCParameters`.

Why EfficientFCParameters:

- `MinimalFCParameters` is fast but too small; it only extracts a limited set of simple statistics.
- `EfficientFCParameters` is much richer and closer to the comprehensive TSFresh feature set.
- It excludes very expensive TSFresh calculators, so it is more practical than the full comprehensive feature set.

### 4.1 Windowed TSFresh Features

For each subject:

1. The signal is divided into 5-second windows with 2.5-second overlap.
2. TSFresh features are computed for each window.
3. Window-level features are aggregated to subject-level features.

Aggregation statistics:

- window mean
- window standard deviation
- window skewness
- window kurtosis

The current feature extraction uses both left and right foot channels:

- left sensor channels
- right sensor channels
- total left force
- total right force

### 4.2 Engineered Gait Features

In addition to TSFresh, engineered gait features are added. These include:

- contact fraction
- contact count
- cadence per minute
- force summary statistics
- active-force summary statistics
- contact duration statistics
- swing duration statistics
- stride interval statistics
- peak force statistics
- impulse statistics
- center-of-pressure features
- left-right force symmetry
- double-support fraction
- no-support fraction
- left-only/right-only support fraction
- left-right correlation and lag

### 4.3 Wavelet Features

Wavelet features are also added using a Morlet continuous wavelet transform.

Signals used:

- left total force
- right total force
- left-right symmetry signal

Wavelet power is summarized across scale bands.

### 4.4 Feature Views / Scopes

Four feature views were evaluated:

| Scope | Meaning |
|---|---|
| `left` | left-foot TSFresh features |
| `right` | right-foot TSFresh features |
| `combined` | left + right TSFresh features |
| `enhanced` | left + right TSFresh + asymmetry + gait + wavelet features |

This helps answer whether one foot, both feet, or engineered features give the best classification performance.

## 5. Why Subject-Level Feature Cache Is Valid

Efficient TSFresh feature extraction is expensive. To avoid recomputing the same subject repeatedly across folds, subject-level feature rows are cached in:

`results/subject_feature_cache/efficient/trim_5s/`

This does not cause leakage because:

- EfficientFCParameters is a fixed feature calculator.
- It does not learn from labels.
- It does not use validation/test distributions to fit parameters.
- The learned steps are still fit inside the fold:
  - imputer
  - variance filter
  - z-score scaler
  - feature selection
  - classifier

So the cache only avoids repeated deterministic computation.

## 6. Correct Fold-Safe Training Flow

For every fold and every model:

1. Load the fold's training, validation, and test feature matrices.
2. Select one feature scope: left, right, combined, or enhanced.
3. Fit preprocessing on training only.
4. Tune model hyperparameters using validation only.
5. Evaluate the chosen model on test only.
6. Store fold-level metrics and predictions.

This is the corrected pipeline:

```text
Fold_i
  Training:
    fit imputer
    fit variance filter
    fit z-score scaler
    fit feature selection
    fit classifier

  Validation:
    transform using training-fitted steps
    evaluate hyperparameter candidates
    select best candidate

  Test:
    transform using training-fitted steps
    evaluate only the final selected model
```

## 7. Preprocessing and Feature Selection

Two feature-selection approaches were tested.

### 7.1 Standard SelectKBest Path

Used for the main classical baseline and neural feature-vector models.

Pipeline:

```text
Median imputation
VarianceThreshold
StandardScaler z-score normalization
SelectKBest(f_classif)
Classifier
```

Candidate `k_best` values for classical models:

- 25
- 50
- 100
- 200
- 400
- all

The exact set depends on how many features remain after variance filtering.

### 7.2 TSFresh `select_features` Path

TSFresh provides its own statistical feature relevance selection. This was tested separately.

Pipeline:

```text
Median imputation
VarianceThreshold
StandardScaler z-score normalization
tsfresh.select_features
Classifier
```

Candidate false-discovery-rate levels:

- 0.05
- 0.10
- 0.20
- 0.50

This answers the question: is TSFresh's built-in feature selection useful?

Conclusion:

- Yes, it is useful in some views.
- It gave the best result for the left-foot view.
- It gave one of the best enhanced-feature results.
- It did not universally beat `SelectKBest`.
- The best combined-view result still came from standard `SelectKBest + Linear SVM`.

## 8. Models Implemented

All models now use feature vectors, not raw signals.

Mamba has not been implemented yet, per the earlier decision.

### 8.1 Classical Models

Three classical classifiers were tested:

1. Random Forest
2. Linear SVM
3. RBF SVM

All use balanced class weights where applicable.

#### Random Forest Grid

```text
n_estimators: 400
max_depth: None, 8, 16
min_samples_leaf: 1, 3
max_features: sqrt
class_weight: balanced
```

#### Linear SVM Grid

```text
kernel: linear
C: 0.001, 0.01, 0.1, 1, 10, 100
class_weight: balanced
```

#### RBF SVM Grid

```text
kernel: rbf
C: 0.1, 1, 10, 100
gamma: scale, 0.001, 0.01, 0.1
class_weight: balanced
```

### 8.2 Feature-Vector Neural Models

The neural models also use TSFresh feature vectors instead of raw signals.

Models tested:

1. 1D ResNet over feature vector
2. LSTM over feature vector sequence
3. GRU over feature vector sequence
4. PatchTST-style transformer over feature vector patches

Training setup:

```text
optimizer: AdamW
loss: BCEWithLogitsLoss with positive-class weighting
max_epochs: 6
patience: 2
batch_size: 16
selection metric: validation F1
```

Neural hyperparameter grid:

```text
k_features: 100, 200, 400
learning_rate: 0.0003, 0.0008
weight_decay: 0.0001, 0.001
```

Validation selects:

- number of selected features
- learning rate
- weight decay
- best epoch via early stopping

## 9. Evaluation Metrics

The required Task 2 metrics are:

- Accuracy
- Sensitivity
- Specificity
- AUC

Additional metrics are also saved:

- Precision
- F1-score
- confusion matrix values: TN, FP, FN, TP

Definitions:

```text
Accuracy    = correct predictions / all predictions
Sensitivity = TP / (TP + FN)
Specificity = TN / (TN + FP)
AUC         = area under ROC curve using model score
```

For this project:

- Sensitivity tells how well patients are detected.
- Specificity tells how well controls are detected.
- AUC tells how well the model ranks patients above controls, independent of a fixed threshold.

## 10. Final Results

### 10.1 Best Model by Feature Scope

| Scope | Best model | Accuracy | Sensitivity | Specificity | AUC | F1 |
|---|---|---:|---:|---:|---:|---:|
| combined | Linear SVM | 83.0% | 86.0% | 80.0% | 0.892 | 0.834 |
| enhanced | Random Forest + TSFresh selection | 83.0% | 82.0% | 84.0% | 0.901 | 0.820 |
| left | Linear SVM + TSFresh selection | 82.0% | 82.0% | 82.0% | 0.912 | 0.810 |
| right | Random Forest | 83.0% | 86.0% | 80.0% | 0.902 | 0.835 |

### 10.2 Best Overall Interpretation

The strongest current non-Mamba baseline is:

```text
Efficient TSFresh feature extraction
+ fold-safe preprocessing
+ Linear SVM or Random Forest
```

The highest accuracy is 83.0%.

Multiple models reach 83.0%, so the final choice depends on what we want to emphasize:

- If emphasizing simple and interpretable baseline: use combined Linear SVM.
- If emphasizing best AUC with balanced left/right results: use left Linear SVM + TSFresh selection.
- If emphasizing engineered features and TSFresh relevance selection: use enhanced Random Forest + TSFresh selection.
- If emphasizing right-foot-only performance: use right Random Forest.

### 10.3 Comparison to Professor Reference

The professor's meeting slide mentioned approximately:

- Linear SVM around 80%
- RBF SVM around 80.6%
- recurrent/deep models higher

Our corrected fold-safe Efficient-feature pipeline now achieves:

- Linear SVM combined: 83.0%
- Random Forest right/enhanced: 83.0%
- best AUC: 0.912

This is now above the 80% Linear SVM reference while using the corrected train/validation/test separation.

## 11. Fold-Level Behavior of Best Models

### 11.1 Combined Linear SVM

| Fold | Validation accuracy | Test accuracy | Sensitivity | Specificity | AUC | Selected params |
|---|---:|---:|---:|---:|---:|---|
| Fold_1 | 80.0% | 90.0% | 80.0% | 100.0% | 0.930 | C=0.01, k=100 |
| Fold_2 | 90.0% | 85.0% | 90.0% | 80.0% | 0.930 | C=0.1, k=100 |
| Fold_3 | 95.0% | 75.0% | 70.0% | 80.0% | 0.820 | C=0.01, k=50 |
| Fold_4 | 90.0% | 80.0% | 90.0% | 70.0% | 0.890 | C=0.01, k=25 |
| Fold_5 | 85.0% | 85.0% | 100.0% | 70.0% | 0.890 | C=0.001, k=25 |

Interpretation:

- Performance is strong on most folds.
- Fold_3 is the weakest fold for this model.
- Selected `k_best` varies, meaning different folds need different feature counts.
- Smaller C values are preferred, suggesting regularization is important.

### 11.2 Enhanced Random Forest + TSFresh Selection

| Fold | Validation accuracy | Test accuracy | Sensitivity | Specificity | AUC | Selected features |
|---|---:|---:|---:|---:|---:|---:|
| Fold_1 | 75.0% | 90.0% | 80.0% | 100.0% | 0.890 | 412 |
| Fold_2 | 90.0% | 80.0% | 90.0% | 70.0% | 0.900 | 1,630 |
| Fold_3 | 90.0% | 70.0% | 50.0% | 90.0% | 0.825 | 1,181 |
| Fold_4 | 85.0% | 90.0% | 100.0% | 80.0% | 0.950 | 386 |
| Fold_5 | 90.0% | 85.0% | 90.0% | 80.0% | 0.940 | 1,172 |

Interpretation:

- This model has the best enhanced-feature result.
- It performs very well on Folds 1, 4, and 5.
- Fold_3 again appears difficult.
- TSFresh selected between 386 and 1,630 features depending on fold.

### 11.3 Left Linear SVM + TSFresh Selection

| Fold | Validation accuracy | Test accuracy | Sensitivity | Specificity | AUC | Selected features |
|---|---:|---:|---:|---:|---:|---:|
| Fold_1 | 80.0% | 75.0% | 60.0% | 90.0% | 0.870 | 391 |
| Fold_2 | 85.0% | 80.0% | 90.0% | 70.0% | 0.900 | 855 |
| Fold_3 | 95.0% | 75.0% | 60.0% | 90.0% | 0.870 | 551 |
| Fold_4 | 85.0% | 90.0% | 100.0% | 80.0% | 0.960 | 617 |
| Fold_5 | 85.0% | 90.0% | 100.0% | 80.0% | 0.960 | 79 |

Interpretation:

- This model gives the best AUC: 0.912 mean AUC.
- It has balanced mean sensitivity and specificity: 82.0% / 82.0%.
- It is a good model to mention if asked about ranking performance rather than just threshold accuracy.

### 11.4 Right Random Forest

| Fold | Validation accuracy | Test accuracy | Sensitivity | Specificity | AUC | Selected params |
|---|---:|---:|---:|---:|---:|---|
| Fold_1 | 75.0% | 90.0% | 80.0% | 100.0% | 0.890 | k=200, leaf=3 |
| Fold_2 | 90.0% | 75.0% | 90.0% | 60.0% | 0.870 | k=100, leaf=3 |
| Fold_3 | 90.0% | 75.0% | 70.0% | 80.0% | 0.840 | k=25, leaf=3 |
| Fold_4 | 85.0% | 85.0% | 90.0% | 80.0% | 0.930 | k=100, leaf=1 |
| Fold_5 | 90.0% | 90.0% | 100.0% | 80.0% | 0.980 | k=200, leaf=1 |

Interpretation:

- Strongest right-foot-only result.
- Mean accuracy is 83.0%.
- Mean sensitivity is high at 86.0%.
- Specificity is 80.0%, so it detects patients slightly better than controls.

## 12. TSFresh Feature Selection Usefulness

The TSFresh selector is useful, but not a universal replacement for SelectKBest.

### Where TSFresh Selection Helped

| Scope | TSFresh-selected model | Accuracy | AUC |
|---|---|---:|---:|
| enhanced | Random Forest + TSFresh selection | 83.0% | 0.901 |
| left | Linear SVM + TSFresh selection | 82.0% | 0.912 |
| right | Random Forest + TSFresh selection | 83.0% | 0.896 |

### Where SelectKBest Was Still Better

| Scope | SelectKBest model | Accuracy | AUC |
|---|---|---:|---:|
| combined | Linear SVM | 83.0% | 0.892 |
| combined | Random Forest | 82.0% | 0.892 |

Presentation explanation:

> TSFresh feature selection is statistically motivated and helpful when the relevant features are sparse and fold-specific. However, in the combined feature view, the simpler SelectKBest path with Linear SVM gave the best accuracy. Therefore, we kept both paths and compared them empirically.

## 13. Neural Model Results

The neural models were tested on extracted feature vectors, not raw signals.

| Scope | Model | Accuracy | Sensitivity | Specificity | AUC | F1 |
|---|---|---:|---:|---:|---:|---:|
| right | PatchTST | 79.0% | 86.0% | 72.0% | 0.888 | 0.803 |
| combined | PatchTST | 78.0% | 82.0% | 74.0% | 0.866 | 0.789 |
| enhanced | PatchTST | 77.0% | 82.0% | 72.0% | 0.866 | 0.779 |
| combined | GRU | 77.0% | 78.0% | 76.0% | 0.826 | 0.767 |
| right | ResNet1D | 76.0% | 82.0% | 70.0% | 0.832 | 0.773 |

Conclusion:

- PatchTST is the best neural feature-vector model.
- Neural models improved with hyperparameter tuning.
- But neural feature-vector models still do not beat the classical SVM/RF models.

Why neural models may underperform:

- Only 60 training subjects per fold.
- Feature vectors are high-dimensional but sample size is small.
- Neural models can overfit quickly.
- Classical models are often stronger for small tabular datasets.
- The neural models are currently using feature vectors, not raw temporal structure.

## 14. What Has Been Completed

### Task 1

Completed:

- demographic summary
- fold distribution summary
- Efficient TSFresh feature extraction
- feature matrix generation for all folds
- PCA/t-SNE notebook sections with train-fitted z-score normalization
- notebook for Task 1 + Task 2

Main notebook:

`Task1_Task2_Efficient_FoldSafe.ipynb`

### Task 2

Completed:

- corrected 5-fold train/validation/test protocol
- Efficient TSFresh feature extraction
- feature-vector classical models
- TSFresh feature selection comparison
- feature-vector neural models
- neural hyperparameter grid
- final metrics: accuracy, sensitivity, specificity, AUC
- result tables and figures
- presentation deck

Main scripts:

- `scripts/run_deliverable2.py`
- `scripts/run_deliverable2_foldsafe.py`

Main presentation:

- `results/Task1_Task2_Efficient_FoldSafe_Results.pptx`

Main summary tables:

- `results/tables/deliverable2_model_summary.csv`
- `results/tables/deliverable2_best_models_by_scope.csv`
- `results/tables/deliverable2_feature_selection_comparison.csv`
- `results/tables/deliverable2_fold_metrics.csv`

## 15. How to Explain This in Presentation

Suggested speaking flow:

### Slide 1: Problem

> The goal is to detect Parkinson's disease from gait signals. The input is force/time-series data from foot sensors, and the output is binary classification: control vs patient.

### Slide 2: Corrected Evaluation Protocol

> The most important correction is fold safety. For each fold, I fit all learned steps only on the training split. The validation split is used only for hyperparameter tuning. The test split is held out until final evaluation.

### Slide 3: Feature Extraction

> I used TSFresh EfficientFCParameters with 5-second windows and 2.5-second step size. The window-level features are aggregated to subject-level features. I also added gait-specific engineered features such as cadence, contact duration, stride intervals, force symmetry, double support, and wavelet features.

### Slide 4: Feature Selection

> I tested two selection strategies: standard SelectKBest and TSFresh's built-in select_features. TSFresh selection helped in left and enhanced views, but SelectKBest was still best for combined Linear SVM.

### Slide 5: Models

> I evaluated Random Forest, Linear SVM, RBF SVM, and feature-vector neural models including ResNet1D, LSTM, GRU, and PatchTST. All models used feature vectors, not raw signals.

### Slide 6: Results

> The best mean test accuracy is 83.0%. Combined Linear SVM achieved 83.0% accuracy and 0.892 AUC. Enhanced Random Forest with TSFresh selection also achieved 83.0% accuracy and 0.901 AUC. Left Linear SVM with TSFresh selection achieved the best AUC, 0.912.

### Slide 7: Interpretation

> The classical models are currently stronger than neural feature-vector models. This is expected because the dataset is small, with only 60 training subjects per fold, and classical models often perform better on small tabular feature sets.

### Slide 8: Next Steps

> The next steps are ablation studies, threshold tuning, feature importance analysis, raw-signal deep models, and eventually Mamba once the baseline is fully defensible.

## 16. Key Takeaways

1. The corrected pipeline is now fold-safe.
2. Efficient TSFresh features improved the results.
3. The best accuracy is 83.0%, above the professor's 80% Linear SVM reference.
4. The best AUC is 0.912.
5. TSFresh feature selection is useful, but not always better than SelectKBest.
6. Classical models currently beat feature-vector neural models.
7. The project is ready for ablation, interpretation, and then raw-signal/Mamba modeling.

## 17. Recommended Next Steps

### Step 1: Run Trim vs No-Trim Ablation

Current model uses 5-second trimming at start and end.

Recommended experiment:

```bash
.venv/bin/python scripts/run_deliverable2.py --feature-set efficient --trim-seconds 0 --selection-metric f1 --max-epochs 6 --patience 2 --batch-size 16 --deep-k-features 100 200 400 --deep-learning-rates 0.0003 0.0008 --deep-weight-decays 0.0001 0.001
```

Compare:

- trim 5s
- trim 0s

This will answer whether removing start/end data improves or hurts the final model.

### Step 2: Compare Feature Sets

Run the same fold-safe protocol with:

- MinimalFCParameters
- curated feature set
- EfficientFCParameters

This gives a clean table:

| Feature set | Runtime | Feature count | Best accuracy | Best AUC |
|---|---:|---:|---:|---:|
| minimal | fast | low | TBD | TBD |
| curated | medium | medium | TBD | TBD |
| efficient | slow | high | 83.0% | 0.912 |

This will justify why Efficient was chosen.

### Step 3: Threshold Tuning on Validation

Right now predictions use a default threshold of 0.5 for models with probability/score output.

Recommended:

- choose threshold on validation split
- optimize balanced accuracy or F1
- apply selected threshold to test split

This may improve sensitivity/specificity balance.

Important: threshold must be selected on validation only, never test.

### Step 4: Add Confidence Intervals

Current report shows fold mean and standard deviation.

Recommended:

- bootstrap confidence intervals across test predictions
- or report 95% confidence intervals across folds

This makes the results more presentation- and publication-ready.

### Step 5: Feature Importance and Interpretation

For Random Forest:

- permutation importance
- impurity-based feature importance

For Linear SVM:

- absolute coefficient ranking after selected features

For TSFresh selected features:

- count which feature calculators appear most often
- identify whether force, rhythm, asymmetry, or wavelet features dominate

This is important for explaining why the model works, not just what accuracy it gets.

### Step 6: Error Analysis

Create a table of subjects that are repeatedly misclassified across folds.

For each misclassified subject:

- subject ID
- true group
- predicted group
- model score
- fold
- demographics if available

This helps answer:

- Are certain patients control-like?
- Are certain controls patient-like?
- Are errors related to age, gender, walking speed, or trial quality?

### Step 7: Demographics-Aware Analysis

Check whether age, gender, height, or weight differ between groups or folds.

Recommended:

- include demographics-only baseline
- include gait + demographics model
- compare whether demographics improve AUC

Be careful: if demographics are included, imputation/scaling must also be fit training-only.

### Step 8: Raw-Signal Deep Learning Baselines

Once feature-vector baselines are finalized, add raw-signal models:

- CNN/ResNet on raw force channels
- LSTM/GRU on raw sequences
- PatchTST on raw windows

This may help recover temporal information lost during feature aggregation.

### Step 9: Implement Mamba

Do this after the baseline is locked.

Recommended Mamba setup:

- input: raw multichannel gait signal or windowed signal
- train fold only
- validation for hyperparameters/early stopping
- test final evaluation

Compare Mamba against:

- combined Linear SVM: 83.0%
- enhanced Random Forest + TSFresh: 83.0%
- right PatchTST feature-vector neural: 79.0%

Mamba should be presented only if it beats or meaningfully complements the current baselines.

### Step 10: Prepare Final Deliverable Tables

For the final report/presentation, include:

1. Fold protocol table
2. Feature extraction table
3. Feature-selection comparison table
4. Best model table
5. Neural model table
6. Confusion matrices
7. Next-step ablation table

## 18. Recommended Final Position for Presentation

Use this as your main conclusion:

> After correcting the train/validation/test protocol, Efficient TSFresh features with classical classifiers provide a strong and defensible non-Mamba baseline. The best models reach 83.0% mean test accuracy, above the 80% Linear SVM reference, while maintaining fold-safe hyperparameter tuning. TSFresh feature selection is useful in some views, especially left and enhanced features, but the best simple baseline remains combined Efficient features with Linear SVM.

## 19. Important Files

| Purpose | File |
|---|---|
| Main notebook | `Task1_Task2_Efficient_FoldSafe.ipynb` |
| Main runner | `scripts/run_deliverable2.py` |
| Fold-safe implementation | `scripts/run_deliverable2_foldsafe.py` |
| Signal/feature helpers | `scripts/analyze_gait.py` |
| Final PPT | `results/Task1_Task2_Efficient_FoldSafe_Results.pptx` |
| Model summary CSV | `results/tables/deliverable2_model_summary.csv` |
| Fold-level metrics | `results/tables/deliverable2_fold_metrics.csv` |
| Best models by scope | `results/tables/deliverable2_best_models_by_scope.csv` |
| Feature-selection comparison | `results/tables/deliverable2_feature_selection_comparison.csv` |
| Neural fold metrics | `results/tables/deliverable2_feature_neural_fold_metrics.csv` |
| Feature audit | `results/tables/deliverable2_fold_safe_feature_audit.csv` |
| Summary figure | `results/figures/deliverable2_metric_summary.png` |
| Confusion matrices | `results/figures/deliverable2_best_confusion_matrices.png` |

