"""
Central configuration: paths, constants, signal/window settings, and the
hyper-parameter grids used by the classical and deep-learning branches.

A few fields (``TSFRESH_FC_PARAMETERS``, ``TSFRESH_N_JOBS``) are overwritten by
the CLI in ``main.py`` before anything else runs.
"""
from __future__ import annotations

import os

import numpy as np

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLITS_DIR = os.path.join(BASE_DIR, "splits")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
CACHE_DIR = os.path.join(RESULTS_DIR, "feature_cache")


def ensure_dirs() -> None:
    for d in (RESULTS_DIR, FIG_DIR, CACHE_DIR):
        os.makedirs(d, exist_ok=True)


# --------------------------------------------------------------------------- #
# General
# --------------------------------------------------------------------------- #
SEED = 42
N_FOLDS = 5

# Optional label appended to result filenames (set by main.py --tag), so repeated
# runs/ablations do not overwrite each other's CSVs and figures.
RUN_TAG = ""
FEET = ("Left", "Right", "Combined")
CLASSICAL_MODELS = ("RandomForest", "SVM-Linear", "SVM-RBF",
                    "LogisticRegression", "ExtraTrees")
DL_MODELS = ("ResNet1D", "LSTM", "Mamba", "MambaFeat", "MambaGait",
             "ResNetGait", "MLPFeat")

# Matched-context controls. The two raw-signal models differ in context as well
# as architecture: 1D-ResNet sees 5 s windows pooled by attention-MIL, Mamba sees
# 30 s crops. These swap the context, so each architecture is also measured on
# the other one's input with everything else unchanged:
#   MambaWin  - the Mamba encoder (DL_MAMBA_CFG) on 5 s windows, ResNet1D's MIL path
#   ResNetSeq - the 1D-ResNet trunk (DL_RESNET_CFG) on 30 s crops, Mamba's crop path
#   ResNetSeq5 / Mamba5 - both architectures on 5 s crops through the SAME crop
#               path as ResNetSeq / Mamba, so the 5 s vs 30 s comparison changes
#               only the context length (same recipe, same seconds of signal per
#               optimiser step, whole recording scored at test time).
# Not in the default roster; run them with --dl-models MambaWin ResNetSeq ...
DL_CONTROL_MODELS = ("MambaWin", "ResNetSeq", "ResNetSeq5", "Mamba5")

# Crop length per crop-path model; anything not listed uses DL_MAMBA_CROP_S (30 s).
# Shorter crops get proportionally more crops per subject and a proportionally
# larger batch (factor DL_MAMBA_CROP_S / length), so steps per epoch and seconds
# of signal per step stay the same as at 30 s.
DL_CROP_S_BY_MODEL = {"ResNetSeq5": 5.0, "Mamba5": 5.0}

# Samples per time-normalised stride for the stride-based MIL models.
DL_GAIT_STRIDE_LEN = 128

# MLPFeat: a small regularised network on the SAME aggregated TSFresh vectors the
# classical models use, so the deep branch is finally given the representation
# that performs best. 60 training subjects, so keep it small.
DL_MLPFEAT_CFG = {
    "hidden": (128, 64),
    "dropout": 0.5,
    "lr": 1e-3,
    "weight_decay": 1e-2,
    "epochs": 200,
    "patience": 30,
    "batch": 16,
    "label_smooth": 0.1,
}

SAMPLING_RATE = 100  # Hz (from format.txt)

# Trim the initial/final standing period (recordings begin/end with the subject
# standing still, which is not gait). Guarded so short recordings are not gutted.
TRIM_START_S = 5.0
TRIM_END_S = 5.0

# --------------------------------------------------------------------------- #
# Enriched per-foot channel layout (13 channels)
#   0..7 : the 8 raw VGRF sensors
#   8    : total force under the foot
#   9    : centre-of-pressure X
#   10   : centre-of-pressure Y
#   11   : heel aggregate (posterior sensors, Y < 0)
#   12   : toe aggregate  (anterior sensors,  Y > 0)
# Sensor (X, Y) coordinates are taken from format.txt.
# --------------------------------------------------------------------------- #
LEFT_COORDS = np.array(
    [[-500, -800], [-700, -400], [-300, -400], [-700, 0],
     [-300, 0], [-700, 400], [-300, 400], [-500, 800]], dtype=float)
RIGHT_COORDS = np.array(
    [[500, -800], [700, -400], [300, -400], [700, 0],
     [300, 0], [700, 400], [300, 400], [500, 800]], dtype=float)
N_CHANNELS_PER_FOOT = 13

# --------------------------------------------------------------------------- #
# TSFresh windowing (classical branch): 5 s window, 2.5 s step.
# --------------------------------------------------------------------------- #
WIN_TS = int(5.0 * SAMPLING_RATE)     # 500 samples
STEP_TS = int(2.5 * SAMPLING_RATE)    # 250 samples
TSFRESH_FC_PARAMETERS = "efficient"   # minimal | efficient | comprehensive
TSFRESH_N_JOBS = 0                    # 0 -> single process (most robust)

# Per-subject aggregation of the window feature vectors.
AGG_FUNCS = ("mean", "std", "skewness", "kurtosis")

# --------------------------------------------------------------------------- #
# Per-fold classical preprocessing
# --------------------------------------------------------------------------- #
SELECT_K = 300          # cap on univariate-selected features (fit on train only)
PCA_MAX = 50            # cap on PCA components for the SVM-RBF representation

# Decision threshold for the classical models:
#   'native'     - the classifier's own boundary (0.5 for probabilities, 0.0 for
#                  SVM margins). Default: every model is fitted with
#                  class_weight="balanced" on balanced folds, so this boundary is
#                  already calibrated.
#   'class_mean' - midpoint between the validation class means.
#   'tuned'      - argmax balanced accuracy on validation.
# Re-tuning on only 20 validation subjects overfits: it cost ~7 accuracy points
# on the Combined representation versus simply using the native boundary.
CLASSICAL_THRESHOLD = "native"

# --------------------------------------------------------------------------- #
# Classical hyper-parameter grids (tuned on train -> validation)
# --------------------------------------------------------------------------- #
RF_GRID = {
    "n_estimators": [300, 600],
    "max_depth": [None, 10, 20],
    "max_features": ["sqrt", 0.3],
}
SVM_LINEAR_GRID = {
    "C": [0.01, 0.1, 1.0, 10.0, 100.0],
}
SVM_RBF_GRID = {
    "C": [0.1, 1.0, 10.0, 100.0],
    "gamma": ["scale", 0.01, 0.001],
}
LOGREG_GRID = {
    "C": [0.01, 0.1, 1.0, 10.0, 100.0],
}
EXTRATREES_GRID = {
    "n_estimators": [300, 600],
    "max_depth": [None, 10, 20],
    "max_features": ["sqrt", 0.3],
}

# --------------------------------------------------------------------------- #
# Deep-learning branch (raw windowed signals)
# --------------------------------------------------------------------------- #
# 5 s / 2.5 s windows: ~2x more training windows than 10 s and a steadier
# per-subject average (each 5 s window still spans several gait cycles).
WIN_DL = int(5.0 * SAMPLING_RATE)     # 500 samples (5 s)
STEP_DL = int(2.5 * SAMPLING_RATE)    # 250 samples (2.5 s)

DL_EPOCHS = 40
DL_BATCH = 64
DL_PATIENCE = 8
DL_WEIGHT_DECAY = 1e-3                # fallback path only; MIL uses per-model cfg
DL_LR_GRID = [1e-3, 3e-4]             # collapsed to [1e-3] when --dl-quick
DL_LR_QUICK = [1e-3]

# Deep ensemble: train the fixed model with these random-init seeds and average
# their subject predictions (Lakshminarayanan et al. 2017). Cuts variance and is
# simple to report. --dl-quick uses a single seed.
DL_ENSEMBLE_SEEDS = [42, 43, 44, 45, 46]

# Light on-the-fly augmentation of TRAIN windows only (data are per-channel
# z-scored, so these magnitudes are in standard-deviation units / samples).
DL_AUGMENT = True
DL_AUG_SCALE = 0.10                   # global amplitude scaling, +/-10 %
DL_AUG_JITTER = 0.05                  # additive Gaussian noise (std)
DL_AUG_SHIFT = 25                     # max circular time shift (+/-0.25 s)

# Attention-MIL head: the ResNet1D becomes a per-window encoder and a learned
# gated-attention pooling produces one subject-level prediction, trained
# end-to-end on the subject label (optimises the metric we actually report,
# instead of averaging independent window predictions). Set False to fall back
# to the window-level training + mean pooling path.
DL_MIL = True
DL_MIL_BAG_BATCH = 4                  # subjects per optimizer step
DL_MIL_MAX_WINDOWS = 48              # cap windows sampled per subject during training
DL_MIL_ATT_DIM = 128                 # attention hidden dimension

# Decision-threshold shrinkage (window-average fallback path only): pull the
# noisy validation threshold this fraction toward the model's boundary.
DL_THR_SHRINK = 0.5

# MIL threshold: Youden's J tuned on validation but CAPPED to this probability
# band (borrowed from the reference 'youden_capped' policy) so the cutoff can
# never run to an extreme on the 20-subject validation set.
DL_THR_CAP_LO = 0.35
DL_THR_CAP_HI = 0.65

# Decision threshold for the deep models. All of them output a softmax
# probability and are trained on balanced folds, so 0.5 is already the
# calibrated boundary. Anything fitted on the 20-subject validation split
# overfits: it left Mamba at sensitivity 0.68 vs specificity 0.84 (biased
# against PD) with accuracy 10 points below its AUC.
#   'native'        - the model's own boundary, 0.5 (default). Fits nothing, so
#                     nothing can overfit the 20-subject validation split.
#   'class_mean'     - midpoint of the validation class means. Statistically tied
#                     with 'native' (paired over 30 model x foot combinations:
#                     delta 0.001, p = 0.82) and clearly better in one case
#                     (ResNet1D/Combined 0.79 vs 0.69), so it is the alternative
#                     worth trying per model.
#   'prior'          - median of the validation scores. Measured WORSE than
#                     native (delta -0.010, p = 0.038).
#   'youden'         - argmax of Youden's J on validation. WORST of all
#                     (delta -0.016, p = 0.012): it fits the cut to 20 subjects
#                     and does not transfer.
#   'youden_capped'  - Youden clipped to [CAP_LO, CAP_HI]; also below native.
# Measured with sweep_thresholds.py over every saved score file.
# Chosen from the full sweep (sweep_thresholds.py, 12 models x 3 feet x 5
# policies): class_mean was the best single policy overall (0.7733 vs native
# 0.7682) and is clearly better on the reported models (0.7742 vs 0.7633),
# almost entirely because it repairs 1D-ResNet, whose native cut is badly placed
# on Combined (0.720 -> 0.780). Principled reason: the classical models are
# fitted with class_weight="balanced" so their native boundary is already
# calibrated, whereas the deep models average probabilities over 5 seeds and
# several crops, which compresses them toward 0.5. Hence classical uses
# 'native' and the deep branch uses 'class_mean'.
DL_THRESHOLD = "class_mean"

# Fixed MIL model config (one per DL model). Trained as a deep ensemble over
# DL_ENSEMBLE_SEEDS -- no validation-based selection, since a 20-subject
# validation set cannot rank configs reliably. Regularization is deliberately
# light: on this task the MIL objective + augmentation + ensembling already
# regularize, and heavier weight decay / dropout under-fit (especially Combined).
DL_RESNET_CFG = {
    "widths": (64, 128, 256),
    "att_dim": 128,
    "dropout": 0.3,
    "lr": 5e-4,
    "weight_decay": 1e-3,
}
# The LSTM encoder is a CNN-LSTM: a small conv stem extracts local features and
# shortens the window (500 -> ~60 steps) before a bi-LSTM. conv_dim sets the
# stem width.
DL_LSTM_CFG = {
    "lstm_hidden": 96,
    "lstm_layers": 1,
    "conv_dim": 64,
    "att_dim": 128,
    "dropout": 0.3,
    "lr": 5e-4,
    "weight_decay": 1e-3,
}

# --------------------------------------------------------------------------- #
# Mamba / selective state-space model
# --------------------------------------------------------------------------- #
# "sequence" (default): Mamba runs over a long multi-second crop of the whole
#   recording and predicts the subject directly. This plays to the SSM's
#   strength -- linear-time long-range modelling -- and lets it observe
#   stride-to-stride variability, which a single 5 s window cannot show.
# "windows": the 5 s window + attention-MIL path used by ResNet1D / CNN-LSTM,
#   kept as a like-for-like ablation against the other encoders.
DL_MAMBA_MODE = "sequence"

DL_MAMBA_CROP_S = 30.0        # seconds per crop (~30 gait cycles)
DL_MAMBA_CROPS_PER_SUBJ = 4   # random crops per subject per epoch (augmentation)
DL_MAMBA_EVAL_CROPS = 6       # evenly spaced crops averaged at inference
DL_MAMBA_BATCH = 8
DL_MAMBA_EPOCHS = 60
DL_MAMBA_PATIENCE = 12
DL_MAMBA_LABEL_SMOOTH = 0.1   # softens targets; helps with only 60 training subjects

# Per-epoch train/validation AUC logging for the probe fit. Answers whether a
# model that peaks at epoch 1 is overfitting (train AUC rises while validation
# falls) or simply not learning (both stay near 0.5). Costs one extra scoring
# pass per epoch, so it is off unless explicitly enabled.
DL_LOG_CURVES = False
DL_LOG_CURVES_NSUB = 20     # training subjects sampled for the train-AUC estimate

# Two-phase fit. Phase 1 trains on the 60 training subjects and early-stops on
# validation to find the epoch budget; phase 2 retrains from scratch on
# train + val (80 subjects) for exactly that many epochs. The classical branch
# already refits on train+val, so without this the deep models learn from 25%
# less data than the classifiers they are compared against.
# Requires DL_THRESHOLD='native': once val is used for training it can no longer
# calibrate a threshold.
# Off by default: enabling several changes at once on the feature-sequence model
# degraded it, so each is tested individually (see DL_FEATMAMBA_* notes).
DL_MAMBA_REFIT_TRAINVAL = False

# Small-data regularisers (0 disables each).
DL_MAMBA_MIXUP_ALPHA = 0.0    # 0.3 to enable: Beta(a,a) mixing of crops + labels
DL_MAMBA_EMA_DECAY = 0.0      # see the EMA window note above; use ~0.99 not 0.999
DL_AUG_CHANNEL_DROP = 0.0     # 0.10 to enable
DL_AUG_TIME_MASK = 0.0        # 0.10 to enable

DL_MAMBA_CFG = {
    "d_model": 128,
    "mamba_layers": 4,
    "d_state": 16,
    "d_conv": 4,
    "expand": 2,
    "conv_dim": 64,
    "downsample": 16,         # 30 s crop (3000) -> ~190 tokens for the SSM stack
    "bidirectional": True,    # gait inside a crop is not causal -> scan both ways
    "att_dim": 128,
    "dropout": 0.3,
    "lr": 5e-4,
    "weight_decay": 1e-3,
}

# --------------------------------------------------------------------------- #
# MambaFeat: Mamba over the sequence of per-window TSFresh feature vectors
# --------------------------------------------------------------------------- #
# One token = one 5 s window described by its TSFresh features, in recording
# order. Combines the strongest representation (TSFresh, AUC ~0.91 in the
# classical branch) with sequence modelling of how the descriptors evolve.
# The first run extracts and caches the un-aggregated per-window features, which
# is slow; later runs reuse the cache.
DL_FEATMAMBA_TOKENS = 48         # windows per crop (~2 min of gait)
DL_FEATMAMBA_SELECT_K = 256      # TSFresh features kept per window (fit on train)
# Criterion used to pick those features. 'f_classif' is an ANOVA F-test, i.e. a
# LINEAR criterion, so in principle it can miss features that separate the
# classes non-linearly. 'mutual_info' was tried for exactly that reason and was
# MEASURABLY WORSE: on the left foot AUC fell 0.876 -> 0.820 and its fold-to-fold
# standard deviation nearly tripled (0.058 -> 0.150). The mutual-information
# estimate is computed from k-nearest-neighbour entropies, which are unreliable
# with ~2700 windows across tens of thousands of candidate features, so the extra
# generality costs more in estimation noise than it recovers. Keep f_classif.
DL_FEATMAMBA_SELECTOR = "f_classif"   # 'f_classif' | 'mutual_info' 
DL_FEATMAMBA_CROPS_PER_SUBJ = 4  # random crops per subject per epoch
DL_FEATMAMBA_EVAL_CROPS = 4      # evenly spaced crops averaged at inference
DL_FEATMAMBA_BATCH = 8
DL_FEATMAMBA_EPOCHS = 60
DL_FEATMAMBA_PATIENCE = 12
DL_FEATMAMBA_LABEL_SMOOTH = 0.1

# --- ablation switches (all OFF = the configuration that scored 0.80 ± 0.055) ---
# Enabling all four at once made things worse (0.80 -> 0.77, variance doubled),
# so they are off by default and meant to be tested ONE AT A TIME. A run is ~9
# min now that the window features are cached, so ablating is cheap.
DL_FEATMAMBA_REFIT_TRAINVAL = False   # two-phase refit on train+val
DL_FEATMAMBA_MIXUP_ALPHA = 0.0        # 0.3 to enable
# EMA note: the averaging window is 1/(1-decay) STEPS. Training here is only
# ~30 steps/epoch x ~30 epochs = ~900 steps, so 0.999 (window 1000) never leaves
# the initialisation. Use ~0.99 (window 100) if enabling at all.
DL_FEATMAMBA_EMA_DECAY = 0.0

# Short recordings: "tile" repeats the sequence, "zero" pads with zeros.
# Tiling is the principled choice (no artificial silence), but it was changed at
# the same time as the others, so it is switchable for a clean ablation.
DL_PAD_MODE = "tile"

DL_FEATMAMBA_CFG = {
    "d_model": 64,
    "mamba_layers": 2,
    "d_state": 16,
    "d_conv": 4,
    "expand": 2,
    "bidirectional": True,
    "att_dim": 64,
    "dropout": 0.4,          # heavier: feature tokens are high-dimensional
    # Defaults are the best MEASURED configuration (0.80 +- 0.055, AUC 0.876).
    # lr 2e-4 + multistat_pool=True was tried and scored 0.79 / AUC 0.882, i.e.
    # inside the noise band, so the better-measured values are kept as default.
    "lr": 8e-4,
    "weight_decay": 1e-2,
    # True concatenates mean/std/max of the token embeddings with the
    # attention-pooled vector, exposing stride-to-stride variability to the head.
    # Principled but not yet shown to help here — kept as an ablation switch.
    "multistat_pool": False,
}

# MambaGait needs its own config. Its tokens are individual strides, and the
# discriminative quantity is the VARIABILITY between them: exploratory analysis showed
# the mean gait profiles of the two groups are similar and only the dispersion
# differs. Attention pooling is a weighted mean, so with multistat_pool=False the
# model literally cannot compute variability — which is why the first MambaGait
# run collapsed to AUC 0.57 on the left foot. It also has only 17 features per
# stride (34 combined), so the encoder must be much smaller than MambaFeat's.
DL_GAITMAMBA_CFG = {
    "d_model": 32,
    "mamba_layers": 1,
    "d_state": 16,
    "d_conv": 4,
    "expand": 2,
    "bidirectional": True,
    "att_dim": 32,
    "dropout": 0.2,          # lighter: 17-34 inputs, not 256
    "lr": 1e-3,
    "weight_decay": 1e-3,
    "multistat_pool": True,  # essential here, not optional
}
