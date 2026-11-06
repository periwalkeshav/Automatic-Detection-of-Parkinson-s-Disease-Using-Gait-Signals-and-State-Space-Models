"""
Raw-window datasets, the training loop, and subject-level aggregation for the
deep-learning branch.

Windows (5 s / 2.5 s step) are encoded and pooled to a subject-level prediction
by gated-attention MIL (ResNet1D or LSTM encoder). Per-channel z-score
normalisation is fit on train windows only, and train windows get light
on-the-fly augmentation. The reported model is a deep ensemble: a fixed config
trained with several random-init seeds, their subject probabilities averaged.
The decision threshold (capped Youden's J) uses the validation split; test is
evaluated once.
"""
from __future__ import annotations

import copy
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

from . import config as C
from .data import foot_signal, make_windows
from .metrics import class_mean_threshold, compute_metrics, youden_capped_threshold
from .mamba_block import mamba_backend
from .models_dl import (MODEL_REGISTRY, FeatureMambaClassifier,
                        MLPFeatClassifier, MambaSeqClassifier, ResNetSeqClassifier,
                        build_mil)


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = _device()


# --------------------------------------------------------------------------- #
# Windowing per split
# --------------------------------------------------------------------------- #
def _split_windows(subjects, labels, lefts, rights, foot):
    X, y_win, groups = [], [], []
    subj_labels = []
    for s_idx, (lab, left, right) in enumerate(zip(labels, lefts, rights)):
        sig = foot_signal(left, right, foot)
        w = make_windows(sig, C.WIN_DL, C.STEP_DL)     # (n, C, win)
        X.append(w)
        y_win.append(np.full(w.shape[0], lab, dtype=np.int64))
        groups.append(np.full(w.shape[0], s_idx, dtype=np.int64))
        subj_labels.append(lab)
    return (np.concatenate(X).astype(np.float32),
            np.concatenate(y_win),
            np.concatenate(groups),
            np.asarray(subj_labels, dtype=int),
            list(subjects))


def needs_aggregate_features(model_name: str) -> bool:
    """Models that consume the per-subject aggregated TSFresh vector."""
    return model_name == "MLPFeat"


def prepare_dl_aggregate(fold: dict, foot: str) -> dict:
    """
    Aggregated TSFresh features, prepared exactly like the classical branch.

    Reuses prepare_classical_fold so the deep model sees the identical
    representation (cleaned, standardised, univariately selected) that the
    classical classifiers score best on — everything fitted on training only.
    """
    from .features import build_feature_matrix
    from .preprocessing import prepare_classical_fold

    idx = {"Left": 0, "Right": 1, "Combined": 2}[foot]
    tr_s, tr_y, tr_l, tr_r = fold["train"]
    va_s, va_y, va_l, va_r = fold["val"]
    te_s, te_y, te_l, te_r = fold["test"]
    xtr = build_feature_matrix(tr_s, tr_l, tr_r)[idx]
    xva = build_feature_matrix(va_s, va_l, va_r)[idx]
    xte = build_feature_matrix(te_s, te_l, te_r)[idx]
    prep = prepare_classical_fold(xtr, np.asarray(tr_y, int), xva, xte)
    Xtr, Xva, Xte = prep.get("sel")
    return {"train": (Xtr.astype(np.float32), np.asarray(tr_y, int)),
            "val": (Xva.astype(np.float32), np.asarray(va_y, int)),
            "test": (Xte.astype(np.float32), np.asarray(te_y, int)),
            "in_features": Xtr.shape[1]}


@torch.no_grad()
def _mlp_scores(model, X):
    model.eval()
    logits = model(torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32)).to(DEVICE))
    return torch.softmax(logits, dim=1)[:, 1].cpu().numpy()


def _fit_mlpfeat(Xtr, ytr, in_features, cfg, seed, max_epochs,
                 Xva=None, yva=None, patience_limit=None):
    torch.manual_seed(seed); np.random.seed(seed)
    rng = np.random.default_rng(seed)
    model = MLPFeatClassifier(in_features, hidden=tuple(cfg["hidden"]),
                              dropout=cfg["dropout"]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                            weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(max_epochs, 1))
    crit = nn.CrossEntropyLoss(label_smoothing=cfg.get("label_smooth", 0.0))

    idx = np.arange(len(Xtr))
    best_auc, best_state, waited = -np.inf, None, 0
    for epoch in range(1, max_epochs + 1):
        model.train(); rng.shuffle(idx)
        for b0 in range(0, len(idx), cfg["batch"]):
            b = idx[b0:b0 + cfg["batch"]]
            if len(b) < 2:            # BatchNorm needs >1 sample
                continue
            xb = torch.from_numpy(Xtr[b]).to(DEVICE)
            yb = torch.from_numpy(ytr[b]).long().to(DEVICE)
            opt.zero_grad(); crit(model(xb), yb).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        sched.step()
        if Xva is None:
            continue
        try:
            auc = roc_auc_score(yva, _mlp_scores(model, Xva))
        except ValueError:
            auc = 0.0
        if auc > best_auc:
            best_auc, waited = auc, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            waited += 1
            if patience_limit and waited >= patience_limit:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_auc


def _train_eval_mlpfeat(model_name, data, fold_id, foot, quick):
    seeds = [C.SEED] if quick else C.DL_ENSEMBLE_SEEDS
    cfg = dict(C.DL_MLPFEAT_CFG)
    Xtr, ytr = data["train"]; Xva, yva = data["val"]; Xte, yte = data["test"]

    members = []
    for seed in seeds:
        m, auc = _fit_mlpfeat(Xtr, ytr, data["in_features"], cfg, seed,
                              cfg["epochs"], Xva=Xva, yva=yva,
                              patience_limit=cfg["patience"])
        members.append(m)
        print(f"        seed {seed}: valAUC={auc:.3f}")

    val_scores = np.mean([_mlp_scores(m, Xva) for m in members], axis=0)
    thr = _pick_dl_threshold(yva, val_scores)
    try:
        val_auc = roc_auc_score(yva, val_scores)
    except ValueError:
        val_auc = float("nan")
    te_scores = np.mean([_mlp_scores(m, Xte) for m in members], axis=0)
    metrics = compute_metrics(yte, (te_scores >= thr).astype(int), te_scores)
    _save_scores(model_name, foot, fold_id, yva, val_scores, yte, te_scores)
    print(f"      {model_name:12s} mlp-ensemble({len(members)} seeds) "
          f"[{data['in_features']}->{cfg['hidden']}] valAUC={val_auc:.3f} | "
          f"test acc={metrics['accuracy']:.3f} auc={metrics['auc']:.3f}")
    return {"branch": "dl", "model": model_name, "foot": foot, "fold": fold_id,
            "params": f"mlp_{cfg['hidden']}_do{cfg['dropout']}_ens{len(members)}",
            **metrics}


def needs_stride_windows(model_name: str) -> bool:
    """MIL models whose instances are time-normalised strides, not fixed windows."""
    return model_name == "ResNetGait"


def prepare_dl_stride_windows(fold: dict, foot: str) -> dict:
    """
    Same MIL structure as prepare_dl_foot, but each instance is one stride.

    Strides are time-normalised to a common length, so the convolution sees the
    same gait phase at the same position in every instance.
    """
    from .gait_features import build_stride_waveform_bags

    out_len = getattr(C, "DL_GAIT_STRIDE_LEN", 128)

    def build(split):
        subjects, labels, lefts, rights = fold[split]
        bags = build_stride_waveform_bags(lefts, rights, foot, out_len)
        X = np.concatenate(bags).astype(np.float32)
        groups = np.concatenate([np.full(len(b), i, dtype=np.int64)
                                 for i, b in enumerate(bags)])
        y_win = np.concatenate([np.full(len(b), lab, dtype=np.int64)
                                for b, lab in zip(bags, labels)])
        return X, y_win, groups, np.asarray(labels, dtype=int), list(subjects)

    tr, va, te = build("train"), build("val"), build("test")
    Xtr = tr[0]
    mean = Xtr.mean(axis=(0, 2), keepdims=True)
    std = Xtr.std(axis=(0, 2), keepdims=True) + 1e-6

    def norm(split):
        X, y, g, sl, sid = split
        return ((X - mean) / std, y, g, sl, sid)

    return {"train": norm(tr), "val": norm(va), "test": norm(te),
            "in_channels": Xtr.shape[1]}


def prepare_dl_foot(fold: dict, foot: str) -> dict:
    """Build train/val/test window tensors for one foot, normalised on train."""
    tr = _split_windows(*fold["train"], foot)
    va = _split_windows(*fold["val"], foot)
    te = _split_windows(*fold["test"], foot)

    # per-channel normalisation fit on train windows
    Xtr = tr[0]
    mean = Xtr.mean(axis=(0, 2), keepdims=True)
    std = Xtr.std(axis=(0, 2), keepdims=True) + 1e-6

    def norm(split):
        X, y, g, sl, sid = split
        return ((X - mean) / std, y, g, sl, sid)

    out = {
        "train": norm(tr),
        "val": norm(va),
        "test": norm(te),
        "in_channels": Xtr.shape[1],
    }
    # Attach the long-sequence view as well, so a sequence model (Mamba) still
    # gets the data it needs even when the caller only asked for windows.
    if any(needs_sequence(m) for m in C.DL_MODELS):
        out["seq"] = prepare_dl_sequence(fold, foot)
    return out


# --------------------------------------------------------------------------- #
# Aggregation + one training run
# --------------------------------------------------------------------------- #
def _subject_scores(win_scores, groups, n_subjects):
    scores = np.zeros(n_subjects, dtype=float)
    for s in range(n_subjects):
        m = groups == s
        scores[s] = win_scores[m].mean() if m.any() else 0.0
    return scores


@torch.no_grad()
def _window_scores(model, X, batch=256):
    """Per-window logit margin (logit_PD - logit_CO).

    Averaging the unsaturated margin over windows gives a smoother, better-
    calibrated subject score than averaging softmax probabilities, which pile up
    near 0/1 and make the decision threshold brittle.
    """
    model.eval()
    out = []
    for i in range(0, len(X), batch):
        xb = torch.from_numpy(X[i:i + batch]).to(DEVICE)
        logits = model(xb)
        out.append((logits[:, 1] - logits[:, 0]).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


class _WindowDataset(Dataset):
    """Windows with optional light augmentation (train only)."""

    def __init__(self, X, y, augment: bool, seed: int):
        self.X = X
        self.y = y
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        x = self.X[i]
        if self.augment:
            x = x * self.rng.uniform(1.0 - C.DL_AUG_SCALE, 1.0 + C.DL_AUG_SCALE)
            x = x + self.rng.normal(0.0, C.DL_AUG_JITTER, x.shape)
            shift = int(self.rng.integers(-C.DL_AUG_SHIFT, C.DL_AUG_SHIFT + 1))
            if shift:
                x = np.roll(x, shift, axis=1)
        return (torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)),
                int(self.y[i]))


def _train_one(model_name, data, lr, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    Xtr, ytr, gtr, sl_tr, _ = data["train"]
    Xva, yva, gva, sl_va, _ = data["val"]

    model = MODEL_REGISTRY[model_name](data["in_channels"]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=C.DL_WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=C.DL_EPOCHS)
    criterion = nn.CrossEntropyLoss()

    loader = DataLoader(
        _WindowDataset(Xtr, ytr, augment=C.DL_AUGMENT, seed=seed),
        batch_size=C.DL_BATCH, shuffle=True, drop_last=False)

    best_auc, best_state, patience = -np.inf, None, 0
    for _ in range(C.DL_EPOCHS):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            opt.step()

        sched.step()
        val_subj = _subject_scores(_window_scores(model, Xva), gva, len(sl_va))
        try:
            auc = roc_auc_score(sl_va, val_subj)
        except ValueError:
            auc = 0.0
        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= C.DL_PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_auc


def _ensemble_subject_scores(members, X, groups, n_subjects):
    """Average subject scores over ensemble members."""
    acc = np.zeros(n_subjects, dtype=float)
    for model in members:
        acc += _subject_scores(_window_scores(model, X), groups, n_subjects)
    return acc / max(len(members), 1)


def train_eval_dl(model_name: str, data: dict, fold_id: int, foot: str,
                  quick: bool = False) -> dict:
    if needs_aggregate_features(model_name):
        return _train_eval_mlpfeat(model_name, data, fold_id, foot, quick)
    if needs_feature_sequence(model_name):
        return _train_eval_featmamba(model_name, data, fold_id, foot, quick)
    if needs_sequence(model_name):
        return _train_eval_mamba_seq(model_name, data, fold_id, foot, quick)
    if C.DL_MIL:
        return _train_eval_mil(model_name, data, fold_id, foot, quick)

    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"{model_name!r} is only implemented for the MIL path; "
                         f"set DL_MIL=True in config.py to use it.")
    lrs = C.DL_LR_QUICK if quick else C.DL_LR_GRID
    seeds = [C.SEED] if quick else C.DL_ENSEMBLE_SEEDS

    # One ensemble member per seed; each member keeps its own best-LR checkpoint.
    members, member_lrs = [], []
    for seed in seeds:
        best_model, best_auc, best_lr = None, -np.inf, None
        for lr in lrs:
            model, auc = _train_one(model_name, data, lr, seed=seed)
            if auc > best_auc:
                best_auc, best_model, best_lr = auc, model, lr
        members.append(best_model)
        member_lrs.append(best_lr)

    # threshold on validation ensemble scores (robust class-mean midpoint)
    Xva, _, gva, sl_va, _ = data["val"]
    val_subj = _ensemble_subject_scores(members, Xva, gva, len(sl_va))
    thr = (1.0 - C.DL_THR_SHRINK) * class_mean_threshold(sl_va, val_subj)
    try:
        val_auc = roc_auc_score(sl_va, val_subj)
    except ValueError:
        val_auc = float("nan")

    # evaluate test at subject level with the ensemble
    Xte, _, gte, sl_te, _ = data["test"]
    te_subj = _ensemble_subject_scores(members, Xte, gte, len(sl_te))
    te_pred = (te_subj >= thr).astype(int)
    metrics = compute_metrics(sl_te, te_pred, te_subj)

    print(f"      {model_name:12s} ensemble={len(members)} lrs={member_lrs} "
          f"valAUC={val_auc:.3f} | test acc={metrics['accuracy']:.3f} "
          f"auc={metrics['auc']:.3f}")

    return {"branch": "dl", "model": model_name, "foot": foot, "fold": fold_id,
            "params": f"ensemble{len(members)}_lrs={member_lrs}", **metrics}


# --------------------------------------------------------------------------- #
# Attention-MIL path (subject = bag of windows, trained end-to-end)
# --------------------------------------------------------------------------- #
def _save_scores(model_name, foot, fold_id, y_val, val_scores, y_te, te_scores):
    """
    Persist subject-level scores so threshold policies can be compared offline.

    Re-training a fold takes ~10 minutes; re-scoring a saved array takes none, so
    any later question about the decision threshold is answered without a rerun.
    """
    if not getattr(C, "SAVE_DL_SCORES", True):
        return
    out_dir = os.path.join(C.RESULTS_DIR, "dl_scores")
    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, f"{model_name}__{foot}__fold{fold_id}.npz"),
             y_val=np.asarray(y_val), val_scores=np.asarray(val_scores),
             y_test=np.asarray(y_te), test_scores=np.asarray(te_scores))


def _pick_dl_threshold(y_val, val_scores) -> float:
    """
    Decision threshold for the deep models, per C.DL_THRESHOLD.

    Subject scores are softmax probabilities from models trained on balanced
    folds, so the native 0.5 boundary is already calibrated; refitting it on 20
    validation subjects overfits and skews sensitivity against specificity.
    """
    policy = getattr(C, "DL_THRESHOLD", "native")
    if policy == "prior":
        # Splits are 50/50 by construction, so the median validation score is the
        # cut that reproduces that prior. Robust (uses all 20 subjects) and does
        # not depend on the scores being calibrated.
        v = np.asarray(val_scores, dtype=float)
        return float(np.median(v)) if v.size else 0.5
    if policy == "native":
        return 0.5
    if policy == "class_mean":
        return class_mean_threshold(y_val, val_scores)
    if policy == "youden_capped":
        return youden_capped_threshold(y_val, val_scores,
                                       C.DL_THR_CAP_LO, C.DL_THR_CAP_HI)
    raise ValueError(f"Unknown DL_THRESHOLD={policy!r}")


def _bags(groups, n_subjects):
    """List of window-index arrays, one per subject."""
    return [np.where(groups == s)[0] for s in range(n_subjects)]


def _augment_batch(X, rng):
    """Light augmentation of a (N, C, T) window batch (train only)."""
    scale = rng.uniform(1.0 - C.DL_AUG_SCALE, 1.0 + C.DL_AUG_SCALE, size=(X.shape[0], 1, 1))
    X = X * scale + rng.normal(0.0, C.DL_AUG_JITTER, X.shape)
    shifts = rng.integers(-C.DL_AUG_SHIFT, C.DL_AUG_SHIFT + 1, size=X.shape[0])
    for k, s in enumerate(shifts):
        if s:
            X[k] = np.roll(X[k], s, axis=1)
    return np.ascontiguousarray(X, dtype=np.float32)


@torch.no_grad()
def _mil_subject_scores(model, X, bags, chunk=256):
    """Per-subject PD probability (softmax over the pooled attention logits)."""
    model.eval()
    scores = np.zeros(len(bags), dtype=float)
    for s, idx in enumerate(bags):
        if len(idx) == 0:
            continue
        hs = []
        for i in range(0, len(idx), chunk):
            xb = torch.from_numpy(np.ascontiguousarray(X[idx[i:i + chunk]])).to(DEVICE)
            hs.append(model.encode(xb))
        logit = model.pool_and_classify(torch.cat(hs, dim=0))
        scores[s] = float(torch.softmax(logit, dim=0)[1].cpu())
    return scores


def _mil_ensemble_scores(members, X, bags):
    acc = np.zeros(len(bags), dtype=float)
    for model in members:
        acc += _mil_subject_scores(model, X, bags)
    return acc / max(len(members), 1)


def _fixed_cfg(model_name):
    """Fixed config for a DL model (ResNet1D, LSTM or Mamba).

    The matched-context controls reuse their architecture's config unchanged:
    MambaWin the Mamba one, ResNetSeq the ResNet one."""
    return dict({"LSTM": C.DL_LSTM_CFG,
                 "Mamba": C.DL_MAMBA_CFG,
                 "MambaWin": C.DL_MAMBA_CFG,
                 "Mamba5": C.DL_MAMBA_CFG}.get(model_name, C.DL_RESNET_CFG))


def _short_cfg(model_name, cfg):
    common = (f"do{cfg['dropout']:.2f} wd{cfg['weight_decay']:.1e} "
              f"lr{cfg['lr']:.1e} a{cfg['att_dim']}")
    if model_name == "LSTM":
        return f"h{cfg['lstm_hidden']} L{cfg['lstm_layers']} {common}"
    if model_name in ("Mamba", "MambaWin", "Mamba5"):
        return (f"d{cfg['d_model']} L{cfg['mamba_layers']} s{cfg['d_state']}"
                f"{' bi' if cfg.get('bidirectional', True) else ''} {common}")
    return f"w{cfg['widths'][0]} {common}"


def _train_one_mil(model_name, data, cfg, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    Xtr, _, gtr, sl_tr, _ = data["train"]
    Xva, _, gva, sl_va, _ = data["val"]
    bags_tr = _bags(gtr, len(sl_tr))
    bags_va = _bags(gva, len(sl_va))

    model = build_mil(model_name, data["in_channels"], cfg).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"],
                           weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=C.DL_EPOCHS)
    criterion = nn.CrossEntropyLoss()

    order = np.arange(len(sl_tr))
    best_auc, best_state, patience = -np.inf, None, 0
    for _ in range(C.DL_EPOCHS):
        model.train()
        rng.shuffle(order)
        for b0 in range(0, len(order), C.DL_MIL_BAG_BATCH):
            batch = order[b0:b0 + C.DL_MIL_BAG_BATCH]
            idx_parts, counts, labels = [], [], []
            for s in batch:
                idx = bags_tr[s]
                if len(idx) > C.DL_MIL_MAX_WINDOWS:      # sample to bound compute
                    idx = rng.choice(idx, C.DL_MIL_MAX_WINDOWS, replace=False)
                idx_parts.append(idx)
                counts.append(len(idx))
                labels.append(int(sl_tr[s]))
            all_idx = np.concatenate(idx_parts)
            Xb = Xtr[all_idx]
            Xb = _augment_batch(Xb, rng) if C.DL_AUGMENT else np.ascontiguousarray(Xb)

            # Encode every window in one pass (proper BatchNorm batch), then
            # attention-pool per subject.
            h = model.encode(torch.from_numpy(Xb).to(DEVICE))
            logits, start = [], 0
            for c in counts:
                logits.append(model.pool_and_classify(h[start:start + c]))
                start += c
            logits = torch.stack(logits, dim=0)
            yb = torch.tensor(labels, dtype=torch.long, device=DEVICE)

            opt.zero_grad()
            criterion(logits, yb).backward()
            opt.step()

        sched.step()
        val_subj = _mil_subject_scores(model, Xva, bags_va)
        try:
            auc = roc_auc_score(sl_va, val_subj)
        except ValueError:
            auc = 0.0
        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= C.DL_PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_auc


def _train_eval_mil(model_name, data, fold_id, foot, quick):
    """Deep ensemble: train a fixed MIL config with several seeds, average them."""
    seeds = [C.SEED] if quick else C.DL_ENSEMBLE_SEEDS
    cfg = _fixed_cfg(model_name)

    members = []
    for seed in seeds:
        model, auc = _train_one_mil(model_name, data, cfg, seed=seed)
        members.append(model)
        print(f"        seed {seed}: valAUC={auc:.3f}")

    Xva, _, gva, sl_va, _ = data["val"]
    bags_va = _bags(gva, len(sl_va))
    val_subj = _mil_ensemble_scores(members, Xva, bags_va)
    # Class-mean midpoint (halfway between the mean PD and mean CO validation
    # score): uses every validation subject, so it is far more stable than the
    # argmax-Youden cutoff, which overfits the 20-subject split and thrashes
    # (high AUC but swinging sensitivity/specificity, especially for the LSTM).
    thr = _pick_dl_threshold(sl_va, val_subj)
    try:
        val_auc = roc_auc_score(sl_va, val_subj)
    except ValueError:
        val_auc = float("nan")

    Xte, _, gte, sl_te, _ = data["test"]
    bags_te = _bags(gte, len(sl_te))
    te_subj = _mil_ensemble_scores(members, Xte, bags_te)
    te_pred = (te_subj >= thr).astype(int)
    metrics = compute_metrics(sl_te, te_pred, te_subj)
    _save_scores(model_name, foot, fold_id, sl_va, val_subj, sl_te, te_subj)

    print(f"      {model_name:12s} MIL deep-ensemble({len(members)} seeds) "
          f"[{_short_cfg(model_name, cfg)}] valAUC={val_auc:.3f} | "
          f"test acc={metrics['accuracy']:.3f} auc={metrics['auc']:.3f}")

    return {"branch": "dl", "model": model_name, "foot": foot, "fold": fold_id,
            "params": f"deep_ensemble_{len(members)}seeds_{_short_cfg(model_name, cfg)}",
            **metrics}


# --------------------------------------------------------------------------- #
# Mamba long-sequence path (one prediction per recording crop)
# --------------------------------------------------------------------------- #
def needs_sequence(model_name: str) -> bool:
    """True when the model consumes recording crops instead of MIL windows."""
    if model_name in ("ResNetSeq", "ResNetSeq5", "Mamba5"):
        return True
    return model_name == "Mamba" and C.DL_MAMBA_MODE == "sequence"


def _crop_seconds(model_name):
    """Crop length for a crop-path model (DL_CROP_S_BY_MODEL, else 30 s)."""
    return float(getattr(C, "DL_CROP_S_BY_MODEL", {}).get(model_name, C.DL_MAMBA_CROP_S))


def _crop_scale(model_name):
    """How many crops replace one 30 s crop, so shorter crops see the same
    seconds of signal per step and per epoch (1 for the 30 s models)."""
    return max(1, int(round(C.DL_MAMBA_CROP_S / _crop_seconds(model_name))))


def prepare_dl_sequence(fold: dict, foot: str) -> dict:
    """Full per-subject signals for one foot, per-channel z-scored on train."""
    def sigs(split):
        subjects, labels, lefts, rights = fold[split]
        xs = [foot_signal(l, r, foot).astype(np.float32) for l, r in zip(lefts, rights)]
        return xs, np.asarray(labels, dtype=int)

    tr_x, tr_y = sigs("train")
    va_x, va_y = sigs("val")
    te_x, te_y = sigs("test")

    allc = np.concatenate(tr_x, axis=1)                 # (C, sum_T) over train only
    mean = allc.mean(axis=1, keepdims=True)
    std = allc.std(axis=1, keepdims=True) + 1e-6
    norm = lambda xs: [((s - mean) / std).astype(np.float32) for s in xs]

    return {"train": (norm(tr_x), tr_y), "val": (norm(va_x), va_y),
            "test": (norm(te_x), te_y), "in_channels": tr_x[0].shape[0]}


def _tile_to(arr, target, axis):
    """Repeat a short sequence cyclically up to `target` along `axis`.

    Zero-padding a short recording would feed the model a long stretch of
    silence that never occurs in real gait; repeating the walk keeps every
    timestep real. Recordings here range from ~12 to ~100 windows, so short
    subjects would otherwise be mostly padding.
    """
    n = arr.shape[axis]
    if n >= target:
        return arr
    if getattr(C, "DL_PAD_MODE", "tile") == "zero":
        shape = list(arr.shape)
        shape[axis] = target
        out = np.zeros(shape, dtype=np.float32)
        idx = [slice(None)] * arr.ndim
        idx[axis] = slice(0, n)
        out[tuple(idx)] = arr
        return out
    reps = int(np.ceil(target / n))
    tiled = np.concatenate([arr] * reps, axis=axis)
    return np.take(tiled, np.arange(target), axis=axis)


def _crop(sig, length, rng=None, start=None):
    """(C, T) -> (C, length): random crop while training, fixed crop at eval."""
    T = sig.shape[1]
    if T < length:
        sig = _tile_to(sig, length, axis=1)
        T = length
    if T == length:
        return np.ascontiguousarray(sig, dtype=np.float32)
    if start is None:
        start = int(rng.integers(0, T - length + 1))
    return sig[:, start:start + length]


def _augment_seq(x, rng):
    """Amplitude scaling, jitter, channel dropout and time masking on a (C, L) crop.

    Channel dropout and time masking force the model to rely on several sensors
    and several parts of the walk rather than latching onto one — the standard
    remedy when there are far more parameters than subjects.
    """
    n_ch, L = x.shape
    x = x * rng.uniform(1.0 - C.DL_AUG_SCALE, 1.0 + C.DL_AUG_SCALE)
    x = x + rng.normal(0.0, C.DL_AUG_JITTER, x.shape)

    p_ch = getattr(C, "DL_AUG_CHANNEL_DROP", 0.0)
    if p_ch > 0:
        drop = rng.random(n_ch) < p_ch
        if drop.any():
            x = x.copy()
            x[drop] = 0.0

    p_t = getattr(C, "DL_AUG_TIME_MASK", 0.0)
    if p_t > 0:
        span = int(p_t * L)
        if span > 0:
            s = int(rng.integers(0, max(L - span, 1)))
            x = x.copy()
            x[:, s:s + span] = 0.0
    return np.ascontiguousarray(x, dtype=np.float32)


class _EMA:
    """Exponential moving average of weights; evaluated instead of the raw model."""

    def __init__(self, model, decay: float):
        self.decay = decay
        self.shadow = {k: v.detach().clone().float() if v.dtype.is_floating_point
                       else v.detach().clone()
                       for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach().float(),
                                                     alpha=1.0 - self.decay)
            else:
                self.shadow[k] = v.detach().clone()

    def state_dict(self, model):
        ref = model.state_dict()
        return {k: v.to(dtype=ref[k].dtype) for k, v in self.shadow.items()}


def _mixup(xb, yb, alpha, rng, n_classes=2):
    """Convex combinations of crops and their one-hot labels (Zhang et al. 2018)."""
    if alpha <= 0:
        return xb, None
    lam = float(rng.beta(alpha, alpha))
    perm = torch.randperm(xb.shape[0], device=xb.device)
    xb = lam * xb + (1.0 - lam) * xb[perm]
    onehot = torch.zeros(yb.shape[0], n_classes, device=yb.device)
    onehot.scatter_(1, yb.view(-1, 1), 1.0)
    soft = lam * onehot + (1.0 - lam) * onehot[perm]
    return xb, soft


@torch.no_grad()
def _seq_subject_scores(model, sigs, length, n_crops=None):
    """P(PD) per subject, averaged over evenly spaced crops (test-time aug).

    At least enough crops are used to cover the whole recording, so a short crop
    length sees as much of the recording as a long one. With 30 s crops and
    recordings under 180 s this is exactly DL_MAMBA_EVAL_CROPS (6), as before.
    """
    n_crops = n_crops or C.DL_MAMBA_EVAL_CROPS
    model.eval()
    out = np.zeros(len(sigs), dtype=float)
    for i, s in enumerate(sigs):
        T = s.shape[1]
        n = max(n_crops, int(np.ceil(T / length)))
        starts = ([0] if T <= length
                  else np.linspace(0, T - length, n).astype(int).tolist())
        crops = np.stack([_crop(s, length, start=int(st)) for st in starts])
        logits = model(torch.from_numpy(crops).to(DEVICE))
        out[i] = float(torch.softmax(logits, dim=1)[:, 1].mean().cpu())
    return out


def _as_sequence_data(data: dict) -> dict:
    """Accept either a pure sequence dict or a window dict carrying a 'seq' view."""
    seq = data.get("seq", data)
    if len(seq["train"]) != 2:
        raise ValueError(
            "Mamba needs long-sequence data but received window tensors. "
            "Make sure main.py is up to date (it should call prepare_dl_sequence "
            "for sequence models), or set DL_MAMBA_MODE='windows' in src/config.py.")
    return seq


def _build_seq_model(model_name, in_channels, cfg):
    """Crop-level classifier: Mamba, or the ResNetSeq matched-context control."""
    if model_name in ("ResNetSeq", "ResNetSeq5"):
        return ResNetSeqClassifier(in_channels, widths=cfg["widths"],
                                   att_dim=cfg["att_dim"], dropout=cfg["dropout"])
    return MambaSeqClassifier(
        in_channels, d_model=cfg["d_model"], n_layers=cfg["mamba_layers"],
        d_state=cfg["d_state"], d_conv=cfg.get("d_conv", 4), expand=cfg.get("expand", 2),
        dropout=cfg["dropout"], conv_dim=cfg.get("conv_dim", 64), att_dim=cfg["att_dim"],
        bidirectional=cfg.get("bidirectional", True),
        downsample=cfg.get("downsample", 16),
        # short crops x many per batch: keep backprop memory small (same function)
        scan="sequential" if _crop_seconds(model_name) < C.DL_MAMBA_CROP_S else "parallel")


def _fit_mamba_seq(Xtr, ytr, in_channels, cfg, seed, max_epochs,
                   Xva=None, yva=None, patience_limit=None, model_name="Mamba"):
    """
    Train one long-sequence model (Mamba, or the ResNetSeq control).

    With a validation split it early-stops on validation AUC and reports the
    epoch that was best; without one it simply trains the whole budget, which is
    how the train+val refit re-uses that epoch count.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    L = int(_crop_seconds(model_name) * C.SAMPLING_RATE)
    scale = _crop_scale(model_name)
    batch = C.DL_MAMBA_BATCH * scale

    model = _build_seq_model(model_name, in_channels, cfg).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                            weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(max_epochs, 1))

    smooth = getattr(C, "DL_MAMBA_LABEL_SMOOTH", 0.0)
    mix_alpha = getattr(C, "DL_MAMBA_MIXUP_ALPHA", 0.0)
    criterion = nn.CrossEntropyLoss(label_smoothing=smooth)

    ema_decay = getattr(C, "DL_MAMBA_EMA_DECAY", 0.0)
    ema = _EMA(model, ema_decay) if ema_decay > 0 else None
    eval_model = copy.deepcopy(model) if ema else model

    def current_eval_model():
        if ema is None:
            return model
        eval_model.load_state_dict(ema.state_dict(model))
        return eval_model

    pool = np.repeat(np.arange(len(Xtr)), C.DL_MAMBA_CROPS_PER_SUBJ * scale)
    best_auc, best_state, best_epoch, waited = -np.inf, None, max_epochs, 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        rng.shuffle(pool)
        for b0 in range(0, len(pool), batch):
            idx = pool[b0:b0 + batch]
            xb = np.stack([_augment_seq(_crop(Xtr[i], L, rng), rng)
                           if C.DL_AUGMENT else _crop(Xtr[i], L, rng) for i in idx])
            xb = torch.from_numpy(xb).to(DEVICE)
            yb = torch.from_numpy(ytr[idx]).long().to(DEVICE)

            xb, soft = _mixup(xb, yb, mix_alpha, rng)
            opt.zero_grad()
            logits = model(xb)
            if soft is None:
                loss = criterion(logits, yb)
            else:                       # soft-target CE for mixed crops
                loss = -(soft * torch.log_softmax(logits, dim=1)).sum(dim=1).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if ema is not None:
                ema.update(model)
        sched.step()

        if Xva is None:                 # refit phase: no validation to watch
            continue
        scored = current_eval_model()
        try:
            auc = roc_auc_score(yva, _seq_subject_scores(scored, Xva, L))
        except ValueError:
            auc = 0.0
        if getattr(C, "DL_LOG_CURVES", False):
            k = min(getattr(C, "DL_LOG_CURVES_NSUB", 20), len(Xtr))
            try:
                tr_auc = roc_auc_score(ytr[:k], _seq_subject_scores(scored, Xtr[:k], L))
            except ValueError:
                tr_auc = float("nan")
            print(f"          ep{epoch:02d}  train_auc {tr_auc:.3f}  val_auc {auc:.3f}")
        if auc > best_auc:
            best_auc, best_epoch, waited = auc, epoch, 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in scored.state_dict().items()}
        else:
            waited += 1
            if patience_limit and waited >= patience_limit:
                break

    if Xva is None:
        final = current_eval_model()
        return copy.deepcopy(final), float("nan"), max_epochs
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_auc, best_epoch


def _train_eval_mamba_seq(model_name, data, fold_id, foot, quick):
    """Deep ensemble of long-sequence Mamba models for one fold / foot."""
    seeds = [C.SEED] if quick else C.DL_ENSEMBLE_SEEDS
    cfg = _fixed_cfg(model_name)
    L = int(_crop_seconds(model_name) * C.SAMPLING_RATE)
    data = _as_sequence_data(data)
    Xtr, ytr = data["train"]
    Xva, yva = data["val"]
    in_ch = data["in_channels"]

    # Refitting on train+val consumes the validation split, so the threshold can
    # no longer be calibrated on it — only the fixed native boundary is valid.
    refit = (getattr(C, "DL_MAMBA_REFIT_TRAINVAL", False)
             and getattr(C, "DL_THRESHOLD", "native") == "native")

    # Phase 1: find the epoch budget on the held-out validation split.
    probe_seed = seeds[0]
    probe_model, probe_auc, best_epoch = _fit_mamba_seq(
        Xtr, ytr, in_ch, cfg, probe_seed, C.DL_MAMBA_EPOCHS,
        Xva=Xva, yva=yva, patience_limit=C.DL_MAMBA_PATIENCE, model_name=model_name)
    print(f"        probe seed {probe_seed}: valAUC={probe_auc:.3f} "
          f"best_epoch={best_epoch}")

    members = []
    if refit:
        # Phase 2: retrain each member on train + val for that many epochs.
        Xfull = list(Xtr) + list(Xva)
        yfull = np.concatenate([ytr, yva])
        # Keep the optimisation budget constant: more subjects means more steps
        # per epoch, so scale the epoch count down by the same ratio. Reusing the
        # raw epoch count would silently train ~33% longer as well as on more data.
        best_epoch = max(1, int(round(best_epoch * len(Xtr) / len(Xfull))))
        print(f"        refit on train+val ({len(Xfull)} subjects) "
              f"for {best_epoch} epochs (step-matched)")
        for seed in seeds:
            model, _, _ = _fit_mamba_seq(Xfull, yfull, in_ch, cfg, seed, best_epoch,
                                         model_name=model_name)
            members.append(model)
    else:
        for seed in seeds:
            model, auc, _ = _fit_mamba_seq(
                Xtr, ytr, in_ch, cfg, seed, C.DL_MAMBA_EPOCHS,
                Xva=Xva, yva=yva, patience_limit=C.DL_MAMBA_PATIENCE,
                model_name=model_name)
            members.append(model)
            print(f"        seed {seed}: valAUC={auc:.3f}")

    # Validation scores must come from a model that never trained on validation.
    # After a train+val refit only the probe qualifies; otherwise use the ensemble.
    val_scores = (_seq_subject_scores(probe_model, Xva, L) if refit
                  else np.mean([_seq_subject_scores(m, Xva, L) for m in members], axis=0))
    thr = _pick_dl_threshold(yva, val_scores)
    try:
        val_auc = roc_auc_score(yva, val_scores)
    except ValueError:
        val_auc = float("nan")

    Xte, yte = data["test"]
    te_scores = np.mean([_seq_subject_scores(m, Xte, L) for m in members], axis=0)
    te_pred = (te_scores >= thr).astype(int)
    metrics = compute_metrics(yte, te_pred, te_scores)
    _save_scores(model_name, foot, fold_id, yva, val_scores, yte, te_scores)

    print(f"      {model_name:12s} seq-ensemble({len(members)} seeds) "
          f"[{_short_cfg(model_name, cfg)}] valAUC={val_auc:.3f} | "
          f"test acc={metrics['accuracy']:.3f} auc={metrics['auc']:.3f}")

    return {"branch": "dl", "model": model_name, "foot": foot, "fold": fold_id,
            "params": (f"seq{_crop_seconds(model_name):g}s_ens{len(members)}seeds"
                       f"{'_refitTV' if refit else ''}"
                       f"{'_mixup' if getattr(C,'DL_MAMBA_MIXUP_ALPHA',0)>0 else ''}"
                       f"{'_ema' if getattr(C,'DL_MAMBA_EMA_DECAY',0)>0 else ''}_"
                       f"{_short_cfg(model_name, cfg)}"),
            **metrics}


# --------------------------------------------------------------------------- #
# MambaFeat: Mamba over per-window TSFresh feature sequences
# --------------------------------------------------------------------------- #
def needs_feature_sequence(model_name: str) -> bool:
    return model_name in ("MambaFeat", "MambaGait")


def feature_source(model_name: str) -> str:
    """'tsfresh' = one token per 5 s window; 'gait' = one token per stride."""
    return "gait" if model_name == "MambaGait" else "tsfresh"


def prepare_dl_feature_sequence(fold: dict, foot: str, source: str = "tsfresh") -> dict:
    """
    Per-subject (n_windows, k) TSFresh feature sequences.

    Cleaning, scaling and feature selection are all fit on the training windows
    only; validation and test are merely transformed, so the fold stays
    leakage-free exactly like the classical branch.
    """
    from sklearn.feature_selection import (SelectKBest, f_classif,
                                           mutual_info_classif)
    from sklearn.preprocessing import StandardScaler

    if source == "gait":
        from .gait_features import build_stride_sequences as build_seqs
    else:
        from .features import build_window_sequences as build_seqs

    def seqs(split):
        subjects, labels, lefts, rights = fold[split]
        return (build_seqs(subjects, lefts, rights, foot),
                np.asarray(labels, dtype=int))

    tr_x, tr_y = seqs("train")
    va_x, va_y = seqs("val")
    te_x, te_y = seqs("test")

    stacked = np.vstack(tr_x)                                   # (sum_W, F)
    stacked = np.where(np.isfinite(stacked), stacked, np.nan)
    col_mean = np.nan_to_num(np.nanmean(stacked, axis=0), nan=0.0)
    stacked = np.where(np.isnan(stacked), col_mean, stacked)
    keep = stacked.std(axis=0) > 0                              # drop constants
    if not keep.any():
        keep = np.ones(stacked.shape[1], dtype=bool)
    stacked = stacked[:, keep]

    scaler = StandardScaler().fit(stacked)
    win_y = np.concatenate([np.full(len(m), y) for m, y in zip(tr_x, tr_y)])
    k = min(C.DL_FEATMAMBA_SELECT_K, stacked.shape[1])
    crit = getattr(C, "DL_FEATMAMBA_SELECTOR", "f_classif")
    if crit == "mutual_info":
        # capture non-linear dependence, which an ANOVA F-test cannot see
        score_fn = lambda X, y: mutual_info_classif(X, y, random_state=C.SEED)
    else:
        score_fn = f_classif
    selector = SelectKBest(score_fn, k=k).fit(scaler.transform(stacked), win_y)

    def transform(mats):
        out = []
        for m in mats:
            m = np.where(np.isfinite(m), m, np.nan)
            m = np.where(np.isnan(m), col_mean, m)[:, keep]
            out.append(selector.transform(scaler.transform(m)).astype(np.float32))
        return out

    return {"train": (transform(tr_x), tr_y), "val": (transform(va_x), va_y),
            "test": (transform(te_x), te_y), "in_features": k}


def _crop_tokens(mat, n_tokens, rng=None, start=None):
    """(W, F) -> (n_tokens, F): random crop while training, fixed crop at eval.

    Recordings run from ~12 to ~100 windows; short ones are tiled rather than
    zero-padded so every token is a real window.
    """
    W = mat.shape[0]
    if W < n_tokens:
        mat = _tile_to(mat, n_tokens, axis=0)
        W = n_tokens
    if W == n_tokens:
        return np.ascontiguousarray(mat, dtype=np.float32)
    if start is None:
        start = int(rng.integers(0, W - n_tokens + 1))
    return mat[start:start + n_tokens]


@torch.no_grad()
def _featseq_scores(model, mats, n_tokens, n_crops=None):
    """P(PD) per subject, averaged over evenly spaced window crops."""
    n_crops = n_crops or C.DL_FEATMAMBA_EVAL_CROPS
    model.eval()
    out = np.zeros(len(mats), dtype=float)
    for i, m in enumerate(mats):
        W = m.shape[0]
        starts = ([0] if W <= n_tokens
                  else np.linspace(0, W - n_tokens, n_crops).astype(int).tolist())
        crops = np.stack([_crop_tokens(m, n_tokens, start=int(s)) for s in starts])
        logits = model(torch.from_numpy(crops).to(DEVICE))
        out[i] = float(torch.softmax(logits, dim=1)[:, 1].mean().cpu())
    return out


def _fit_featmamba(Xtr, ytr, in_features, cfg, seed, max_epochs,
                   Xva=None, yva=None, patience_limit=None):
    """Train one feature-sequence Mamba (mixup + EMA; early stops only if given val)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    T = C.DL_FEATMAMBA_TOKENS

    model = FeatureMambaClassifier(
        in_features, d_model=cfg["d_model"], n_layers=cfg["mamba_layers"],
        d_state=cfg["d_state"], d_conv=cfg.get("d_conv", 4),
        expand=cfg.get("expand", 2), dropout=cfg["dropout"], att_dim=cfg["att_dim"],
        bidirectional=cfg.get("bidirectional", True),
        multistat_pool=cfg.get("multistat_pool", True)).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                            weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(max_epochs, 1))

    criterion = nn.CrossEntropyLoss(
        label_smoothing=getattr(C, "DL_FEATMAMBA_LABEL_SMOOTH", 0.0))
    mix_alpha = getattr(C, "DL_FEATMAMBA_MIXUP_ALPHA", 0.0)
    ema_decay = getattr(C, "DL_FEATMAMBA_EMA_DECAY", 0.0)
    ema = _EMA(model, ema_decay) if ema_decay > 0 else None
    eval_model = copy.deepcopy(model) if ema else model

    def current_eval_model():
        if ema is None:
            return model
        eval_model.load_state_dict(ema.state_dict(model))
        return eval_model

    pool = np.repeat(np.arange(len(Xtr)), C.DL_FEATMAMBA_CROPS_PER_SUBJ)
    best_auc, best_state, best_epoch, waited = -np.inf, None, max_epochs, 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        rng.shuffle(pool)
        for b0 in range(0, len(pool), C.DL_FEATMAMBA_BATCH):
            idx = pool[b0:b0 + C.DL_FEATMAMBA_BATCH]
            xb = np.stack([_crop_tokens(Xtr[i], T, rng) for i in idx])
            if C.DL_AUGMENT:                       # jitter in standardised units
                xb = xb + rng.normal(0.0, C.DL_AUG_JITTER, xb.shape).astype(np.float32)
            xb = torch.from_numpy(np.ascontiguousarray(xb, dtype=np.float32)).to(DEVICE)
            yb = torch.from_numpy(ytr[idx]).long().to(DEVICE)

            xb, soft = _mixup(xb, yb, mix_alpha, rng)
            opt.zero_grad()
            logits = model(xb)
            loss = (criterion(logits, yb) if soft is None else
                    -(soft * torch.log_softmax(logits, dim=1)).sum(dim=1).mean())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if ema is not None:
                ema.update(model)
        sched.step()

        if Xva is None:                            # refit phase: nothing to watch
            continue
        scored = current_eval_model()
        try:
            auc = roc_auc_score(yva, _featseq_scores(scored, Xva, T))
        except ValueError:
            auc = 0.0
        if getattr(C, "DL_LOG_CURVES", False):
            k = min(getattr(C, "DL_LOG_CURVES_NSUB", 20), len(Xtr))
            try:
                tr_auc = roc_auc_score(ytr[:k], _featseq_scores(scored, Xtr[:k], T))
            except ValueError:
                tr_auc = float("nan")
            print(f"          ep{epoch:02d}  train_auc {tr_auc:.3f}  val_auc {auc:.3f}")
        if auc > best_auc:
            best_auc, best_epoch, waited = auc, epoch, 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in scored.state_dict().items()}
        else:
            waited += 1
            if patience_limit and waited >= patience_limit:
                break

    if Xva is None:
        return copy.deepcopy(current_eval_model()), float("nan"), max_epochs
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_auc, best_epoch


def _train_eval_featmamba(model_name, data, fold_id, foot, quick):
    """Deep ensemble of feature-sequence Mamba models for one fold / foot."""
    seeds = [C.SEED] if quick else C.DL_ENSEMBLE_SEEDS
    cfg = dict(C.DL_GAITMAMBA_CFG if model_name == "MambaGait"
               else C.DL_FEATMAMBA_CFG)
    T = C.DL_FEATMAMBA_TOKENS
    Xtr, ytr = data["train"]
    Xva, yva = data["val"]
    in_feat = data["in_features"]

    refit = (getattr(C, "DL_FEATMAMBA_REFIT_TRAINVAL", False)
             and getattr(C, "DL_THRESHOLD", "native") == "native")

    probe_seed = seeds[0]
    probe_model, probe_auc, best_epoch = _fit_featmamba(
        Xtr, ytr, in_feat, cfg, probe_seed, C.DL_FEATMAMBA_EPOCHS,
        Xva=Xva, yva=yva, patience_limit=C.DL_FEATMAMBA_PATIENCE)
    print(f"        probe seed {probe_seed}: valAUC={probe_auc:.3f} "
          f"best_epoch={best_epoch}")

    members = []
    if refit:
        Xfull = list(Xtr) + list(Xva)
        yfull = np.concatenate([ytr, yva])
        # Keep the optimisation budget constant: more subjects means more steps
        # per epoch, so scale the epoch count down by the same ratio. Reusing the
        # raw epoch count would silently train ~33% longer as well as on more data.
        best_epoch = max(1, int(round(best_epoch * len(Xtr) / len(Xfull))))
        print(f"        refit on train+val ({len(Xfull)} subjects) "
              f"for {best_epoch} epochs (step-matched)")
        for seed in seeds:
            model, _, _ = _fit_featmamba(Xfull, yfull, in_feat, cfg, seed, best_epoch)
            members.append(model)
    else:
        for seed in seeds:
            model, auc, _ = _fit_featmamba(
                Xtr, ytr, in_feat, cfg, seed, C.DL_FEATMAMBA_EPOCHS,
                Xva=Xva, yva=yva, patience_limit=C.DL_FEATMAMBA_PATIENCE)
            members.append(model)
            print(f"        seed {seed}: valAUC={auc:.3f}")

    # Validation scores must come from a model that never trained on validation.
    val_scores = (_featseq_scores(probe_model, Xva, T) if refit
                  else np.mean([_featseq_scores(m, Xva, T) for m in members], axis=0))
    thr = _pick_dl_threshold(yva, val_scores)
    try:
        val_auc = roc_auc_score(yva, val_scores)
    except ValueError:
        val_auc = float("nan")

    Xte, yte = data["test"]
    te_scores = np.mean([_featseq_scores(m, Xte, T) for m in members], axis=0)
    te_pred = (te_scores >= thr).astype(int)
    metrics = compute_metrics(yte, te_pred, te_scores)
    _save_scores(model_name, foot, fold_id, yva, val_scores, yte, te_scores)

    print(f"      {model_name:12s} featseq-ensemble({len(members)} seeds) "
          f"[tok{T} k{data['in_features']} d{cfg['d_model']} L{cfg['mamba_layers']}] "
          f"valAUC={val_auc:.3f} | test acc={metrics['accuracy']:.3f} "
          f"auc={metrics['auc']:.3f}")

    return {"branch": "dl", "model": model_name, "foot": foot, "fold": fold_id,
            "params": (f"featseq_tok{T}_k{in_feat}_ens{len(members)}seeds"
                       f"{'_refitTV' if refit else ''}"
                       f"{'_mixup' if getattr(C,'DL_FEATMAMBA_MIXUP_ALPHA',0)>0 else ''}"
                       f"{'_ema' if getattr(C,'DL_FEATMAMBA_EMA_DECAY',0)>0 else ''}_"
                       f"d{cfg['d_model']}L{cfg['mamba_layers']}"),
            **metrics}
