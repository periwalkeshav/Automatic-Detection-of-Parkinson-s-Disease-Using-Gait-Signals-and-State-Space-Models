"""
Deep-learning architectures for the raw-signal branch.

Window encoders (ResNet1D, CNN-LSTM, Mamba) all expose the same interface —
(N, C, T) windows in, (N, embed_dim) embeddings out — so they plug into the
shared attention-MIL head and the same training/evaluation protocol.

Further encoders such as GRU or PatchTST can be
added by defining a builder and registering it in ``MODEL_REGISTRY``.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.utils.checkpoint

from .mamba_block import BiMambaLayer


class BasicBlock1d(nn.Module):
    """Two-conv residual block with optional stride-2 downsampling."""

    def __init__(self, c_in: int, c_out: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(c_in, c_out, kernel_size=7, stride=stride,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(c_out)
        self.conv2 = nn.Conv1d(c_out, c_out, kernel_size=7, stride=1,
                               padding=3, bias=False)
        self.bn2 = nn.BatchNorm1d(c_out)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = None
        if stride != 1 or c_in != c_out:
            self.downsample = nn.Sequential(
                nn.Conv1d(c_in, c_out, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(c_out),
            )

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class ResNet1D(nn.Module):
    """Compact 1D-ResNet for multichannel gait windows."""

    def __init__(self, in_channels: int, n_classes: int = 2,
                 widths=(64, 128, 256)):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, widths[0], kernel_size=7, stride=2,
                      padding=3, bias=False),
            nn.BatchNorm1d(widths[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )
        blocks = []
        c_prev = widths[0]
        for i, w in enumerate(widths):
            stride = 1 if i == 0 else 2
            blocks.append(BasicBlock1d(c_prev, w, stride=stride))
            blocks.append(BasicBlock1d(w, w, stride=1))
            c_prev = w
        self.blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.3)
        self.embed_dim = c_prev
        self.fc = nn.Linear(c_prev, n_classes)

    def features(self, x):                       # x: (B, C, T) -> (B, embed_dim)
        x = self.stem(x)
        x = self.blocks(x)
        return self.pool(x).squeeze(-1)

    def forward(self, x):                        # x: (B, C, T) -> (B, n_classes)
        return self.fc(self.dropout(self.features(x)))


class ResNetEncoder(nn.Module):
    """ResNet1D as a window encoder: (N, C, T) -> (N, embed_dim)."""

    def __init__(self, in_channels: int, widths=(64, 128, 256)):
        super().__init__()
        self.net = ResNet1D(in_channels, widths=widths)
        self.embed_dim = self.net.embed_dim

    def forward(self, x):
        return self.net.features(x)


class LSTMEncoder(nn.Module):
    """CNN-LSTM window encoder: (N, C, T) -> (N, embed_dim).

    A short conv stem extracts local features and downsamples the window (500 ->
    ~60 steps), then a bi-LSTM models the temporal dynamics; outputs are
    mean-pooled over time. The conv stem makes the RNN far more effective than a
    plain LSTM over the raw 100 Hz signal.
    """

    def __init__(self, in_channels: int, hidden: int = 96, layers: int = 1,
                 dropout: float = 0.3, conv_dim: int = 64):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, conv_dim, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(conv_dim), nn.ReLU(inplace=True),
            nn.MaxPool1d(2),                                     # /4
            nn.Conv1d(conv_dim, conv_dim, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(conv_dim), nn.ReLU(inplace=True),
            nn.MaxPool1d(2),                                     # /8  -> ~60 steps
        )
        self.lstm = nn.LSTM(input_size=conv_dim, hidden_size=hidden,
                            num_layers=layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0,
                            bidirectional=True)
        self.embed_dim = hidden * 2

    def forward(self, x):                        # x: (N, C, T)
        x = self.stem(x)                         # (N, conv_dim, T/8)
        x = x.transpose(1, 2)                    # (N, T/8, conv_dim)
        out, _ = self.lstm(x)                    # (N, T/8, 2*hidden)
        return out.mean(dim=1)                   # (N, embed_dim)


class MambaEncoder(nn.Module):
    """Conv stem + bidirectional Mamba window encoder: (N, C, T) -> (N, embed_dim).

    The same conv stem as the CNN-LSTM shortens the 5 s window (500 -> ~60
    tokens) and lifts the 13/26 raw channels into a d_model embedding; a stack of
    bidirectional selective state-space (Mamba) layers then models the temporal
    dynamics, and the tokens are mean-pooled into one window embedding.

    Memory: a MIL step encodes every sampled window of several bags at once
    (4 bags x 48 windows = 192 sequences), so the pure-PyTorch parallel scan needs
    ~20 GB for the backward pass. This encoder therefore uses the sequential scan
    (identical recurrence, only one state per step kept) and checkpoints each
    Mamba layer during training. Neither changes the computed function; the
    long-sequence and feature-sequence models keep the parallel scan.
    """

    def __init__(self, in_channels: int, d_model: int = 128, n_layers: int = 2,
                 d_state: int = 16, d_conv: int = 4, expand: int = 2,
                 dropout: float = 0.3, conv_dim: int = 64, bidirectional: bool = True,
                 downsample: int = 8, scan: str = "sequential", checkpoint: bool = True):
        super().__init__()
        self.stem = _conv_stem(in_channels, conv_dim, downsample)   # 500 -> ~60 tokens
        self.proj = nn.Linear(conv_dim, d_model)
        self.layers = nn.ModuleList([
            BiMambaLayer(d_model, d_state=d_state, d_conv=d_conv, expand=expand,
                         dropout=dropout, bidirectional=bidirectional, scan=scan)
            for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.embed_dim = d_model
        self.checkpoint = checkpoint

    def forward(self, x):                        # x: (N, C, T)
        x = self.stem(x)                         # (N, conv_dim, T/8)
        x = self.proj(x.transpose(1, 2))         # (N, T/8, d_model)
        for layer in self.layers:
            if self.checkpoint and self.training and torch.is_grad_enabled():
                # recompute the layer in backward instead of storing its
                # activations; the RNG state is replayed, so dropout matches
                x = torch.utils.checkpoint.checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
        return self.norm(x).mean(dim=1)          # (N, d_model)


def _conv_stem(in_channels: int, conv_dim: int, downsample: int = 8) -> nn.Sequential:
    """Conv front-end reducing the time axis by `downsample` (a power of two >= 4).

    Fewer, richer tokens keep the state-space stack cheap: a 30 s crop at 100 Hz
    (3000 samples) becomes ~190 tokens at /16, each covering ~0.16 s — still far
    finer than a gait cycle, so stride-to-stride structure is preserved.
    """
    if downsample < 4 or (downsample & (downsample - 1)) != 0:
        raise ValueError(f"downsample must be a power of two >= 4, got {downsample}")
    layers = [
        nn.Conv1d(in_channels, conv_dim, kernel_size=7, stride=2, padding=3, bias=False),
        nn.BatchNorm1d(conv_dim), nn.ReLU(inplace=True),
        nn.MaxPool1d(2),                                         # /4
        nn.Conv1d(conv_dim, conv_dim, kernel_size=5, stride=1, padding=2, bias=False),
        nn.BatchNorm1d(conv_dim), nn.ReLU(inplace=True),
    ]
    for _ in range(int(math.log2(downsample)) - 2):              # remaining /2 stages
        layers.append(nn.MaxPool1d(2))
    return nn.Sequential(*layers)


class AttentionPool(nn.Module):
    """Gated-attention pooling over sequence tokens: (B, L, d) -> (B, d)."""

    def __init__(self, d_model: int, att_dim: int = 128):
        super().__init__()
        self.V = nn.Linear(d_model, att_dim)
        self.U = nn.Linear(d_model, att_dim)
        self.w = nn.Linear(att_dim, 1)

    def forward(self, x):
        gate = torch.tanh(self.V(x)) * torch.sigmoid(self.U(x))
        att = torch.softmax(self.w(gate), dim=1)          # (B, L, 1)
        return (att * x).sum(dim=1)


class MambaSeqClassifier(nn.Module):
    """
    Long-sequence Mamba classifier: one prediction per recording crop.

    A conv stem downsamples a multi-second crop (30 s = 3000 samples -> ~375
    tokens) and lifts the raw channels into d_model; a stack of bidirectional
    selective state-space layers models the full crop in linear time; gated
    attention pools the tokens and a linear head predicts PD vs control.

    Unlike the window + MIL encoders, this sees many consecutive gait cycles at
    once, so it can represent stride-to-stride variability directly — the
    property that separated the two groups in the data analysis.
    """

    def __init__(self, in_channels: int, d_model: int = 128, n_layers: int = 4,
                 d_state: int = 16, d_conv: int = 4, expand: int = 2,
                 dropout: float = 0.3, conv_dim: int = 64, att_dim: int = 128,
                 bidirectional: bool = True, downsample: int = 16, n_classes: int = 2,
                 scan: str = "parallel"):
        super().__init__()
        self.stem = _conv_stem(in_channels, conv_dim, downsample)
        self.proj = nn.Linear(conv_dim, d_model)
        self.layers = nn.ModuleList([
            BiMambaLayer(d_model, d_state=d_state, d_conv=d_conv, expand=expand,
                         dropout=dropout, bidirectional=bidirectional, scan=scan)
            for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.pool = AttentionPool(d_model, att_dim)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x):                        # x: (B, C, T) -> (B, n_classes)
        x = self.stem(x)
        x = self.proj(x.transpose(1, 2))         # (B, T/8, d_model)
        for layer in self.layers:
            x = layer(x)
        z = self.pool(self.norm(x))              # (B, d_model)
        return self.head(self.dropout(z))


class ResNetSeqClassifier(nn.Module):
    """
    1D-ResNet over a long recording crop: the matched-context control for
    MambaSeqClassifier.

    The ResNet1D stem and residual stages are kept but the global average pool is
    replaced by the same gated attention pooling over time positions that the
    Mamba model uses. The trunk downsamples by 16, exactly like the Mamba conv
    stem, so a 30 s crop becomes ~190 tokens for both; only the layers that
    process the tokens differ.
    """

    def __init__(self, in_channels: int, widths=(64, 128, 256), att_dim: int = 128,
                 dropout: float = 0.3, n_classes: int = 2):
        super().__init__()
        trunk = ResNet1D(in_channels, widths=widths)
        self.stem, self.blocks = trunk.stem, trunk.blocks
        self.norm = nn.LayerNorm(trunk.embed_dim)
        self.pool = AttentionPool(trunk.embed_dim, att_dim)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(trunk.embed_dim, n_classes)

    def forward(self, x):                        # x: (B, C, T) -> (B, n_classes)
        x = self.blocks(self.stem(x))            # (B, width, T/16)
        z = self.pool(self.norm(x.transpose(1, 2)))
        return self.head(self.dropout(z))


class FeatureMambaClassifier(nn.Module):
    """
    Mamba over the time-ordered sequence of per-window TSFresh feature vectors.

    Each token is one 5 s window described by its selected TSFresh features, and
    the tokens are in recording order — so, unlike chopping a static feature
    vector into arbitrary chunks, the sequence axis really is time and the
    state-space model's inductive bias applies. It combines the two things that
    worked best separately: TSFresh descriptors (the strongest representation in
    the classical branch) and sequence modelling of how they change stride to
    stride.
    """

    def __init__(self, in_features: int, d_model: int = 64, n_layers: int = 2,
                 d_state: int = 16, d_conv: int = 4, expand: int = 2,
                 dropout: float = 0.4, att_dim: int = 64,
                 bidirectional: bool = True, n_classes: int = 2,
                 multistat_pool: bool = True):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_features, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.layers = nn.ModuleList([
            BiMambaLayer(d_model, d_state=d_state, d_conv=d_conv, expand=expand,
                         dropout=dropout, bidirectional=bidirectional)
            for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.pool = AttentionPool(d_model, att_dim)
        self.dropout = nn.Dropout(dropout)
        # Attention pooling is a weighted mean, so it cannot express how much the
        # descriptors FLUCTUATE across the walk — yet stride-to-stride variability
        # is what separates PD from controls, and is exactly what the classical
        # branch captures with its std/skew/kurtosis aggregation. Concatenating
        # dispersion statistics gives the head that information explicitly.
        self.multistat_pool = multistat_pool
        head_in = d_model * 4 if multistat_pool else d_model
        self.head = nn.Linear(head_in, n_classes)

    def forward(self, x):                    # x: (B, W, F) -> (B, n_classes)
        x = self.proj(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        z = self.pool(x)
        if self.multistat_pool:
            z = torch.cat([z, x.mean(dim=1), x.std(dim=1), x.amax(dim=1)], dim=-1)
        return self.head(self.dropout(z))


class MLPFeatClassifier(nn.Module):
    """
    Regularised MLP on the aggregated TSFresh feature vector.

    This is the one representation that has never been given to a deep model:
    the classical classifiers reach their best scores on exactly these
    per-subject aggregates, while every deep model so far has been fed either
    the raw signal or per-window sequences. With only 60 training subjects the
    network must be small and heavily regularised, so it is closer to a
    non-linear logistic regression than to a deep network — which is
    appropriate, since the input is already highly informative.
    """

    def __init__(self, in_features: int, hidden=(128, 64), dropout: float = 0.5,
                 n_classes: int = 2):
        super().__init__()
        layers, d = [], in_features
        for h in hidden:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.GELU(),
                       nn.Dropout(dropout)]
            d = h
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(d, n_classes)

    def forward(self, x):                    # (B, F) -> (B, n_classes)
        return self.head(self.body(x))


class AttentionMIL(nn.Module):
    """
    Gated-attention MIL head (Ilse et al., 2018) over any window encoder.

    A subject is a bag of windows: every window is encoded, a learned gated
    attention weights them, and the weighted sum is classified once per subject.
    Trained end-to-end on the subject label.
    """

    def __init__(self, encoder: nn.Module, embed_dim: int, n_classes: int = 2,
                 att_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.encoder = encoder
        self.att_V = nn.Linear(embed_dim, att_dim)
        self.att_U = nn.Linear(embed_dim, att_dim)
        self.att_w = nn.Linear(att_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, n_classes)

    def encode(self, windows):                   # (N, C, T) -> (N, embed_dim)
        return self.encoder(windows)

    def pool_and_classify(self, h):              # (n_i, embed_dim) -> (n_classes,)
        gate = torch.tanh(self.att_V(h)) * torch.sigmoid(self.att_U(h))
        att = torch.softmax(self.att_w(gate), dim=0)          # (n_i, 1)
        z = (att * h).sum(dim=0, keepdim=True)                # (1, embed_dim)
        return self.classifier(self.dropout(z)).squeeze(0)    # (n_classes,)


def build_mil(model_name: str, in_channels: int, cfg: dict) -> nn.Module:
    """Build an AttentionMIL model with the encoder for `model_name` and a
    hyper-parameter config `cfg` sampled by the DL search."""
    if model_name in ("ResNet1D", "ResNetGait"):
        encoder = ResNetEncoder(in_channels, widths=cfg["widths"])
    elif model_name == "LSTM":
        encoder = LSTMEncoder(in_channels, hidden=cfg["lstm_hidden"],
                              layers=cfg["lstm_layers"], dropout=cfg["dropout"],
                              conv_dim=cfg.get("conv_dim", 64))
    elif model_name in ("Mamba", "MambaWin"):
        encoder = MambaEncoder(in_channels, d_model=cfg["d_model"],
                               n_layers=cfg["mamba_layers"], d_state=cfg["d_state"],
                               d_conv=cfg.get("d_conv", 4), expand=cfg.get("expand", 2),
                               dropout=cfg["dropout"], conv_dim=cfg.get("conv_dim", 64),
                               bidirectional=cfg.get("bidirectional", True),
                               downsample=cfg.get("downsample", 8))
    else:
        raise ValueError(f"Unknown DL model for MIL: {model_name!r}")
    return AttentionMIL(encoder, encoder.embed_dim, att_dim=cfg["att_dim"],
                        dropout=cfg["dropout"])


def build_resnet1d(in_channels: int) -> nn.Module:
    return ResNet1D(in_channels=in_channels, n_classes=2)


# Window-level models for the (legacy) DL_MIL=False fallback path.
MODEL_REGISTRY = {
    "ResNet1D": build_resnet1d,
}
