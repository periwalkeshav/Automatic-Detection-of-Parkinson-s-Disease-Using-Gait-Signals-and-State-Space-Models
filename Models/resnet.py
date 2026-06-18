class ResidualBlock1D(nn.Module):
    """
    Residual block from professor's architecture, adapted with GELU.
    kernel_size=7 throughout: larger receptive field vs k=3/5 — important
    for the 47-window feature sequences in this dataset.
    Shortcut handles both channel changes (1×1 conv) and stride mismatch
    (stride argument) so the residual addition is always shape-safe.
    No dropout here — dropout only after the final global pool (avoids
    BN/dropout interaction that corrupts running statistics at test time).
    """
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch,  out_ch, 7, stride=stride, padding=3, bias=False)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 7, stride=1,      padding=3, bias=False)
        self.bn2   = nn.BatchNorm1d(out_ch)
        self.act   = nn.GELU()
        self.shortcut = (
            nn.Sequential(nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                          nn.BatchNorm1d(out_ch))
            if stride != 1 or in_ch != out_ch else nn.Identity())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r   = self.shortcut(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + r)   # post-activation, standard ResNet


class ResNet1D(nn.Module):
    """
    Hybrid 1D-ResNet: professor's architecture (k=7, stride-based downsampling,
    progressive channel expansion, end-only dropout) scaled to the GaitPaDB
    dataset (60 training subjects, T=47 windows after PCA).

    3 stages (not 4): C → 2C → 4C gives 36K–141K params for d_model in [8, 16].
    Stem: k=9 (compromise — large enough for temporal context, not overly
    aggressive for T=47) with stride=2 → T: 47 → 24 → 12 → 6 → 1.
    d_model maps to base_channels; sweep in HP grid as [8, 16].
    """
    def __init__(self, f_in: int, num_classes: int = 2,
                 dropout: float = 0.3, d_model: int = 16) -> None:
        super().__init__()
        C = d_model
        self.stem = nn.Sequential(
            nn.Conv1d(f_in, C, kernel_size=9, stride=2, padding=4, bias=False),
            nn.BatchNorm1d(C), nn.GELU())
        self.layer1 = nn.Sequential(ResidualBlock1D(C,   C,   stride=1),
                                     ResidualBlock1D(C,   C,   stride=1))
        self.layer2 = nn.Sequential(ResidualBlock1D(C,   C*2, stride=2),
                                     ResidualBlock1D(C*2, C*2, stride=1))
        self.layer3 = nn.Sequential(ResidualBlock1D(C*2, C*4, stride=2),
                                     ResidualBlock1D(C*4, C*4, stride=1))
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Dropout(dropout), nn.Linear(C*4, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # (B, f_in, T)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.head(x)