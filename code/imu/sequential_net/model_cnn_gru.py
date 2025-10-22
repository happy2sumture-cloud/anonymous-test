# model_cnn_gru.py
import torch
import torch.nn as nn

class CNNGRU(nn.Module):
    def __init__(self, input_dim, num_classes, chs=(128,128,128), ks=5, dropout=0.2, gru_h=256, gru_layers=1):
        super().__init__()
        layers = []
        in_ch = input_dim
        for c in chs:
            layers += [
                nn.Conv1d(in_ch, c, ks, padding=ks//2),
                nn.BatchNorm1d(c),
                nn.ReLU(),
                nn.Dropout(dropout)
            ]
            in_ch = c
        self.cnn = nn.Sequential(*layers)
        self.gru = nn.GRU(input_size=in_ch, hidden_size=gru_h,
                          num_layers=gru_layers, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(
            nn.Linear(gru_h*2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):     # x: (B, D, T)
        h = self.cnn(x)       # (B, C, T)
        h = h.transpose(1, 2) # (B, T, C)
        out, _ = self.gru(h)  # (B, T, 2H)
        g = out.mean(dim=1)   # 全局平均，也可换成 attention
        return self.head(g)
