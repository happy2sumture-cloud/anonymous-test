# model_bilstm_attn.py
import torch
import torch.nn as nn

class AttnPool(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.w = nn.Linear(d, 1)
    def forward(self, H):             # H: (B, T, D)
        a = self.w(H).squeeze(-1)     # (B, T)
        a = torch.softmax(a, dim=1)
        return (H * a.unsqueeze(-1)).sum(dim=1)  # (B, D)

class BiLSTMAttn(nn.Module):
    def __init__(self, input_dim, num_classes, hidden=256, layers=2, dropout=0.3):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden)
        self.lstm = nn.LSTM(hidden, hidden, num_layers=layers, batch_first=True,
                            dropout=dropout, bidirectional=True)
        self.attn = AttnPool(hidden*2)
        self.head = nn.Sequential(
            nn.Linear(hidden*2, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, num_classes)
        )

    def forward(self, x):  # x: (B, D, T)
        x = x.transpose(1, 2)         # (B, T, D)
        x = self.proj(x)              # (B, T, H)
        H, _ = self.lstm(x)           # (B, T, 2H)
        g = self.attn(H)              # (B, 2H)
        return self.head(g)
