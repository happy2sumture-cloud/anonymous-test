# model_transformer.py
import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)        # (T, D)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (T,1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)   # 偶数维
        pe[:, 1::2] = torch.cos(position * div_term)   # 奇数维
        # 注册为buffer，随 model.to(device) 一起迁移；persistent=False 防止state_dict过大
        self.register_buffer('pe', pe.unsqueeze(0), persistent=False)  # (1, T, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        T = x.size(1)
        pe = self.pe[:, :T, :].to(device=x.device, dtype=x.dtype)
        return x + pe


class IMUTransformer(nn.Module):
    def __init__(self, input_dim, num_classes,
                 d_model=64, nhead=4, num_layers=2,
                 dim_feedforward=128, dropout=0.3, use_cls=False, max_len=2000):
        super().__init__()
        self.use_cls = use_cls

        # 投影 + LayerNorm
        self.proj = nn.Linear(input_dim, d_model)
        self.proj_norm = nn.LayerNorm(d_model)

        self.posenc = PositionalEncoding(d_model, max_len=max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
            activation="gelu"   # 比 relu 更稳定
        )
        self.enc = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        if use_cls:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        self.norm = nn.LayerNorm(d_model)

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(d_model, num_classes)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, D, T) -> (B, T, D)
        x = x.transpose(1, 2)
        x = self.proj(x)
        x = self.proj_norm(x)               # 归一化投影
        x = self.posenc(x)                  # 位置编码

        if self.use_cls:
            B = x.size(0)
            cls = self.cls_token.expand(B, -1, -1)
            x = torch.cat([cls, x], dim=1)

        h = self.enc(x)
        h = h[:, 0] if self.use_cls else h.mean(dim=1)  # 平均池化更稳定
        h = self.norm(h)
        return self.head(h)
