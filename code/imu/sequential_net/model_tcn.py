# model_tcn.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class Chomp1d(nn.Module):
    def __init__(self, chomp):
        super().__init__()
        self.chomp = chomp
    def forward(self, x):
        return x[:, :, :-self.chomp].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout):
        super().__init__()
        self.conv1 = nn.utils.weight_norm(
            nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride,
                      padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.utils.weight_norm(
            nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride,
                      padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x); out = self.chomp1(out); out = self.relu1(out); out = self.drop1(out)
        out = self.conv2(out); out = self.chomp2(out); out = self.relu2(out); out = self.drop2(out)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCN(nn.Module):
    def __init__(self, input_dim, num_classes, channels=(64,128,128), ks=3, dropout=0.2):
        super().__init__()
        layers = []
        prev = input_dim
        for i, ch in enumerate(channels):
            dilation = 2 ** i
            padding = (ks - 1) * dilation
            layers += [TemporalBlock(prev, ch, ks, 1, dilation, padding, dropout)]
            prev = ch
        self.net = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(prev, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):       # x: (B, D, T)
        h = self.net(x)
        return self.head(h)
