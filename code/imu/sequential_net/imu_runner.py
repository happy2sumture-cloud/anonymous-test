#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, os, random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset_imu import ImuClips   # 你的数据集实现（已不再包含 limit_per_exp）

# -------------------- 公共工具 --------------------
def set_seed(seed: int = 2025):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def topk_accuracy(logits, y, ks=(1,3,5,10)):
    maxk = max(ks)
    _, pred = logits.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(y.view(1, -1).expand_as(pred))
    res = {}
    for k in ks:
        corr_k = correct[:k].reshape(-1).float().sum(0).item()
        res[f"top{k}"] = corr_k / y.size(0)
    return res

def accuracy_per_class(logits, y, num_classes):
    pred = logits.argmax(dim=1).cpu().numpy()
    y = y.cpu().numpy()
    hit = np.zeros(num_classes); cnt = np.zeros(num_classes)
    for yy, pp in zip(y, pred):
        cnt[yy] += 1; hit[yy] += (yy == pp)
    cls_acc = np.divide(hit, np.maximum(cnt, 1), out=np.zeros_like(hit), where=cnt>0)
    return float(cls_acc.mean())

def build_model(arch: str, input_dim: int, num_classes: int, dropout: float):
    arch = arch.lower()
    if arch == "tcn":
        from model_tcn import TCN
        return TCN(input_dim=input_dim, num_classes=num_classes, dropout=dropout)
    elif arch == "transformer":
        from model_transformer import IMUTransformer
        return IMUTransformer(input_dim=input_dim, num_classes=num_classes,
                              d_model=256, nhead=8, num_layers=4,
                              dim_feedforward=512, dropout=0.1)
    elif arch == "bilstm_attn":
        from model_bilstm_attn import BiLSTMAttn
        return BiLSTMAttn(input_dim=input_dim, num_classes=num_classes,
                          hidden=256, layers=2, dropout=0.3)
    elif arch == "cnn_gru":
        from model_cnn_gru import CNNGRU
        return CNNGRU(input_dim=input_dim, num_classes=num_classes,
                      chs=(128,128,128), ks=5, dropout=0.2,
                      gru_h=256, gru_layers=1)
    else:
        raise ValueError(f"Unknown arch: {arch}")

# -------------------- 训练流程 --------------------
def run_train(args):
    set_seed(args.seed)

    # 标签映射（保留 label_map.json）
    with open(args.label_map, "r", encoding="utf-8") as f:
        label_map = json.load(f)
    num_classes = len(label_map)

    # 训练集（统计归一化）
    train_set = ImuClips(args.train_csv, label_map,
                         train=True, compute_norm=True, augment=args.augment)
    norm = train_set.norm_stats()

    # 验证集
    if args.val_csv:
        val_set = ImuClips(args.val_csv, label_map,
                           train=False, compute_norm=False, norm_stats=norm)
    else:
        n = len(train_set); n_val = max(1, int(0.1 * n)); n_tr = n - n_val
        train_set, val_set = torch.utils.data.random_split(train_set, [n_tr, n_val])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=False)
    val_loader   = DataLoader(val_set,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, pin_memory=True, drop_last=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 自动推断 input_dim
    if args.input_dim is None:
        sample_x, _ = next(iter(train_loader))
        input_dim = sample_x.shape[1]  # (B, D, T)
        del sample_x
    else:
        input_dim = int(args.input_dim)

    # 模型/优化器/调度器
    model = build_model(args.arch, input_dim=input_dim, num_classes=num_classes, dropout=args.dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)

    print(f"[INFO] TRAIN | arch={args.arch} | input_dim={input_dim} | num_classes={num_classes}")

    best_top1 = 0.0
    for epoch in range(1, args.epochs + 1):
        # ---- Train
        model.train()
        total, loss_sum, top1_sum = 0, 0.0, 0.0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits = model(x)
            loss = criterion(logits, y)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optim.step()

            bs = x.size(0)
            total    += bs
            loss_sum += loss.item() * bs
            top1_sum += (logits.argmax(1) == y).float().sum().item()
        sched.step()
        print(f"[Epoch {epoch}] train loss={loss_sum/total:.4f} acc@1={top1_sum/total:.4f}")

        # ---- Val
        model.eval()
        vt, vloss, vtop1 = 0, 0.0, 0.0
        all_logits, all_labels = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)

                bs = x.size(0)
                vt    += bs
                vloss += loss.item() * bs
                vtop1 += (logits.argmax(1) == y).float().sum().item()

                all_logits.append(logits.cpu()); all_labels.append(y.cpu())

        all_logits = torch.cat(all_logits, 0); all_labels = torch.cat(all_labels, 0)
        tops = topk_accuracy(all_logits, all_labels, ks=(1,3,5,10))
        pi = float(vtop1 / vt)
        pc = accuracy_per_class(all_logits, all_labels, num_classes)
        print(f"[Val] loss={vloss/vt:.4f} P-I@1={pi:.4f} P-C@1={pc:.4f} "
              f"tops={{top1:{tops['top1']:.4f}, top3:{tops['top3']:.4f}, top5:{tops['top5']:.4f}, top10:{tops['top10']:.4f}}}")

        # 保存最好
        if tops["top1"] > best_top1:
            best_top1 = tops["top1"]
            os.makedirs(os.path.dirname(args.ckpt_out), exist_ok=True)
            # 自动拼接文件名
            base, ext = os.path.splitext(args.ckpt_out)
            ckpt_name = f"{base}_best_epoch{epoch:03d}_top1-{best_top1:.4f}{ext}"
            torch.save({
                "model": model.state_dict(),
                "norm": norm,
                "label_map": label_map,
                "epoch": epoch,
                "arch": args.arch,
                "input_dim": input_dim,
            }, ckpt_name)
            print(f"  ↳ saved best checkpoint to {ckpt_name}")

    print("[DONE] Training finished.]")

# -------------------- 测试流程 --------------------
def run_test(args):
    set_seed(args.seed)

    # 加载 ckpt
    ckpt = torch.load(args.ckpt, map_location="cpu")
    label_map = ckpt["label_map"]; num_classes = len(label_map)
    norm = ckpt["norm"]
    arch = ckpt.get("arch", "tcn")
    input_dim = ckpt.get("input_dim")

    print(f"[INFO] TEST | arch={arch}, epoch={ckpt.get('epoch')}, input_dim={input_dim}")

    # 构造测试集（不再有 limit_per_exp）
    test_set = ImuClips(args.test_csv, label_map,
                        train=False, compute_norm=False, norm_stats=norm)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(arch, input_dim=input_dim, num_classes=num_classes, dropout=0.0).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # 评估
    all_logits, all_labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            all_logits.append(logits.cpu()); all_labels.append(y.cpu())

    all_logits = torch.cat(all_logits, 0)
    all_labels = torch.cat(all_labels, 0)

    tops = topk_accuracy(all_logits, all_labels, ks=(1,3,5,10))
    pi = (all_logits.argmax(1) == all_labels).float().mean().item()
    pc = accuracy_per_class(all_logits, all_labels, num_classes)

    print("========== TEST RESULT ==========")
    print(f"P-I@1 : {pi:.4f}")
    print(f"P-C@1 : {pc:.4f}")
    print(f"Top1  : {tops['top1']:.4f}")
    print(f"Top3  : {tops['top3']:.4f}")
    print(f"Top5  : {tops['top5']:.4f}")
    print(f"Top10 : {tops['top10']:.4f}")

# -------------------- 主入口 --------------------
def main():
    ap = argparse.ArgumentParser(description="IMU 手语识别统一入口（训练/测试）")
    sub = ap.add_subparsers(dest="mode", required=True)

    # ---- train 子命令
    ap_tr = sub.add_parser("train", help="训练模型并保存 ckpt")
    ap_tr.add_argument("--arch", type=str, default="tcn",
                       choices=["tcn","transformer","bilstm_attn","cnn_gru"])
    ap_tr.add_argument("--train_csv", required=True)
    ap_tr.add_argument("--val_csv", default=None)
    ap_tr.add_argument("--label_map", required=True)  # 保留 label_map.json
    ap_tr.add_argument("--batch_size", type=int, default=64)
    ap_tr.add_argument("--epochs", type=int, default=50)
    ap_tr.add_argument("--lr", type=float, default=1e-3)
    ap_tr.add_argument("--dropout", type=float, default=0.2)
    ap_tr.add_argument("--augment", action="store_true")
    ap_tr.add_argument("--workers", type=int, default=4)
    ap_tr.add_argument("--seed", type=int, default=2025)
    ap_tr.add_argument("--input_dim", type=int, default=None,
                       help="可覆盖自动推断（通常不需要）")
    ap_tr.add_argument("--ckpt_out", type=str, default="best_imu_tcn.pt",
                       help="保存的权重路径（会自动加上 epoch 和 acc）")
    ap_tr.set_defaults(func=run_train)

    # ---- test 子命令
    ap_te = sub.add_parser("test", help="用 ckpt 在 test.csv 上评估")
    ap_te.add_argument("--test_csv", required=True)
    ap_te.add_argument("--ckpt", type=str, default="best_imu_tcn.pt")
    ap_te.add_argument("--batch_size", type=int, default=128)
    ap_te.add_argument("--workers", type=int, default=4)
    ap_te.add_argument("--seed", type=int, default=2025)
    ap_te.set_defaults(func=run_test)

    args = ap.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
