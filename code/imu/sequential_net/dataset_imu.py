#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# ================== 可调参数（可被 __init__ 覆盖） ==================
KEEP_COLS = [
    "AccX", "AccY", "AccZ",
    "GX",  "GY",  "GZ",
    "AsX", "AsY", "AsZ",
    "AngleX", "AngleY", "AngleZ",
]
TARGET_HZ = 50
TARGET_T  = 128
TIME_FMT  = "%Y-%m-%d %H:%M:%S.%f"
# ===================================================================

def _clip_id_from_dir(clip_dir: Path) -> int:
    return int(str(clip_dir.name).split("_")[-1])

def _exp_dir_from_clip_dir(clip_dir: Path) -> Path:
    # .../exp_xxx/imu_clips/clip_n -> .../exp_xxx
    return clip_dir.parent.parent

def _read_clip_csv(clip_dir: Path,
                   side: str,
                   keep_cols: List[str],
                   strict: bool = True) -> Optional[pd.DataFrame]:
    """
    读取单侧 CSV，并按固定格式解析时间。
    - strict=True: 若解析后为空或缺列，抛异常（训练期用 _load_one 时使用）
    - strict=False: 若异常/为空，返回 None（用于数据集构建期的预筛）
    """
    cid = _clip_id_from_dir(clip_dir)
    fp = clip_dir / f"{side}_clip_{cid}.csv"
    try:
        if not fp.exists():
            raise FileNotFoundError(fp)

        df = pd.read_csv(fp, low_memory=False)
        if "AlignedTime" not in df.columns:
            raise KeyError(f"{fp} 缺少 AlignedTime 列")

        # 固定格式解析，避免推断警告与性能损失
        df["AlignedTime"] = pd.to_datetime(df["AlignedTime"], format=TIME_FMT, errors="coerce")
        df = df.dropna(subset=["AlignedTime"]).reset_index(drop=True)
        if len(df) == 0:
            raise ValueError(f"{fp} 解析 AlignedTime 后为空（期望格式 {TIME_FMT}）")

        missing = [c for c in keep_cols if c not in df.columns]
        if missing:
            raise KeyError(f"{fp} 缺少列: {missing}")

        return df[["AlignedTime"] + keep_cols].sort_values("AlignedTime").reset_index(drop=True)
    except Exception as e:
        if strict:
            raise
        else:
            print(f"[SKIP] {e}")
            return None

def _resample(df: pd.DataFrame, cols: List[str], hz: int) -> pd.DataFrame:
    df = df.set_index("AlignedTime").sort_index()
    rule = f"{int(1000 / hz)}ms"   # 'L' 已废弃，改用 'ms'
    out = df[cols].resample(rule).mean()
    out = out.interpolate("time").dropna()
    out.reset_index(inplace=True)
    return out

def _fix_length(x: np.ndarray, T: int) -> np.ndarray:
    n = len(x)
    if n == T: return x
    if n == 0: return np.zeros((T, x.shape[1]), dtype=np.float32)
    if n > T:
        idx = np.linspace(0, n - 1, T).astype(np.int64)
        return x[idx]
    reps = int(np.ceil(T / n))
    return np.tile(x, (reps, 1))[:T]

def _zscore(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    std = np.where(std < 1e-6, 1.0, std)
    return (x - mean) / std


class ImuClips(Dataset):
    """
    读取两列 CSV：clip_path,label
    - clip_path: 目录，含 left_clip_{id}.csv / right_clip_{id}.csv
    - label: 标签字符串（用 label_map 映射）
    输出：x:(D,T)、y:int；其中 D = len(keep_cols)*2
    """

    def __init__(self,
                 csv_list_path: str,
                 label_map: Dict[str, int],
                 train: bool = True,
                 compute_norm: bool = False,
                 norm_stats: Optional[Dict[str, np.ndarray]] = None,
                 target_hz: int = TARGET_HZ,
                 target_T: int = TARGET_T,
                 keep_cols: Optional[List[str]] = None,
                 augment: bool = False):
        """
        注意：本版本 **不再做两层过滤**，也 **移除了 limit_per_exp**。
        直接按 CSV 中的路径读取，若样本有问题：
          - 统计均值/方差阶段：跳过坏样本并提示
          - 训练/取样阶段：严格读取，若坏样本会抛错（方便定位）
        """
        self.csv_list_path = csv_list_path
        self.df = pd.read_csv(csv_list_path)
        assert "clip_path" in self.df.columns and "label" in self.df.columns, \
            f"{csv_list_path} 必须包含列 clip_path,label"

        self.label_map = label_map
        self.train = train
        self.augment = augment
        self.hz = target_hz
        self.T = target_T
        self.keep_cols = keep_cols if keep_cols is not None else KEEP_COLS

        # 直接读取 CSV 列表，映射 label（不再做任何过滤/存在性检查）
        self.items: List[Tuple[Path, int]] = []
        skipped_label = 0
        for _, row in self.df.iterrows():
            clip = Path(str(row["clip_path"]).strip())
            lab_str = str(row["label"]).strip()
            if lab_str not in self.label_map:
                skipped_label += 1
                print(f"[SKIP] label 未在 label_map 中：{lab_str}（clip={clip}）")
                continue
            y = self.label_map[lab_str]
            self.items.append((clip, y))

        if len(self.items) == 0:
            raise RuntimeError("数据集为空：所有样本均被跳过（可能 label_map 不匹配）。")

        # 统计归一化：为了健壮，统计阶段对坏样本 try/except 跳过；真正训练阶段仍严格
        if compute_norm:
            xs = []
            K = min(200, len(self.items))  # 采样部分 clip 估计统计提速
            used = 0
            for i in range(K):
                clip_dir, _ = self.items[i]
                try:
                    x_i = self._load_one(clip_dir)  # 严格读
                    xs.append(x_i)
                    used += 1
                except Exception as e:
                    print(f"[NORM] 跳过坏样本（统计阶段）: {clip_dir} | {e}")
                    continue
            if used == 0:
                raise RuntimeError("统计均值/方差失败：前 K 个样本均不可用。请检查数据。")
            X = np.concatenate(xs, axis=0)      # (used*T, D)
            self.mean = X.mean(axis=0, keepdims=True).astype(np.float32)
            self.std  = X.std(axis=0, keepdims=True).astype(np.float32)
        else:
            if norm_stats is None:
                raise ValueError("compute_norm=False 时必须提供 norm_stats")
            self.mean = norm_stats["mean"]
            self.std  = norm_stats["std"]

        print(f"[DATA] 读取 {len(self.df)} 行 | 生效样本 {len(self.items)} | label 跳过 {skipped_label}")

    def norm_stats(self) -> Dict[str, np.ndarray]:
        return {"mean": self.mean, "std": self.std}

    def _load_one(self, clip_dir: Path) -> np.ndarray:
        # 训练/取样阶段保持严格：坏样本直接抛错以便定位
        dl = _read_clip_csv(clip_dir, "left",  self.keep_cols, strict=True)
        dr = _read_clip_csv(clip_dir, "right", self.keep_cols, strict=True)
        dl = _resample(dl, self.keep_cols, self.hz)
        dr = _resample(dr, self.keep_cols, self.hz)
        xl = dl[self.keep_cols].to_numpy(dtype=np.float32)
        xr = dr[self.keep_cols].to_numpy(dtype=np.float32)
        xl = _fix_length(xl, self.T)
        xr = _fix_length(xr, self.T)
        x = np.concatenate([xl, xr], axis=1)  # (T, D_left + D_right)
        return x

    def _augment(self, x: np.ndarray) -> np.ndarray:
        if np.random.rand() < 0.5:
            x = x + np.random.normal(0, 0.01, size=x.shape).astype(np.float32)
        if np.random.rand() < 0.3:
            x = (x * np.random.uniform(0.9, 1.1)).astype(np.float32)
        if np.random.rand() < 0.3:
            idx = np.random.choice(self.T, size=max(1, self.T // 10), replace=False)
            x[idx] *= (1 + np.random.uniform(-0.05, 0.05, size=(len(idx), x.shape[1])))
        return x

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        clip_dir, y = self.items[i]
        x = self._load_one(clip_dir)          # 严格读取（遇坏样本会抛错）
        if self.train and self.augment:
            x = self._augment(x)
        x = _zscore(x, self.mean, self.std)
        x = x.transpose(1, 0)  # (D, T) for 1D-CNN/TCN/Transformer
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)


















# class ImuClips(Dataset):
#     """
#     读取两列 CSV：clip_path,label
#     - clip_path: 目录，含 left_clip_{id}.csv / right_clip_{id}.csv
#     - label: 标签字符串（用 label_map 映射）
#     输出：x:(D,T)、y:int；其中 D = len(keep_cols)*2
#     """

#     def __init__(self,
#                  csv_list_path: str,
#                  label_map: Dict[str, int],
#                  train: bool = True,
#                  compute_norm: bool = False,
#                  norm_stats: Optional[Dict[str, np.ndarray]] = None,
#                  target_hz: int = TARGET_HZ,
#                  target_T: int = TARGET_T,
#                  keep_cols: Optional[List[str]] = None,
#                  augment: bool = False,
#                  limit_per_exp: Optional[int] = None):
#         """
#         Args:
#             csv_list_path: 含 clip_path,label 的 CSV
#             label_map: {label_str: int_id}
#             train: 是否训练集（仅影响是否做增强）
#             compute_norm: 是否从当前集合估计均值/方差（建议只在训练集上启用）
#             norm_stats: 已有 {"mean":..., "std":...}（验证/测试集必传或 compute_norm=True）
#             target_hz: 重采样频率
#             target_T: 固定帧数
#             keep_cols: 使用的通道列表；None 则用 KEEP_COLS
#             augment: 是否做轻量数据增强（仅 train=True 时生效）
#             limit_per_exp: 每个 exp 仅保留前 N 个 clip；None/<=0 则不限制
#                            该限制对 train/val/test 一致生效
#         """
#         self.csv_list_path = csv_list_path
#         self.df = pd.read_csv(csv_list_path)
#         assert "clip_path" in self.df.columns and "label" in self.df.columns, \
#             f"{csv_list_path} 必须包含列 clip_path,label"

#         self.label_map = label_map
#         self.train = train
#         self.augment = augment
#         self.hz = target_hz
#         self.T = target_T
#         self.keep_cols = keep_cols if keep_cols is not None else KEEP_COLS
#         self.limit_per_exp = (None if limit_per_exp is None or int(limit_per_exp) <= 0
#                               else int(limit_per_exp))

#         # 读取 CSV 样本（先根据文件存在性构建列表）
#         raw_items: List[Tuple[Path, str]] = []
#         for _, row in self.df.iterrows():
#             clip = Path(str(row["clip_path"]).strip())
#             lab_str = str(row["label"]).strip()
#             if not clip.exists():
#                 print(f"[SKIP] clip 目录不存在：{clip}")
#                 continue
#             cid = _clip_id_from_dir(clip)
#             l_csv = clip / f"left_clip_{cid}.csv"
#             r_csv = clip / f"right_clip_{cid}.csv"
#             if not (l_csv.exists() and r_csv.exists()):
#                 print(f"[SKIP] 缺少左右手 CSV：{l_csv} / {r_csv}")
#                 continue
#             raw_items.append((clip, lab_str))

#         # 进一步：内容有效性预筛（时间可解析且非空）
#         validated: List[Tuple[Path, str]] = []
#         for clip, lab_str in raw_items:
#             dl = _read_clip_csv(clip, "left",  self.keep_cols, strict=False)
#             dr = _read_clip_csv(clip, "right", self.keep_cols, strict=False)
#             if dl is None or dr is None or len(dl) == 0 or len(dr) == 0:
#                 # _read_clip_csv(strict=False) 已打印了详细原因
#                 # 这里再补充一条 clip 级别的汇总提示
#                 print(f"[SKIP] 无效 clip（解析后为空）：{clip}")
#                 continue
#             validated.append((clip, lab_str))

#         # 按 exp 分组，保留前 N 个 clip（如果设置了 limit）
#         if self.limit_per_exp is not None:
#             grouped = defaultdict(list)
#             for clip, lab_str in validated:
#                 grouped[str(_exp_dir_from_clip_dir(clip))].append(
#                     (clip, lab_str, _clip_id_from_dir(clip))
#                 )
#             filtered: List[Tuple[Path, str]] = []
#             for _, triplets in grouped.items():
#                 triplets.sort(key=lambda x: x[2])  # 按 clip_id 升序
#                 take = triplets[: self.limit_per_exp]
#                 filtered.extend([(c, l) for (c, l, _) in take])
#             validated = filtered

#         kept_before = len(self.df)
#         kept_after_files = len(raw_items)
#         kept_after_valid = len(validated)
#         print(f"[DATA] 输入 {kept_before} 行 | 通过文件存在性 {kept_after_files} | 通过内容校验 {kept_after_valid}")

#         # 生成最终样本
#         self.items: List[Tuple[Path, int]] = []
#         for clip, lab_str in validated:
#             if lab_str not in self.label_map:
#                 print(f"[SKIP] label 未在 label_map 中：{lab_str}（clip={clip}）")
#                 continue
#             y = self.label_map[lab_str]
#             self.items.append((clip, y))

#         if len(self.items) == 0:
#             raise RuntimeError("数据集为空：所有样本均被跳过。请检查 CSV 与原始数据。")

#         # 统计归一化
#         if compute_norm:
#             xs = []
#             K = min(200, len(self.items))  # 采样部分 clip 估计统计提速
#             for i in range(K):
#                 clip_dir, _ = self.items[i]
#                 xs.append(self._load_one(clip_dir))
#             X = np.concatenate(xs, axis=0)      # (K*T, D)
#             self.mean = X.mean(axis=0, keepdims=True).astype(np.float32)
#             self.std  = X.std(axis=0, keepdims=True).astype(np.float32)
#         else:
#             if norm_stats is None:
#                 raise ValueError("compute_norm=False 时必须提供 norm_stats")
#             self.mean = norm_stats["mean"]
#             self.std  = norm_stats["std"]

#     def norm_stats(self) -> Dict[str, np.ndarray]:
#         return {"mean": self.mean, "std": self.std}

#     def _load_one(self, clip_dir: Path) -> np.ndarray:
#         # 这里使用 strict=True，若训练中仍遇到不一致数据，可暴露异常，方便定位
#         dl = _read_clip_csv(clip_dir, "left",  self.keep_cols, strict=True)
#         dr = _read_clip_csv(clip_dir, "right", self.keep_cols, strict=True)
#         dl = _resample(dl, self.keep_cols, self.hz)
#         dr = _resample(dr, self.keep_cols, self.hz)
#         xl = dl[self.keep_cols].to_numpy(dtype=np.float32)
#         xr = dr[self.keep_cols].to_numpy(dtype=np.float32)
#         xl = _fix_length(xl, self.T)
#         xr = _fix_length(xr, self.T)
#         x = np.concatenate([xl, xr], axis=1)  # (T, D_left + D_right)
#         return x

#     def _augment(self, x: np.ndarray) -> np.ndarray:
#         if np.random.rand() < 0.5:
#             x = x + np.random.normal(0, 0.01, size=x.shape).astype(np.float32)
#         if np.random.rand() < 0.3:
#             x = (x * np.random.uniform(0.9, 1.1)).astype(np.float32)
#         if np.random.rand() < 0.3:
#             idx = np.random.choice(self.T, size=max(1, self.T // 10), replace=False)
#             x[idx] *= (1 + np.random.uniform(-0.05, 0.05, size=(len(idx), x.shape[1])))
#         return x

#     def __len__(self) -> int:
#         return len(self.items)

#     def __getitem__(self, i: int):
#         clip_dir, y = self.items[i]
#         x = self._load_one(clip_dir)
#         if self.train and self.augment:
#             x = self._augment(x)
#         x = _zscore(x, self.mean, self.std)
#         x = x.transpose(1, 0)  # (D, T) for 1D-CNN/TCN/Transformer
#         return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)
