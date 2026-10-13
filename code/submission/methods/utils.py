"""通用工具：预算解析、类别索引、原型统计、图像形状归一化。"""

from __future__ import annotations

from typing import Any

import numpy as np

NUM_CLASSES = 10


# --------------------------------------------------------------------------- #
# 预算解析（兼容 task["poison_budget"] / 各种命名与比例写法）
# --------------------------------------------------------------------------- #
def resolve_budget(task: dict, images: np.ndarray, default_rate: float = 0.05) -> int:
    """从 task 中解析投毒预算（被修改样本数的上限）。

    优先使用显式预算字段（赛题规范为 ``task["poison_budget"]``）；
    若只给了比例则按样本数换算；都没有则回退到默认 5%。
    """
    n = len(images)
    for key in ("poison_budget", "budget", "n_modified", "max_modified", "k"):
        if key in task and task[key] is not None:
            try:
                v = int(task[key])
            except (TypeError, ValueError):
                continue
            if v > 0:
                return int(min(v, n))
    for key in ("poison_rate", "poison_ratio", "rate"):
        if key in task and task[key] is not None:
            try:
                r = float(task[key])
            except (TypeError, ValueError):
                continue
            if 0.0 < r <= 1.0:
                return int(min(max(1, int(r * n)), n))
    return int(min(max(1, int(default_rate * n)), n))


# --------------------------------------------------------------------------- #
# 基本统计
# --------------------------------------------------------------------------- #
def as_image_array(images: Any) -> np.ndarray:
    """统一为 [N,1,28,28] float32（兼容 [N,784]、[N,28,28] 输入）。"""
    arr = np.asarray(images, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr.reshape(arr.shape[0], 1, 28, 28)
    elif arr.ndim == 3:
        arr = arr.reshape(arr.shape[0], 1, *arr.shape[1:])
    elif arr.ndim != 4:
        raise ValueError(f"不支持的图像形状: {arr.shape}")
    return arr


def flatten(images: np.ndarray) -> np.ndarray:
    return np.asarray(images, dtype=np.float32).reshape(len(images), -1)


def class_indices(labels: np.ndarray) -> dict:
    labels = np.asarray(labels, dtype=np.int64)
    return {c: np.flatnonzero(labels == c) for c in range(NUM_CLASSES)}


def centroids(flat: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """类内均值原型（最近质心分类器的决策依据）。"""
    flat = np.asarray(flat, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    cent = np.zeros((NUM_CLASSES, flat.shape[1]), dtype=np.float64)
    for c in range(NUM_CLASSES):
        m = labels == c
        if m.any():
            cent[c] = flat[m].mean(axis=0)
    return cent


def nearest_class_map(flat: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """每个类别的「原型最近异类」，用于选择标签翻转目标。"""
    cent = centroids(flat, labels)
    d = np.linalg.norm(cent[:, None, :] - cent[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    return d.argmin(axis=1).astype(np.int64)
