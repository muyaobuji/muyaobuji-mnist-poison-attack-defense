"""公共工具：预算解析、类别统计、特征表示。"""

from __future__ import annotations

import numpy as np

NUM_CLASSES = 10


def flatten(images: np.ndarray) -> np.ndarray:
    return np.asarray(images, dtype=np.float32).reshape(len(images), -1)


def class_indices(labels: np.ndarray) -> dict:
    labels = np.asarray(labels, dtype=np.int64)
    return {c: np.flatnonzero(labels == c) for c in range(NUM_CLASSES)}


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n: int = NUM_CLASSES) -> np.ndarray:
    """行 = 真实类别，列 = 预测类别，返回计数矩阵。"""
    cm = np.zeros((n, n), dtype=np.int64)
    for t, p in zip(np.asarray(y_true, dtype=np.int64), np.asarray(y_pred, dtype=np.int64)):
        cm[int(t), int(p)] += 1
    return cm


def row_normalize(mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat, dtype=np.float64)
    denom = mat.sum(axis=1, keepdims=True)
    return mat / np.maximum(denom, 1e-12)


def nearest_class_map(probe_feats: np.ndarray, probe_labels: np.ndarray) -> np.ndarray:
    """用类内均值模板的欧氏距离，给出每个类别「最容易混淆」的目标类别。

    返回长度 10 的数组 target_of[c]，且 target_of[c] != c。
    """
    feats = np.asarray(probe_feats, dtype=np.float64)
    labels = np.asarray(probe_labels, dtype=np.int64)
    centroids = np.zeros((NUM_CLASSES, feats.shape[1]), dtype=np.float64)
    for c in range(NUM_CLASSES):
        m = labels == c
        centroids[c] = feats[m].mean(axis=0) if m.any() else 0.0
    dist = np.linalg.norm(centroids[:, None, :] - centroids[None, :, :], axis=2)
    np.fill_diagonal(dist, np.inf)
    return dist.argmin(axis=1).astype(np.int64)
