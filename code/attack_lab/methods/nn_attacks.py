"""基于纯 NumPy 代理模型的梯度类攻击方法（无 torch 依赖）。

包含：
  - grad_flip        : 翻转标签 + FGSM 型输入扰动（放大错误）
  - emn_noise        : 误差最小化噪声（EMN 简化版，交替优化）
  - influence_select : 用输入梯度范数/损失估计样本「影响力」，挑选最毒的样本翻转
"""

from __future__ import annotations

import numpy as np

from .attacks import AttackContext, _flip_labels
from .common import NUM_CLASSES, class_indices, confusion_matrix, flatten, nearest_class_map
from .nn import NumpyMLP


# --------------------------------------------------------------------------- #
# 公共：选样本 + 选目标
# --------------------------------------------------------------------------- #
def _select_by_confusion(ctx: AttackContext, k: int):
    """按「该类→目标类」的测试集混淆量降序，轮转选取 k 个样本索引。"""
    labels = np.asarray(ctx.labels, dtype=np.int64)
    target_of = ctx.proxies.get("target_of")
    if target_of is None:
        target_of = nearest_class_map(flatten(ctx.images), labels)
    cls_idx = class_indices(labels)

    if ctx.test_proba is not None and ctx.test_labels is not None:
        cm = confusion_matrix(ctx.test_labels, ctx.test_proba.argmax(axis=1))
        score = np.array([cm[c, int(target_of[c])] for c in range(NUM_CLASSES)], dtype=np.float64)
    else:
        score = np.ones(NUM_CLASSES)
    order = np.argsort(-score)

    ptr = {c: 0 for c in range(NUM_CLASSES)}
    idx = []
    while len(idx) < k:
        progressed = False
        for c in order:
            if len(idx) >= k:
                break
            p = ptr[c]
            if p >= len(cls_idx[c]):
                continue
            idx.append(cls_idx[c][p])
            ptr[c] = p + 1
            progressed = True
        if not progressed:
            break
    return np.asarray(idx, dtype=np.int64), np.asarray(target_of, dtype=np.int64)


def _normalize_rows(g: np.ndarray) -> np.ndarray:
    """把每个样本的梯度归一化到单位 L2 范数，避免交叉熵梯度尺度极小导致扰动无效。"""
    g = np.asarray(g, dtype=np.float32)
    nrm = np.linalg.norm(g, axis=1, keepdims=True)
    return g / np.maximum(nrm, 1e-12)


def _fit_surrogate(ctx: AttackContext, images: np.ndarray, labels: np.ndarray,
                   epochs: int = 12, seed: int = 42) -> NumpyMLP:
    flat = flatten(images) if images.ndim == 4 else np.asarray(images, dtype=np.float32)
    mlp = NumpyMLP(seed=seed)
    mlp.fit(flat, np.asarray(labels, dtype=np.int64), epochs=epochs, seed=seed)
    return mlp


# --------------------------------------------------------------------------- #
# 1. 标签翻转 + FGSM 型扰动
# --------------------------------------------------------------------------- #
def grad_flip(ctx: AttackContext, eps: float = 0.15, epochs: int = 12,
              ascent: bool = True, seed: int = 42) -> dict:
    """翻转标签后沿「对投毒标签的损失上升方向」扰动输入，使样本更难被拟合正确。"""
    labels = np.asarray(ctx.labels, dtype=np.int64)
    idx, target_of = _select_by_confusion(ctx, ctx.budget)
    new = np.array([int(target_of[int(labels[i])]) for i in idx], dtype=np.int64)

    flat = flatten(ctx.images).astype(np.float32)
    mlp = _fit_surrogate(ctx, flat, labels, epochs=epochs, seed=seed)
    g = _normalize_rows(mlp.input_grad(flat[idx], new))
    if not ascent:
        g = -g
    flat[idx] = np.clip(flat[idx] + eps * g, 0.0, 1.0)

    return {
        "images": flat.reshape(np.asarray(ctx.images).shape),
        "labels": _flip_labels(labels, idx, new),
        "indices": idx,
        "note": f"标签翻转 + FGSM 扰动 eps={eps}（NumPy 代理）",
    }


# --------------------------------------------------------------------------- #
# 2. 误差最小化噪声（EMN 简化版）
# --------------------------------------------------------------------------- #
def emn_noise(ctx: AttackContext, steps: int = 40, eps: float = 0.2, inner: int = 40,
              lr_noise: float = 0.04, seed: int = 42, target_mode: str = "near") -> dict:
    """逐样本优化噪声，使代理模型在「投毒标签」上损失极小（让样本难以被正确学到）。

    交替进行：内层用含噪样本训练代理模型；外层沿损失下降方向更新噪声（投影到 eps 球）。
    """
    labels = np.asarray(ctx.labels, dtype=np.int64)
    idx, target_of = _select_by_confusion(ctx, ctx.budget)
    if target_mode == "far":
        # 目标改为「特征最远类」，与最近类策略互补
        target_of = _farthest_target(flatten(ctx.images), labels)
    new = np.array([int(target_of[int(labels[i])]) for i in idx], dtype=np.int64)

    flat = flatten(ctx.images).astype(np.float32)
    base = flat[idx].copy()
    mlp = NumpyMLP(seed=seed)
    rng = np.random.default_rng(seed)

    Xp = base.copy()
    for _ in range(steps):
        # 内层：在含噪投毒样本上训练代理（少量 epoch，模拟污染效果）
        perm = rng.permutation(len(idx))
        for _ in range(inner):
            for i in range(0, len(idx), 128):
                sub = perm[i:i + 128]
                mlp.step(Xp[sub], new[sub])
        # 外层：下降噪声，使代理在投毒标签上进一步"学会"
        g = _normalize_rows(mlp.input_grad(Xp, new))
        Xp = np.clip(Xp - lr_noise * g, base - eps, base + eps)
        Xp = np.clip(Xp, 0.0, 1.0)

    flat[idx] = Xp
    return {
        "images": flat.reshape(np.asarray(ctx.images).shape),
        "labels": _flip_labels(labels, idx, new),
        "indices": idx,
        "note": f"EMN 逐样本噪声 {steps}×{inner} 步, eps={eps}, target={target_mode}",
    }


def _farthest_target(feats: np.ndarray, labels: np.ndarray) -> np.ndarray:
    feats = np.asarray(feats, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    cent = np.zeros((NUM_CLASSES, feats.shape[1]))
    for c in range(NUM_CLASSES):
        m = labels == c
        if m.any():
            cent[c] = feats[m].mean(axis=0)
    d = np.linalg.norm(cent[:, None, :] - cent[None, :, :], axis=2)
    np.fill_diagonal(d, -np.inf)
    return d.argmax(axis=1).astype(np.int64)


# --------------------------------------------------------------------------- #
# 3. 影响力选样：用损失/梯度估计每个样本的破坏力
# --------------------------------------------------------------------------- #
def influence_select(ctx: AttackContext, epochs: int = 12, seed: int = 42,
                     target_strategy: str = "nearest", weight: str = "loss_grad") -> dict:
    """用代理模型估计「翻转该样本」对干净损失的提升，取 top-k 翻转。

    代理评分（无需重训）：
        score_i = L(x_i, y'_i) + ||∇_x L(x_i, y'_i)||  （错标签下的损失与敏感度）
    """
    labels = np.asarray(ctx.labels, dtype=np.int64)
    flat = flatten(ctx.images).astype(np.float32)

    target_of = ctx.proxies.get("target_of")
    if target_of is None:
        target_of = nearest_class_map(flat, labels)
    if target_strategy == "farthest":
        target_of = _farthest_target(flat, labels)
    target_of = np.asarray(target_of, dtype=np.int64)

    mlp = _fit_surrogate(ctx, flat, labels, epochs=epochs, seed=seed)
    logits, acts = mlp.forward(flat, cache=True)
    y_flip = target_of[labels]
    loss, gl = mlp.softmax_ce(logits, y_flip)
    _, _, dX = mlp.backward(acts, gl)
    per_sample = -np.log(np.maximum(
        mlp.predict_proba(flat)[np.arange(len(labels)), y_flip], 1e-12))
    if weight == "loss_grad":
        score = per_sample + np.linalg.norm(dX, axis=1)
    elif weight == "loss":
        score = per_sample
    else:
        score = np.linalg.norm(dX, axis=1)

    # 类别轮转，避免预算被单一类别吃光
    cls_idx = class_indices(labels)
    order = {c: cls_idx[c][np.argsort(-score[cls_idx[c]])] for c in range(NUM_CLASSES)}
    ptr = {c: 0 for c in range(NUM_CLASSES)}
    idx = []
    while len(idx) < ctx.budget:
        progressed = False
        for c in range(NUM_CLASSES):
            if len(idx) >= ctx.budget:
                break
            p = ptr[c]
            if p >= len(order[c]):
                continue
            idx.append(order[c][p])
            ptr[c] = p + 1
            progressed = True
        if not progressed:
            break
    idx = np.asarray(idx, dtype=np.int64)
    new = np.array([int(target_of[int(labels[i])]) for i in idx], dtype=np.int64)
    return {
        "images": flat.reshape(np.asarray(ctx.images).shape),
        "labels": _flip_labels(labels, idx, new),
        "indices": idx,
        "note": f"影响力选样（{weight}, target={target_strategy}）取 top-{len(idx)}",
    }
