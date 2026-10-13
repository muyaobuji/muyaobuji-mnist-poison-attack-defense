"""第三批攻击：针对「最近质心 / kNN 型统一模型」的结构化攻击。

核心洞察：最近质心分类器的决策完全由各类原型决定，
    argmin_c ||x - μ_c||²  =  argmax_c ( 2·x·μ_c - ||μ_c||² )
因此投毒只需改变 μ_c。把被投毒样本放进「目标类」区域并改标签，
等价于同时（a）让目标类多一个正确投票，（b）让源类少一个正确投票，
当样本与源类原型差异较大时是「双重打击」。
"""

from __future__ import annotations

import numpy as np

from .attacks import AttackContext, _flip_labels
from .common import NUM_CLASSES, class_indices, confusion_matrix, flatten, nearest_class_map


def _centroids(flat: np.ndarray, labels: np.ndarray) -> np.ndarray:
    cent = np.zeros((NUM_CLASSES, flat.shape[1]), dtype=np.float64)
    for c in range(NUM_CLASSES):
        m = labels == c
        if m.any():
            cent[c] = flat[m].mean(axis=0)
    return cent


def _pair_priority(ctx: AttackContext, target_of: np.ndarray) -> np.ndarray:
    """按「测试集上 c -> target_of[c] 的混淆量」给类别排序（大者优先）。"""
    if ctx.test_proba is not None and ctx.test_labels is not None:
        cm = confusion_matrix(ctx.test_labels, ctx.test_proba.argmax(axis=1))
        score = np.array([cm[c, int(target_of[c])] for c in range(NUM_CLASSES)], dtype=np.float64)
    else:
        score = np.ones(NUM_CLASSES, dtype=np.float64)
    return np.argsort(-score)


def _pick_round_robin(pools: dict, order: np.ndarray, budget: int) -> np.ndarray:
    """按类别优先级轮转取样本，避免预算被单一类别耗尽。"""
    ptr = {c: 0 for c in pools}
    idx = []
    while len(idx) < budget:
        progressed = False
        for c in order:
            if len(idx) >= budget:
                break
            p = ptr[int(c)]
            if p >= len(pools[int(c)]):
                continue
            idx.append(pools[int(c)][p])
            ptr[int(c)] = p + 1
            progressed = True
        if not progressed:
            break
    return np.asarray(idx, dtype=np.int64)


def _lever_pools(flat, labels, target_of, prefer_far: bool, max_pool: int = 200):
    """构造各类别的候选池（可选用「离本类原型最远」的杠杆样本）。

    prefer_far=True 时优先挑离源类原型远的样本（替换后对源类原型拉动更大）。
    """
    cls_idx = class_indices(labels)
    cent = _centroids(flat, labels)
    pools = {}
    for c in range(NUM_CLASSES):
        ci = cls_idx[c]
        if len(ci) == 0:
            pools[c] = ci
            continue
        d_self = np.linalg.norm(flat[ci] - cent[c], axis=1)
        if prefer_far:
            order = np.argsort(-d_self)
        else:
            # 优先挑「离目标类原型近」的样本（替换代价小、目标类原型拉动小）
            t = int(target_of[c])
            d_tgt = np.linalg.norm(flat[ci] - cent[t], axis=1)
            order = np.argsort(d_tgt)
        sub = ci[order]
        pools[c] = sub[:max_pool] if len(sub) > max_pool else sub
    return pools, cent


# --------------------------------------------------------------------------- #
# 1. 图像替换（标签不变）：把源类样本换成目标类样本
# --------------------------------------------------------------------------- #
def image_replacement(ctx: AttackContext, prefer_far: bool = True,
                      target_mode: str = "nearest") -> dict:
    """把预算内的源类样本替换为目标类的近邻样本，标签保持不变。

    对最近质心模型：源类原型被拉向目标类，目标类原型被拉离源类。
    """
    images = np.asarray(ctx.images, dtype=np.float32)
    labels = np.asarray(ctx.labels, dtype=np.int64)
    flat = flatten(images).copy()
    target_of = ctx.proxies.get("target_of")
    if target_of is None:
        target_of = nearest_class_map(flat, labels)
    target_of = np.asarray(target_of, dtype=np.int64)

    order = _pair_priority(ctx, target_of)
    pools, cent = _lever_pools(flat, labels, target_of, prefer_far)
    idx = _pick_round_robin(pools, order, ctx.budget)

    cls_idx = class_indices(labels)
    out = flat.copy()
    for i in idx:
        a = int(labels[i])
        b = int(target_of[a])
        cand = cls_idx[b]
        if len(cand) == 0:
            continue
        # 用目标类原型 + 该样本与源类原型的差，构造一个"更贴近目标类"的图像
        delta = flat[i] - cent[a]
        cand_img = np.clip(cent[b] + 0.6 * delta, 0.0, 1.0)
        out[i] = cand_img.astype(np.float32)

    return {
        "images": out.reshape(images.shape),
        "labels": labels.copy(),
        "indices": idx,
        "note": f"图像替换（clean-label, prefer_far={prefer_far}）",
    }


# --------------------------------------------------------------------------- #
# 2. 图像替换 + 标签翻转（双重打击）
# --------------------------------------------------------------------------- #
def replacement_plus_flip(ctx: AttackContext, prefer_far: bool = True) -> dict:
    """把源类样本改成目标类图像，同时把标签改成目标类。"""
    images = np.asarray(ctx.images, dtype=np.float32)
    labels = np.asarray(ctx.labels, dtype=np.int64)
    flat = flatten(images).copy()
    target_of = ctx.proxies.get("target_of")
    if target_of is None:
        target_of = nearest_class_map(flat, labels)
    target_of = np.asarray(target_of, dtype=np.int64)

    order = _pair_priority(ctx, target_of)
    pools, cent = _lever_pools(flat, labels, target_of, prefer_far)
    idx = _pick_round_robin(pools, order, ctx.budget)

    out = flat.copy()
    new = labels.copy()
    for i in idx:
        a = int(labels[i])
        b = int(target_of[a])
        delta = flat[i] - cent[a]
        out[i] = np.clip(cent[b] + 0.6 * delta, 0.0, 1.0).astype(np.float32)
        new[i] = b

    return {
        "images": out.reshape(images.shape),
        "labels": new,
        "indices": idx,
        "note": f"图像替换 + 标签翻转（prefer_far={prefer_far}）",
    }


# --------------------------------------------------------------------------- #
# 3. 杠杆标签翻转：优先翻转「离本类原型远」的样本
# --------------------------------------------------------------------------- #
def lever_label_flip(ctx: AttackContext, target_mode: str = "nearest") -> dict:
    """翻转样本标签，但优先选「离本类原型远」的样本（对原型拉动最大）。"""
    labels = np.asarray(ctx.labels, dtype=np.int64)
    flat = flatten(ctx.images)
    target_of = ctx.proxies.get("target_of")
    if target_of is None:
        target_of = nearest_class_map(flat, labels)
    target_of = np.asarray(target_of, dtype=np.int64)
    if target_mode == "farthest":
        cent = _centroids(flat, labels)
        d = np.linalg.norm(cent[:, None, :] - cent[None, :, :], axis=2)
        np.fill_diagonal(d, -np.inf)
        target_of = d.argmax(axis=1).astype(np.int64)

    order = _pair_priority(ctx, target_of)
    pools, _ = _lever_pools(flat, labels, target_of, prefer_far=True)
    idx = _pick_round_robin(pools, order, ctx.budget)
    new = np.array([int(target_of[int(labels[i])]) for i in idx], dtype=np.int64)

    return {
        "images": np.asarray(ctx.images, dtype=np.float32).copy(),
        "labels": _flip_labels(labels, idx, new),
        "indices": idx,
        "note": f"杠杆标签翻转（离本类原型最远优先, target={target_mode}）",
    }
