"""选择策略模块：样本选择 × 目标类选择 的组合化标签投毒。

实验动机：同样 5% 预算，改「选哪些样本」和「改成哪一类」会带来数量级差异。
本模块把两者拆成独立策略，便于网格搜索找出最强组合。
"""

from __future__ import annotations

import numpy as np

from .attacks import AttackContext, _flip_labels
from .common import NUM_CLASSES, class_indices, confusion_matrix, flatten, nearest_class_map


# --------------------------------------------------------------------------- #
# 样本选择策略：返回 (类别 -> 该类别内按优先级排序的索引)
# --------------------------------------------------------------------------- #
def order_samples(ctx: AttackContext, strategy: str) -> dict:
    labels = np.asarray(ctx.labels, dtype=np.int64)
    cls_idx = class_indices(labels)
    flat = flatten(ctx.images)
    proba = ctx.surro_proba
    out = {}

    if strategy == "random":
        rng = np.random.default_rng(ctx.seed)
        for c in range(NUM_CLASSES):
            ci = cls_idx[c].copy()
            rng.shuffle(ci)
            out[c] = ci

    elif strategy == "confident":
        conf = _true_class_conf(ctx)
        for c in range(NUM_CLASSES):
            ci = cls_idx[c]
            out[c] = ci[np.argsort(-conf[ci])]

    elif strategy == "uncertain":
        conf = _true_class_conf(ctx)
        for c in range(NUM_CLASSES):
            ci = cls_idx[c]
            # 二类边际最小 = 最接近决策边界
            out[c] = ci[np.argsort(conf[ci])]

    elif strategy == "dense":
        # 类内距离均值最小的样本 = 最典型（k-NN 密度代理）
        for c in range(NUM_CLASSES):
            ci = cls_idx[c]
            if len(ci) <= 1:
                out[c] = ci
                continue
            sub = flat[ci]
            idx = np.arange(len(ci)) if len(ci) <= 400 else np.random.default_rng(ctx.seed).choice(
                len(ci), 400, replace=False)
            d = np.linalg.norm(sub[idx][:, None, :] - sub[None, :, :], axis=2)
            mean_d = d.mean(axis=1)
            order_local = np.argsort(mean_d)
            out[c] = ci[idx][order_local]

    elif strategy == "sparse":
        for c in range(NUM_CLASSES):
            ci = cls_idx[c]
            if len(ci) <= 1:
                out[c] = ci
                continue
            sub = flat[ci]
            idx = np.arange(len(ci)) if len(ci) <= 400 else np.random.default_rng(ctx.seed).choice(
                len(ci), 400, replace=False)
            d = np.linalg.norm(sub[idx][:, None, :] - sub[None, :, :], axis=2)
            mean_d = d.mean(axis=1)
            out[c] = ci[idx][np.argsort(-mean_d)]

    elif strategy == "margin_other":
        # 对「原类」概率低、对目标类概率高的样本（最容易被推过去）
        target_of = ctx.proxies.get("target_of")
        for c in range(NUM_CLASSES):
            ci = cls_idx[c]
            if proba is None:
                out[c] = ci
                continue
            t = int(target_of[c])
            marg = proba[ci, c] - proba[ci, t]
            out[c] = ci[np.argsort(marg)]

    else:
        raise ValueError(f"未知样本选择策略: {strategy}")
    return out


def _true_class_conf(ctx: AttackContext) -> np.ndarray:
    if ctx.surro_proba is None:
        return np.ones(ctx.n, dtype=np.float64)
    return ctx.surro_proba[np.arange(ctx.n), np.asarray(ctx.labels, dtype=np.int64)]


# --------------------------------------------------------------------------- #
# 目标类策略：返回长度 10 的数组
# --------------------------------------------------------------------------- #
def target_classes(ctx: AttackContext, strategy: str) -> np.ndarray:
    labels = np.asarray(ctx.labels, dtype=np.int64)
    target_of = ctx.proxies.get("target_of")
    if target_of is None:
        target_of = nearest_class_map(flatten(ctx.images), labels)

    if strategy == "nearest":
        return np.asarray(target_of, dtype=np.int64)

    if strategy == "confusion":
        if ctx.test_proba is None or ctx.test_labels is None:
            return np.asarray(target_of, dtype=np.int64)
        cm = confusion_matrix(ctx.test_labels, ctx.test_proba.argmax(axis=1))
        m = np.array(cm, dtype=np.float64, copy=True)
        np.fill_diagonal(m, -1.0)
        return m.argmax(axis=1).astype(np.int64)

    if strategy == "random":
        rng = np.random.default_rng(ctx.seed + 1)
        return (np.arange(NUM_CLASSES) + rng.integers(1, NUM_CLASSES, size=NUM_CLASSES)) % NUM_CLASSES

    if strategy == "symmetric":
        # 固定对称置换：把每个类改成 (c+5)%10，最大化与近邻类的语义冲突
        return ((np.arange(NUM_CLASSES) + 5) % NUM_CLASSES).astype(np.int64)

    if strategy == "farthest":
        feats = flatten(ctx.images).astype(np.float64)
        cent = np.zeros((NUM_CLASSES, feats.shape[1]))
        for c in range(NUM_CLASSES):
            m = labels == c
            if m.any():
                cent[c] = feats[m].mean(axis=0)
        d = np.linalg.norm(cent[:, None, :] - cent[None, :, :], axis=2)
        np.fill_diagonal(d, -np.inf)
        return d.argmax(axis=1).astype(np.int64)

    raise ValueError(f"未知目标类策略: {strategy}")


# --------------------------------------------------------------------------- #
# 组合攻击：按类别轮转消耗预算（保证多类同时受损）
# --------------------------------------------------------------------------- #
def selective_flip(ctx: AttackContext, sample_strategy: str = "confident",
                   target_strategy: str = "nearest", per_class_cap: bool = True) -> dict:
    """样本选择策略 × 目标类策略 的组合标签投毒。"""
    labels = np.asarray(ctx.labels, dtype=np.int64)
    order = order_samples(ctx, sample_strategy)
    targets = target_classes(ctx, target_strategy)
    sizes = {c: len(order[c]) for c in range(NUM_CLASSES)}

    idx_all, new_all = [], []
    ptr = {c: 0 for c in range(NUM_CLASSES)}
    remaining = ctx.budget

    if per_class_cap:
        # 轮转：每轮每个类取 1 个，保证受损类别尽量多
        while remaining > 0:
            progressed = False
            for c in range(NUM_CLASSES):
                if remaining <= 0:
                    break
                p = ptr[c]
                if p >= sizes[c]:
                    continue
                idx_all.append(order[c][p])
                new_all.append(int(targets[c]))
                ptr[c] = p + 1
                remaining -= 1
                progressed = True
            if not progressed:
                break
    else:
        # 按类别规模比例分配
        total = sum(sizes.values())
        for c in np.argsort(-np.array([sizes[c] for c in range(NUM_CLASSES)])):
            if remaining <= 0:
                break
            take = min(sizes[c], remaining)
            idx_all.extend(order[c][:take].tolist())
            new_all.extend([int(targets[c])] * take)
            remaining -= take

    idx = np.asarray(idx_all, dtype=np.int64)
    new = np.asarray(new_all, dtype=np.int64)
    return {
        "images": np.asarray(ctx.images, dtype=np.float32).copy(),
        "labels": _flip_labels(labels, idx, new),
        "indices": idx,
        "note": f"select={sample_strategy} target={target_strategy} cap={per_class_cap}",
    }


# --------------------------------------------------------------------------- #
# 最终候选：标签翻转 + 特征推移（复合攻击）
# --------------------------------------------------------------------------- #
def flip_to_predicted(ctx: AttackContext, pool: str = "hard",
                      prefer_borderline: bool = False) -> dict:
    """把「代理模型已判错/最不确定」的训练样本标签改成代理模型的预测类。

    原理：这些样本处于本类的边缘（模型自己都判错），把它们的标签改成模型的
    预测类，会让模型把「难分样本」的特征区域整体划给邻类，从而波及干净测试
    集中同类的难样本——这是对 MLP 最有效的可复现增强之一。
    """
    labels = np.asarray(ctx.labels, dtype=np.int64)
    proba = ctx.surro_proba
    if proba is None:
        raise ValueError("flip_to_predicted 需要 surro_proba")
    pred = proba.argmax(axis=1)
    true_conf = proba[np.arange(ctx.n), labels]
    margin = np.sort(proba, axis=1)[:, -1] - np.sort(proba, axis=1)[:, -2]

    if pool == "hard":
        score = -true_conf                      # 越不确定越优先
    elif pool == "misclassified":
        score = np.where(pred != labels, -1e9, -true_conf)
    elif pool == "borderline":
        score = margin                          # 边际越小越优先
    else:
        score = -true_conf

    cls_idx = class_indices(labels)
    order = {}
    for c in range(NUM_CLASSES):
        ci = cls_idx[c]
        order[c] = ci[np.argsort(score[ci])]

    ptr = {c: 0 for c in range(NUM_CLASSES)}
    idx_all, new_all = [], []
    remaining = ctx.budget
    while remaining > 0:
        progressed = False
        for c in range(NUM_CLASSES):
            if remaining <= 0:
                break
            p = ptr[c]
            if p >= len(order[c]):
                continue
            i = int(order[c][p])
            tgt = int(pred[i])
            if tgt == int(labels[i]):
                # 模型判对且高置信：改成一个近邻类仍有价值
                tgt = int(ctx.proxies.get("target_of", np.zeros(NUM_CLASSES, dtype=np.int64))[c])
            idx_all.append(i)
            new_all.append(tgt)
            ptr[c] = p + 1
            remaining -= 1
            progressed = True
        if not progressed:
            break

    idx = np.asarray(idx_all, dtype=np.int64)
    new = np.asarray(new_all, dtype=np.int64)
    return {
        "images": np.asarray(ctx.images, dtype=np.float32).copy(),
        "labels": _flip_labels(labels, idx, new),
        "indices": idx,
        "note": f"改为代理预测类（pool={pool}）",
    }


def compound_flip_shift(ctx: AttackContext, sample_strategy: str = "confident",
                        target_strategy: str = "nearest", pixel_budget: int = 30,
                        scale: float = 0.8) -> dict:
    """先按策略选样本并翻转标签，再把图像朝目标类模板推移若干像素。

    同时破坏监督信号与特征分布：投毒样本既带错误标签，又处于两类之间的歧义带。
    """
    base = selective_flip(ctx, sample_strategy, target_strategy)
    idx = base["indices"]
    labels = np.asarray(ctx.labels, dtype=np.int64)
    targets = target_classes(ctx, target_strategy)

    flat = flatten(ctx.images).copy()
    cent = np.zeros((NUM_CLASSES, flat.shape[1]), dtype=np.float64)
    for c in range(NUM_CLASSES):
        m = labels == c
        if m.any():
            cent[c] = flat[m].mean(axis=0)

    for i in idx:
        a = int(labels[i])
        b = int(targets[a])
        delta = np.clip(cent[b] - cent[a], 0.0, None)
        if delta.max() <= 0:
            continue
        top = np.argsort(-delta)[:pixel_budget]
        flat[i, top] = np.clip(flat[i, top] + scale * delta[top], 0.0, 1.0)

    return {
        "images": flat.reshape(np.asarray(ctx.images).shape),
        "labels": base["labels"],
        "indices": idx,
        "note": f"复合：select={sample_strategy} target={target_strategy} "
                f"pixels={pixel_budget} scale={scale}",
    }
