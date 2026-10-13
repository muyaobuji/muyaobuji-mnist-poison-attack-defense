"""评测端视角的攻击方法对比（只用训练数据，不依赖任何测试集信息）。

这是提交包里真正能用的方法空间。所有候选方法只吃 images/labels/budget，
在真实基准上对多个候选统一模型评测。
"""

from __future__ import annotations

import numpy as np

from .common import NUM_CLASSES, class_indices, flatten, nearest_class_map


def _margin(proba: np.ndarray) -> np.ndarray:
    s = np.sort(proba, axis=1)
    return s[:, -1] - s[:, -2]


def _runner_up(proba: np.ndarray) -> np.ndarray:
    return np.argsort(-proba, axis=1)[:, 1].astype(np.int64)


def runnerup_flip(ctx) -> dict:
    """把标签改成代理模型的第二候选类（最易混淆类）。"""
    labels = np.asarray(ctx.labels, dtype=np.int64)
    proba = ctx.surro_proba
    target = _runner_up(proba)
    cls_idx = class_indices(labels)
    score = _margin(proba)
    order = {c: cls_idx[c][np.argsort(score[cls_idx[c]])] for c in range(NUM_CLASSES)}

    ptr = {c: 0 for c in range(NUM_CLASSES)}
    idx, new = [], []
    while len(idx) < ctx.budget:
        progressed = False
        for c in range(NUM_CLASSES):
            if len(idx) >= ctx.budget:
                break
            p = ptr[c]
            if p >= len(order[c]):
                continue
            i = int(order[c][p])
            ptr[c] = p + 1
            b = int(target[i])
            if b == int(labels[i]):
                continue
            idx.append(i)
            new.append(b)
            progressed = True
        if not progressed:
            break
    idx = np.asarray(idx, dtype=np.int64)
    new = np.asarray(new, dtype=np.int64)
    return {"images": np.asarray(ctx.images, dtype=np.float32).copy(),
            "labels": _flip(labels, idx, new), "indices": idx,
            "note": "改为代理第二候选类（runner-up）"}


def runnerup_flip_confident(ctx) -> dict:
    """优先翻转代理模型「高置信且判对」的样本到第二候选类（最大矛盾）。"""
    labels = np.asarray(ctx.labels, dtype=np.int64)
    proba = ctx.surro_proba
    target = _runner_up(proba)
    true_conf = proba[np.arange(len(labels)), labels]
    cls_idx = class_indices(labels)
    # 置信度降序（越自信的样本被改标签，矛盾越大）
    order = {c: cls_idx[c][np.argsort(-true_conf[cls_idx[c]])] for c in range(NUM_CLASSES)}

    ptr = {c: 0 for c in range(NUM_CLASSES)}
    idx, new = [], []
    while len(idx) < ctx.budget:
        progressed = False
        for c in range(NUM_CLASSES):
            if len(idx) >= ctx.budget:
                break
            p = ptr[c]
            if p >= len(order[c]):
                continue
            i = int(order[c][p])
            ptr[c] = p + 1
            b = int(target[i])
            if b == int(labels[i]):
                continue
            idx.append(i)
            new.append(b)
            progressed = True
        if not progressed:
            break
    idx = np.asarray(idx, dtype=np.int64)
    new = np.asarray(new, dtype=np.int64)
    return {"images": np.asarray(ctx.images, dtype=np.float32).copy(),
            "labels": _flip(labels, idx, new), "indices": idx,
            "note": "高置信样本改为第二候选类"}


def runnerup_flip_shift(ctx, pixel_budget: int = 30, scale: float = 0.8) -> dict:
    """第二候选类翻转 + 把图像朝目标类模板推移（复合）。"""
    base = runnerup_flip(ctx)
    idx = np.asarray(base["indices"], dtype=np.int64)
    labels = np.asarray(ctx.labels, dtype=np.int64)
    proba = ctx.surro_proba
    target = _runner_up(proba)

    flat = flatten(ctx.images).copy()
    cent = np.zeros((NUM_CLASSES, flat.shape[1]), dtype=np.float64)
    for c in range(NUM_CLASSES):
        m = labels == c
        if m.any():
            cent[c] = flat[m].mean(axis=0)

    for i in idx:
        a = int(labels[i])
        b = int(target[i])
        delta = np.clip(cent[b] - cent[a], 0.0, None)
        if delta.max() <= 0:
            continue
        top = np.argsort(-delta)[:pixel_budget]
        flat[i, top] = np.clip(flat[i, top] + scale * delta[top], 0.0, 1.0)

    return {"images": flat.reshape(np.asarray(ctx.images).shape),
            "labels": base["labels"], "indices": idx,
            "note": "第二候选类翻转 + 像素推移"}


def runnerup_flip_mix(ctx, blend: float = 0.5, sigma: float = 0.05, seed: int = 42) -> dict:
    """第二候选类翻转 + 与目标类最近样本混合（合成陷阱，但目标类来自真实标签）。"""
    rng = np.random.default_rng(seed)
    base = runnerup_flip(ctx)
    idx = np.asarray(base["indices"], dtype=np.int64)
    labels = np.asarray(ctx.labels, dtype=np.int64)
    target = _runner_up(ctx.surro_proba)
    cls_idx = class_indices(labels)
    flat = flatten(ctx.images).copy()

    for i in idx:
        b = int(target[i])
        cand = cls_idx[b]
        if len(cand) == 0:
            continue
        d = np.linalg.norm(flat[cand] - flat[i], axis=1)
        j = int(cand[int(d.argmin())])
        flat[i] = np.clip((1.0 - blend) * flat[i] + blend * flat[j]
                          + rng.normal(0.0, sigma, size=flat[i].shape).astype(np.float32),
                          0.0, 1.0).astype(np.float32)

    return {"images": flat.reshape(np.asarray(ctx.images).shape),
            "labels": base["labels"], "indices": idx,
            "note": f"第二候选类翻转 + 跨类混合 blend={blend}"}


def concentrated_runnerup(ctx, n_src: int = 1) -> dict:
    """集中式：把少数几个类的样本标签改成「该类最常被判成的类」。"""
    labels = np.asarray(ctx.labels, dtype=np.int64)
    proba = ctx.surro_proba
    cls_idx = class_indices(labels)
    runner = _runner_up(proba)

    # 每个类被判成谁最多
    best_target = np.zeros(NUM_CLASSES, dtype=np.int64)
    conf = np.zeros(NUM_CLASSES)
    for c in range(NUM_CLASSES):
        ci = cls_idx[c]
        if len(ci) == 0:
            continue
        cnt = np.bincount(runner[ci], minlength=NUM_CLASSES).astype(np.float64)
        cnt[c] = -1
        best_target[c] = int(cnt.argmax())
        # 该类的样本「以为是别的类」的平均概率（越大越容易被推走）
        conf[c] = float(np.mean([proba[i, cnt.argmax()] for i in ci[:2000]]))
    src_order = np.argsort(-conf)[:max(1, n_src)]

    out_labels = labels.copy()
    touched = []
    remaining = ctx.budget
    for c in src_order:
        if remaining <= 0:
            break
        take = min(remaining, len(cls_idx[c]))
        idx = cls_idx[c][:take]
        out_labels[idx] = int(best_target[c])
        touched.append(idx)
        remaining -= take
    touched = np.concatenate(touched) if touched else np.array([], dtype=np.int64)
    return {"images": np.asarray(ctx.images, dtype=np.float32).copy(),
            "labels": out_labels, "indices": touched,
            "note": f"集中式 n_src={n_src} 目标=各类最常被误判类"}


def _flip(labels, idx, new):
    out = np.asarray(labels, dtype=np.int64).copy()
    out[idx] = new
    return out
