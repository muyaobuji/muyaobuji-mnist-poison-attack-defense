"""第四批攻击：集中式（concentrated）投毒与测试驱动投毒。

实验动机：分散到 10 个类的标签噪声只产生 1% 级下降；把预算集中到少数
类对上会形成「高密度错误簇」，对决策边界与原型统计的破坏大得多。
"""

from __future__ import annotations

import numpy as np

from .attacks import AttackContext, _flip_labels
from .common import NUM_CLASSES, class_indices, confusion_matrix, flatten, nearest_class_map


def concentrated_flip(ctx: AttackContext, n_src: int = 1, target_mode: str = "nearest",
                      use_confusion: bool = True) -> dict:
    """把预算集中投给 n_src 个源类（每类的目标类为最易混淆类）。

    参数
    ----
    n_src      : 集中攻击的源类数量（1 = 全部预算打一类）
    target_mode: nearest（原型最近类）/ confusion（测试集最常被误判类）/ farthest
    """
    labels = np.asarray(ctx.labels, dtype=np.int64)
    flat = flatten(ctx.images)
    cls_idx = class_indices(labels)
    cent = _centroids(flat, labels)

    if target_mode == "farthest":
        d = np.linalg.norm(cent[:, None, :] - cent[None, :, :], axis=2)
        np.fill_diagonal(d, -np.inf)
        target_of = d.argmax(axis=1).astype(np.int64)
    elif target_mode == "confusion" and ctx.test_proba is not None:
        cm = confusion_matrix(ctx.test_labels, ctx.test_proba.argmax(axis=1))
        m = cm.astype(np.float64).copy()
        np.fill_diagonal(m, -1.0)
        target_of = m.argmax(axis=1).astype(np.int64)
    else:
        target_of = np.asarray(
            ctx.proxies.get("target_of", nearest_class_map(flat, labels)), dtype=np.int64)

    # 源类优先级：该类样本量足够、且其目标类混淆贡献大者优先
    if ctx.test_proba is not None and ctx.test_labels is not None:
        cm = confusion_matrix(ctx.test_labels, ctx.test_proba.argmax(axis=1))
        score = np.array([cm[c, int(target_of[c])] for c in range(NUM_CLASSES)], dtype=np.float64)
    else:
        score = np.array([len(cls_idx[c]) for c in range(NUM_CLASSES)], dtype=np.float64)
    src_order = np.argsort(-score)[:max(1, n_src)]

    per_src = max(1, ctx.budget // max(1, len(src_order)))
    idx_all, new_all = [], []
    remaining = ctx.budget
    for c in src_order:
        take = min(per_src, len(cls_idx[c]), remaining)
        if take <= 0:
            continue
        chosen = cls_idx[c][:take]
        idx_all.append(chosen)
        new_all.append(np.full(take, int(target_of[c]), dtype=np.int64))
        remaining -= take
    if remaining > 0:                      # 预算还有余量则补给第一个源类
        c = int(src_order[0])
        used = len(cls_idx[c][:per_src])
        extra = min(remaining, len(cls_idx[c]) - used)
        if extra > 0:
            idx_all.append(cls_idx[c][used:used + extra])
            new_all.append(np.full(extra, int(target_of[c]), dtype=np.int64))

    idx = np.concatenate(idx_all) if idx_all else np.array([], dtype=np.int64)
    new = np.concatenate(new_all) if new_all else np.array([], dtype=np.int64)
    src_desc = ",".join(str(int(c)) for c in src_order)
    return {
        "images": np.asarray(ctx.images, dtype=np.float32).copy(),
        "labels": _flip_labels(labels, idx, new),
        "indices": idx,
        "note": f"集中式标签翻转 n_src={n_src} 源类=[{src_desc}] target={target_mode}",
    }


def test_driven_poison(ctx: AttackContext, sigma: float = 0.05, mode: str = "nearest",
                       seed: int = 42) -> dict:
    """测试驱动投毒：用本地测试集样本改造出「陷阱样本」。

    思路（对 kNN / 原型类模型最有效）：对每个易被误判的测试样本 t（真实类 a、
    最近邻类 b），把它轻微扰动后以标签 b 放进训练集，使 b 在 t 附近形成高密度
    投票区。注意这里只用测试集**图像**来构造投毒样本，不依赖测试标签以外的信息。
    """
    rng = np.random.default_rng(seed)
    images = np.asarray(ctx.images, dtype=np.float32)
    labels = np.asarray(ctx.labels, dtype=np.int64)
    if ctx.test_proba is None or ctx.test_labels is None:
        raise ValueError("test_driven_poison 需要本地测试集信息")

    proba = ctx.test_proba
    pred = proba.argmax(axis=1)
    tlab = np.asarray(ctx.test_labels, dtype=np.int64)
    # 选「当前预测正确、但第二候选概率接近」的测试样本（边界样本）
    top2 = np.argsort(-proba, axis=1)[:, :2]
    correct = pred == tlab
    margin = proba[np.arange(len(proba)), top2[:, 0]] - proba[np.arange(len(proba)), top2[:, 1]]
    cand = np.flatnonzero(correct)
    order = cand[np.argsort(margin[cand])]          # 边际最小者优先
    chosen = order[:ctx.budget]

    test_images = ctx.test_images
    flat = flatten(images).copy()
    cls_idx = class_indices(labels)
    cent = _centroids(flatten(images), labels)

    out_labels = labels.copy()
    # 用真实训练样本承载投毒（避免分布外样本被简单过滤）
    idx = []
    new = []
    for rank, t in enumerate(chosen):
        a = int(tlab[t])
        b = int(top2[t, 1])
        # 从源类 a 里挑一个尚未使用的样本，把它改成「靠近测试样本 t 的图像 + 标签 b」
        pool = cls_idx[a]
        if rank >= len(pool):
            break
        i = int(pool[rank])
        x = flatten(test_images[t:t + 1])[0]
        noise = rng.normal(0.0, sigma, size=x.shape).astype(np.float32)
        flat[i] = np.clip(0.5 * x + 0.5 * flat[i] + noise, 0.0, 1.0)
        idx.append(i)
        new.append(b)
    if not idx:
        raise RuntimeError("test_driven_poison 未产生投毒样本")

    idx = np.asarray(idx, dtype=np.int64)
    new = np.asarray(new, dtype=np.int64)
    return {
        "images": flat.reshape(images.shape),
        "labels": _flip_labels(labels, idx, new),
        "indices": idx,
        "note": f"测试驱动陷阱投毒（{len(idx)} 个, sigma={sigma}）",
    }


def _centroids(flat: np.ndarray, labels: np.ndarray) -> np.ndarray:
    flat = np.asarray(flat, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    cent = np.zeros((NUM_CLASSES, flat.shape[1]), dtype=np.float64)
    for c in range(NUM_CLASSES):
        m = labels == c
        if m.any():
            cent[c] = flat[m].mean(axis=0)
    return cent
