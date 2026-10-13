"""第二批攻击方法：特征空间冲突（clean-label / 混合样本）与污染优化。

动机：少量「对称标签噪声」对 MNIST 只是弱攻击（实测 5% 预算下 ASR≈0.014）。
要突破判定线，必须让被投毒样本在特征空间中制造真正的冲突：
  - 让样本位于两类之间的歧义带，使决策边界退化；
  - 或直接优化投毒样本，使干净测试损失最大化。
"""

from __future__ import annotations

import numpy as np

from .attacks import AttackContext, _flip_labels
from .common import NUM_CLASSES, class_indices, confusion_matrix, flatten, nearest_class_map


# --------------------------------------------------------------------------- #
# 1. 类对交换：把 A 类样本改成 B、B 类样本改成 A（集中式、高密度冲突）
# --------------------------------------------------------------------------- #
def class_pair_swap(ctx: AttackContext) -> dict:
    """在预算内选取若干 (a,b) 类对做双向标签交换。

    直觉：与「随机撒点」不同，双向交换会在特征空间中把两个类的标签
    整体对调，迫使模型把 a 的区域学成 b、b 的区域学成 a，边界大规模退化。
    """
    labels = np.asarray(ctx.labels, dtype=np.int64)
    cls_idx = class_indices(labels)
    cm = None
    if ctx.test_proba is not None and ctx.test_labels is not None:
        cm = confusion_matrix(ctx.test_labels, ctx.test_proba.argmax(axis=1))

    # 候选类对评分：测试集上 a<->b 的互混淆量
    pairs = []
    for a in range(NUM_CLASSES):
        for b in range(a + 1, NUM_CLASSES):
            score = 0.0
            if cm is not None:
                score = float(cm[a, b] + cm[b, a])
            if score <= 0:
                # 退化到特征距离（越近越易混）
                score = 1.0
            pairs.append((score, a, b))
    pairs.sort(reverse=True)

    idx_all, new_all = [], []
    remaining = ctx.budget
    for _, a, b in pairs:
        if remaining < 2:
            break
        ca, cb = cls_idx[a], cls_idx[b]
        take_a = min(len(ca), max(1, remaining // 2))
        take_b = min(len(cb), remaining - take_a)
        take_a = min(take_a, remaining - take_b)
        if take_a <= 0 and take_b <= 0:
            continue
        if take_a > 0:
            idx_all.append(ca[:take_a]); new_all.append(np.full(take_a, b, dtype=np.int64))
        if take_b > 0:
            idx_all.append(cb[:take_b]); new_all.append(np.full(take_b, a, dtype=np.int64))
        remaining -= take_a + take_b
        if remaining <= 0:
            break

    idx = np.concatenate(idx_all) if idx_all else np.array([], dtype=np.int64)
    new = np.concatenate(new_all) if new_all else np.array([], dtype=np.int64)
    return {
        "images": np.asarray(ctx.images, dtype=np.float32).copy(),
        "labels": _flip_labels(labels, idx, new),
        "indices": idx,
        "note": "高混淆类对双向标签交换",
    }


# --------------------------------------------------------------------------- #
# 2. 特征混合投毒：x_poison = (1-r)*x + r*x_other，保留原标签（clean-label）
# --------------------------------------------------------------------------- #
def _nearest_other_class(images: np.ndarray, labels: np.ndarray,
                         target_of: np.ndarray, max_pixels: int = 300):
    """为每个样本找「目标类」中像素空间最近的样本索引。"""
    flat = flatten(images)
    out = np.full(len(labels), -1, dtype=np.int64)
    cls_idx = class_indices(labels)
    for c in range(NUM_CLASSES):
        src = cls_idx[c]
        if len(src) == 0:
            continue
        tgt_cls = int(target_of[c])
        cand = cls_idx[tgt_cls]
        if len(cand) == 0:
            continue
        # 用均值模板筛选候选，减少距离计算量
        src_sub = src if len(src) <= max_pixels else src[:max_pixels]
        d = np.linalg.norm(flat[src_sub][:, None, :] - flat[cand][None, :, :], axis=2)
        out[src_sub] = cand[d.argmin(axis=1)]
    return out


def feature_mix_poison(ctx: AttackContext, ratio: float = 0.5, keep_label: bool = True,
                       mix: str = "linear", nearest: bool = True) -> dict:
    """把被选中样本与「目标类」样本混合，制造两可样本。

    参数
    ----
    ratio      : 混合比例 r，x' = (1-r)x + r*x_target
    keep_label : True=clean-label（保留原标签）；False=同时翻转标签
    mix        : linear（线性混合）| max（取较大像素，产生更亮的歧义样本）
    nearest    : True=用像素空间最近的跨类样本混合；False=用类均值模板
    """
    images = np.asarray(ctx.images, dtype=np.float32)
    labels = np.asarray(ctx.labels, dtype=np.int64)
    target_of = ctx.proxies.get("target_of")
    if target_of is None:
        target_of = nearest_class_map(flatten(images), labels)

    # 选哪些样本：优先目标类最容易混淆的来源类
    if ctx.test_proba is not None and ctx.test_labels is not None:
        cm = confusion_matrix(ctx.test_labels, ctx.test_proba.argmax(axis=1))
        score = np.array([cm[c, int(target_of[c])] for c in range(NUM_CLASSES)], dtype=np.float64)
        order = np.argsort(-score)
    else:
        order = np.arange(NUM_CLASSES)

    cls_idx = class_indices(labels)
    chosen = []
    remaining = ctx.budget
    # 轮转各类别取样本（类内按「与目标类距离」升序优先）
    per_class_pool = {}
    for c in range(NUM_CLASSES):
        per_class_pool[c] = cls_idx[c].copy()
    for c in order:
        if remaining <= 0:
            break
        pool = per_class_pool[c]
        take = min(len(pool), remaining)
        if take > 0:
            chosen.append(pool[:take])
            remaining -= take
    if remaining > 0:
        for c in order:
            pool = per_class_pool[c]
            if len(pool) > 0 and remaining > 0:
                extra = min(len(pool), remaining)
                chosen.append(pool[:extra])
                remaining -= extra
    idx = np.concatenate(chosen) if chosen else np.array([], dtype=np.int64)

    flat = flatten(images).copy()
    if nearest:
        partner = _nearest_other_class(images, labels, target_of)
    else:
        partner = None

    cent = np.zeros((NUM_CLASSES, flat.shape[1]), dtype=np.float64)
    for c in range(NUM_CLASSES):
        m = labels == c
        if m.any():
            cent[c] = flat[m].mean(axis=0)

    for i in idx:
        a = int(labels[i])
        b = int(target_of[a])
        if nearest and partner[i] >= 0:
            x_t = flat[partner[i]]
        else:
            x_t = cent[b].astype(np.float32)
        if mix == "max":
            flat[i] = np.clip(np.maximum(flat[i], ratio * x_t), 0.0, 1.0)
        else:
            flat[i] = np.clip((1.0 - ratio) * flat[i] + ratio * x_t, 0.0, 1.0)

    out_labels = labels.copy()
    if not keep_label:
        out_labels = _flip_labels(labels, idx, np.array([int(target_of[int(labels[i])]) for i in idx]))
    return {
        "images": flat.reshape(images.shape),
        "labels": out_labels,
        "indices": idx,
        "note": f"特征混合 ratio={ratio} keep_label={keep_label} mix={mix} nearest={nearest}",
    }


# --------------------------------------------------------------------------- #
# 3. 像素级「歧义化」：把目标类模板的特征像素加到源样本上（保留原标签）
# --------------------------------------------------------------------------- #
def ambiguity_patch(ctx: AttackContext, pixel_budget: int = 40, scale: float = 0.6,
                    target: str = "centroid") -> dict:
    """在源样本上叠加目标类的判别性像素，使其跨在两类之间，标签保持不变。"""
    images = np.asarray(ctx.images, dtype=np.float32)
    labels = np.asarray(ctx.labels, dtype=np.int64)
    target_of = ctx.proxies.get("target_of")
    if target_of is None:
        target_of = nearest_class_map(flatten(images), labels)

    if ctx.test_proba is not None and ctx.test_labels is not None:
        cm = confusion_matrix(ctx.test_labels, ctx.test_proba.argmax(axis=1))
        score = np.array([cm[c, int(target_of[c])] for c in range(NUM_CLASSES)], dtype=np.float64)
        order = np.argsort(-score)
    else:
        order = np.arange(NUM_CLASSES)

    cls_idx = class_indices(labels)
    chosen, remaining = [], ctx.budget
    for c in order:
        if remaining <= 0:
            break
        take = min(len(cls_idx[c]), remaining)
        if take > 0:
            chosen.append(cls_idx[c][:take])
            remaining -= take
    idx = np.concatenate(chosen) if chosen else np.array([], dtype=np.int64)

    flat = flatten(images).copy()
    cent = np.zeros((NUM_CLASSES, flat.shape[1]), dtype=np.float64)
    for c in range(NUM_CLASSES):
        m = labels == c
        if m.any():
            cent[c] = flat[m].mean(axis=0)

    for i in idx:
        a = int(labels[i])
        b = int(target_of[a])
        delta = np.clip(cent[b] - cent[a], 0.0, None)
        top = np.argsort(-delta)[:pixel_budget]
        flat[i, top] = np.clip(flat[i, top] + scale * delta[top], 0.0, 1.0)

    return {
        "images": flat.reshape(images.shape),
        "labels": labels.copy(),
        "indices": idx,
        "note": f"clean-label 歧义化 patch（{pixel_budget} 像素, scale={scale}）",
    }


# --------------------------------------------------------------------------- #
# 4. 不可学习样本（Error-Minimizing Noise，简化版，逐样本噪声）
# --------------------------------------------------------------------------- #
def unlearnable_noise(ctx: AttackContext, steps: int = 120, lr: float = 0.05,
                      eps: float = 0.25, n_sub: int = 400, seed: int = 42) -> dict:
    """优化逐样本噪声，使代理模型在「错误标签」上的损失最小化（EMN 简化版）。

    这是 poison 优化的思路：不是随机噪声，而是让投毒样本容易被误拟合、
    从而在干净测试上制造错误。噪声约束在 eps 球内，保证图像仍像手写数字。
    """
    import torch
    import torch.nn as nn

    images = np.asarray(ctx.images, dtype=np.float32)
    labels = np.asarray(ctx.labels, dtype=np.int64)
    target_of = ctx.proxies.get("target_of")
    if target_of is None:
        target_of = nearest_class_map(flatten(images), labels)

    if ctx.test_proba is not None and ctx.test_labels is not None:
        cm = confusion_matrix(ctx.test_labels, ctx.test_proba.argmax(axis=1))
        score = np.array([cm[c, int(target_of[c])] for c in range(NUM_CLASSES)], dtype=np.float64)
        order = np.argsort(-score)
    else:
        order = np.arange(NUM_CLASSES)

    cls_idx = class_indices(labels)
    chosen, remaining = [], ctx.budget
    for c in order:
        if remaining <= 0:
            break
        take = min(len(cls_idx[c]), remaining)
        if take > 0:
            chosen.append(cls_idx[c][:take])
            remaining -= take
    idx = np.concatenate(chosen) if chosen else np.array([], dtype=np.int64)

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    sel = idx if len(idx) <= n_sub else rng.choice(idx, size=n_sub, replace=False)

    X = torch.from_numpy(flatten(images)[sel]).clone()
    y_poison = torch.from_numpy(np.array([int(target_of[int(labels[i])]) for i in sel],
                                         dtype=np.int64))

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(784, 256), nn.ReLU(),
                                     nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 10))

        def forward(self, x):
            return self.net(x)

    net = MLP()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    lossf = nn.CrossEntropyLoss()
    base = X.clone()
    for _ in range(steps):
        # 内层：让模型在该样本上（按投毒标签）损失最小
        for _ in range(3):
            opt.zero_grad()
            loss = lossf(net(X), y_poison)
            loss.backward()
            opt.step()
        # 外层：更新噪声，使损失进一步下降
        noise = (X - base).detach().clone().requires_grad_(True)
        delta = torch.clamp(noise, -eps, eps)
        Xn = torch.clamp(base + delta, 0.0, 1.0)
        loss = lossf(net(Xn), y_poison)
        g, = torch.autograd.grad(loss, delta)
        with torch.no_grad():
            X = torch.clamp(X - lr * g.sign(), 0.0, 1.0)
            X = torch.clamp(X, base - eps, base + eps)
            X = torch.clamp(X, 0.0, 1.0)

    flat = flatten(images).copy()
    with torch.no_grad():
        flat[sel] = X.numpy()
    out_labels = _flip_labels(labels, sel, np.array([int(target_of[int(labels[i])]) for i in sel]))
    return {
        "images": flat.reshape(images.shape),
        "labels": out_labels,
        "indices": sel,
        "note": f"EMN 简化版噪声优化（{len(sel)} 样本, {steps} 步, eps={eps}）",
    }
