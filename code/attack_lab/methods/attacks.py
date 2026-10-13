"""实验用攻击方法集合。

每个方法签名统一为：
    fn(ctx: AttackContext) -> dict
返回：
    {"images": 投毒图像, "labels": 投毒标签, "indices": 被修改样本索引, "note": 说明}

ctx 由 attack_lab.harness 构造，携带：images / labels / budget / seed /
surrogate 预测概率 / 类别索引等。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .common import NUM_CLASSES, class_indices, confusion_matrix, flatten, nearest_class_map


# --------------------------------------------------------------------------- #
# 上下文
# --------------------------------------------------------------------------- #
@dataclass
class AttackContext:
    images: np.ndarray                     # [N,1,28,28] float32
    labels: np.ndarray                     # [N] int64
    budget: int                            # 允许修改的最大样本数
    seed: int = 42
    # 代理模型（评分模型同构 MLP，干净训练）给出的信息
    surro_proba: np.ndarray | None = None  # [N,10] 训练样本上的预测概率
    surro_pred: np.ndarray | None = None   # [N] 训练样本上的预测标签
    test_proba: np.ndarray | None = None   # [M,10] 测试集预测概率（仅实验用）
    test_labels: np.ndarray | None = None
    test_images: np.ndarray | None = None  # [M,1,28,28] 本地测试集图像（仅实验用）
    proxies: dict = field(default_factory=dict)  # 其他代理（knn 等）

    @property
    def n(self) -> int:
        return len(self.labels)

    def class_idx(self) -> dict:
        return class_indices(self.labels)


def _flip_labels(labels: np.ndarray, idx: np.ndarray, new: np.ndarray) -> np.ndarray:
    out = np.asarray(labels, dtype=np.int64).copy()
    out[idx] = new
    return out


# --------------------------------------------------------------------------- #
# A. 随机标签翻转（Baseline 复现）
# --------------------------------------------------------------------------- #
def random_label_flip(ctx: AttackContext) -> dict:
    rng = np.random.default_rng(ctx.seed)
    idx = rng.choice(ctx.n, size=ctx.budget, replace=False)
    old = ctx.labels[idx]
    # 在其余 9 类中等概率随机选一个
    new = (old + rng.integers(1, NUM_CLASSES, size=len(idx))) % NUM_CLASSES
    return {
        "images": np.asarray(ctx.images, dtype=np.float32).copy(),
        "labels": _flip_labels(ctx.labels, idx, new),
        "indices": idx,
        "note": "随机标签翻转 Baseline",
    }


# --------------------------------------------------------------------------- #
# B. 类内均值模板最近类翻转
# --------------------------------------------------------------------------- #
def centroid_nearest_flip(ctx: AttackContext) -> dict:
    target_of = ctx.proxies.get("target_of")
    if target_of is None:
        target_of = nearest_class_map(flatten(ctx.images), ctx.labels)
    return _flip_by_target_map(ctx, target_of, note="翻转到类内均值模板最近类")


# --------------------------------------------------------------------------- #
# C. 代理模型混淆矩阵选靶（行内最易被误判的类别）
# --------------------------------------------------------------------------- #
def confusion_pair_flip(ctx: AttackContext) -> dict:
    if ctx.test_proba is None or ctx.test_labels is None:
        raise ValueError("confusion_pair_flip 需要 test_proba / test_labels")
    cm = confusion_matrix(ctx.test_labels, ctx.test_proba.argmax(axis=1))
    target_of = _top_confusion_target(cm)
    return _flip_by_target_map(ctx, target_of, note="翻转到测试集最易混淆类（混淆矩阵选靶）")


def _top_confusion_target(cm: np.ndarray) -> np.ndarray:
    """对每个真实类 c，取其最常被误判成的类别作为翻转目标。"""
    m = np.array(cm, dtype=np.float64, copy=True)
    np.fill_diagonal(m, -1.0)          # 排除自身
    # 剔除被自己正确分类比例很小的情况（无信息），仍取最大误判
    return m.argmax(axis=1).astype(np.int64)


# --------------------------------------------------------------------------- #
# D. 贪心预算分配：按「每单位预算削弱的正确分类数」排序，全局择优
# --------------------------------------------------------------------------- #
def greedy_budget_alloc(ctx: AttackContext) -> dict:
    """把预算集中投放到影响最大的若干 (源类 -> 目标类) 对上。

    单位预算收益 proxy = 该类被误判到目标类的比例 / 该类训练样本数占比，
    使小类别也能在单位预算下产生大影响。
    """
    if ctx.test_proba is None or ctx.test_labels is None:
        raise ValueError("greedy_budget_alloc 需要 test_proba / test_labels")
    cm = confusion_matrix(ctx.test_labels, ctx.test_proba.argmax(axis=1))
    n_test = cm.sum()
    cls_idx = ctx.class_idx()

    pairs = []
    for a in range(NUM_CLASSES):
        for b in range(NUM_CLASSES):
            if a == b:
                continue
            benefit = cm[a, b] / n_test          # 该误判对整体错误的贡献
            capacity = len(cls_idx[a])           # 可翻转的样本上限
            if benefit <= 0 or capacity == 0:
                continue
            # 单位预算收益：把 a 类样本改成 b 能带来多少整体错误
            pairs.append((benefit / max(capacity, 1), benefit, a, b, capacity))

    pairs.sort(reverse=True)
    idx_all, new_all = [], []
    remaining = ctx.budget
    for _, _, a, b, capacity in pairs:
        if remaining <= 0:
            break
        take = min(remaining, capacity)
        chosen = cls_idx[a][:take]
        idx_all.append(chosen)
        new_all.append(np.full(take, b, dtype=np.int64))
        remaining -= take

    if not idx_all:
        return random_label_flip(ctx)
    idx = np.concatenate(idx_all)
    new = np.concatenate(new_all)
    return {
        "images": np.asarray(ctx.images, dtype=np.float32).copy(),
        "labels": _flip_labels(ctx.labels, idx, new),
        "indices": idx,
        "note": f"贪心预算分配（{len(pairs)} 个候选对，用满 {len(idx)}/{ctx.budget}）",
    }


# --------------------------------------------------------------------------- #
# E. 难样本优先：代理模型高置信度样本优先翻转
# --------------------------------------------------------------------------- #
def confident_hard_flip(ctx: AttackContext) -> dict:
    """优先翻转代理模型最「确信」属于其原标签的样本，制造最大矛盾。"""
    if ctx.surro_proba is None:
        raise ValueError("confident_hard_flip 需要 surro_proba")
    conf = ctx.surro_proba[np.arange(ctx.n), np.asarray(ctx.labels, dtype=np.int64)]
    target_of = ctx.proxies.get("target_of")
    if target_of is None:
        target_of = nearest_class_map(flatten(ctx.images), ctx.labels)

    idx_all, new_all = [], []
    remaining = ctx.budget
    cls_idx = ctx.class_idx()
    # 各类别内部按置信度降序，轮转取，保证类别均衡
    order = {}
    for c in range(NUM_CLASSES):
        ci = cls_idx[c]
        order[c] = ci[np.argsort(-conf[ci])]
    ptr = {c: 0 for c in range(NUM_CLASSES)}
    while remaining > 0:
        progressed = False
        for c in range(NUM_CLASSES):
            if remaining <= 0:
                break
            p = ptr[c]
            if p >= len(order[c]):
                continue
            idx_all.append(order[c][p])
            new_all.append(int(target_of[c]))
            ptr[c] = p + 1
            remaining -= 1
            progressed = True
        if not progressed:
            break
    idx = np.asarray(idx_all, dtype=np.int64)
    new = np.asarray(new_all, dtype=np.int64)
    return {
        "images": np.asarray(ctx.images, dtype=np.float32).copy(),
        "labels": _flip_labels(ctx.labels, idx, new),
        "indices": idx,
        "note": "高置信度样本优先 + 最近类目标",
    }


# --------------------------------------------------------------------------- #
# F. 翻转 + 特征扰动（把图像朝目标类轻微推移）
# --------------------------------------------------------------------------- #
def flip_with_shift(ctx: AttackContext, pixel_budget: int = 30, step: float = 0.9) -> dict:
    """翻转标签，同时把图像中属于目标类典型区域的像素朝目标类推移。

    做法：用类内均值模板差异 delta = clip(centroid[b] - centroid[a], 0, 1)，
    选出差异最大的 pixel_budget 个像素，把 x 在这些像素上朝 b 的模板移动。
    """
    base = random_label_flip(ctx) if ctx.proxies.get("target_of") is None else \
        _flip_by_target_map(ctx, ctx.proxies["target_of"])
    target_of = ctx.proxies.get("target_of")
    if target_of is None:
        target_of = nearest_class_map(flatten(ctx.images), ctx.labels)
    idx = base["indices"]

    feats = flatten(ctx.images)
    labels = np.asarray(ctx.labels, dtype=np.int64)
    centroids = np.zeros((NUM_CLASSES, feats.shape[1]), dtype=np.float64)
    for c in range(NUM_CLASSES):
        m = labels == c
        if m.any():
            centroids[c] = feats[m].mean(axis=0)

    images = np.asarray(ctx.images, dtype=np.float32).copy()
    flat = images.reshape(len(images), -1)
    for i in idx:
        a = int(labels[i])
        b = int(target_of[a])
        delta = np.clip(centroids[b] - centroids[a], 0.0, None)
        top = np.argsort(-delta)[:pixel_budget]
        flat[i, top] = np.clip(flat[i, top] + step * delta[top], 0.0, 1.0)
    images = flat.reshape(images.shape)

    return {
        "images": images,
        "labels": base["labels"],
        "indices": idx,
        "note": f"标签翻转 + 朝目标类模板推移 {pixel_budget} 个像素",
    }


# --------------------------------------------------------------------------- #
# G. 梯度增强扰动（代理模型梯度方向放大错误）
# --------------------------------------------------------------------------- #
def gradient_amplified_flip(ctx: AttackContext, eps: float = 0.2, ascend: bool = True) -> dict:
    """翻转标签后，沿代理模型对「投毒标签」的损失上升方向扰动像素。

    需要 ctx.proxies["grad_fn"]：返回 loss 对输入图像（展平）的梯度。
    """
    grad_fn = ctx.proxies.get("grad_fn")
    if grad_fn is None:
        raise ValueError("gradient_amplified_flip 需要 proxies['grad_fn']")
    base = _flip_by_target_map(ctx, ctx.proxies["target_of"])
    idx = base["indices"]
    images = np.asarray(ctx.images, dtype=np.float32).copy()
    flat = images.reshape(len(images), -1)
    poisoned_labels = base["labels"]
    g = grad_fn(ctx.images[idx], poisoned_labels[idx])
    g = np.asarray(g, dtype=np.float32).reshape(len(idx), -1)
    if not ascend:
        g = -g
    flat[idx] = np.clip(flat[idx] + eps * np.sign(g), 0.0, 1.0)
    return {
        "images": flat.reshape(images.shape),
        "labels": poisoned_labels,
        "indices": idx,
        "note": f"标签翻转 + FGSM 型扰动 eps={eps}",
    }


# --------------------------------------------------------------------------- #
# 工具
# --------------------------------------------------------------------------- #
def _flip_by_target_map(ctx: AttackContext, target_of: np.ndarray, note: str, quota=None) -> dict:
    """按 target_of（类->目标类）均匀分配预算执行标签翻转。"""
    labels = np.asarray(ctx.labels, dtype=np.int64)
    cls_idx = class_indices(labels)
    idx_all, new_all = [], []
    remaining = ctx.budget
    sizes = np.array([len(cls_idx[c]) for c in range(NUM_CLASSES)], dtype=np.float64)
    if quota is None:
        quota = sizes / max(sizes.sum(), 1.0)
    # 按类别规模比例分配，向上取整后再截断到预算
    alloc = np.floor(quota * ctx.budget).astype(int)
    while alloc.sum() < ctx.budget:
        # 余量给还能出的类别
        for c in np.argsort(-sizes):
            if alloc.sum() >= ctx.budget:
                break
            if alloc[c] < len(cls_idx[c]):
                alloc[c] += 1
    for c in range(NUM_CLASSES):
        take = min(int(alloc[c]), len(cls_idx[c]), remaining)
        if take <= 0:
            continue
        chosen = cls_idx[c][:take]
        idx_all.append(chosen)
        new_all.append(np.full(take, int(target_of[c]), dtype=np.int64))
        remaining -= take
        if remaining <= 0:
            break
    idx = np.concatenate(idx_all) if idx_all else np.array([], dtype=np.int64)
    new = np.concatenate(new_all) if new_all else np.array([], dtype=np.int64)
    return {
        "images": np.asarray(ctx.images, dtype=np.float32).copy(),
        "labels": _flip_labels(labels, idx, new),
        "indices": idx,
        "note": note,
    }
