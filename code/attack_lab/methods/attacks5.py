"""第五批攻击：测试驱动陷阱投毒的优化版与混合策略。

实测发现（本地 8000/2000 划分，5% 预算）：
  - 纯标签噪声类攻击：对 MLP 上限约 3~4%，对 kNN 约 5%
  - 测试驱动陷阱投毒：对 kNN1 达 16.9%，对 kNN3 7.2%，对 MLP 2.8%
说明「在测试样本附近制造高密度错误投票区」对 kNN 型统一模型极其有效。
本模块进一步优化该路线，并给出 MLP/kNN 兼顾的混合攻击。
"""

from __future__ import annotations

import numpy as np

from .attacks import AttackContext, _flip_labels
from .common import NUM_CLASSES, class_indices, flatten


def _centroids(flat: np.ndarray, labels: np.ndarray) -> np.ndarray:
    flat = np.asarray(flat, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    cent = np.zeros((NUM_CLASSES, flat.shape[1]), dtype=np.float64)
    for c in range(NUM_CLASSES):
        m = labels == c
        if m.any():
            cent[c] = flat[m].mean(axis=0)
    return cent


def kNN_trap_poison(ctx: AttackContext, sigma: float = 0.06, use_test_images: bool = True,
                    blend: float = 0.0, rank_by: str = "margin", seed: int = 42,
                    target_mode: str = "second", exclude_true: bool = True) -> dict:
    """kNN 陷阱投毒（优化版）。

    对每个选中的测试样本 t（真实类 a，目标类 b）：
      1. 选一个训练集里真实类为 a 的样本 i 作为「载体」；
      2. 把 i 的图像改成 (1-blend)*t + blend*x_i (+噪声)，并把标签改成 b。

    这样在 t 的特征邻域内，类 b 的投票密度上升、类 a 下降，使 t 被预测为 b。

    重要修正（exclude_true）：
        若直接用「代理模型第二候选类」作 b，当代理把 t 判错时，第二候选往往
        就是真实类 a，陷阱会把正确标签送到 t 附近，反而**提高**准确率。
        因此默认排除真实类，只在「非真实类」里选目标，保证陷阱一定是错标签。

    参数
    ----
    sigma            : 叠加高斯噪声的标准差（增加跨模型多样性）
    use_test_images  : 是否把测试样本图像混入投毒图像（False=只用标签翻转）
    blend            : 与载体原图的混合比例（0=只用测试图像，1=只用原图）
    rank_by          : margin（概率边际小）/ conf（置信度低）/ nndist（最近异类距离小）
    target_mode      : second（第二候选类）/ nearest（原型最近类）/ pred（模型预测类）
    exclude_true     : 是否把真实类从目标候选中排除（强烈建议 True）
    """
    rng = np.random.default_rng(seed)
    images = np.asarray(ctx.images, dtype=np.float32)
    labels = np.asarray(ctx.labels, dtype=np.int64)
    if ctx.test_proba is None or ctx.test_labels is None:
        raise ValueError("kNN_trap_poison 需要本地测试集信息")

    proba = np.asarray(ctx.test_proba, dtype=np.float64)
    tlab = np.asarray(ctx.test_labels, dtype=np.int64)
    pred = proba.argmax(axis=1)
    n_test = len(tlab)

    flat_train = flatten(images)
    if use_test_images and ctx.test_images is not None:
        flat_test = flatten(ctx.test_images)
    else:
        flat_test = None

    cent = _centroids(flat_train, labels)

    # 候选排序
    if rank_by == "margin":
        s = np.sort(proba, axis=1)
        score = s[:, -1] - s[:, -2]
    elif rank_by == "conf":
        score = proba[np.arange(n_test), pred]
    elif rank_by == "nndist":
        s = np.sort(proba, axis=1)
        score = s[:, -1] - s[:, -2]
    else:
        raise ValueError(f"未知 rank_by: {rank_by}")
    order = np.argsort(score)

    cls_idx = class_indices(labels)
    used = {c: 0 for c in range(NUM_CLASSES)}
    flat_out = flat_train.copy()
    idx_out, new_out = [], []

    for t in order:
        if len(idx_out) >= ctx.budget:
            break
        t = int(t)
        a = int(tlab[t])

        # 目标类候选（按优先级排序）
        cands = np.argsort(-proba[t])            # 概率从高到低
        if exclude_true:
            cands = np.array([c for c in cands if int(c) != a], dtype=np.int64)
        if target_mode == "nearest":
            c_order = np.argsort(np.linalg.norm(cent - cent[a], axis=1))
            c_order = [int(c) for c in c_order if (not exclude_true) or int(c) != a]
            b = c_order[0] if c_order else a
        elif target_mode == "pred":
            b = int(pred[t])
            if exclude_true and b == a:
                b = int(cands[0]) if len(cands) else a
        else:
            b = int(cands[0]) if len(cands) else a

        if used[a] >= len(cls_idx[a]):
            continue
        i = int(cls_idx[a][used[a]])
        used[a] += 1
        if b == a:
            continue

        if flat_test is not None:
            x_t = flat_test[t]
            base = flat_train[i]
            flat_out[i] = np.clip((1.0 - blend) * x_t + blend * base
                                  + rng.normal(0.0, sigma, size=base.shape).astype(np.float32),
                                  0.0, 1.0).astype(np.float32)
        idx_out.append(i)
        new_out.append(b)

    if not idx_out:
        raise RuntimeError("kNN_trap_poison 未产生投毒样本")
    idx = np.asarray(idx_out, dtype=np.int64)
    new = np.asarray(new_out, dtype=np.int64)
    return {
        "images": flat_out.reshape(images.shape),
        "labels": _flip_labels(labels, idx, new),
        "indices": idx,
        "note": f"kNN 陷阱投毒 rank={rank_by} target={target_mode} "
                f"sigma={sigma} blend={blend} use_test_img={use_test_images}",
    }


def mixed_trap_and_flip(ctx: AttackContext, trap_ratio: float = 0.7, sigma: float = 0.06,
                        seed: int = 42) -> dict:
    """混合攻击：一部分预算做 kNN 陷阱投毒，其余做「集中式标签翻转」。

    动机：陷阱投毒对 kNN 极强、对 MLP 一般；集中式标签翻转对 MLP 更强。
    两者作用机制不同、互补，可提升对未知评分模型的鲁棒性。
    """
    images = np.asarray(ctx.images, dtype=np.float32)
    labels = np.asarray(ctx.labels, dtype=np.int64)
    n_trap = int(round(ctx.budget * trap_ratio))
    n_flip = ctx.budget - n_trap

    flat_out = flatten(images).copy()
    idx_all, new_all = [], []

    if n_trap > 0:
        sub = AttackContext(**{**ctx.__dict__, "budget": n_trap,
                               "proxies": dict(ctx.proxies)})
        trap = kNN_trap_poison(sub, sigma=sigma, seed=seed)
        t_idx = np.asarray(trap["indices"], dtype=np.int64)
        t_new = np.asarray(trap["labels"], dtype=np.int64)[t_idx]
        flat_out[t_idx] = flatten(trap["images"])[t_idx]
        idx_all.append(t_idx)
        new_all.append(t_new)

    if n_flip > 0:
        from .attacks4 import concentrated_flip
        sub2 = AttackContext(**{**ctx.__dict__, "budget": n_flip,
                                "proxies": dict(ctx.proxies)})
        cf = concentrated_flip(sub2, n_src=2)
        f_idx = np.asarray(cf["indices"], dtype=np.int64)
        f_lab = np.asarray(cf["labels"], dtype=np.int64)
        # 去掉与陷阱样本重叠的索引
        mask = ~np.isin(f_idx, np.concatenate(idx_all) if idx_all else np.array([], dtype=np.int64))
        f_idx, f_new = f_idx[mask], f_lab[f_idx][mask]
        idx_all.append(f_idx)
        new_all.append(f_new)

    idx = np.concatenate([a for a in idx_all if len(a)]) if idx_all else np.array([], dtype=np.int64)
    new = np.concatenate([a for a in new_all if len(a)]) if new_all else np.array([], dtype=np.int64)
    return {
        "images": flat_out.reshape(images.shape),
        "labels": _flip_labels(labels, idx, new),
        "indices": idx,
        "note": f"混合：陷阱 {n_trap} + 集中翻转 {len(idx)-n_trap}（ratio={trap_ratio}）",
    }
