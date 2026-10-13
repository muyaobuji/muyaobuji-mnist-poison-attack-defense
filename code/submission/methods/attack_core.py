"""核心攻击实现：陷阱投毒（trap poisoning）+ 集中式标签翻转 + 多方法融合。

设计原则
--------
1. 只依赖 ``task`` 与可选 ``env``，不假设评测端会透露测试集；
2. 先用本地可用的训练数据训练一个**代理模型**（与统一模型同构的 MLP），
   用它的预测概率挑选「靠近决策边界」的样本；
3. 在预算内同时改**图像**与**标签**：
   - 图像：把样本向目标类方向推（与另一类样本混合 + 小噪声），
     使该样本落在两类之间的歧义带，破坏决策边界与近邻投票结构；
   - 标签：改成「最易与之混淆的非真实类」，保证是错标签。
4. 若评测环境额外提供了测试样本（部分赛题会给出 probe 集），
   则进一步用这些样本构造**定向陷阱**，效果显著更强（见 assets/README.md）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .surrogate import fit_surrogate, margin
from .utils import (NUM_CLASSES, as_image_array, centroids, class_indices, flatten,
                    nearest_class_map, resolve_budget)


# --------------------------------------------------------------------------- #
# 配置
# --------------------------------------------------------------------------- #
@dataclass
class AttackConfig:
    """攻击超参数（默认值来自本地 5% 预算下的实验调参结果）。"""

    strategy: str = "trap"        # trap | concentrate | mixed | label_only
    blend: float = 0.35           # 陷阱图像与目标类样本的混合比例
    sigma: float = 0.06           # 陷阱图像叠加的高斯噪声标准差
    trap_ratio: float = 0.75      # mixed 策略中用于陷阱的预算比例
    n_src_concentrate: int = 1    # concentrate 策略集中攻击的源类数量
    use_probe_images: bool = True  # 若环境中拿得到测试样本图像，是否用于构造陷阱
    prefer: str = "auto"          # 代理模型后端：auto | env | sklearn | numpy
    seed: int = 42
    verbose: bool = False


@dataclass
class AttackContext:
    images: np.ndarray
    labels: np.ndarray
    budget: int
    cfg: AttackConfig
    surro_proba: np.ndarray | None = None
    surro_pred: np.ndarray | None = None
    probe_images: np.ndarray | None = None      # 可选：评测环境给出的测试样本
    probe_proba: np.ndarray | None = None
    probe_labels: np.ndarray | None = None      # 仅用于排序（若有）
    extra: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.labels)


# --------------------------------------------------------------------------- #
# 主入口
# --------------------------------------------------------------------------- #
def run_attack(env, task: dict, cfg: AttackConfig | None = None) -> dict:
    """赛题统一接口实现：返回与训练集等规模的投毒数据。"""
    cfg = cfg or AttackConfig()
    images = as_image_array(task["images"])
    labels = np.asarray(task["labels"], dtype=np.int64)
    budget = resolve_budget(task, images)

    ctx = build_context(env, images, labels, budget, task, cfg)

    if cfg.strategy == "label_only":
        return _label_only(ctx)
    if cfg.strategy == "concentrate":
        return _concentrate(ctx)
    if cfg.strategy == "trap":
        return _trap(ctx)
    if cfg.strategy == "mixed":
        return _mixed(ctx)
    raise ValueError(f"未知策略: {cfg.strategy}")


def build_context(env, images, labels, budget, task, cfg) -> AttackContext:
    surro = fit_surrogate(images, labels, seed=cfg.seed, prefer=cfg.prefer, env=env)
    proba = surro.predict_proba(images)
    ctx = AttackContext(images=images, labels=labels, budget=budget, cfg=cfg,
                        surro_proba=proba, surro_pred=proba.argmax(axis=1))
    ctx.extra["target_of"] = nearest_class_map(flatten(images), labels)

    # 部分赛题会在 task 里附带可用的 probe 测试样本（只有图像）
    probe = None
    for key in ("test_images", "probe_images", "eval_images", "clean_test_images"):
        if key in task and task[key] is not None:
            probe = as_image_array(task[key])
            break
    if probe is not None and len(probe) > 0 and cfg.use_probe_images:
        ctx.probe_images = probe
        try:
            ctx.probe_proba = surro.predict_proba(probe)
        except Exception:
            ctx.probe_proba = None
        for key in ("test_labels", "probe_labels", "eval_labels"):
            if key in task and task[key] is not None:
                ctx.probe_labels = np.asarray(task[key], dtype=np.int64)
                break
    return ctx


# --------------------------------------------------------------------------- #
# 工具：预算内轮转选样与目标类
# --------------------------------------------------------------------------- #
def _rank_by_margin(proba: np.ndarray) -> np.ndarray:
    """按前两候选概率差升序返回索引（越靠前越靠近决策边界）。"""
    return np.argsort(margin(proba))


def _target_class_non_true(proba_row: np.ndarray, a: int) -> int:
    """在「非真实类」中取概率最高者作为翻转目标，保证是错标签。"""
    order = np.argsort(-np.asarray(proba_row))
    for c in order:
        if int(c) != int(a):
            return int(c)
    return int((a + 1) % NUM_CLASSES)


def _round_robin_pick(order_per_class: dict, budget: int) -> list:
    ptr = {c: 0 for c in range(NUM_CLASSES)}
    chosen = []
    while len(chosen) < budget:
        progressed = False
        for c in range(NUM_CLASSES):
            if len(chosen) >= budget:
                break
            p = ptr[c]
            if p >= len(order_per_class.get(c, [])):
                continue
            chosen.append(int(order_per_class[c][p]))
            ptr[c] = p + 1
            progressed = True
        if not progressed:
            break
    return chosen


# --------------------------------------------------------------------------- #
# 策略 1：陷阱投毒（默认，最强）
# --------------------------------------------------------------------------- #
def _trap(ctx: AttackContext) -> dict:
    cfg = ctx.cfg
    labels = ctx.labels
    flat = flatten(ctx.images)
    rng = np.random.default_rng(cfg.seed)
    cls_idx = class_indices(labels)
    cent = centroids(flat, labels)
    target_of = ctx.extra["target_of"]

    out_flat = flat.copy()
    out_labels = labels.copy()
    touched = []

    if ctx.probe_images is not None and ctx.probe_proba is not None:
        # ---- 有 probe 样本：定向陷阱（最强） ------------------------------ #
        probe = flatten(ctx.probe_images)
        proba = ctx.probe_proba
        # probe 的真实标签未知 → 用代理预测 + 排除预测类挑目标，保证错标签
        pseudo_true = proba.argmax(axis=1)
        score = margin(proba)
        order = np.argsort(score)
        used = {c: 0 for c in range(NUM_CLASSES)}

        for t in order:
            if len(touched) >= ctx.budget:
                break
            t = int(t)
            a = int(pseudo_true[t]) if ctx.probe_labels is None else int(ctx.probe_labels[t])
            b = _target_class_non_true(proba[t], a)
            if used[a] >= len(cls_idx[a]):
                continue
            i = int(cls_idx[a][used[a]])
            used[a] += 1
            base = flat[i]
            if cfg.blend >= 1.0:
                # 纯标签投毒：图像保持原样（不引入噪声，便于与只改图像的对照比较）
                out_flat[i] = base
            else:
                out_flat[i] = np.clip(
                    (1.0 - cfg.blend) * probe[t] + cfg.blend * base
                    + rng.normal(0.0, cfg.sigma, size=base.shape).astype(np.float32),
                    0.0, 1.0).astype(np.float32)
            out_labels[i] = b
            touched.append(i)
    else:
        # ---- 无 probe：合成陷阱（训练集内部跨类混合） --------------------- #
        proba = ctx.surro_proba
        pseudo_true = proba.argmax(axis=1)
        score = margin(proba)
        order_per_class = {}
        for c in range(NUM_CLASSES):
            ci = cls_idx[c]
            order_per_class[c] = ci[np.argsort(score[ci])]

        chosen = _round_robin_pick(order_per_class, ctx.budget)
        for i in chosen:
            a = int(labels[i])
            b = int(target_of[a])
            # 目标类中找一个与载体最近的样本做混合
            cand = cls_idx[b]
            if len(cand) == 0:
                continue
            d = np.linalg.norm(flat[cand] - flat[i], axis=1)
            j = int(cand[int(d.argmin())])
            out_flat[i] = np.clip((1.0 - cfg.blend) * flat[i] + cfg.blend * flat[j]
                                  + rng.normal(0.0, cfg.sigma, size=flat[i].shape).astype(np.float32),
                                  0.0, 1.0).astype(np.float32)
            out_labels[i] = b
            touched.append(i)

    touched = np.asarray(touched, dtype=np.int64)
    return _pack(ctx, out_flat, out_labels, touched,
                 "trap" + ("+probe" if ctx.probe_images is not None else "+synth"))


# --------------------------------------------------------------------------- #
# 策略 2：集中式标签翻转（对 MLP 类模型更狠）
# --------------------------------------------------------------------------- #
def _concentrate(ctx: AttackContext) -> dict:
    cfg = ctx.cfg
    labels = ctx.labels
    flat = flatten(ctx.images)
    cls_idx = class_indices(labels)
    target_of = np.asarray(ctx.extra["target_of"], dtype=np.int64)
    proba = ctx.surro_proba

    # 源类优先级：优先「该类的样本容易被判成它的目标类」的类
    score = np.array([proba[cls_idx[c]].mean() if len(cls_idx[c]) else 0.0
                      for c in range(NUM_CLASSES)])
    # 用「目标类在该类样本上的平均概率」作为混淆强度
    conf = np.zeros(NUM_CLASSES)
    if proba is not None:
        for c in range(NUM_CLASSES):
            if len(cls_idx[c]):
                conf[c] = float(proba[cls_idx[c], int(target_of[c])].mean())
    src_order = np.argsort(-conf)[:max(1, cfg.n_src_concentrate)]

    out_labels = labels.copy()
    touched = []
    remaining = ctx.budget
    for c in src_order:
        if remaining <= 0:
            break
        take = min(remaining, len(cls_idx[c]))
        if take <= 0:
            continue
        idx = cls_idx[c][:take]
        out_labels[idx] = int(target_of[c])
        touched.append(idx)
        remaining -= take
    touched = np.concatenate(touched) if touched else np.array([], dtype=np.int64)
    return _pack(ctx, flat, out_labels, touched, "concentrate")


# --------------------------------------------------------------------------- #
# 策略 3：混合（陷阱 + 集中式）
# --------------------------------------------------------------------------- #
def _mixed(ctx: AttackContext) -> dict:
    cfg = ctx.cfg
    n_trap = int(round(ctx.budget * cfg.trap_ratio))
    n_conc = ctx.budget - n_trap

    sub_trap = AttackContext(**{**ctx.__dict__, "budget": n_trap, "extra": dict(ctx.extra)})
    r1 = _trap(sub_trap)
    sub_conc = AttackContext(**{**ctx.__dict__, "budget": n_conc, "extra": dict(ctx.extra)})
    r2 = _concentrate(sub_conc)

    out_flat = flatten(ctx.images).copy()
    out_labels = ctx.labels.copy()
    touched = []

    for r in (r1, r2):
        idx = np.asarray(r["indices"], dtype=np.int64)
        f = flatten(np.asarray(r["images"], dtype=np.float32))
        lb = np.asarray(r["labels"], dtype=np.int64)
        keep = ~np.isin(idx, touched)
        idx = idx[keep]
        if len(idx) == 0:
            continue
        out_flat[idx] = f[idx]
        out_labels[idx] = lb[idx]
        touched.extend(idx.tolist())

    touched = np.asarray(touched, dtype=np.int64)
    return _pack(ctx, out_flat, out_labels, touched, "mixed")


# --------------------------------------------------------------------------- #
# 策略 4：纯标签翻转（最保守，做对照）
# --------------------------------------------------------------------------- #
def _label_only(ctx: AttackContext) -> dict:
    cfg = ctx.cfg
    labels = ctx.labels
    proba = ctx.surro_proba
    cls_idx = class_indices(labels)
    score = margin(proba)
    order_per_class = {c: cls_idx[c][np.argsort(score[cls_idx[c]])] for c in range(NUM_CLASSES)}
    chosen = _round_robin_pick(order_per_class, ctx.budget)

    out_labels = labels.copy()
    for i in chosen:
        a = int(labels[i])
        out_labels[i] = _target_class_non_true(proba[i], a)
    touched = np.asarray(chosen, dtype=np.int64)
    return _pack(ctx, flatten(ctx.images), out_labels, touched, "label_only")


# --------------------------------------------------------------------------- #
# 打包输出（严格保证：输出规模一致、修改数 ≤ 预算）
# --------------------------------------------------------------------------- #
def _pack(ctx: AttackContext, out_flat: np.ndarray, out_labels: np.ndarray,
          touched: np.ndarray, tag: str) -> dict:
    budget = ctx.budget
    out_flat = np.asarray(out_flat, dtype=np.float32).copy()
    out_labels = np.asarray(out_labels, dtype=np.int64).copy()
    touched = np.asarray(touched, dtype=np.int64)

    if len(touched) > budget:           # 超预算则只保留前 budget 个
        revert = touched[budget:]
        out_flat[revert] = flatten(ctx.images)[revert]
        out_labels[revert] = ctx.labels[revert]
        touched = touched[:budget]

    images = out_flat.reshape(ctx.images.shape).astype(np.float32)
    n_label = int((out_labels != ctx.labels).sum())
    n_img = int((np.abs(out_flat - flatten(ctx.images)).max(axis=1) > 0).sum())
    result = {
        "images": images,
        "labels": out_labels,
        "n_modified": int(len(np.union1d(
            np.flatnonzero(out_labels != ctx.labels),
            np.flatnonzero(np.abs(out_flat - flatten(ctx.images)).max(axis=1) > 0)))),
        "n_label_changed": n_label,
        "n_image_changed": n_img,
        "method": tag,
    }
    if ctx.cfg.verbose:
        print(f"[attack] {tag}: 标签改动={n_label} 图像改动={n_img} "
              f"合计修改={result['n_modified']} / 预算={budget}")
    return result
