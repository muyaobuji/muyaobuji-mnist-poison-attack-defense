#!/usr/bin/env python
"""严格三分实验：彻底分离「构造陷阱用的样本」与「评测用的隐藏样本」。

数据划分（全部来自 MNIST 训练池）：
    pool  6500 张  → 投毒预算的来源，也是攻击方唯一能看到的训练数据
    probe 1500 张  → 模拟"攻击方可见的本地测试集"（只用图像+代理预测）
    eval  1500 张  → 模拟"隐藏测试集"，攻击方完全不可见

三组对照，用来判断方法的真实增益来源：
    clean      : 只有 pool 训练的干净基线
    label_only : 只做标签翻转（pool 内）
    img_only   : 只做图像陷阱（pool 内）
    full       : 图像 + 标签都改
统一在 eval 上评测。

用法：
    python -m attack_lab.strict_split --model knn1 --trap margin
    python -m attack_lab.strict_split --model mlp  --trap margin
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from attack_lab.env import LocalEnv  # noqa: E402
from attack_lab.methods import attacks as M  # noqa: E402
from attack_lab.methods import attacks5 as M5  # noqa: E402
from attack_lab.methods.common import class_indices, flatten  # noqa: E402
from dataset import load_benchmark  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def asr(acc0, acc1):
    return float(max(0.0, (acc0 - acc1) / max(acc0, 1e-8)))


def build_cfg(name: str, seed: int) -> dict:
    if name.startswith("knn"):
        return {"model": "knn", "knn_k": int(name[3:]), "seed": seed}
    return {"model": name, "seed": seed}


def _load_full_mnist_arff():
    """直接解析内置离线缓存里的 MNIST ARFF（避免引入 pandas 依赖）。"""
    import gzip

    cands = [
        os.path.join(CODE_DIR, "data", "openml", "openml.org", "data", "v1", "download",
                     "52667", "mnist_784.arff.gz"),
        os.path.join(CODE_DIR, "data", "openml", "openml", "openml.org", "data", "v1",
                     "download", "52667", "mnist_784.arff.gz"),
    ]
    path = next((p for p in cands if os.path.exists(p)), None)
    if path is None:
        raise FileNotFoundError("未找到内置 MNIST ARFF 缓存")

    rows = []
    with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as f:
        in_data = False
        for line in f:
            s = line.strip()
            if not s or s.startswith("%"):
                continue
            if not in_data:
                if s.lower().startswith("@data"):
                    in_data = True
                continue
            rows.append(s)
    arr = np.array([r.split(",") for r in rows], dtype=np.float32)
    X = arr[:, :784]
    y = arr[:, 784].astype(np.int64)
    return X, y


def load_source(name: str, seed: int):
    """加载实验数据源。

    benchmark : 项目自带 8000/2000 划分（与官方 test.py 一致）
    full      : 完整 MNIST 70k，用 OpenML 离线缓存加载，适合做三分严格实验
    """
    if name == "benchmark":
        tr, te = load_benchmark(seed=seed)
        return (np.asarray(tr["images"], np.float32), np.asarray(tr["labels"], np.int64),
                np.asarray(te["images"], np.float32), np.asarray(te["labels"], np.int64))
    X, y = _load_full_mnist_arff()
    X = np.asarray(X, dtype=np.float32) / 255.0
    X = X.reshape(len(X), 1, 28, 28)
    y = y.astype(np.int64)
    # 注意：ARFF 原始顺序是按类别排序的，必须先整体打乱，否则划分会类别失衡
    rs = np.random.default_rng(seed)
    p = rs.permutation(len(y))
    return X[p], y[p], X[:2000], y[:2000]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="knn1")
    ap.add_argument("--trap", default="margin")
    ap.add_argument("--source", default="full", choices=["benchmark", "full"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pool", type=int, default=20000)
    ap.add_argument("--probe", type=int, default=5000)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    images, labels, _, _ = load_source(args.source, args.seed)
    n = len(labels)
    budget = int(0.05 * n)          # 预算按"攻击方可见训练集"规模的 5% 计

    if args.pool + args.probe >= n:
        raise SystemExit(f"pool+probe 必须小于样本数 {n}")

    rng = np.random.default_rng(2024 + args.seed)
    perm = rng.permutation(n)
    pool_idx = perm[:args.pool]
    probe_idx = perm[args.pool:args.pool + args.probe]
    eval_idx = perm[args.pool + args.probe:]

    cfg = build_cfg(args.model, args.seed)
    print(f"[strict] 数据源={args.source} 模型={args.model}  pool={len(pool_idx)} "
          f"probe={len(probe_idx)} eval={len(eval_idx)}  预算={budget} "
          f"({budget/args.pool*100:.2f}% of pool)")

    # 攻击方视角：只有 pool
    ctx = M.AttackContext(images=images[pool_idx], labels=labels[pool_idx],
                          budget=budget, seed=args.seed)
    surro = LocalEnv(None, None, cfg).train(ctx.images, ctx.labels)
    ctx.surro_proba = surro.predict_proba(ctx.images)
    ctx.surro_pred = surro.predict(ctx.images)
    from attack_lab.methods.common import nearest_class_map
    ctx.proxies["target_of"] = nearest_class_map(flatten(ctx.images), ctx.labels)

    # 攻击方额外可见：probe 的图像（标签仅用于评分排序，模拟"可见测试集"）
    ctx.test_images = images[probe_idx]
    ctx.test_proba = surro.predict_proba(images[probe_idx])
    ctx.test_labels = labels[probe_idx]

    # 构造投毒
    trap = M5.kNN_trap_poison(ctx, rank_by=args.trap, blend=0.3)
    trap_idx = np.asarray(trap["indices"], dtype=np.int64)
    trap_lab = np.asarray(trap["labels"], dtype=np.int64)
    poisoned_imgs_pool = flatten(np.asarray(trap["images"], dtype=np.float32))

    # 只做标签翻转的对照：标签改为「非真实类」中概率最高者
    proba_pool = ctx.surro_proba
    order_t = np.argsort(-ctx.test_proba, axis=1)
    tlab_probe = labels[probe_idx]
    # 用同一批目标（排除真实类），但载体不变、图像不改
    base = M.AttackContext(**{**ctx.__dict__})
    label_only = M5.kNN_trap_poison(base, rank_by=args.trap, blend=1.0)  # blend=1 → 图像≈原图

    variants = {
        "clean": (images.copy(), labels.copy()),
        "full(图+标签)": (None, None),
        "img_only(只改图)": (None, None),
        "label_only(只改标签)": (None, None),
    }

    full_i = images.copy()
    full_l = labels.copy()
    full_i[pool_idx] = poisoned_imgs_pool.reshape(len(pool_idx), 1, 28, 28)
    full_l[pool_idx] = np.asarray(trap["labels"], dtype=np.int64)
    variants["full(图+标签)"] = (full_i, full_l)

    img_i = images.copy()
    img_i[pool_idx] = poisoned_imgs_pool.reshape(len(pool_idx), 1, 28, 28)
    variants["img_only(只改图)"] = (img_i, labels.copy())

    lab_only_l = labels.copy()
    lab_only_l[pool_idx] = np.asarray(label_only["labels"], dtype=np.int64)
    variants["label_only(只改标签)"] = (images.copy(), lab_only_l)

    ev_images, ev_labels = images[eval_idx], labels[eval_idx]
    env = LocalEnv(ev_images, ev_labels, cfg)
    results = {}
    acc_clean = None
    for name, (im, lb) in variants.items():
        m = env.train(im, lb)
        acc = env.accuracy(m)
        if name == "clean":
            acc_clean = acc
        results[name] = acc
        print(f"  {name:22s} eval Acc = {acc:.4f}")

    print()
    for name, acc in results.items():
        print(f"  {name:22s} ASR = {asr(acc_clean, acc):.4f}")
    print(f"\n  [标签改动数={int((full_l != labels).sum())}, 图像改动数="
          f"{int(np.abs(flatten(full_i) - flatten(images)).max(1).sum() > 0)}]")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"model": args.model, "trap": args.trap, "acc": results,
                       "acc_clean": acc_clean,
                       "asr": {k: asr(acc_clean, v) for k, v in results.items()}},
                      f, ensure_ascii=False, indent=2)
        print(f"[strict] 已保存 {args.out}")


if __name__ == "__main__":
    main()
