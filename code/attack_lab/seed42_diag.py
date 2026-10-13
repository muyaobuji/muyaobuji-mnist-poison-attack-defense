#!/usr/bin/env python
"""诊断：seed=42 下，对每个源类做集中式标签翻转，看实际造成的准确率下降。

目的：找出 5% 预算下能稳定突破 Δ=5 个百分点的投毒配方（供提交的攻击代码使用）。
"""

from __future__ import annotations

import os
import sys

import numpy as np

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from attack_lab.env import LocalEnv  # noqa: E402
from attack_lab.methods.common import class_indices, flatten  # noqa: E402
from dataset import load_benchmark  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def centroids(flat, labels, C=10):
    cent = np.zeros((C, flat.shape[1]))
    for c in range(C):
        m = labels == c
        if m.any():
            cent[c] = flat[m].mean(0)
    return cent


def main():
    seed = 42
    train, test = load_benchmark(seed=seed)
    images = np.asarray(train["images"], dtype=np.float32)
    labels = np.asarray(train["labels"], dtype=np.int64)
    n = len(labels)
    budget = max(1, int(0.05 * n))
    flat = flatten(images)
    cls_idx = class_indices(labels)
    cent = centroids(flat, labels)
    C = 10

    env = LocalEnv(test["images"], test["labels"], {"model": "mlp", "seed": seed})
    acc_clean = env.accuracy(env.train(images, labels))
    print(f"seed={seed} 预算={budget} Acc_clean={acc_clean:.4f}")
    print(f"各类样本数: {[len(cls_idx[c]) for c in range(C)]}\n")

    dist = np.linalg.norm(cent[:, None, :] - cent[None, :, :], axis=2)
    np.fill_diagonal(dist, np.inf)
    nearest = dist.argmin(1)

    print("=== 单源类集中翻转 400 个样本 -> 最近原型类 ===")
    results = []
    for c in range(C):
        ci = cls_idx[c]
        take = min(budget, len(ci))
        t = int(nearest[c])
        pl = labels.copy()
        pl[ci[:take]] = t
        acc = env.accuracy(env.train(images, pl))
        drop = acc_clean - acc
        results.append((drop, c, t, take))
        mark = "  <== 过线" if drop >= 0.05 else ""
        print(f"  类 {c} -> {t}: 翻转 {take} 个  Acc={acc:.4f}  下降={drop*100:+.2f}pp{mark}")

    results.sort(reverse=True)
    print(f"\n最佳单源类: 类 {results[0][1]} -> {results[0][2]}，下降 {results[0][0]*100:+.2f}pp")

    # 对最佳源类，试不同目标类
    best_c = results[0][1]
    print(f"\n=== 源类 {best_c} 换不同目标类 ===")
    for t in range(C):
        if t == best_c:
            continue
        pl = labels.copy()
        ci = cls_idx[best_c]
        pl[ci[:min(budget, len(ci))]] = t
        acc = env.accuracy(env.train(images, pl))
        print(f"  类 {best_c} -> {t}: 下降={(acc_clean-acc)*100:+.2f}pp")

    # 组合：预算拆分到两个最优类
    print("\n=== 两源类各 200 ===")
    top2 = [results[0][1], results[1][1]]
    pl = labels.copy()
    for c in top2:
        ci = cls_idx[c]
        pl[ci[:budget // 2]] = int(nearest[c])
    acc = env.accuracy(env.train(images, pl))
    print(f"  类 {top2} 各 {budget//2}: 下降={(acc_clean-acc)*100:+.2f}pp")


if __name__ == "__main__":
    main()
