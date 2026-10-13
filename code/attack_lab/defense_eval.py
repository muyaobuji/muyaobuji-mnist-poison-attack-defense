#!/usr/bin/env python
"""攻防对阵评测：多攻击 × 多防御，算 RecoveryRate / EffectiveASR。

指标（与基准文档一致）：
    RecoveryRate = clip((Acc_defense - Acc_attack) / (Acc_clean - Acc_attack), 0, 1)
    EffectiveASR = max(0, (Acc_clean - Acc_defense) / Acc_clean)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from attack_lab.env import LocalEnv  # noqa: E402
from attack_lab.harness import parse_model_specs  # noqa: E402
from attack_lab.methods import attacks as M  # noqa: E402
from attack_lab.methods.common import flatten, nearest_class_map  # noqa: E402
from dataset import load_benchmark  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


# --------------------------------------------------------------------------- #
# 防御方法
# --------------------------------------------------------------------------- #
def defense_augment(images, labels, seed=42, max_shift=2, copies=1):
    """Baseline：数据增强（每样本附加 copies 份随机平移副本）。"""
    images = np.asarray(images, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    rng = np.random.default_rng(seed)
    n = images.shape[0]
    imgs, lbls = [images], [labels]
    for _ in range(copies):
        aug = images.copy()
        sh = rng.integers(-max_shift, max_shift + 1, size=(n, 2))
        for i in range(n):
            aug[i] = np.roll(aug[i], shift=(int(sh[i, 0]), int(sh[i, 1])), axis=(1, 2))
        imgs.append(aug)
        lbls.append(labels.copy())
    return {"images": np.concatenate(imgs, 0).astype(np.float32),
            "labels": np.concatenate(lbls, 0).astype(np.int64),
            "n_augmented": n * copies}


def defense_clean(images, labels, seed=42, k=5, mode="relabel", copies=1, keep_ratio=0.9):
    """标签清洗防御：用 kNN 交叉投票找出可疑（疑似被改标签）的样本并纠正/剔除。

    原理
    ----
    投毒样本的标签与其特征邻域的真实标签不一致。用「留一法」式的近邻投票
    （把样本自身的标签排除在投票外）估计其"特征决定的标签"，若与给定标签
    冲突则判为可疑，按 mode 处理：
        relabel  —— 改成投票得到的标签
        drop     —— 直接从训练集移除
        weight   —— 保留但置于低权重组（等价于降权，通过复制低分样本实现）

    参数
    ----
    k          : 近邻数
    mode       : relabel | drop | weight
    copies     : 清洗后是否再做数据增强（0=不做）
    keep_ratio : 对剩余样本的比例（drop 模式下最多剔除 (1-keep_ratio) 的样本）
    """
    from sklearn.neighbors import NearestNeighbors

    images = np.asarray(images, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    n = len(labels)
    X = flatten(images).astype(np.float32)

    # 每个样本的「特征决定标签」= 排除自身及其同类近邻后的多数投票
    nn = NearestNeighbors(n_neighbors=min(k + 2, n)).fit(X)
    _, ind = nn.kneighbors(X)

    cleaned = labels.copy()
    suspicious = np.zeros(n, dtype=bool)
    for i in range(n):
        neigh = ind[i][1:k + 2]                       # 排除自身
        votes = np.bincount(labels[neigh], minlength=10)
        # 排除与自身同标签的邻居，避免"抄袭"自己原来的错误标签
        same = labels[neigh] == labels[i]
        if same.any():
            votes[int(labels[i])] -= int(same.sum())
        feat_label = int(votes.argmax()) if votes.sum() > 0 else int(labels[i])
        if feat_label != int(labels[i]) and votes[feat_label] >= max(2, k // 2):
            suspicious[i] = True
            cleaned[i] = feat_label

    if mode == "relabel":
        out_images, out_labels = images, cleaned
    elif mode == "drop":
        keep = np.flatnonzero(~suspicious)
        max_drop = int(n * (1.0 - keep_ratio))
        if len(keep) < n - max_drop:
            # 可疑样本太多时，按"可疑程度"只剔除最靠前的一部分
            order = np.flatnonzero(suspicious)
            keep = np.setdiff1d(np.arange(n), order[:max_drop])
        out_images, out_labels = images[keep], labels[keep]
    elif mode == "weight":
        # 降权近似：可疑样本只保留一次，干净样本额外复制一份
        keep_clean = np.flatnonzero(~suspicious)
        out_images = np.concatenate([images, images[keep_clean]], 0)
        out_labels = np.concatenate([cleaned, labels[keep_clean]], 0)
    else:
        raise ValueError(f"未知 mode: {mode}")

    if copies > 0:
        aug = defense_augment(out_images, out_labels, seed=seed, copies=copies)
        return aug

    return {"images": out_images.astype(np.float32), "labels": out_labels.astype(np.int64),
            "n_augmented": 0}


DEFENSES = {
    "none": lambda im, lb, seed: {"images": im, "labels": lb, "n_augmented": 0},
    "augment": lambda im, lb, seed: defense_augment(im, lb, seed=seed, copies=1),
    "clean_relabel": lambda im, lb, seed: defense_clean(im, lb, seed=seed, mode="relabel", copies=1),
    "clean_drop": lambda im, lb, seed: defense_clean(im, lb, seed=seed, mode="drop", copies=1),
    "clean_weight": lambda im, lb, seed: defense_clean(im, lb, seed=seed, mode="weight", copies=1),
    "clean_relabel_noaug": lambda im, lb, seed: defense_clean(im, lb, seed=seed, mode="relabel", copies=0),
}


# --------------------------------------------------------------------------- #
# 攻击方法
# --------------------------------------------------------------------------- #
def make_attacks():
    def random_flip(ctx):
        return M.random_label_flip(ctx)

    def centered_flip(ctx):
        from attack_lab.methods.attacks4 import concentrated_flip
        return concentrated_flip(ctx, n_src=1)

    return {"random_flip": random_flip, "concentrated_flip": centered_flip}


def recovery_rate(a_clean, a_attack, a_defense):
    denom = a_clean - a_attack
    if denom <= 1e-6:
        return None
    return float(np.clip((a_defense - a_attack) / denom, 0.0, 1.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="mlp,knn1")
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--poison-rate", type=float, default=0.05)
    ap.add_argument("--defenses", default="")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    train, test = load_benchmark(seed=42)
    images = np.asarray(train["images"], dtype=np.float32)
    labels = np.asarray(train["labels"], dtype=np.int64)
    budget = max(1, int(args.poison_rate * len(labels)))
    seeds = [int(s) for s in args.seeds.split(",") if s]
    dnames = [d for d in (args.defenses.split(",") if args.defenses else list(DEFENSES)) if d]

    print(f"[对阵阵评] 攻击×防御  recovery 矩阵 | 预算={budget} seeds={seeds}\n")
    records = []
    attacks = make_attacks()

    for mname, mcfg in parse_model_specs(args.models, seeds[0]):
        print(f"===== 统一模型: {mname} =====")
        env = LocalEnv(test["images"], test["labels"], mcfg)
        for seed in seeds:
            # 干净与投毒基线
            a_clean = env.accuracy(env.train(images, labels))
            ctx = M.AttackContext(images=images, labels=labels, budget=budget, seed=seed)
            for aname, afn in attacks.items():
                out = afn(ctx)
                p_img = np.asarray(out["images"], dtype=np.float32)
                p_lab = np.asarray(out["labels"], dtype=np.int64)
                a_attack = env.accuracy(env.train(p_img, p_lab))
                row = [f"  seed={seed} {aname:18s} Acc_attack={a_attack:.4f}"]
                for dname in dnames:
                    d = DEFENSES[dname](p_img, p_lab, seed)
                    a_def = env.accuracy(env.train(np.asarray(d["images"], dtype=np.float32),
                                                   np.asarray(d["labels"], dtype=np.int64)))
                    rec = recovery_rate(a_clean, a_attack, a_def)
                    eff = max(0.0, (a_clean - a_def) / max(a_clean, 1e-8))
                    records.append({"model": mname, "seed": seed, "attack": aname,
                                    "defense": dname, "acc_clean": a_clean,
                                    "acc_attack": a_attack, "acc_defense": a_def,
                                    "recovery_rate": rec, "effective_asr": eff,
                                    "n_train_out": int(len(d["labels"]))})
                    row.append(f"{dname}={('%.2f' % rec) if rec is not None else 'n/a'}")
                print(" ".join(row))
        print()

    print("=" * 88)
    print("防御方法汇总（RecoveryRate 均值，越大越好；EffectiveASR 越小越好）")
    print("=" * 88)
    for dname in dnames:
        sub = [r for r in records if r["defense"] == dname and r["recovery_rate"] is not None]
        if not sub:
            print(f"  {dname:22s} 无有效对阵（攻击未造成下降）")
            continue
        rec = float(np.mean([r["recovery_rate"] for r in sub]))
        eff = float(np.mean([r["effective_asr"] for r in sub]))
        print(f"  {dname:22s} RecoveryRate={rec:.4f}  EffectiveASR={eff:.4f}  "
              f"对阵数={len(sub)}")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        print(f"\n[对阵阵评] 已保存 {args.out}")


if __name__ == "__main__":
    main()
