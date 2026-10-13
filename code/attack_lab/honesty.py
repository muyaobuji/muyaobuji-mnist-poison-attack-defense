#!/usr/bin/env python
"""诚实性实验：在「攻击方看不到测试集」的前提下评测陷阱投毒。

背景：本地 `data/test.npz` 是基准给出的干净测试集，但平台隐藏验证集未必与其
一致。因此必须确认两个版本的效果：
  A. 有测试集信息（transductive，本地可见）—— 上界参考
  B. 无测试集信息（只从训练集合成陷阱）—— 保证可迁移

同时验证「陷阱是否只对见过的测试样本有效」：把陷阱对照另一份独立测试样本
（从训练集里切出的 held-out 集）评测。

用法：
    python -m attack_lab.honesty --trap-mode test_info
    python -m attack_lab.honesty --trap-mode synth
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
from attack_lab.methods.common import class_indices, flatten, nearest_class_map  # noqa: E402
from dataset import load_benchmark  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def asr(acc_clean, acc):
    return float(max(0.0, (acc_clean - acc) / max(acc_clean, 1e-8)))


def synth_traps(ctx: M.AttackContext, rank_by: str = "margin", sigma: float = 0.06,
                blend: float = 0.35, seed: int = 42) -> dict:
    """无测试集版陷阱投毒：用「跨类混合」合成歧义陷阱点。

    对每个目标类 a，用代理模型挑出「最不确定/边际最小」的训练样本，
    再与最近异类 b 的样本混合，图像放到两类之间、标签改为 b。
    """
    rng = np.random.default_rng(seed)
    images = np.asarray(ctx.images, dtype=np.float32)
    labels = np.asarray(ctx.labels, dtype=np.int64)
    flat = flatten(images)
    proba = ctx.surro_proba
    if proba is None:
        raise ValueError("synth_traps 需要 surro_proba")

    target_of = np.asarray(ctx.proxies.get("target_of",
                                           nearest_class_map(flat, labels)), dtype=np.int64)
    cls_idx = class_indices(labels)
    srt = np.sort(proba, axis=1)
    margin = srt[:, -1] - srt[:, -2]
    true_conf = proba[np.arange(len(labels)), labels]

    if rank_by == "margin":
        score = margin
    elif rank_by == "conf":
        score = true_conf
    else:
        score = -true_conf

    # 各类别内按 score 升序（越靠前越「靠近边界」）
    order = {c: cls_idx[c][np.argsort(score[cls_idx[c]])] for c in range(10)}
    ptr = {c: 0 for c in range(10)}
    out = flat.copy()
    idx_out, new_out = [], []

    while len(idx_out) < ctx.budget:
        progressed = False
        for a in range(10):
            if len(idx_out) >= ctx.budget:
                break
            if ptr[a] >= len(order[a]):
                continue
            i = int(order[a][ptr[a]])
            ptr[a] += 1
            b = int(target_of[a])
            cand = cls_idx[b]
            if len(cand) == 0:
                continue
            # 找一个与之最近的异类样本作为混合对象
            d = np.linalg.norm(flat[cand] - flat[i], axis=1)
            j = int(cand[int(d.argmin())])
            out[i] = np.clip((1.0 - blend) * flat[i] + blend * flat[j]
                             + rng.normal(0.0, sigma, size=flat[i].shape).astype(np.float32),
                             0.0, 1.0).astype(np.float32)
            idx_out.append(i)
            new_out.append(b)
            progressed = True
        if not progressed:
            break

    idx = np.asarray(idx_out, dtype=np.int64)
    new = np.asarray(new_out, dtype=np.int64)
    return {"images": out.reshape(images.shape),
            "labels": M._flip_labels(labels, idx, new),
            "indices": idx,
            "note": f"合成陷阱（无测试集）rank={rank_by} blend={blend}"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trap-mode", choices=["test_info", "synth", "both"], default="both")
    parser.add_argument("--model", default="knn1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    train, test = load_benchmark(seed=args.seed)
    labels = np.asarray(train["labels"], dtype=np.int64)
    images = np.asarray(train["images"], dtype=np.float32)
    budget = max(1, int(0.05 * len(labels)))

    cfg = {"model": "knn" if args.model.startswith("knn") else args.model,
           "seed": args.seed, "knn_k": 1}
    clean_env = LocalEnv(test["images"], test["labels"], cfg)
    acc_clean = clean_env.accuracy(clean_env.train(images, labels))
    print(f"[诚实性] 评分模型={args.model}  Acc_clean={acc_clean:.4f}  预算={budget}")

    # 从训练集切一份 held-out，用于检验陷阱是否只对"见过的"样本有效
    rng = np.random.default_rng(1234)
    perm = rng.permutation(len(labels))
    hold_n = 1500
    hold_idx, keep_idx = perm[:hold_n], perm[hold_n:]
    print(f"[诚实性] held-out 独立测试集 {hold_n} 张，攻击方只能看到 {len(keep_idx)} 张训练样本\n")

    # 攻击方视角：只有 keep 部分
    ctx = M.AttackContext(images=images[keep_idx], labels=labels[keep_idx], budget=budget,
                          seed=args.seed)
    surro = LocalEnv(None, None, cfg).train(ctx.images, ctx.labels)
    ctx.surro_proba = surro.predict_proba(ctx.images)
    ctx.surro_pred = surro.predict(ctx.images)
    ctx.proxies["target_of"] = nearest_class_map(flatten(ctx.images), ctx.labels)

    # A. 有测试集信息：攻击方额外看到 held-out 的**图像**（不看标签）
    ctxA = M.AttackContext(**{**ctx.__dict__})
    ctxA.test_images = images[hold_idx]
    ctxA.test_proba = surro.predict_proba(images[hold_idx])
    ctxA.test_labels = np.asarray(labels[hold_idx])   # 仅用于排序，模拟"可见测试集"

    results = {}
    for mode in (["test_info", "synth"] if args.trap_mode == "both" else [args.trap_mode]):
        if mode == "test_info":
            out = M5.kNN_trap_poison(ctxA, rank_by="margin", blend=0.3)
        else:
            out2 = M5.kNN_trap_poison(ctxA, use_test_images=False)
            out = synth_traps(ctx)
        # 把投毒数据放回完整训练集
        full_images = images.copy()
        full_labels = labels.copy()
        full_images[keep_idx] = np.asarray(out["images"], dtype=np.float32)
        full_labels[keep_idx] = np.asarray(out["labels"], dtype=np.int64)
        n_lab = int((full_labels != labels).sum())
        n_img = int((np.abs(flatten(full_images) - flatten(images)).max(axis=1) > 0).sum())
        print(f"  [{mode}] 投毒统计: 标签改动={n_lab} 图像改动={n_img} "
              f"keep集内标签改动={int((full_labels[keep_idx] != labels[keep_idx]).sum())}")

        for eval_name, ev_images, ev_labels, train_slice in [
            ("held-out(未见过的测试集)", images[hold_idx], labels[hold_idx], keep_idx),
            ("基准 test.npz", test["images"], test["labels"], np.arange(len(labels))),
        ]:
            env = LocalEnv(ev_images, ev_labels, cfg)
            # 干净基线必须只用攻击方可见的训练子集，避免把 held-out 自己训进去
            acc0 = env.accuracy(env.train(images[train_slice], labels[train_slice]))
            acc1 = env.accuracy(env.train(full_images, full_labels))
            print(f"  [{mode:9s}] 评测于 {eval_name:22s}: Acc {acc0:.4f} -> {acc1:.4f}  "
                  f"ASR={asr(acc0, acc1):.4f}")
            results[f"{mode}|{eval_name}"] = {"acc_clean": acc0, "acc_attack": acc1,
                                              "asr": asr(acc0, acc1)}
        print()

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"model": args.model, "acc_clean": acc_clean,
                       "results": results}, f, ensure_ascii=False, indent=2)
        print(f"[诚实性] 已保存 {args.out}")


if __name__ == "__main__":
    main()
