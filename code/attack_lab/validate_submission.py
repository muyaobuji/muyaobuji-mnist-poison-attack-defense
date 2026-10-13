#!/usr/bin/env python
"""用提交包（submission/attack.py）在真实基准上端到端评测。

这是"所提交即所测"的验证：直接调用提交包的 attack(env, task)，
再用多种候选统一模型评测，输出最终指标。

用法：
    python -m attack_lab.validate_submission --eval-models mlp,knn1,knn3,centroid
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time

import numpy as np

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from attack_lab.env import LocalEnv  # noqa: E402
from attack_lab.harness import MODEL_SPECS, parse_model_specs  # noqa: E402
from dataset import load_benchmark  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

SUBMISSION_DIR = os.path.join(CODE_DIR, "submission")


def load_submission():
    """按文件路径加载提交包入口，模拟评测端 import。"""
    path = os.path.join(SUBMISSION_DIR, "attack.py")
    spec = importlib.util.spec_from_file_location("submission_attack", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["submission_attack"] = mod
    spec.loader.exec_module(mod)
    return mod


def untargeted_asr(acc0, acc1):
    return float(max(0.0, (acc0 - acc1) / max(acc0, 1e-8)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-models", default="mlp,knn1,knn3,knn5,centroid")
    ap.add_argument("--poison-rate", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--repeat", type=int, default=1, help="重复次数（换种子做稳健性检验）")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    train, test = load_benchmark(seed=args.seed)
    images = np.asarray(train["images"], dtype=np.float32)
    labels = np.asarray(train["labels"], dtype=np.int64)
    budget = max(1, int(args.poison_rate * len(labels)))

    sub = load_submission()
    print(f"[validate] 已加载提交包: {SUBMISSION_DIR}")
    print(f"[validate] 训练集 {images.shape}  测试集 {np.asarray(test['images']).shape}  "
          f"预算 {budget} ({args.poison_rate*100:.0f}%)\n")

    records = []
    for rep in range(args.repeat):
        seed = args.seed + rep
        task = {"images": images, "labels": labels, "poison_budget": budget}
        env = LocalEnv(test["images"], test["labels"], {"model": "mlp", "seed": seed})

        t0 = time.time()
        out = sub.attack(env, task)
        t_atk = time.time() - t0

        p_images = np.asarray(out["images"], dtype=np.float32)
        p_labels = np.asarray(out["labels"], dtype=np.int64)
        n_lab = int((p_labels != labels).sum())
        n_img = int((np.abs(p_images.reshape(len(p_images), -1)
                            - images.reshape(len(images), -1)).max(axis=1) > 0).sum())
        n_mod = int(len(np.union1d(np.flatnonzero(p_labels != labels),
                                   np.flatnonzero(np.abs(
                                       p_images.reshape(len(p_images), -1)
                                       - images.reshape(len(images), -1)).max(axis=1) > 0))))
        print(f"--- 第 {rep+1} 次 (seed={seed}) 攻击耗时 {t_atk:.1f}s | "
              f"标签改动 {n_lab} 图像改动 {n_img} 合计 {n_mod} / 预算 {budget} ---")

        eval_models = parse_model_specs(args.eval_models, seed)
        for mname, mcfg in eval_models:
            e = LocalEnv(test["images"], test["labels"], mcfg)
            m0 = e.train(images, labels)
            acc0 = e.accuracy(m0)
            m1 = e.train(p_images, p_labels)
            acc1 = e.accuracy(m1)
            rec = {"repeat": rep, "seed": seed, "model": mname, "acc_clean": acc0,
                   "acc_attack": acc1, "accuracy_drop": acc0 - acc1,
                   "untargeted_asr": untargeted_asr(acc0, acc1),
                   "n_modified": n_mod, "poison_rate": n_mod / len(labels),
                   "over_budget": bool(n_mod > budget), "t_attack_s": round(t_atk, 2)}
            records.append(rec)
            flag = "过线" if rec["untargeted_asr"] >= 0.05 else ("有分" if rec["untargeted_asr"] > 0 else "无效")
            print(f"    {mname:12s} Acc {acc0:.4f} -> {acc1:.4f}  "
                  f"ASR={rec['untargeted_asr']:.4f}  [{flag}]")
        print()

    # 汇总
    print("=" * 78)
    print("汇总（提交包实测）")
    print("=" * 78)
    models = sorted({r["model"] for r in records})
    summary = {}
    for m in models:
        vals = [r["untargeted_asr"] for r in records if r["model"] == m]
        summary[m] = {"asr_mean": float(np.mean(vals)), "asr_min": float(np.min(vals)),
                      "asr_max": float(np.max(vals)), "n_runs": len(vals)}
        print(f"  {m:12s} ASR 均值={summary[m]['asr_mean']:.4f}  "
              f"[{summary[m]['asr_min']:.4f}, {summary[m]['asr_max']:.4f}]")
    print(f"  预算合规: {all(not r['over_budget'] for r in records)}")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "records": records}, f,
                      ensure_ascii=False, indent=2)
        print(f"\n[validate] 已保存 {args.out}")


if __name__ == "__main__":
    main()
