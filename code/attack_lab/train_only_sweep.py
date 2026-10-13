#!/usr/bin/env python
"""只用训练数据的攻击方法横扫对比（评测端视角）。

所有候选方法只吃 task 的 images/labels/budget，在真实基准上对多个候选统一
模型评测，用来确定提交包的默认方法。

用法：
    python -m attack_lab.train_only_sweep --eval-models mlp,knn1,knn3,knn5 --seeds 42,43,44
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
from attack_lab.methods import attacks4 as M4  # noqa: E402
from attack_lab.methods import train_only as T  # noqa: E402
from attack_lab.methods.common import flatten, nearest_class_map  # noqa: E402
from dataset import load_benchmark  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

CANDIDATES = {
    "random_label_flip": (M.random_label_flip, False),
    "centroid_nearest_flip": (M.centroid_nearest_flip, False),
    "concentrated_flip": (lambda c: M4.concentrated_flip(c, n_src=1), False),
    "runnerup_flip": (T.runnerup_flip, True),
    "runnerup_flip_confident": (T.runnerup_flip_confident, True),
    "runnerup_flip_shift": (T.runnerup_flip_shift, True),
    "runnerup_flip_mix": (T.runnerup_flip_mix, True),
    "concentrated_runnerup": (T.concentrated_runnerup, True),
}


def asr(a0, a1):
    return float(max(0.0, (a0 - a1) / max(a0, 1e-8)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-models", default="mlp,knn1,knn3,knn5")
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--poison-rate", type=float, default=0.05)
    ap.add_argument("--methods", default="")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    names = [x for x in (args.methods.split(",") if args.methods else list(CANDIDATES)) if x]
    seeds = [int(s) for s in args.seeds.split(",") if s]

    train, test = load_benchmark(seed=seeds[0])
    images = np.asarray(train["images"], dtype=np.float32)
    labels = np.asarray(train["labels"], dtype=np.int64)
    budget = max(1, int(args.poison_rate * len(labels)))

    print(f"[sweep] 训练集={images.shape} 预算={budget} "
          f"候选模型={args.eval_models} seeds={seeds}\n")

    records = []
    for name in names:
        fn, needs_surro = CANDIDATES.get(name, (None, False))
        if fn is None:
            print(f"!! 跳过 {name}（未找到实现）")
            continue
        for seed in seeds:
            ctx = M.AttackContext(images=images, labels=labels, budget=budget, seed=seed)
            if needs_surro:
                cfg = {"model": "mlp", "seed": seed}
                surro = LocalEnv(test["images"], test["labels"], cfg).train(images, labels)
                ctx.surro_proba = surro.predict_proba(images)
                ctx.surro_pred = surro.predict(images)
                ctx.proxies["target_of"] = nearest_class_map(flatten(images), labels)

            t0 = time.time()
            try:
                out = fn(ctx)
            except Exception as exc:
                print(f"!! {name} (seed={seed}) 失败: {type(exc).__name__}: {exc}")
                continue
            t_atk = time.time() - t0

            p_images = np.asarray(out["images"], dtype=np.float32)
            p_labels = np.asarray(out["labels"], dtype=np.int64)
            n_img = int((np.abs(p_images.reshape(len(p_images), -1)
                                - images.reshape(len(images), -1)).max(axis=1) > 0).sum())
            n_lab = int((p_labels != labels).sum())

            for mname, mcfg in parse_model_specs(args.eval_models, seed):
                e = LocalEnv(test["images"], test["labels"], mcfg)
                acc0 = e.accuracy(e.train(images, labels))
                acc1 = e.accuracy(e.train(p_images, p_labels))
                records.append({"method": name, "seed": seed, "model": mname,
                                "acc_clean": acc0, "acc_attack": acc1,
                                "untargeted_asr": asr(acc0, acc1),
                                "n_label": n_lab, "n_img": n_img, "t_attack_s": round(t_atk, 2)})

        sub = [r for r in records if r["method"] == name]
        if sub:
            models = sorted({r["model"] for r in sub})
            cells = "  ".join(
                f"{m}={np.mean([r['untargeted_asr'] for r in sub if r['model'] == m]):.4f}"
                for m in models)
            print(f"  {name:26s} {cells}   (n_lab={sub[0]['n_label']} "
                  f"n_img={sub[0]['n_img']} {sub[0]['t_attack_s']}s)")

    # 汇总排名（按所有模型 ASR 均值）
    print("\n" + "=" * 84)
    print("排名（所有 seed × 所有模型 的 ASR 均值）")
    print("=" * 84)
    models = sorted({r["model"] for r in records})
    agg = {}
    for name in names:
        sub = [r for r in records if r["method"] == name]
        if not sub:
            continue
        agg[name] = float(np.mean([r["untargeted_asr"] for r in sub]))
    header = "  ".join(f"{m:>9s}" for m in models)
    print(f"  {'method':26s} {'mean':>8s}  {header}")
    for name, val in sorted(agg.items(), key=lambda kv: -kv[1]):
        cells = "  ".join(
            f"{np.mean([r['untargeted_asr'] for r in records if r['method'] == name and r['model'] == m]):9.4f}"
            for m in models)
        print(f"  {name:26s} {val:8.4f}  {cells}")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"records": records, "aggregate": agg}, f, ensure_ascii=False, indent=2)
        print(f"\n[sweep] 已保存 {args.out}")


if __name__ == "__main__":
    main()
