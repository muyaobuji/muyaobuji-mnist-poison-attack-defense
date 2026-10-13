#!/usr/bin/env python
"""攻击方实验脚手架：批量跑攻击方法 -> 统一模型训练 -> 指标对比。

用法：
    python -m attack_lab.harness --stage batch                # 批量对比方法（默认 Model-A/MLP）
    python -m attack_lab.harness --stage grid                 # 选择策略 × 目标策略 网格搜索
    python -m attack_lab.harness --stage poison --method X    # 导出某个方法的投毒数据
    python -m attack_lab.harness --stage evaluate --input X   # 评测已导出的投毒数据
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time

import numpy as np

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from attack import attack as baseline_attack  # noqa: E402
from attack_lab.env import LocalEnv  # noqa: E402
from attack_lab.methods import attacks as M  # noqa: E402
from attack_lab.methods import attacks2 as M2  # noqa: E402
from attack_lab.methods import attacks3 as M3  # noqa: E402
from attack_lab.methods import attacks4 as M4  # noqa: E402
from attack_lab.methods import attacks5 as M5  # noqa: E402
from attack_lab.methods import nn_attacks as MN  # noqa: E402
from attack_lab.methods import selectors as S  # noqa: E402
from attack_lab.methods.common import class_indices, flatten, nearest_class_map  # noqa: E402
from dataset import load_benchmark  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

RESULTS_DIR = os.path.join(CODE_DIR, "attack_lab", "results")


# --------------------------------------------------------------------------- #
# 指标
# --------------------------------------------------------------------------- #
def untargeted_asr(acc_clean: float, acc_attack: float) -> float:
    return float(max(0.0, (acc_clean - acc_attack) / max(acc_clean, 1e-8)))


def attack_success(acc_clean: float, acc_attack: float, delta: float = 0.05) -> bool:
    return bool(acc_attack < acc_clean - delta)


# --------------------------------------------------------------------------- #
# 上下文构造（代理模型先验）
# --------------------------------------------------------------------------- #
def build_context(train: dict, test: dict, budget: int, seed: int,
                  surro_cfg: dict | None = None) -> M.AttackContext:
    images = np.asarray(train["images"], dtype=np.float32)
    labels = np.asarray(train["labels"], dtype=np.int64)

    cfg = {"model": "mlp", "seed": seed}
    if surro_cfg:
        cfg.update(surro_cfg)
    env = LocalEnv(test["images"], test["labels"], cfg)
    model = env.train(images, labels)

    ctx = M.AttackContext(
        images=images,
        labels=labels,
        budget=budget,
        seed=seed,
        surro_proba=model.predict_proba(images),
        surro_pred=model.predict(images),
        test_proba=model.predict_proba(test["images"]),
        test_labels=np.asarray(test["labels"], dtype=np.int64),
        test_images=np.asarray(test["images"], dtype=np.float32),
    )
    ctx.proxies["target_of"] = nearest_class_map(flatten(images), labels)
    ctx.proxies["class_idx"] = class_indices(labels)
    return ctx


# --------------------------------------------------------------------------- #
# 方法注册表：(函数, 说明)
# --------------------------------------------------------------------------- #
METHODS = {
    # --- 标签篡改类 ---
    "random_label_flip": (M.random_label_flip, False),
    "centroid_nearest_flip": (M.centroid_nearest_flip, False),
    "confusion_pair_flip": (M.confusion_pair_flip, False),
    "greedy_budget_alloc": (M.greedy_budget_alloc, False),
    "confident_hard_flip": (M.confident_hard_flip, False),
    "class_pair_swap": (M2.class_pair_swap, False),
    "baseline_project_attack": (None, False),
    # --- 特征扰动 / 冲突类 ---
    "flip_with_shift": (M.flip_with_shift, False),
    "mix_r50_keep": (lambda c: M2.feature_mix_poison(c, ratio=0.5, keep_label=True), False),
    "mix_r70_keep": (lambda c: M2.feature_mix_poison(c, ratio=0.7, keep_label=True), False),
    "mix_r70_flip": (lambda c: M2.feature_mix_poison(c, ratio=0.7, keep_label=False), False),
    "mix_max_r70_keep": (lambda c: M2.feature_mix_poison(c, ratio=0.7, keep_label=True, mix="max"), False),
    "ambiguity_patch_40": (lambda c: M2.ambiguity_patch(c, pixel_budget=40, scale=0.6), False),
    "ambiguity_patch_80": (lambda c: M2.ambiguity_patch(c, pixel_budget=80, scale=0.6), False),
    # --- 梯度 / 优化类（纯 NumPy） ---
    "grad_flip_e15": (lambda c: MN.grad_flip(c, eps=0.15), False),
    "grad_flip_e30": (lambda c: MN.grad_flip(c, eps=0.30), False),
    "influence_lossgrad": (lambda c: MN.influence_select(c, weight="loss_grad"), False),
    "influence_loss": (lambda c: MN.influence_select(c, weight="loss"), False),
    "influence_grad": (lambda c: MN.influence_select(c, weight="grad"), False),
    "emn_noise_e20": (lambda c: MN.emn_noise(c, eps=0.2), False),
    # --- 选择策略组合（单点） ---
    "sel_conf_near": (lambda c: S.selective_flip(c, "confident", "nearest"), False),
    "sel_unc_near": (lambda c: S.selective_flip(c, "uncertain", "nearest"), False),
    "sel_conf_conf": (lambda c: S.selective_flip(c, "confident", "confusion"), False),
    "compound_conf_near": (lambda c: S.compound_flip_shift(c, "confident", "nearest"), False),
    "compound_unc_near": (lambda c: S.compound_flip_shift(c, "uncertain", "nearest"), False),
    # --- 针对最近质心 / kNN 的结构化攻击 ---
    "image_replacement": (lambda c: M3.image_replacement(c, prefer_far=True), False),
    "image_replacement_near": (lambda c: M3.image_replacement(c, prefer_far=False), False),
    "replacement_plus_flip": (lambda c: M3.replacement_plus_flip(c, prefer_far=True), False),
    "lever_label_flip": (lambda c: M3.lever_label_flip(c), False),
    "lever_flip_farthest": (lambda c: M3.lever_label_flip(c, target_mode="farthest"), False),
    # --- 第四批：集中式 / 测试驱动 ---
    "concentrated_n1": (lambda c: M4.concentrated_flip(c, n_src=1), False),
    "concentrated_n2": (lambda c: M4.concentrated_flip(c, n_src=2), False),
    "concentrated_n3": (lambda c: M4.concentrated_flip(c, n_src=3), False),
    "concentrated_n5": (lambda c: M4.concentrated_flip(c, n_src=5), False),
    "test_driven_poison": (lambda c: M4.test_driven_poison(c), False),
    # --- 第五批：kNN 陷阱投毒优化 + 混合策略 ---
    "trap_margin_blend0": (lambda c: M5.kNN_trap_poison(c, rank_by="margin", blend=0.0), False),
    "trap_margin_blend3": (lambda c: M5.kNN_trap_poison(c, rank_by="margin", blend=0.3), False),
    "trap_margin_blend5": (lambda c: M5.kNN_trap_poison(c, rank_by="margin", blend=0.5), False),
    "trap_conf_blend3": (lambda c: M5.kNN_trap_poison(c, rank_by="conf", blend=0.3), False),
    "trap_nndist_blend3": (lambda c: M5.kNN_trap_poison(c, rank_by="nndist", blend=0.3), False),
    "trap_labelonly": (lambda c: M5.kNN_trap_poison(c, use_test_images=False), False),
    "mixed_trap70": (lambda c: M5.mixed_trap_and_flip(c, trap_ratio=0.7), False),
    "mixed_trap50": (lambda c: M5.mixed_trap_and_flip(c, trap_ratio=0.5), False),
    "mixed_trap85": (lambda c: M5.mixed_trap_and_flip(c, trap_ratio=0.85), False),
}

# 评分模型候选：名称 -> 训练配置覆盖
MODEL_SPECS = {
    "mlp": {"model": "mlp"},
    "mlp_wide": {"model": "mlp_wide"},
    "cnn": {"model": "cnn"},
    "knn3": {"model": "knn", "knn_k": 3},
    "knn1": {"model": "knn", "knn_k": 1},
    "knn5": {"model": "knn", "knn_k": 5},
    "centroid": {"model": "centroid"},
    "centroid_l2": {"model": "centroid", "nc_norm": "l2"},
    "centroid_cos": {"model": "centroid", "nc_metric": "cos"},
    "centroid_median": {"model": "centroid", "nc_mode": "median"},
    "centroid_trim": {"model": "centroid", "nc_mode": "trim"},
    "centroid_std": {"model": "centroid", "nc_std": "pixel"},
    "centroid_std_l2": {"model": "centroid", "nc_std": "pixel", "nc_norm": "l2"},
}


def parse_model_specs(text: str, seed: int) -> list:
    names = [t.strip() for t in text.split(",") if t.strip()]
    out = []
    for n in names:
        if n not in MODEL_SPECS:
            raise ValueError(f"未知评分模型: {n}（可选 {sorted(MODEL_SPECS)}）")
        cfg = dict(MODEL_SPECS[n])
        cfg["seed"] = seed
        out.append((n, cfg))
    return out


def run_method(name: str, ctx: M.AttackContext):
    if name == "baseline_project_attack":
        out = baseline_attack(ctx.images, ctx.labels,
                              poison_rate=ctx.budget / ctx.n, seed=ctx.seed)
        changed = np.flatnonzero(np.asarray(out["labels"]) != ctx.labels)
        return {"images": out["images"], "labels": out["labels"],
                "indices": changed, "note": "项目自带 attack.py 基线"}
    fn, _ = METHODS[name]
    return fn(ctx)


# --------------------------------------------------------------------------- #
# 评测
# --------------------------------------------------------------------------- #
def evaluate_case(name, ctx, test, eval_models, acc_clean_map, verbose=True, note=None):
    """在多个候选评分模型上评测同一份投毒数据。"""
    t0 = time.time()
    out = run_method(name, ctx)
    t_attack = time.time() - t0

    images = np.asarray(out["images"], dtype=np.float32)
    labels = np.asarray(out["labels"], dtype=np.int64)
    changed = np.flatnonzero(labels != ctx.labels)
    img_changed = np.flatnonzero(
        np.abs(images.reshape(len(images), -1) - ctx.images.reshape(len(images), -1)).max(axis=1) > 0
    )
    n_modified = int(len(np.union1d(changed, img_changed)))

    per_model = {}
    for mname, mcfg in eval_models:
        env = LocalEnv(test["images"], test["labels"], mcfg)
        model = env.train(images, labels)
        acc = env.accuracy(model)
        acc_clean = acc_clean_map[mname]
        per_model[mname] = {
            "acc_attack": acc,
            "acc_clean": acc_clean,
            "accuracy_drop": acc_clean - acc,
            "untargeted_asr": untargeted_asr(acc_clean, acc),
            "attack_success": attack_success(acc_clean, acc),
        }

    rec = {
        "method": name,
        "note": note if note is not None else out.get("note", ""),
        "n_modified": n_modified,
        "poison_rate": n_modified / ctx.n,
        "over_budget": bool(n_modified > ctx.budget),
        "n_label_changed": int(len(changed)),
        "n_image_changed": int(len(img_changed)),
        "t_attack_s": round(t_attack, 2),
        "models": per_model,
    }
    if verbose:
        cells = " ".join(f"{mn}={per_model[mn]['untargeted_asr']:.3f}" for mn, _ in eval_models)
        print(f"  {name:30s} n={n_modified:<4d} [{t_attack:5.1f}s] {cells}")
    return rec


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="攻击方批量实验")
    parser.add_argument("--stage", choices=["batch", "grid", "poison", "evaluate"], default="batch")
    parser.add_argument("--methods", default="", help="逗号分隔的方法名，空=全部")
    parser.add_argument("--method", default="influence_lossgrad")
    parser.add_argument("--sample-strategies", default="confident,uncertain,dense,random")
    parser.add_argument("--target-strategies", default="nearest,confusion,farthest,random")
    parser.add_argument("--model", default="mlp", help="单个评分模型（--stage poison/evaluate 用）")
    parser.add_argument("--eval-models", default="mlp,centroid,centroid_l2,knn1,knn3",
                        help="逗号分隔的候选评分模型")
    parser.add_argument("--poison-rate", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-size", type=int, default=8000)
    parser.add_argument("--test-size", type=int, default=2000)
    parser.add_argument("--tag", default="")
    parser.add_argument("--input", default="")
    parser.add_argument("--output-dir", default=RESULTS_DIR)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    train, test = load_benchmark(train_size=args.train_size, test_size=args.test_size, seed=args.seed)
    labels = np.asarray(train["labels"], dtype=np.int64)
    budget = max(1, int(args.poison_rate * len(labels)))
    eval_models = parse_model_specs(args.eval_models, args.seed)
    eval_cfg = dict(eval_models[0][1])

    print(f"\n=== 攻击实验 | 候选评分模型={[m for m,_ in eval_models]} | 预算={budget} "
          f"({args.poison_rate*100:.1f}%) | seed={args.seed} ===")

    acc_clean_map = {}
    for mname, mcfg in eval_models:
        t0 = time.time()
        env = LocalEnv(test["images"], test["labels"], mcfg)
        model = env.train(train["images"], labels)
        acc_clean_map[mname] = env.accuracy(model)
        print(f"[干净基线] {mname:16s} Acc_clean = {acc_clean_map[mname]:.4f} "
              f"({time.time()-t0:.1f}s)")
    print()

    if args.stage == "evaluate":
        poisoned = np.load(args.input)
        images = poisoned["images"].astype(np.float32)
        plabels = poisoned["labels"].astype(np.int64)
        n_mod = int(len(np.flatnonzero(plabels != labels)))
        for mname, mcfg in eval_models:
            env = LocalEnv(test["images"], test["labels"], mcfg)
            model = env.train(images, plabels)
            acc = env.accuracy(model)
            print(f"  {mname:16s} Acc_attack={acc:.4f} "
                  f"ASR={untargeted_asr(acc_clean_map[mname], acc):.4f}")
        print(f"  n_modified={n_mod}")
        return

    ctx = build_context(train, test, budget, args.seed)

    tag = args.tag or args.model
    report = os.path.join(args.output_dir,
                          f"attack_{args.stage}_{tag}.json")

    if args.stage == "grid":
        s_strats = [s for s in args.sample_strategies.split(",") if s]
        t_strats = [s for s in args.target_strategies.split(",") if s]
        combos = list(itertools.product(s_strats, t_strats))
        records = []
        print(f"[grid] {len(combos)} 个组合\n")
        for ss, ts in combos:
            name = f"sel_{ss}_{ts}"
            try:
                rec = evaluate_case(name, ctx, test, eval_models, acc_clean_map,
                                    note=f"select={ss} target={ts}")
                records.append(rec)
            except Exception as exc:
                print(f"!! {name} 失败: {type(exc).__name__}: {exc}")
            _dump(report, acc_clean_map, args, records, budget)
        _rank(records, report)
        return

    if args.stage == "poison":
        out = run_method(args.method, ctx)
        path = os.path.join(args.output_dir, f"poisoned_{args.method}.npz")
        np.savez_compressed(path, images=np.asarray(out["images"], dtype=np.float32),
                            labels=np.asarray(out["labels"], dtype=np.int64))
        print(f"[poison] 已保存投毒数据: {path}")
        return

    names = [m for m in (args.methods.split(",") if args.methods else list(METHODS)) if m]
    records = []
    for name in names:
        try:
            records.append(evaluate_case(name, ctx, test, eval_models, acc_clean_map))
        except Exception as exc:
            print(f"!! {name} 失败: {type(exc).__name__}: {exc}")
        _dump(report, acc_clean_map, args, records, budget)
    _rank(records, report)


def _dump(path, acc_clean_map, args, records, budget):
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"acc_clean": acc_clean_map, "eval_models": args.eval_models,
                   "budget": budget, "seed": args.seed, "poison_rate": args.poison_rate,
                   "records": records}, f, ensure_ascii=False, indent=2)


def _rank(records, report):
    print(f"\n[harness] 报告已保存: {report}")
    if not records:
        return
    model_names = list(records[0]["models"].keys())
    print("\n排名（按各模型 ASR 均值）:")
    for r in records:
        r["_avg"] = float(np.mean([r["models"][m]["untargeted_asr"] for m in model_names]))
    records.sort(key=lambda r: -r["_avg"])
    header = "  ".join(f"{m:>10s}" for m in model_names)
    print(f"  {'method':32s} {'avg':>7s}  {header}")
    for i, r in enumerate(records[:20], 1):
        cells = "  ".join(f"{r['models'][m]['untargeted_asr']:10.4f}" for m in model_names)
        print(f"{i:2d}. {r['method']:32s} {r['_avg']:7.4f}  {cells}")


if __name__ == "__main__":
    main()
