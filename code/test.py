#!/usr/bin/env python
"""测试代码 —— 手写数字分类数据投毒基准（Benchmark）指标计算。

流程：
    攻击评测（run_attack.sh）：
        干净训练集 --训练--> Acc_clean
        干净训练集 --攻击--> 投毒训练集 --训练--> Acc_attack
        指标：PoisonRate / UntargetedASR / AttackSuccess
    防御评测（run_defense.sh）：
        投毒训练集 --防御--> 清洗后训练集 --训练--> Acc_defense
        指标：RecoveryRate / EffectiveASR

指标公式（与基准文档一致，Acc 为干净测试集上的分类准确率）：
    Accuracy        = 正确预测数 / 样本总数
    PoisonRate      = N_modified / N_train                       （≤ 5%）
    UntargetedASR   = max(0, (Acc_clean - Acc_attack) / Acc_clean)
    AttackSuccess   = Acc_attack < Acc_clean - delta             （delta = 0.05）
    RecoveryRate    = clip((Acc_defense - Acc_attack) /
                           (Acc_clean  - Acc_attack), 0, 1)
    EffectiveASR    = max(0, (Acc_clean - Acc_defense) / Acc_clean)

用法：
    python test.py --stage attack      # 攻击方 Baseline 评测（默认）
    python test.py --stage defense     # 防守方 Baseline 评测
    python test.py --stage all         # 完整攻防评测
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from attack import attack
from dataset import load_benchmark
from defense import defend

# --------------------------------------------------------------------------- #
# 统一分类模型（基准训练协议：MLP，隐层 [256,128]，Adam，与文档一致）
# --------------------------------------------------------------------------- #
def train_model(images: np.ndarray, labels: np.ndarray, seed: int = 42):
    """在给定数据上训练统一分类模型，返回具有 predict 接口的对象。"""
    from sklearn.neural_network import MLPClassifier

    X = np.asarray(images, dtype=np.float32).reshape(len(images), -1)
    y = np.asarray(labels, dtype=np.int64)
    est = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=128,
        learning_rate_init=1e-3,
        max_iter=40,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=5,
        random_state=seed,
    )
    est.fit(X, y)
    return est


def accuracy(model, images: np.ndarray, labels: np.ndarray) -> float:
    """Accuracy = 正确预测数 / 样本总数。"""
    pred = model.predict(np.asarray(images, dtype=np.float32).reshape(len(images), -1))
    labels = np.asarray(labels)
    return float(np.mean(np.asarray(pred) == labels))


# --------------------------------------------------------------------------- #
# 指标计算（严格对齐基准文档公式）
# --------------------------------------------------------------------------- #
def untargeted_asr(acc_clean: float, acc_attack: float) -> float:
    """非定向攻击成功率：归一化整体性能下降，越大攻击越成功。"""
    return float(max(0.0, (acc_clean - acc_attack) / max(acc_clean, 1e-8)))


def attack_success(acc_clean: float, acc_attack: float, delta: float = 0.05) -> bool:
    """攻击成功判定：Acc_attack < Acc_clean - delta。"""
    return bool(acc_attack < acc_clean - delta)


def recovery_rate(acc_clean: float, acc_attack: float, acc_defense: float):
    """防御恢复率：clip((Acc_defense-Acc_attack)/(Acc_clean-Acc_attack), 0, 1)。

    攻击未造成有效下降（分母≈0）时返回 None，不计入统计。
    """
    denom = acc_clean - acc_attack
    if denom <= 1e-6:
        return None
    return float(np.clip((acc_defense - acc_attack) / denom, 0.0, 1.0))


def effective_asr(acc_clean: float, acc_defense: float) -> float:
    """防御后残余攻击效果（用于全对阵矩阵的攻击-对抗意义）。"""
    return float(max(0.0, (acc_clean - acc_defense) / max(acc_clean, 1e-8)))


# --------------------------------------------------------------------------- #
# 评测入口
# --------------------------------------------------------------------------- #
def evaluate(
    stage: str = "attack",
    train_size: int = 8000,
    test_size: int = 2000,
    poison_rate: float = 0.05,
    delta: float = 0.05,
    seed: int = 42,
    output_dir: str = "results",
):
    train, test = load_benchmark(train_size=train_size, test_size=test_size, seed=seed)
    n_train = len(train["labels"])
    budget = max(1, int(poison_rate * n_train))

    t0 = time.time()
    # 1) 干净基线：干净训练集 -> 模型 -> Acc_clean
    model_clean = train_model(train["images"], train["labels"], seed=seed)
    acc_clean = accuracy(model_clean, test["images"], test["labels"])
    print(f"\n[评测] 干净训练: Acc_clean = {acc_clean:.4f}  ({time.time()-t0:.1f}s)")

    # 2) 攻击：随机标签翻转 -> 投毒训练集 -> 模型 -> Acc_attack
    t0 = time.time()
    poisoned = attack(train["images"], train["labels"],
                      poison_rate=poison_rate, seed=seed)
    n_modified = int((poisoned["labels"] != train["labels"]).sum())
    model_atk = train_model(poisoned["images"], poisoned["labels"], seed=seed)
    acc_attack = accuracy(model_atk, test["images"], test["labels"])
    print(f"[评测] 投毒训练: Acc_attack = {acc_attack:.4f}  ({time.time()-t0:.1f}s)")

    result = {
        "stage": stage,
        "train_size": n_train,
        "test_size": len(test["labels"]),
        "poison_rate_limit": poison_rate,
        "poison_budget": budget,
        "n_modified": n_modified,
        "actual_poison_rate": n_modified / n_train,
        "acc_clean": acc_clean,
        "acc_attack": acc_attack,
        "untargeted_asr": untargeted_asr(acc_clean, acc_attack),
        "attack_success": attack_success(acc_clean, acc_attack, delta),
        "delta": delta,
    }

    # 3) 防御：置信学习纠正 -> 清洗后训练集 -> 模型 -> Acc_defense
    if stage in ("defense", "all"):
        t0 = time.time()
        cleaned = defend(poisoned["images"], poisoned["labels"], seed=seed)
        model_def = train_model(cleaned["images"], cleaned["labels"], seed=seed)
        acc_defense = accuracy(model_def, test["images"], test["labels"])
        print(f"[评测] 防御训练: Acc_defense = {acc_defense:.4f}  "
              f"(增强至 {cleaned['n_augmented'] + n_train} 样本, {time.time()-t0:.1f}s)")
        rec = recovery_rate(acc_clean, acc_attack, acc_defense)
        result.update({
            "acc_defense": acc_defense,
            "defense_augmented_samples": cleaned["n_augmented"] + n_train,
            "recovery_rate": rec,
            "effective_asr": effective_asr(acc_clean, acc_defense),
        })

    return result


def print_report(result: dict):
    print("\n" + "=" * 62)
    print(f"基准: MNIST 手写数字分类数据投毒  |  阶段: {result['stage']}")
    print("=" * 62)
    print(f"训练集规模        : {result['train_size']}   测试集规模: {result['test_size']}")
    print(f"投毒预算          : {result['poison_budget']}  (PoisonRate ≤ {result['poison_rate_limit']})")
    print(f"实际修改样本数    : {result['n_modified']}  -> PoisonRate = {result['actual_poison_rate']:.4f}")
    print("-" * 62)
    print(f"Acc_clean         = {result['acc_clean']:.4f}")
    print(f"Acc_attack        = {result['acc_attack']:.4f}")
    print(f"UntargetedASR     = max(0,(Acc_clean-Acc_attack)/Acc_clean) = {result['untargeted_asr']:.4f}")
    print(f"AttackSuccess     = Acc_attack < Acc_clean - {result['delta']:.2f}  ->  {result['attack_success']}")
    if "acc_defense" in result:
        print(f"Acc_defense       = {result['acc_defense']:.4f}")
        print(f"RecoveryRate      = clip((Acc_defense-Acc_attack)/(Acc_clean-Acc_attack),0,1) = "
              f"{result['recovery_rate'] if result['recovery_rate'] is not None else 'n/a'}")
        print(f"EffectiveASR      = max(0,(Acc_clean-Acc_defense)/Acc_clean) = {result['effective_asr']:.4f}")
    print("=" * 62)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="手写数字分类数据投毒基准：Baseline 指标计算",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--stage", choices=["attack", "defense", "all"],
                        default="attack", help="评测阶段")
    parser.add_argument("--train-size", type=int, default=8000)
    parser.add_argument("--test-size", type=int, default=2000)
    parser.add_argument("--poison-rate", type=float, default=0.05)
    parser.add_argument("--delta", type=float, default=0.05, help="攻击成功判定阈值(个百分点)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="results/metrics.json")
    args = parser.parse_args()

    res = evaluate(stage=args.stage, train_size=args.train_size,
                   test_size=args.test_size, poison_rate=args.poison_rate,
                   delta=args.delta, seed=args.seed)
    print_report(res)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f"\n[test] 指标已保存: {args.output}")
