#!/usr/bin/env python
"""verify_attack.py —— 给提交的 attack.py 做真实数据端到端验证。

用法（把 attack.py 放在本文件同目录，或用 --attack 指定路径）：

    python verify_attack.py                       # 默认找同目录 attack.py
    python verify_attack.py --attack D:\\sub\\attack.py
    python verify_attack.py --data  ../data       # 指定基准数据目录

做的事情：
    1. 从离线缓存加载基准 MNIST（8000 训练 / 2000 干净测试，seed=42）
    2. 训练干净基线模型 -> Acc_clean
    3. 调用提交的 attack(images, labels, poison_rate, seed)
    4. 检查合规性：输出规模一致、修改数 ≤ 5% 预算、取值合法
    5. 训练投毒模型 -> Acc_attack，算 UntargetedASR 与是否过线
    6. 额外在 kNN 统一模型上复测一遍（跨模型稳健性）

依赖：numpy、scikit-learn
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


# --------------------------------------------------------------------------- #
# 指标
# --------------------------------------------------------------------------- #
def untargeted_asr(acc_clean: float, acc_attack: float) -> float:
    return float(max(0.0, (acc_clean - acc_attack) / max(acc_clean, 1e-8)))


def score_of(asr: float) -> int:
    """按平台评分档换算得分。"""
    if asr >= 0.10:
        return 100
    if asr >= 0.05:
        return 85
    if asr >= 0.03:
        return 70
    if asr > 0:
        return 60
    return 0


# --------------------------------------------------------------------------- #
# 数据
# --------------------------------------------------------------------------- #
def load_data(data_dir: str, train_size: int = 8000, test_size: int = 2000, seed: int = 42):
    """优先读 data/train.npz + data/test.npz，缺失时用 OpenML 缓存重建。"""
    tr_path = os.path.join(data_dir, "train.npz")
    te_path = os.path.join(data_dir, "test.npz")
    if os.path.exists(tr_path) and os.path.exists(te_path):
        t, e = np.load(tr_path), np.load(te_path)
        print(f"[data] 命中缓存 {tr_path}")
        return (t["images"].astype(np.float32), t["labels"].astype(np.int64),
                e["images"].astype(np.float32), e["labels"].astype(np.int64))

    print("[data] 未找到 npz 缓存，尝试从 OpenML 离线缓存重建 …")
    sys.path.insert(0, os.path.dirname(os.path.abspath(data_dir)))
    from dataset import load_benchmark
    tr, te = load_benchmark(train_size=train_size, test_size=test_size, seed=seed)
    return (np.asarray(tr["images"], np.float32), np.asarray(tr["labels"], np.int64),
            np.asarray(te["images"], np.float32), np.asarray(te["labels"], np.int64))


# --------------------------------------------------------------------------- #
# 模型
# --------------------------------------------------------------------------- #
def train_mlp(x, y, seed=42):
    from sklearn.neural_network import MLPClassifier
    est = MLPClassifier(hidden_layer_sizes=(256, 128), activation="relu", solver="adam",
                        alpha=1e-4, batch_size=128, learning_rate_init=1e-3, max_iter=40,
                        early_stopping=True, validation_fraction=0.1, n_iter_no_change=5,
                        random_state=seed)
    est.fit(x.reshape(len(x), -1), y)
    return est


def train_knn(x, y, k=1):
    from sklearn.neighbors import KNeighborsClassifier
    est = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
    est.fit(x.reshape(len(x), -1), y)
    return est


def acc(model, x, y) -> float:
    return float(np.mean(model.predict(x.reshape(len(x), -1)) == y))


# --------------------------------------------------------------------------- #
# 加载提交代码
# --------------------------------------------------------------------------- #
def load_attack(path: str):
    spec = importlib.util.spec_from_file_location("submitted_attack", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["submitted_attack"] = mod
    spec.loader.exec_module(mod)
    if not hasattr(mod, "attack"):
        raise SystemExit(f"❌ {path} 里没有找到 attack 函数")
    return mod.attack


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attack", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                     "attack.py"))
    ap.add_argument("--data", default="")
    ap.add_argument("--poison-rate", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    data_dir = args.data or os.path.join(os.path.dirname(here), "data")
    if not os.path.isdir(data_dir):
        data_dir = os.path.join(here, "data")

    print("=" * 74)
    print("提交代码真实数据验证")
    print("=" * 74)
    print(f"  attack.py : {args.attack}")
    print(f"  数据目录  : {data_dir}\n")

    attack_fn = load_attack(args.attack)
    x_tr, y_tr, x_te, y_te = load_data(data_dir, seed=args.seed)
    n = len(y_tr)
    budget = max(1, int(args.poison_rate * n))
    print(f"[data] 训练集 {x_tr.shape}  测试集 {x_te.shape}  预算 {budget} "
          f"({args.poison_rate*100:.0f}%)\n")

    # ---- 调用攻击 ----
    out = attack_fn(x_tr, y_tr, poison_rate=args.poison_rate, seed=args.seed)
    p_img = np.asarray(out["images"], dtype=np.float32)
    p_lab = np.asarray(out["labels"], dtype=np.int64)

    chg_lab = np.flatnonzero(p_lab != y_tr)
    chg_img = np.flatnonzero(np.abs(p_img.reshape(len(p_img), -1)
                                    - x_tr.reshape(len(x_tr), -1)).max(axis=1) > 0)
    n_mod = int(len(np.union1d(chg_lab, chg_img)))

    # ---- 合规性检查 ----
    print("合规性检查")
    checks = [
        ("输出图像规模与原训练集一致", p_img.shape == x_tr.shape),
        ("输出标签规模与原训练集一致", p_lab.shape == y_tr.shape),
        (f"修改样本数 {n_mod} ≤ 预算 {budget}", n_mod <= budget),
        ("标签取值合法 {0..9}", set(np.unique(p_lab).tolist()) <= set(range(10))),
        ("像素值域 [0,1]", float(p_img.min()) >= 0.0 and float(p_img.max()) <= 1.0),
        ("确实修改了样本", n_mod > 0),
    ]
    for name, ok in checks:
        print(f"  [{'OK  ' if ok else 'FAIL'}] {name}")
    if not all(ok for _, ok in checks):
        print("\n❌ 存在不合规项，评测可能直接判 0 分")
    print(f"  （标签改动 {len(chg_lab)} 个，图像改动 {len(chg_img)} 个，合计 {n_mod} 个）\n")

    # ---- 统一模型评测 ----
    for mname, builder, builder_kw in [
        ("MLP(256,128)  【基准统一模型】", train_mlp, {}),
        ("kNN(k=1)      【平台提示的最近质心系】", train_knn, {"k": 1}),
    ]:
        m0 = builder(x_tr, y_tr, seed=args.seed, **builder_kw) if builder is train_mlp \
            else builder(x_tr, y_tr, **builder_kw)
        a0 = acc(m0, x_te, y_te)
        m1 = builder(p_img, p_lab, seed=args.seed, **builder_kw) if builder is train_mlp \
            else builder(p_img, p_lab, **builder_kw)
        a1 = acc(m1, x_te, y_te)
        asr = untargeted_asr(a0, a1)
        print(f"{mname}")
        print(f"  Acc_clean = {a0:.4f}   Acc_attack = {a1:.4f}   "
              f"下降 = {(a0-a1)*100:+.2f} 个百分点")
        print(f"  UntargetedASR = {asr:.4f}  →  评分档 {score_of(asr)} 分"
              f"{'   ✅ 过攻击成功线(Δ=5pp)' if a1 < a0 - 0.05 else ''}\n")

    print("=" * 74)
    print("验证完成：以上为真实 MNIST 数据上的实测结果。")
    print("=" * 74)


if __name__ == "__main__":
    main()
