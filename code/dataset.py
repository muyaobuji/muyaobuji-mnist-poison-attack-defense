#!/usr/bin/env python
"""数据集代码 —— 手写数字分类数据投毒基准（Benchmark）数据集处理。

职责：
1. 优先加载本地缓存（data/train.npz, data/test.npz），避免重复下载与划分；
2. 否则通过 sklearn 的 fetch_openml 从 OpenML 下载 MNIST（mnist_784，数据集 id=554，
   开源地址 https://www.openml.org/d/554，本目录 data/openml 已内置离线缓存）；
3. 网络与缓存均不可用时，回退到离线合成数据，保证评测流程总能跑通；
4. 统一输出格式（基准输入输出规范）：

       train = {"images": [N_train, 1, 28, 28] float32，像素取值 [0,1] 已归一化,
                "labels": [N_train] int64，取值 {0,1,...,9}}
       test  = {"images": [N_test, 1, 28, 28] float32,
                "labels": [N_test] int64}

    基准固定划分为 8,000 张训练图像与 2,000 张测试图像（seed=42 打乱抽样），
    测试集在整个评测过程中保持干净、不参与任何训练，模拟“隐藏测试集”。

用法：
    python dataset.py                 # 生成/加载基准数据并打印规模
"""

from __future__ import annotations

import os

import numpy as np

# --------------------------------------------------------------------------- #
# 常量（与基准文档一致）
# --------------------------------------------------------------------------- #
DEFAULT_TRAIN_SIZE = 8000
DEFAULT_TEST_SIZE = 2000
DEFAULT_SEED = 42
CACHE_DIR = "data"


# --------------------------------------------------------------------------- #
# 数据预处理
# --------------------------------------------------------------------------- #
def to_image_format(images: np.ndarray, normalize: bool = True) -> np.ndarray:
    """统一转为 [N, 1, 28, 28] float32，并按需归一化到 [0,1]。"""
    images = np.asarray(images, dtype=np.float32)
    if images.ndim == 2:
        n = images.shape[0]
        images = images.reshape(n, 1, 28, 28)
    elif images.ndim == 3:
        images = images.reshape(images.shape[0], 1, 28, 28)
    elif images.ndim != 4:
        raise ValueError(f"不支持的图像形状: {images.shape}")
    if normalize:
        if images.max() > 1.5:  # 原始像素范围 [0,255] -> [0,1]
            images = images / 255.0
        images = np.clip(images, 0.0, 1.0)
    return images


# --------------------------------------------------------------------------- #
# 数据加载
# --------------------------------------------------------------------------- #
def _load_mnist_openml(cache_dir: str):
    """从 OpenML 拉取 MNIST（优先命中 data/openml 内置缓存）。"""
    from sklearn.datasets import fetch_openml

    os.makedirs(cache_dir, exist_ok=True)
    bunch = fetch_openml(
        "mnist_784",
        version=1,
        as_frame=False,
        data_home=cache_dir,
    )
    X = bunch.data.astype(np.float32)  # [N, 784]
    y = bunch.target.astype(np.int64)  # '0'..'9' -> int
    return X, y


def _make_synthetic(n_total: int, seed: int = 42):
    """离线合成 10 类“类数字”图像数据（无网络且无缓存时的兜底）。

    为每个类别生成一张随机稀疏模板，样本 = 类模板 + 高斯噪声，
    保证 MLP 能学到较高准确率，使攻击/防御指标可观测。
    """
    rng = np.random.default_rng(seed)
    n_classes = 10
    templates = np.zeros((n_classes, 28, 28), dtype=np.float32)
    for c in range(n_classes):
        for _ in range(3 + c % 3):
            cy, cx = rng.integers(4, 24, size=2)
            sigma = rng.uniform(1.5, 3.0)
            yy, xx = np.mgrid[0:28, 0:28]
            templates[c] += np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma ** 2))
    templates = np.clip(templates / templates.max(axis=(1, 2), keepdims=True), 0, 1)

    per_class = n_total // n_classes
    X = np.zeros((per_class * n_classes, 28, 28), dtype=np.float32)
    y = np.zeros(per_class * n_classes, dtype=np.int64)
    for c in range(n_classes):
        s, e = c * per_class, (c + 1) * per_class
        noise = rng.normal(0.0, 0.25, size=(per_class, 28, 28)).astype(np.float32)
        X[s:e] = np.clip(templates[c][None] + noise, 0.0, 1.0)
        y[s:e] = c
    perm = rng.permutation(len(X))
    return X[perm].astype(np.float32), y[perm].astype(np.int64)


def load_benchmark(
    train_size: int = DEFAULT_TRAIN_SIZE,
    test_size: int = DEFAULT_TEST_SIZE,
    seed: int = DEFAULT_SEED,
    cache_dir: str = CACHE_DIR,
    use_cache: bool = True,
) -> tuple[dict, dict]:
    """加载基准训练集与测试集，返回 (train, test) 各为 {images, labels} 字典。

    首次调用会从 OpenML（或内置缓存）加载并落盘 data/train.npz、data/test.npz；
    之后直接读取本地 npz，保证划分完全一致、结果可复现。
    """
    train_path = os.path.join(cache_dir, "train.npz")
    test_path = os.path.join(cache_dir, "test.npz")

    if use_cache and os.path.exists(train_path) and os.path.exists(test_path):
        t = np.load(train_path)
        e = np.load(test_path)
        train = {"images": t["images"], "labels": t["labels"]}
        test = {"images": e["images"], "labels": e["labels"]}
        print(f"[dataset] 命中本地缓存: {train_path}, {test_path}")
        return train, test

    rng = np.random.default_rng(seed)
    total_needed = train_size + test_size

    X, y = None, None
    try:
        X, y = _load_mnist_openml(cache_dir)
        print(f"[dataset] 从 OpenML 加载 MNIST 成功: {X.shape}")
    except Exception as exc:  # 网络/缓存均不可用
        print(f"[dataset] OpenML 加载失败（{exc}），回退到合成数据。")
        X, y = None, None

    if X is None:
        X, y = _make_synthetic(max(total_needed * 3, 20000), seed)

    n = X.shape[0]
    total_needed = min(total_needed, n)
    perm = rng.permutation(n)[:total_needed]
    X, y = X[perm], y[perm]
    tr = min(train_size, max(total_needed - 500, int(total_needed * 0.8)))
    Xtr, ytr = X[:tr], y[:tr]
    Xte, yte = X[tr:total_needed], y[tr:total_needed]

    train = {
        "images": to_image_format(Xtr),
        "labels": ytr.astype(np.int64),
    }
    test = {
        "images": to_image_format(Xte),
        "labels": yte.astype(np.int64),
    }

    os.makedirs(cache_dir, exist_ok=True)
    np.savez_compressed(train_path, images=train["images"], labels=train["labels"])
    np.savez_compressed(test_path, images=test["images"], labels=test["labels"])
    print(f"[dataset] 已保存基准数据缓存: {train_path}, {test_path}")
    return train, test


def save_npz(path: str, images: np.ndarray, labels: np.ndarray) -> None:
    """将 {images, labels} 数据保存为 .npz 文件（供攻击/防御结果落盘）。"""
    np.savez_compressed(path, images=np.asarray(images, dtype=np.float32),
                        labels=np.asarray(labels, dtype=np.int64))
    print(f"[dataset] 已保存 {path}（{len(labels)} 个样本）")


def load_npz(path: str) -> dict:
    """读取 .npz 文件为 {images, labels} 字典。"""
    d = np.load(path)
    return {"images": d["images"], "labels": d["labels"]}


if __name__ == "__main__":
    tr, te = load_benchmark()
    print(f"train.images: {tr['images'].shape}  dtype={tr['images'].dtype}  "
          f"值域 [{tr['images'].min():.3f}, {tr['images'].max():.3f}]")
    print(f"train.labels: {tr['labels'].shape}  类别 {sorted(set(tr['labels'].tolist()))}")
    print(f"test.images:  {te['images'].shape}  test.labels: {te['labels'].shape}")
