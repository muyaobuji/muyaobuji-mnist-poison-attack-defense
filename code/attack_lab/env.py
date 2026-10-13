#!/usr/bin/env python
"""统一分类模型环境 —— 对标赛题《统一分类模型》接口，供攻击方本地自测。

赛题接口：
    model = env.train(images=train_images, labels=train_labels,
                      sample_weights=sample_weights, config=train_config)
    pred  = model.predict(test_images)

公开阶段为 Model-A（MLP[256,128]）。本文件额外提供 CNN 变体，
用于检测攻击方法的跨模型泛化能力（隐藏验证可能换 Model-B/Model-C）。
"""

from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
# 模型封装
# --------------------------------------------------------------------------- #
class SklearnModel:
    """包装 sklearn 估计器，暴露 predict 接口。"""

    def __init__(self, est):
        self.est = est

    def predict(self, images: np.ndarray) -> np.ndarray:
        X = np.asarray(images, dtype=np.float32).reshape(len(images), -1)
        return np.asarray(self.est.predict(X)).astype(np.int64)

    def predict_proba(self, images: np.ndarray) -> np.ndarray:
        X = np.asarray(images, dtype=np.float32).reshape(len(images), -1)
        return np.asarray(self.est.predict_proba(X), dtype=np.float64)


# --------------------------------------------------------------------------- #
# 默认训练配置（与基准文档一致）
# --------------------------------------------------------------------------- #
DEFAULT_CONFIG = {
    "model": "mlp",              # mlp | mlp_wide | cnn
    "hidden_layer_sizes": (256, 128),
    "activation": "relu",
    "solver": "adam",
    "alpha": 1e-4,
    "batch_size": 128,
    "learning_rate_init": 1e-3,
    "max_iter": 40,
    "early_stopping": True,
    "validation_fraction": 0.1,
    "n_iter_no_change": 5,
    "seed": 42,
}


def _subset_by_weights(images, labels, weights, rng):
    """MLPClassifier 不支持样本权重；用按权重抽样近似（防御方若返回权重时使用）。"""
    if weights is None:
        return images, labels
    w = np.asarray(weights, dtype=np.float64)
    w = np.clip(w, 0.0, None)
    if w.sum() <= 0:
        return images, labels
    p = w / w.sum()
    idx = rng.choice(len(labels), size=len(labels), replace=True, p=p)
    return images[idx], labels[idx]


class NearestCentroidModel:
    """最近质心分类器（平台提示的"统一模型"候选）。

    支持原型统计量、距离度量与特征标准化开关，用于枚举可能的评分器实现。
    """

    def __init__(self, protos, normalize="none", metric="l2", dim=784, std=None):
        self.protos = np.asarray(protos, dtype=np.float32)
        self.normalize = normalize
        self.metric = metric
        self.dim = dim
        self.std = None if std is None else np.asarray(std, dtype=np.float32)

    def _prep(self, X):
        X = np.asarray(X, dtype=np.float32).reshape(len(X), -1)
        if self.std is not None:
            X = X / self.std
        if self.normalize == "l2":
            X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
        return X

    def _dist(self, X):
        X = self._prep(X)
        P = self.protos
        if self.std is not None:
            P = P / self.std
        if self.normalize == "l2":
            P = P / np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-12)
        if self.metric == "cos":
            Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
            Pn = P / np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-12)
            return -(Xn @ Pn.T)
        d = (X ** 2).sum(1, keepdims=True) - 2 * (X @ P.T) + (P ** 2).sum(1)[None, :]
        return d

    def predict(self, images):
        return self._dist(images).argmin(axis=1).astype(np.int64)

    def predict_proba(self, images):
        d = self._dist(images)
        d = d - d.min(axis=1, keepdims=True)
        w = np.exp(-d / max(float(np.median(d) + 1e-9), 1e-6))
        return w / w.sum(axis=1, keepdims=True)


def _train_nearest_centroid(images, labels, cfg):
    X = np.asarray(images, dtype=np.float32).reshape(len(images), -1)
    y = np.asarray(labels, dtype=np.int64)
    mode = str(cfg.get("nc_mode", "mean")).lower()
    std = None
    if str(cfg.get("nc_std", "none")).lower() in ("pixel", "true", "1", "std"):
        std = X.std(axis=0).astype(np.float32)
        std[std < 1e-6] = 1.0
    protos = np.zeros((10, X.shape[1]), dtype=np.float32)
    for c in range(10):
        sub = X[y == c]
        if len(sub) == 0:
            continue
        if mode == "median":
            protos[c] = np.median(sub, axis=0)
        elif mode == "trim":
            s = np.sort(sub, axis=0)
            n = len(s)
            protos[c] = s[int(0.05 * n):max(int(0.95 * n), int(0.05 * n) + 1)].mean(axis=0)
        else:
            protos[c] = sub.mean(axis=0)
    return NearestCentroidModel(protos, normalize=str(cfg.get("nc_norm", "none")),
                                metric=str(cfg.get("nc_metric", "l2")), std=std)


def _train_knn(images, labels, cfg):
    from sklearn.neighbors import KNeighborsClassifier

    X = np.asarray(images, dtype=np.float32).reshape(len(images), -1)
    y = np.asarray(labels, dtype=np.int64)
    k = int(cfg.get("knn_k", 3))
    est = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
    est.fit(X, y)
    return SklearnModel(est)


# --------------------------------------------------------------------------- #
# 环境
# --------------------------------------------------------------------------- #
class LocalEnv:
    """攻击方的本地评测环境。

    参数
    ----
    test_images / test_labels : 干净测试集（模拟隐藏测试集）
    config : 训练配置覆盖项
    """

    def __init__(self, test_images=None, test_labels=None, config=None):
        self.test_images = test_images
        self.test_labels = test_labels
        self.config = dict(DEFAULT_CONFIG)
        if config:
            self.config.update(config)

    # -- 赛题统一训练接口 ---------------------------------------------------- #
    def train(self, images, labels, sample_weights=None, config=None):
        cfg = dict(self.config)
        if config:
            cfg.update(config)

        images = np.asarray(images, dtype=np.float32)
        labels = np.asarray(labels, dtype=np.int64)
        rng = np.random.default_rng(int(cfg.get("seed", 42)))
        images, labels = _subset_by_weights(images, labels, sample_weights, rng)

        model_name = str(cfg.get("model", "mlp")).lower()
        if model_name in ("centroid", "nearest_centroid", "nc"):
            return _train_nearest_centroid(images, labels, cfg)
        if model_name in ("knn",):
            return _train_knn(images, labels, cfg)
        if model_name in ("mlp", "mlp_wide"):
            from sklearn.neural_network import MLPClassifier

            hidden = tuple(cfg["hidden_layer_sizes"])
            if model_name == "mlp_wide":
                hidden = tuple(cfg.get("wide_layer_sizes", (512, 256, 128)))
            est = MLPClassifier(
                hidden_layer_sizes=hidden,
                activation=str(cfg.get("activation", "relu")),
                solver=str(cfg.get("solver", "adam")),
                alpha=float(cfg.get("alpha", 1e-4)),
                batch_size=int(cfg.get("batch_size", 128)),
                learning_rate_init=float(cfg.get("learning_rate_init", 1e-3)),
                max_iter=int(cfg.get("max_iter", 40)),
                early_stopping=bool(cfg.get("early_stopping", True)),
                validation_fraction=float(cfg.get("validation_fraction", 0.1)),
                n_iter_no_change=int(cfg.get("n_iter_no_change", 5)),
                random_state=int(cfg.get("seed", 42)),
            )
            X = images.reshape(len(images), -1)
            est.fit(X, labels)
            return SklearnModel(est)

        if model_name == "cnn":
            return _train_cnn(images, labels, cfg)

        raise ValueError(f"未知模型类型: {model_name}")

    # -- 便捷评测 ------------------------------------------------------------ #
    def accuracy(self, model, images=None, labels=None) -> float:
        images = self.test_images if images is None else images
        labels = self.test_labels if labels is None else labels
        pred = model.predict(images)
        return float(np.mean(pred == np.asarray(labels)))


def _train_cnn(images, labels, cfg):
    """小型 CNN（PyTorch 可选）。用于跨模型泛化性检验。"""
    try:
        import torch
        import torch.nn as nn
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"CNN 变体需要 torch：{exc}") from exc

    seed = int(cfg.get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)

    class SmallCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Flatten(), nn.Linear(64 * 7 * 7, 128), nn.ReLU(), nn.Linear(128, 10),
            )

        def forward(self, x):
            return self.net(x)

    X = torch.from_numpy(np.asarray(images, dtype=np.float32))
    y = torch.from_numpy(np.asarray(labels, dtype=np.int64))
    net = SmallCNN()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    lossf = nn.CrossEntropyLoss()
    n = len(y)
    bs = int(cfg.get("batch_size", 128))
    epochs = int(cfg.get("max_iter", 15))
    for _ in range(epochs):
        perm = torch.randperm(n)
        net.train()
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = lossf(net(X[idx]), y[idx])
            loss.backward()
            opt.step()

    class TorchModel:
        def predict(self, imgs):
            net.eval()
            with torch.no_grad():
                out = net(torch.from_numpy(np.asarray(imgs, dtype=np.float32)))
                return out.argmax(dim=1).numpy().astype(np.int64)

        def predict_proba(self, imgs):
            net.eval()
            with torch.no_grad():
                out = net(torch.from_numpy(np.asarray(imgs, dtype=np.float32)))
                return torch.softmax(out, dim=1).numpy()

    return TorchModel()
