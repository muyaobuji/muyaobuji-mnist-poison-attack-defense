"""代理模型：攻击方在本地训练一个与统一模型同构的分类器，用于选择投毒样本。

优先使用 sklearn 的 MLPClassifier（与基准统一模型同构，先验最强）；
若运行环境缺少 sklearn（或 env.train 不可用），自动回退到内置的纯 NumPy MLP，
保证提交包在最小依赖下也能工作。
"""

from __future__ import annotations

import numpy as np

from .numpy_mlp import NumpyMLP
from .utils import flatten

NUM_CLASSES = 10


class Surrogate:
    """带 predict / predict_proba 的代理模型包装。"""

    def __init__(self, impl, kind: str):
        self.impl = impl
        self.kind = kind

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32).reshape(len(X), -1)
        return np.asarray(self.impl.predict(X)).astype(np.int64)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32).reshape(len(X), -1)
        return np.asarray(self.impl.predict_proba(X), dtype=np.float64)


def fit_surrogate(images: np.ndarray, labels: np.ndarray, seed: int = 42,
                  prefer: str = "auto", env=None) -> Surrogate:
    """训练代理模型。

    参数
    ----
    images : [N,1,28,28] 或 [N,784]
    labels : [N]
    prefer : auto | sklearn | numpy | env
        - env：优先调用评测环境自带的 env.train（白盒程度最高、也最省时间）
        - auto：env -> sklearn -> numpy 依次尝试
    env    : 评测环境对象（可选），需提供 train(images=..., labels=...) 接口
    """
    X = flatten(images) if np.asarray(images).ndim > 2 else np.asarray(images, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)

    order = {
        "env": ["env"],
        "sklearn": ["sklearn"],
        "numpy": ["numpy"],
        "auto": ["env", "sklearn", "numpy"],
    }.get(prefer, ["env", "sklearn", "numpy"])

    for impl_name in order:
        try:
            if impl_name == "env" and env is not None and hasattr(env, "train"):
                model = env.train(images=images, labels=labels)
                if hasattr(model, "predict_proba"):
                    return Surrogate(model, "env")
            elif impl_name == "sklearn":
                from sklearn.neural_network import MLPClassifier  # 延迟导入

                est = MLPClassifier(
                    hidden_layer_sizes=(256, 128), activation="relu", solver="adam",
                    alpha=1e-4, batch_size=128, learning_rate_init=1e-3, max_iter=40,
                    early_stopping=True, validation_fraction=0.1, n_iter_no_change=5,
                    random_state=seed,
                )
                est.fit(X, y)
                return Surrogate(est, "sklearn")
            elif impl_name == "numpy":
                mlp = NumpyMLP(seed=seed)
                mlp.fit(X, y, epochs=20, seed=seed)
                return Surrogate(mlp, "numpy")
        except Exception:       # 任一后端不可用就尝试下一个
            continue

    raise RuntimeError("无法训练代理模型：env/sklearn/NumPy 后端均不可用")


def second_choice(proba: np.ndarray) -> np.ndarray:
    """每个样本的「第二候选类」（概率次高者），用于选择标签翻转目标。"""
    return np.argsort(-np.asarray(proba), axis=1)[:, 1].astype(np.int64)


def margin(proba: np.ndarray) -> np.ndarray:
    """前两候选的概率差（越小越靠近决策边界）。"""
    s = np.sort(np.asarray(proba), axis=1)
    return (s[:, -1] - s[:, -2]).astype(np.float64)
