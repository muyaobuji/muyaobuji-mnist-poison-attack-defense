"""纯 NumPy 的 MLP 与梯度工具（零第三方依赖，供攻击方法内部使用）。

作用：攻击方需要一个「代理模型」来获得梯度或损失信息。为保持提交包
轻量、跨环境可跑，这里用 NumPy 手写一个与基准同构的 MLP（784-256-128-10，
ReLU，Adam），只用于攻击方的内部优化，不参与评分。
"""

from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
# 基础层
# --------------------------------------------------------------------------- #
def _he_init(rng, fan_in, fan_out):
    return rng.normal(0.0, np.sqrt(2.0 / fan_in), size=(fan_in, fan_out)).astype(np.float32)


class NumpyMLP:
    """784-256-128-10 的 ReLU MLP，带 Adam 优化器与反向传播。"""

    def __init__(self, dims=(784, 256, 128, 10), seed: int = 42, lr: float = 1e-3):
        rng = np.random.default_rng(seed)
        self.dims = tuple(dims)
        self.W = [_he_init(rng, self.dims[i], self.dims[i + 1]) for i in range(len(self.dims) - 1)]
        self.b = [np.zeros(self.dims[i + 1], dtype=np.float32) for i in range(len(self.dims) - 1)]
        self.lr = lr
        self._mW = [np.zeros_like(w) for w in self.W]
        self._vW = [np.zeros_like(w) for w in self.W]
        self._mb = [np.zeros_like(b) for b in self.b]
        self._vb = [np.zeros_like(b) for b in self.b]
        self._t = 0

    # -- 前向 ---------------------------------------------------------------- #
    def forward(self, X: np.ndarray, cache: bool = False):
        """X: [N, 784] -> logits [N, 10]。cache=True 时返回中间激活。"""
        h = np.asarray(X, dtype=np.float32)
        acts = [h]
        for i in range(len(self.W) - 1):
            h = h @ self.W[i] + self.b[i]
            h = np.maximum(h, 0.0)
            acts.append(h)
        logits = h @ self.W[-1] + self.b[-1]
        if cache:
            return logits, acts
        return logits

    # -- 损失与反向 ---------------------------------------------------------- #
    @staticmethod
    def softmax_ce(logits: np.ndarray, y: np.ndarray):
        """返回 (平均交叉熵损失, 对 logits 的梯度)。"""
        z = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(z)
        p = exp / exp.sum(axis=1, keepdims=True)
        n = len(y)
        loss = float(-np.log(np.maximum(p[np.arange(n), y], 1e-12)).mean())
        g = p
        g[np.arange(n), y] -= 1.0
        return loss, (g / n).astype(np.float32)

    def backward(self, acts, grad_logits):
        """标准的 MLP 反传，返回 (dW, db, dX)。"""
        dW = [None] * len(self.W)
        db = [None] * len(self.b)
        g = grad_logits
        for i in range(len(self.W) - 1, -1, -1):
            dW[i] = acts[i].T @ g
            db[i] = g.sum(axis=0)
            if i > 0:
                g = (g @ self.W[i].T) * (acts[i] > 0)
        dX = g @ self.W[0].T if len(self.W) else g
        return dW, db, dX

    # -- 训练 ---------------------------------------------------------------- #
    def step(self, X: np.ndarray, y: np.ndarray, beta1: float = 0.9, beta2: float = 0.999,
             eps: float = 1e-8):
        logits, acts = self.forward(X, cache=True)
        loss, gl = self.softmax_ce(logits, y)
        dW, db, _ = self.backward(acts, gl)
        self._t += 1
        for i in range(len(self.W)):
            self._mW[i] = beta1 * self._mW[i] + (1 - beta1) * dW[i]
            self._vW[i] = beta2 * self._vW[i] + (1 - beta2) * dW[i] ** 2
            mhat = self._mW[i] / (1 - beta1 ** self._t)
            vhat = self._vW[i] / (1 - beta2 ** self._t)
            self.W[i] -= self.lr * mhat / (np.sqrt(vhat) + eps)
            self._mb[i] = beta1 * self._mb[i] + (1 - beta1) * db[i]
            self._vb[i] = beta2 * self._vb[i] + (1 - beta2) * db[i] ** 2
            mhat_b = self._mb[i] / (1 - beta1 ** self._t)
            vhat_b = self._vb[i] / (1 - beta2 ** self._t)
            self.b[i] -= self.lr * mhat_b / (np.sqrt(vhat_b) + eps)
        return loss

    def fit(self, X, y, epochs: int = 20, batch_size: int = 128, seed: int = 42, verbose: bool = False):
        rng = np.random.default_rng(seed)
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        n = len(y)
        for ep in range(epochs):
            perm = rng.permutation(n)
            total = 0.0
            for i in range(0, n, batch_size):
                idx = perm[i:i + batch_size]
                total += self.step(X[idx], y[idx])
            if verbose and (ep + 1) % 5 == 0:
                print(f"  [NumpyMLP] epoch {ep+1}/{epochs} loss={total/max(1,n//batch_size):.4f}")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X).argmax(axis=1).astype(np.int64)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        z = self.forward(X)
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    # -- 输入梯度 ------------------------------------------------------------ #
    def input_grad(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """返回交叉熵损失对输入 X 的梯度（用于 FGSM 型扰动）。"""
        logits, acts = self.forward(X, cache=True)
        _, gl = self.softmax_ce(logits, y)
        _, _, dX = self.backward(acts, gl)
        return dX
