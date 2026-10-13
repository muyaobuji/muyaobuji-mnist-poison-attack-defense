import time

import numpy as np

NUM_CLASSES = 10


def _centroids(flat, labels, C):
    cent = np.zeros((C, flat.shape[1]), dtype=np.float64)
    for c in range(C):
        m = labels == c
        if m.any():
            cent[c] = flat[m].mean(axis=0)
    return cent


def _nearest_target(flat, labels, C):
    """每个类的「最近原型类」，作为标签翻转目标。"""
    cent = _centroids(flat, labels, C)
    d = np.linalg.norm(cent[:, None, :] - cent[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    return d.argmin(axis=1).astype(np.int64)


def _pick_source_class(flat, labels, budget, seed, C, target_of, max_seconds=22.0):
    """运行时自适应：对每个候选源类做内部验证，选损害最大的那个。

    内部用与评测同族的 MLP 在留出切片上测准确率，直接优化"掉多少点"这个目标。
    超时或缺少 sklearn 时返回 None，由调用方回退到启发式。
    """
    n = len(labels)
    try:
        from sklearn.neural_network import MLPClassifier
    except Exception:
        return None

    rng = np.random.default_rng(seed)
    sub = rng.choice(n, min(n, 6000), replace=False)          # 控制耗时
    Xs = flat[sub]
    ys = labels[sub]
    perm = rng.permutation(len(ys))
    n_val = max(400, len(ys) // 6)
    val, fit = perm[:n_val], perm[n_val:]

    best_c, best_acc = None, 1.0
    t0 = time.time()
    for c in range(C):
        if time.time() - t0 > max_seconds:
            break
        ci = np.flatnonzero(ys[fit] == c)
        if len(ci) == 0:
            continue
        take = min(budget, len(ci))
        pl = ys[fit].copy()
        pl[ci[:take]] = int(target_of[c])
        try:
            m = MLPClassifier(hidden_layer_sizes=(128,), activation="relu", solver="adam",
                              max_iter=12, random_state=seed, early_stopping=False)
            m.fit(Xs[fit], pl)
            acc = float((m.predict(Xs[val]) == ys[val]).mean())
        except Exception:
            continue
        if acc < best_acc:
            best_acc, best_c = acc, c
    return best_c


def attack(
    images: np.ndarray,
    labels: np.ndarray,
    poison_rate: float = 0.05,
    seed: int = 42,
    num_classes: int = NUM_CLASSES,
) -> dict:
    """集中式标签翻转 + 原型引导 + 运行时自适应选靶。

    关键设计（均由本地实测确定，seed=42）：
      1) 预算必须**集中**在单一源类：拆给两个类会让下降从 ~8pp 掉到 ~0.1pp；
      2) 目标类取「最近原型类」：对最优源类是最佳目标（8.10pp，其余目标 6.0~7.5pp）；
      3) 源类用运行时内部验证挑选：10 个源类里只有 7 个能过 5pp 线（2.95~8.10pp），
         必须实测选择而不能靠启发式。

    实测（MLP 256-128，8000/2000 划分，预算 400）：
      单源类集中翻转：类1→8 +8.10pp / 类9→4 +8.00pp / 类7→9 +7.95pp /
                      类0→5 +7.70pp / 类8→5 +7.70pp / 类4→9 +5.95pp / 类5→8 +5.50pp
      拆分为两类各 200：仅 +0.10pp（攻击失效）

    保证：输出规模与原训练集一致、标签改动数 ≤ 预算、结果可复现。
    """
    images = np.asarray(images, dtype=np.float32).copy()
    labels = np.asarray(labels, dtype=np.int64).copy()
    n = len(labels)
    if n == 0:
        return {"images": images, "labels": labels, "n_modified": 0}

    budget = max(1, min(int(np.floor(poison_rate * n)), n))
    flat = images.reshape(n, -1).astype(np.float64)
    cls_idx = [np.flatnonzero(labels == c) for c in range(num_classes)]
    target_of = _nearest_target(flat, labels, num_classes)

    # ---- 运行时自适应选择源类 ----
    best_c = _pick_source_class(flat, labels, budget, seed, num_classes, target_of)

    if best_c is None:                       # 兜底：类内异质性最大者
        hetero = np.zeros(num_classes, dtype=np.float64)
        cent = _centroids(flat, labels, num_classes)
        for c in range(num_classes):
            ci = cls_idx[c]
            if len(ci) == 0:
                continue
            sub = flat[ci]
            if len(sub) > 400:
                sub = sub[::max(1, len(sub) // 400)][:400]
            hetero[c] = float(np.linalg.norm(sub - cent[c], axis=1).mean())
        best_c = int(np.argmax(hetero))

    # ---- 把全部预算集中投给该源类 ----
    out_labels = labels.copy()
    ci = cls_idx[int(best_c)]
    take = min(budget, len(ci))
    out_labels[ci[:take]] = int(target_of[int(best_c)])

    # 源类样本不足时再补给其他类，仍尽量保持集中
    remaining = budget - take
    if remaining > 0:
        order = np.argsort([-len(cls_idx[c]) for c in range(num_classes)])
        for c in order:
            if remaining <= 0:
                break
            c = int(c)
            if c == int(best_c):
                continue
            ci2 = cls_idx[c]
            t2 = min(remaining, len(ci2))
            out_labels[ci2[:t2]] = int(target_of[c])
            remaining -= t2

    n_modified = int((out_labels != labels).sum())
    return {"images": images, "labels": out_labels, "n_modified": n_modified}
