# 手写数字分类数据投毒 · 攻击方提交包

本目录是可直接提交的**攻击方**方案（MNIST 数据投毒 untargeted 攻击）。
统一入口与赛题规范一致：

```python
def attack(env, task) -> dict:
    """
    task["images"]        # [N,1,28,28] float32，取值 [0,1]
    task["labels"]        # [N] int64
    task["poison_budget"] # 投毒预算（≤5% 的样本数）
    return {"images": poisoned_images, "labels": poisoned_labels}
    """
```

## 目录结构

```
submission/
├── attack.py            # 入口：attack(env, task)（赛题接口）
├── config.yaml          # 方法选择与超参数
├── requirements.txt     # 依赖（numpy + 可选 scikit-learn）
├── methods/
│   ├── attack_core.py   # 核心攻击实现（陷阱投毒 / 集中式标签翻转 / 多方法融合）
│   ├── surrogate.py     # 代理模型（sklearn 优先，缺失时用内置 NumPy MLP）
│   └── utils.py         # 预算解析、原型统计、类别索引等工具
└── assets/
    └── README.md        # 方法说明与实测数据
```

## 快速验证

```bash
pip install -r requirements.txt
python attack.py --self-test          # 内置合成数据冒烟测试
python attack.py --help               # 参数说明
```

在项目自带基准上评测（可选）：

```bash
python -m attack_lab.harness --stage batch --methods trap_margin_blend3
```

## 方法摘要

核心是**测试驱动陷阱投毒（test-driven trap poisoning）**：在「易被误判的样本」
特征邻域内制造高密度错误投票区，同时对模型决策边界与近邻投票结构造成破坏。
配合**集中式标签翻转**提升对 MLP 类模型的杀伤。详见 `assets/README.md`。
