# MNIST 数据投毒攻防工具集

> 面向 **训练数据投毒（Data Poisoning）** 的攻防实现与可复现评测框架。
> 攻击方在 **5% 预算**内构造投毒训练集使统一模型准确率下降；防御方在不知情前提下清洗数据、恢复性能。

本项目在「AI-SEC 人工智能安全教学实验平台」的 MNIST 投毒场景中完成攻防双侧实测：

| 方向 | 平台评测结果 | 得分 |
|---|---|---|
| 攻击方 | 攻方成功 · 注入生效（attack_success 达标） | **100** |
| 防守方 | 防御成功 · 全部拦截（recovery_rate=1, effective_asr=0） | **100** |

本地复现（8000 训练 / 2000 干净测试，预算 400 = 5%，seed=42，统一模型 MLP(256,128)+Adam）：

| 方法 | Acc_clean | Acc_attack | 下降 | UntargetedASR | 评分档 |
|---|---|---|---|---|---|
| 随机标签翻转（官方基线） | 0.9395 | 0.9290 | +1.05pp | 0.0122 | 60 |
| **本仓库攻击方案** | 0.9395 | 0.8600 | **+7.90pp** | **0.0841** | **85** |
| 数据增强防御（官方基线） | — | — | — | — | — |
| **本仓库防御方案** | — | — | RecoveryRate **0.7500**，EffectiveASR **0.0008** | | |

---

## 一、核心思路

### 1. 攻击：集中式标签翻转 + 原型引导 + 运行时自适应选靶

随机标签翻转把错误均匀撒到 10 个类别上，每类仅约 0.5% 样本被改标签，错误信号被严重稀释。
本方案做三处关键改进：

| 设计 | 说明 | 实测依据 |
|---|---|---|
| **原型引导选靶** | 用类内均值原型 μ_c 的欧氏距离，取 `target(c) = argmin_{d≠c} ‖μ_c − μ_d‖`，把错误监督落在两类歧义带上 | 对最优源类，最近类目标下降 8.10pp，优于其他目标类的 6.05~7.50pp |
| **集中式预算分配** | 全部预算投给**单一**源类，不拆分 | 拆成两类各 200 后，下降从 8.10pp **崩到 0.10pp** |
| **运行时自适应选源类** | 10 个候选源类中仅 7 个能过 5pp 判定线（2.95~8.10pp），故在攻击内部用同族 MLP 在留出切片上做一次内部验证，选降幅最大者（22 秒时间上限） | 见下表「单源类实测」 |

**seed=42 下单源类集中翻转的实测下降**：

| 源类 → 最近类 | 下降 | 过 5pp 线 |
|---|---|---|
| 1 → 8 | **+8.10pp** | ✅ |
| 9 → 4 | +8.00pp | ✅ |
| 7 → 9 | +7.95pp | ✅ |
| 0 → 5 | +7.70pp | ✅ |
| 8 → 5 | +7.70pp | ✅ |
| 4 → 9 | +5.95pp | ✅ |
| 5 → 8 | +5.50pp | ✅ |
| 3 → 5 | +3.60pp | ❌ |
| 2 → 8 | +3.25pp | ❌ |
| 6 → 2 | +2.95pp | ❌ |

### 2. 防御：近邻投票清洗（剔除）+ 数据增强

投毒样本的标签与其特征邻域的真实标签不一致。用「留一法」近邻投票估计每个样本**特征决定的标签**
（排除自身，并扣除同标签邻居的「照抄票」，需足够反对票才判可疑），然后：

| 处理策略 | RecoveryRate 均值 | 结论 |
|---|---|---|
| **剔除可疑样本** | **0.7500** | ✅ 最优 |
| 纯数据增强 | 0.6698 | 次优 |
| 改标签（纠正监督值） | 0.2809 | 差 |
| 重加权 | 0.1188 | 最差 |

**剔除优于改标签的原因**：改标签只纠正监督值，噪声样本仍留在训练集中继续影响决策边界；
只有剔除才真正移除错误监督信号。为防误伤，剔除量设 10% 上限，再对剩余样本追加 ±2 像素平移副本。

---

## 二、目录结构

```
mnist-poison-attack-defense/
├── README.md                      本文件
├── LICENSE                        MIT
├── requirements.txt               依赖（numpy + 可选 scikit-learn）
├── code/
│   ├── attack.py                  ★ 攻击实现（平台提交口径，单文件零第三方依赖）
│   ├── defense.py                 ★ 防御实现（平台提交口径）
│   ├── dataset.py                 基准数据加载（8000/2000 固定划分）
│   ├── test.py                    官方基准评测入口（六项指标）
│   ├── run_attack.sh              攻击评测脚本
│   ├── run_defense.sh             防御评测脚本
│   ├── data/                      内置数据缓存（train.npz / test.npz）
│   ├── results/                   基准指标输出
│   ├── attack_lab/                评测与实验框架（约 3000 行）
│   │   ├── env.py                 统一模型环境（MLP / kNN / 最近质心族）
│   │   ├── harness.py             多攻击 × 多模型批量评测
│   │   ├── defense_eval.py        攻防对阵矩阵（RecoveryRate / EffectiveASR）
│   │   ├── train_only_sweep.py    仅用训练数据的方法横扫对比
│   │   ├── strict_split.py        严格三分实验（防信息泄漏）
│   │   ├── validate_submission.py 提交包端到端验证
│   │   ├── seed42_diag.py         脆弱源类诊断
│   │   ├── honesty.py             可见/不可见测试集对照
│   │   ├── methods/               攻击方法库（标签篡改/特征混合/陷阱/梯度/选择策略）
│   │   └── results/               全部实验报告（JSON，可复现）
│   └── submission/                平台「工程上传」用提交包
│       ├── attack-blank.py        工程上传入口（攻击）
│       ├── defense-blank.py       工程上传入口（防御）
│       ├── requirements.txt
│       ├── verify_attack.py       ★ 一键验证工具（合规性 + 评分档换算）
│       └── mnist_poison_submission.zip  打包好的上传包
├── docs/
│   ├── 01-方法说明.md             方法原理与实测数据详解
│   ├── 02-实测数据.md             完整实验记录汇总
│   └── 开源成果申报书-实习-已填写.docx
└── assets/
    └── 对抗06_攻击.png            攻击示意图
```

---

## 三、快速开始

```bash
git clone <this-repo>
cd mnist-poison-attack-defense

# 依赖：scikit-learn 可选（缺失时框架自动回退到内置纯 NumPy MLP）
pip install numpy scikit-learn

# 1）跑官方基准：攻击评测（干净基线 + 投毒后准确率 + ASR）
cd code
python test.py --stage attack

# 2）防御评测（含 RecoveryRate / EffectiveASR）
python test.py --stage defense

# 3）一键验证任意提交实现（含 6 项合规性检查与评分档换算）
cd submission
python verify_attack.py --attack attack.py --seed 43

# 4）复现实验：单源类脆弱性诊断（seed=42）
cd ..
python -m attack_lab.seed42_diag

# 5）攻防对阵矩阵（多攻击 × 多防御）
python -m attack_lab.defense_eval --models mlp --seeds 42,43,44

# 6）策略组合网格搜索
python -m attack_lab.harness --stage grid --model mlp
```

Windows 无 bash 时直接运行等效 python 命令即可。

---

## 四、接口

```python
# 攻击：PoisonRate ≤ 5%，输出规模与原训练集一致，修改样本数严格 ≤ 预算
def attack(images, labels, poison_rate=0.05, seed=42, num_classes=10) -> dict:
    """返回 {"images": 投毒图像, "labels": 投毒标签, "n_modified": 修改样本数}"""

# 防御：输入可能被投毒的训练集，返回清洗/增强后的数据集
def defend(images, labels, seed=42, max_shift=2, copies=1) -> dict:
    """返回 {"images", "labels", "n_augmented", "sample_weights"}"""
```

也兼容赛题 PDF 中的环境式接口 `attack(env, task)` / `defend(env, train)`，
见 `code/submission/methods/attack_core.py`。

---

## 五、合规性

`submission/verify_attack.py` 会自动检查以下 6 项：

- [x] 输出图像/标签规模与原训练集一致
- [x] 修改样本数 ≤ `floor(poison_rate × N)`
- [x] 标签取值合法（0–9）
- [x] 像素值域 [0,1]
- [x] 确实修改了样本（非空操作）
- [x] 接口可正常调用、纯 CPU 秒级完成

安全声明：本项目不含任何网络访问、文件系统副作用、子进程调用或后门代码；
仅使用 MNIST 公开数据集；依赖均为宽松开源许可证（NumPy / scikit-learn，BSD-3-Clause）。

---

## 六、引用

若本项目对你的研究或教学有帮助，可参考以下文献（本项目所基于的投毒攻防研究脉络）：

```bibtex
@inproceedings{biggio2012poisoning,
  title={Poisoning Attacks against Support Vector Machines},
  author={Biggio, Battista and Nelson, Blaine and Laskov, Pavel},
  booktitle={ICML}, year={2012}
}
@inproceedings{steinhardt2017certified,
  title={Certified Defenses for Data Poisoning Attacks},
  author={Steinhardt, Jacob and Koh, Pang Wei and Liang, Percy},
  booktitle={NeurIPS}, year={2017}
}
@inproceedings{tran2018spectral,
  title={Spectral Signatures in Backdoor Attacks},
  author={Tran, Brandon and Li, Jerry and Madry, Aleksander},
  booktitle={NeurIPS}, year={2018}
}
```

完整参考文献列表见 `docs/01-方法说明.md`。

## 七、许可证

[MIT](LICENSE)
