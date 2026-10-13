#!/usr/bin/env bash
# =============================================================================
# 攻击方终端脚本 —— 手写数字分类数据投毒基准（Baseline：随机标签翻转）
#
# 在新机器上以默认参数运行：
#     pip install -r requirements.txt
#     bash run_attack.sh
#
# 输出：Acc_clean / Acc_attack / PoisonRate / UntargetedASR / AttackSuccess
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"

# 指定解释器：默认 python，可用 PYTHON 环境变量覆盖
PY="${PYTHON:-python}"

echo ">>> [run_attack] 攻击方 Baseline 评测（随机标签翻转, poison_rate=0.05）"
"$PY" test.py --stage attack
