#!/usr/bin/env bash
# =============================================================================
# 防守方终端脚本 —— 手写数字分类数据投毒基准（Baseline：数据增强）
#
# 在新机器上以默认参数运行：
#     pip install -r requirements.txt
#     bash run_defense.sh
#
# 输出：Acc_clean / Acc_attack / Acc_defense / RecoveryRate / EffectiveASR
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"

# 指定解释器：默认 python，可用 PYTHON 环境变量覆盖
PY="${PYTHON:-python}"

echo ">>> [run_defense] 防守方 Baseline 评测（数据增强）"
"$PY" test.py --stage defense
