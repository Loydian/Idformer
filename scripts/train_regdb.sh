#!/usr/bin/env bash
# Usage: bash scripts/train_regdb.sh REGDB_ROOT SWINV2_PRETRAIN OUTPUT_DIR TRIAL SYSU_CKPT|none [extra main.py args]
#   TRIAL      split 1-10
#   SYSU_CKPT  SYSU-MM01 checkpoint to fine-tune from, or "none"
set -euo pipefail
if [[ $# -lt 5 ]]; then
  echo "Usage: $0 REGDB_ROOT SWINV2_PRETRAIN OUTPUT_DIR TRIAL SYSU_CKPT|none [extra main.py args]" >&2
  exit 2
fi
data_root=$(cd "$1" && pwd)
pretrain=$(cd "$(dirname "$2")" && pwd)/$(basename "$2")
mkdir -p "$3"
output_dir=$(cd "$3" && pwd)
trial=$4
[[ $trial =~ ^([1-9]|10)$ ]] || { echo "TRIAL must be an integer from 1 to 10" >&2; exit 2; }
finetune_args=()
if [[ $5 != none ]]; then
  finetune_args=(--finetune "$(cd "$(dirname "$5")" && pwd)/$(basename "$5")")
fi
shift 5
cd "$(dirname "${BASH_SOURCE[0]}")/.."

python main.py --dataset regdb --data-path "$data_root" --pretrain "$pretrain" --output-dir "$output_dir" \
  --width 144 --height 288 --stride 3 --dim 1024 --checkpoint-stages 2 \
  --batch-size 4 --num-pos 4 --epochs 120 --warmup 24 --lr-warmup-decay 0.01 --cos \
  --weight-decay 0.0001 --momentum 0.9 --nesterov --square 0 --amp --seed 0 \
  --workers 8 --test-batch 32 --pin --kl 0 --lr 0.02 \
  --re 0.5 --trial "$trial" --test-mode 2 1 \
  ${finetune_args[@]+"${finetune_args[@]}"} "$@"
