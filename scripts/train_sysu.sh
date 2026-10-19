#!/usr/bin/env bash
# Usage: bash scripts/train_sysu.sh SYSU_ROOT SWINV2_PRETRAIN OUTPUT_DIR [extra main.py args]
set -euo pipefail
if [[ $# -lt 3 ]]; then
  echo "Usage: $0 SYSU_ROOT SWINV2_PRETRAIN OUTPUT_DIR [extra main.py args]" >&2
  exit 2
fi
data_root=$(cd "$1" && pwd)
pretrain=$(cd "$(dirname "$2")" && pwd)/$(basename "$2")
mkdir -p "$3"
output_dir=$(cd "$3" && pwd)
shift 3
cd "$(dirname "${BASH_SOURCE[0]}")/.."

python main.py --dataset sysu --data-path "$data_root" --pretrain "$pretrain" --output-dir "$output_dir" \
  --width 144 --height 288 --stride 3 --dim 1024 --checkpoint-stages 2 \
  --batch-size 4 --num-pos 4 --epochs 120 --warmup 24 --lr-warmup-decay 0.01 --cos \
  --weight-decay 0.0001 --momentum 0.9 --nesterov --square 0 --amp --seed 0 \
  --workers 8 --test-batch 32 --pin --kl 0 --lr 0.005 \
  --two --augc --cre --re 0.5 --ga 0.2 --mode all --test-mode 1 2 \
  "$@"
