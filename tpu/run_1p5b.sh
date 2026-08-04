#!/bin/bash
# 1.5B single-core trial run: proves the TPU port scales past 0.5B.
# Same shape as run_mtrial.sh but with the 1.5B Coder model.
set -e
cd /root/heretic
export PJRT_DEVICE=TPU
export XLA_USE_BF16=1
export TOKENIZERS_PARALLELISM=false
N_TRIALS="${N_TRIALS:-3}"
SAVE_DIR="${SAVE_DIR:-/root/heretic/exported_model}"
echo "=== MTRIAL15 start $(date +%H:%M:%S) trials=$N_TRIALS ==="
heretic \
  --model=Qwen/Qwen2.5-Coder-1.5B-Instruct \
  --dtypes=bfloat16 --quantization=none --device-map=auto \
  --tpu-cores=1 --no-tpu-use-fsdp \
  --batch-size=2 --n-trials="$N_TRIALS" --n-startup-trials=2 --seed=42 \
  --trial-index=0 --export-strategy=merge --checkpoint-action=continue --model-action=save \
  --save-directory="$SAVE_DIR" \
  --max-response-length=20 --row-normalization=full \
  --full-normalization-lora-rank=3 --orthogonalize-direction \
  --winsorization-quantile=1.0 \
  --good-prompts='{"dataset":"mlabonne/harmless_alpaca","split":"train[:2]","column":"text"}' \
  --bad-prompts='{"dataset":"mlabonne/harmful_behaviors","split":"train[:2]","column":"text"}'
echo "=== MTRIAL15 done $(date +%H:%M:%S) ==="
