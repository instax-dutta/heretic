#!/bin/bash
# The big one: 200-trial TPU abliteration of Qwen/Qwen3.5-4B.
# NO --tpu-cores / --tpu-use-fsdp flags on purpose: auto-detection must
# pick up 8 cores + FSDP by itself (XLA_USE_SPMD=1 before client init).
set -e
cd /root/heretic
export PJRT_DEVICE=TPU
export XLA_USE_BF16=1
export TOKENIZERS_PARALLELISM=false
N_TRIALS="${N_TRIALS:-200}"
SAVE_DIR="${SAVE_DIR:-/root/heretic/exported_model}"
echo "=== QWEN35-4B start $(date +%H:%M:%S) trials=$N_TRIALS ==="
heretic \
  --model=Qwen/Qwen3.5-4B \
  --dtypes=bfloat16 --quantization=none --device-map=auto \
  --batch-size=2 --n-trials="$N_TRIALS" --n-startup-trials=1 --seed=42 \
  --trial-index=0 --export-strategy=merge --checkpoint-action=continue --model-action=save \
  --save-directory="$SAVE_DIR" \
  --max-response-length=20 --row-normalization=full \
  --full-normalization-lora-rank=3 --orthogonalize-direction \
  --winsorization-quantile=1.0 \
  --good-prompts='{"dataset":"mlabonne/harmless_alpaca","split":"train[:2]","column":"text"}' \
  --bad-prompts='{"dataset":"mlabonne/harmful_behaviors","split":"train[:2]","column":"text"}'
echo "=== QWEN35-4B done $(date +%H:%M:%S) ==="