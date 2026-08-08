#!/bin/bash
# 7-8B multi-core FSDP run: the big-model TPU test.
# 8 cores, SPMD FSDP, small trial count to validate the path end-to-end.
set -e
cd /root/heretic
export PJRT_DEVICE=TPU
export XLA_USE_BF16=1
export TOKENIZERS_PARALLELISM=false
N_TRIALS="${N_TRIALS:-2}"
SAVE_DIR="${SAVE_DIR:-/root/heretic/exported_model}"
echo "=== MTRIAL7B start $(date +%H:%M:%S) trials=$N_TRIALS ==="
heretic \
  --model=Qwen/Qwen2.5-Coder-7B-Instruct \
  --dtypes=bfloat16 --quantization=none --device-map=auto \
  --tpu-cores=8 --tpu-use-fsdp \
  --batch-size=2 --n-trials="$N_TRIALS" --n-startup-trials=1 --seed=42 \
  --trial-index=0 --export-strategy=merge --checkpoint-action=continue --model-action=save \
  --save-directory="$SAVE_DIR" \
  --max-response-length=20 --row-normalization=full \
  --full-normalization-lora-rank=3 --orthogonalize-direction \
  --winsorization-quantile=1.0 \
  --good-prompts='{"dataset":"mlabonne/harmless_alpaca","split":"train[:2]","column":"text"}' \
  --bad-prompts='{"dataset":"mlabonne/harmful_behaviors","split":"train[:2]","column":"text"}'
echo "=== MTRIAL7B done $(date +%H:%M:%S) ==="
