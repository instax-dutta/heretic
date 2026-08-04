#!/bin/bash
# Multi-trial speed benchmark on TPU: amortize the one-time XLA compile
# across N trials in a single process. Baseline: RTX 4060 8GB does
# ~18s/trial (200 trials in ~1h). TPU should be ~10-60s/trial with
# single-execution batching; compile (~7 min incl. model load) is fixed.
set -e
cd /root/heretic
export PJRT_DEVICE=TPU
export XLA_USE_BF16=1
export TOKENIZERS_PARALLELISM=false
N_TRIALS="${N_TRIALS:-10}"
SAVE_DIR="${SAVE_DIR:-/root/heretic/exported_model}"
echo "=== MTRIAL start $(date +%H:%M:%S) trials=$N_TRIALS ==="
heretic \
  --model=Qwen/Qwen2.5-Coder-0.5B-Instruct \
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
echo "=== MTRIAL done $(date +%H:%M:%S) ==="
