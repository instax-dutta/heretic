#!/bin/bash
set -e
cd /root/heretic
export PJRT_DEVICE=TPU
export XLA_USE_BF16=1
export TOKENIZERS_PARALLELISM=false
heretic \
  --model=Qwen/Qwen2.5-Coder-0.5B-Instruct \
  --dtypes=bfloat16 --quantization=none --device-map=auto \
  --tpu-cores=8 --tpu-use-fsdp \
  --batch-size=2 --n-trials=1 --n-startup-trials=1 \
  --max-response-length=20 --row-normalization=full \
  --full-normalization-lora-rank=3 --orthogonalize-direction \
  --winsorization-quantile=1.0 \
  --good-prompts='{"dataset":"mlabonne/harmless_alpaca","split":"train[:10]","column":"text"}' \
  --bad-prompts='{"dataset":"mlabonne/harmful_behaviors","split":"train[:10]","column":"text"}'
