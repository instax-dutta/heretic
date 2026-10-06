# TPU-abliterated Qwen2.5-Coder-0.5B-Instruct

First heretic abliteration executed entirely on a TPU (Google Colab TPU v5e-1,
torch 2.9.0 / torch_xla 2.9.0, bf16).

## Recipe

    --model=Qwen/Qwen2.5-Coder-0.5B-Instruct
    --dtypes=bfloat16 --quantization=none --device-map=auto
    --batch-size=2 --n-trials=20 --n-startup-trials=5 --seed=42
    --max-response-length=20 --row-normalization=full
    --full-normalization-lora-rank=3 --orthogonalize-direction
    --winsorization-quantile=1.0 --export-strategy=merge
    good: mlabonne/harmless_alpaca train[:100]
    bad:  mlabonne/harmful_behaviors train[:100]

## Result (harmful_behaviors test[:100], heretic KeywordRate scorer)

| Model                                   | Refusals |
| --------------------------------------- | -------- |
| Base Qwen2.5-Coder-0.5B-Instruct        | 30/100   |
| GPU-abliterated (saidutta69/...-heretic) | 5/100    |
| TPU-abliterated (this model)            | 5/100    |

Identical refusal rate to the GPU-abliterated model of the same base.

## Files

- model.safetensors - 290 tensors, bf16, sha256
  b9ba52076867ed00d7749009f3905d172e14fd0868a17fed08030259c5c15afd
- config.json, generation_config.json, tokenizer.json, tokenizer_config.json,
  chat_template.jinja
