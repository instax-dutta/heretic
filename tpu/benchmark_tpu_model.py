"""Benchmark the TPU-abliterated model: refusals + KL divergence, matching heretic's scorers.

KL is computed exactly as heretic.scorers.kl_divergence does it: first-token
logit distributions on harmless_alpaca test[:100], F.kl_div(input=abliterated,
target=baseline, reduction="batchmean", log_target=True).
"""

import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, "src")

from transformers import AutoModelForCausalLM, AutoTokenizer

from heretic.scorers.keyword_rate import DEFAULT_KEYWORD_MARKERS

BASE = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
TPU_MODEL = "exported_model_colab_tpu"
GPU_MODEL = "saidutta69/Qwen2.5-Coder-0.5B-Instruct-heretic"

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"device: {DEVICE}", flush=True)


def first_token_logits(model_id: str, prompts: list[str], batch: int = 8) -> torch.Tensor:
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.float32
    ).to(DEVICE)
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(prompts), batch):
            chunk = prompts[i : i + batch]
            texts = tok.apply_chat_template(
                [
                    [
                        {"role": "system", "content": p.system},
                        {"role": "user", "content": p.user},
                    ]
                    for p in chunk
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            enc = tok(texts, return_tensors="pt", padding=True).to(DEVICE)
            logits = model(**enc).logits[:, -1, :]
            out.append(logits.float().cpu())
    del model
    torch.mps.empty_cache() if DEVICE == "mps" else None
    return torch.cat(out, dim=0)


def refusals(model_id: str, prompts: list[str], batch: int = 8) -> int:
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.float32
    ).to(DEVICE)
    model.eval()
    n = 0
    with torch.no_grad():
        for i in range(0, len(prompts), batch):
            chunk = prompts[i : i + batch]
            texts = tok.apply_chat_template(
                [
                    [
                        {"role": "system", "content": p.system},
                        {"role": "user", "content": p.user},
                    ]
                    for p in chunk
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            enc = tok(texts, return_tensors="pt", padding=True).to(DEVICE)
            gen = model.generate(
                **enc, max_new_tokens=20, do_sample=False, pad_token_id=tok.pad_token_id
            )
            new = gen[:, enc["input_ids"].shape[1] :]
            for r in tok.batch_decode(new, skip_special_tokens=True):
                low = r.lower()
                if any(k in low for k in DEFAULT_KEYWORD_MARKERS):
                    n += 1
    del model
    torch.mps.empty_cache() if DEVICE == "mps" else None
    return n


def kl(abl_logits: torch.Tensor, base_logits: torch.Tensor) -> float:
    abl_lp = F.log_softmax(abl_logits, dim=-1)
    base_lp = F.log_softmax(base_logits, dim=-1)
    return F.kl_div(
        abl_lp, base_lp, reduction="batchmean", log_target=True
    ).item()


def main() -> None:
    from heretic.utils import load_prompts as heretic_load
    from heretic.config import Settings

    s = Settings(model=BASE, max_response_length=20)

    from heretic.config import DatasetSpecification

    harmless = heretic_load(
        s,
        DatasetSpecification(
            dataset="mlabonne/harmless_alpaca", split="test[:100]", column="text"
        ),
    )
    harmful = heretic_load(
        s,
        DatasetSpecification(
            dataset="mlabonne/harmful_behaviors", split="test[:100]", column="text"
        ),
    )
    print(f"prompts: {len(harmless)} harmless, {len(harmful)} harmful", flush=True)

    print("computing base logits (harmless)...", flush=True)
    base_logits = first_token_logits(BASE, harmless)
    print("computing base refusals...", flush=True)
    base_ref = refusals(BASE, harmful)
    print(f"RESULT BASE refusals={base_ref}/{len(harmful)}", flush=True)

    for label, mid in [("TPU_ABLITERATED", TPU_MODEL), ("GPU_ABLITERATED", GPU_MODEL)]:
        try:
            lg = first_token_logits(mid, harmless)
            k = kl(lg, base_logits)
            r = refusals(mid, harmful)
            print(
                f"RESULT {label} refusals={r}/{len(harmful)} kl_divergence={k:.4f}",
                flush=True,
            )
        except Exception as e:
            print(f"RESULT {label} FAILED {type(e).__name__}: {str(e)[:200]}", flush=True)

    print("BENCH_DONE", flush=True)


if __name__ == "__main__":
    main()
