# Handoff - TPU port status and next session plan (speed focus)

Session date: 2026-08-04. Kaggle TPU v5e-8 session expired; the exported abliterated model was
pulled locally to `exported_model/` (gitignored, 988MB bf16 safetensors).

## What is DONE (committed)

Full E2E 1-trial abliteration pipeline runs and completes on TPU:
model load -> KL baseline (100 prompts) -> per-layer residual directions -> 1 trial
(abliterate + score) -> study completes -> trial restore -> abliterate -> CPU merge ->
`save_pretrained` -> "Model saved to /root/heretic/exported_model." -> clean exit.

Measured result (trial #1, random startup trial): refusal keywords 45/100 -> 26/100,
KL divergence vs baseline 0.0244. Direction index 16.97, attn.o_proj max_weight 1.09
@ layer 18.89 (min 0.96, dist 4.98), mlp.down_proj max_weight 0.83 @ 20.99 (min 0.24, dist 8.06).

Artifact: `exported_model/` locally. NOT yet validated (chat + refusal check) - first task of next session.

## The core TPU constraints (all measured, all committed as code+comments)

1. **Every XLA execution costs ~2.4GB (seq=256/bs=2) or ~1.3GB (seq=128/bs=1) HBM, never reclaimed.**
   Allocator evicts ~1.4-1.8GB under pressure but net is +1.1-1.9GB/exec; hard crash at 16.9GB
   (~6-12 execs). No cache-clear/release API exists in torch_xla 2.8. Consequence: all TPU
   batched methods run ALL prompts in ONE execution (`get_responses_batched`, `get_residuals_batched`,
   `get_logits_batched`), and outputs are CPU-offloaded per step (`offload_outputs_to_cpu or _is_tpu`).
   Prompt sets are kept small (good/bad = train[:2], 2+2).
2. **Laziness is our friend for abliteration**: the whole weight-mutation loop compiles to ~1 exec.
3. **padding="max_length" (128) is mandatory** - plain `padding=True` = one XLA recompile per unique
   batch shape. Compile dominates wall time (~10 min one-time per process, includes model load).
4. **SPMD only for explicit multi-core FSDP** (`--tpu-use-fsdp --tpu-cores>1`). Single-core runs must
   not SPMD (broken memory probing, null-data crashes).
5. **torch.cat on XLA lazy tensors crashes** torch_xla 2.8 - materialize to CPU first
   (already done in get_residuals_batched/get_logits_batched). `mark_step(wait=True)` everywhere.
6. **Merge/export must happen on CPU**: `self.model.to("cpu")` before extracting LoRA adapters
   (clone of XLA lazy tensors after abliteration = OSError EPERM, errno 1, when HBM near capacity).
7. **Interactive menus need CLI flags on non-tty runs**: `--trial-index --export-strategy --checkpoint-action
   --model-action --save-directory`. Resume reloads settings from the study snapshot, which wipes CLI
   values - main.py re-applies these 5 (4 flags + save-directory is a flag in run script; actually 4
   flags are re-applied in main.py ~line 407; save_directory comes via the script).

## Perf reference points

- Single 1-trial run: ~15 min wall (dominated by one-time load + XLA compile); the trial itself ~43s
  (journal timestamps), including 100-prompt keyword generation + 100-prompt KL in single execs.
- User's RTX 4060 8GB: ~18s/trial, 200 trials ~1h. TPU wins at scale (big-batch single execs,
  compile amortizes), NOT at 1-trial/1-core-small-batch configs.

## Next session priorities (SPEED)

1. Validate `exported_model/` locally (0.5B bf16 on CPU, chat + refusal keywords).
2. Spin up a fresh Kaggle TPU v5e-8 VM, re-run `tpu/bootstrap.sh`, scp `tpu/run_1trial.sh`,
   rerun to confirm reproducibility (full E2E incl. export).
3. Speed work (the real goal):
   - Amortize the ~10 min compile: run MORE trials per process. HBM allows ~9-10 execs/process;
     with single-exec batching, a 10-20 trial run should fit if each trial is ~2-3 execs
     (abliterate lazy + score). Budget the executor budget precisely (see TPU_PLAN.md phases).
   - 1-2B single-core (same pattern, bigger batch where HBM allows).
   - 7-8B VL on 8-core SPMD FSDP (needs HF token; smoke_test.py covers multi-core FSDP path).
4. `smoke_test.py` (PASS 10/10) covers single-core; add a speed/throughput benchmark for the
   executor budget decisions.

## Repro / debug scripts

`tpu/repro/`: repro_cat.py (torch.cat XLA crash), repro_kl.py (memory accumulation), repro_pipe.py
(pipeline), repro_wait.py (mark_step wait). Logs referenced in code comments; VM gone.

## Getting a VM again

Kaggle notebook/VM: TPU v5e-8. SSH alias `kaggle` (port-forwarded localhost:9191 per runbook in
commit d2af194). Never `pip install torch` from PyPI on the VM (must stay torch_xla 2.8-compatible);
use `tpu/bootstrap.sh`.
