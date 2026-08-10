# tpu/AGENTS.md

## Purpose

- TPU port operations: run scripts, smoke tests, handoff, study checkpoint sync, VM bootstrap — the operational center for Kaggle TPU v5e-8 runs

## Ownership

- `run_*.sh` - per-model run recipes (`run_vl3b.sh` = Qwen2.5-VL-3B-Instruct 200-trial abliteration, `run_1trial.sh` / `run_mtrial.sh` quick checks, `run_7b.sh`, `run_1p5b.sh`, `run_qwen35_4b.sh`)
- `smoke_test.py` - staged TPU env validation (device, cores, SPMD, multichip)
- `HANDOFF.md` - TPU port status: SPMD/FSDP segfault root cause and fix, session wins
- `bootstrap.sh` - fresh-VM setup: clone, deps (torch+cu128, torch_xla), smoke test
- `eval100.py` - one-shot x/100 harmful_behaviors eval against exported model vs base
- `repro/` - standalone repro scripts (spmd_matmul.py, multidevice_ar.py, etc.) used to isolate TPU bugs
- `resume/` - off-VM Optuna journal checkpoints (crash insurance; synced from VM every few minutes during runs)

## Local Contracts

- Run flags live ONLY in `tpu/run_*.sh` — the sole source of truth; no inline flags for TPU runs
- Always auto-detection: env vars PJRT_DEVICE=TPU, XLA_USE_BF16=1, XLA_USE_SPMD=1 (SPMD BEFORE any XLA init)
- SSH alias: `kaggle` (zrok tunnel to the VM). VM env: /root/heretic, /tmp/<model>.log
- Setting up a VM: source via tar over ssh (repo is private), then `bash /root/setup_vm.sh` (torch 2.8.0+cu128, torch_xla 2.8.x, `pip install -e /root/heretic[dev]`)
- Study resume: patch stored `n_additional_trials > 0` in the jsonl user_attrs before `--checkpoint-action=continue`, or validation fails
- Sync the live study (`/root/heretic/checkpoints/*.jsonl`) to `resume/` every few minutes during long runs

## Work Guidance

- Before changing run recipes, confirm the equivalent notebook (`notebooks/`) and `src/heretic/AGENTS.md` contracts stay consistent
- When a VM dies: re-tunnel, push `resume/*.jsonl` back to `/root/heretic/checkpoints/`, relaunch the same run script — study continues
- Eval/upload of the final model: `eval100.py` for the x/100 score, then model card + `huggingface-cli` upload

## Verification

- `smoke_test.py` on a fresh VM is the entry check (device:0 cores count, SPMD mode)
- Completed run: grep the run log for `Optimization finished` and `exported_model/` presence

## Child DOX Index

- No child AGENTS.md files