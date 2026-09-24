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
- Always auto-detection: no `--tpu-cores`/`--tpu-use-fsdp` flags; env vars PJRT_DEVICE=TPU, XLA_USE_BF16=1, XLA_USE_SPMD=1 (SPMD BEFORE any XLA init)
- Kaggle container TPU env does NOT reach SSH shells: source `/root/tpu_env.sh` (Kaggle sets `TPU_SKIP_MDS_QUERY=1`, `TPU_ACCELERATOR_TYPE=v5litepod-8`, `TPU_CHIPS_PER_HOST_BOUNDS=2,4,1`, worker id 0 in PID 1 environ; regenerate from `/proc/1/environ` on a fresh VM). Wrong/invented bounds (e.g. 2,2,2) cause "Mesh build failed, duplicate coordinate assignment" - copy PID 1's values, never guess
- Only ONE process may hold the vfio TPU devices: kill leftover heretic/ipykernel processes (pkill -9) before starting any run, else init dies silently with `open(/dev/vfio/N): Device or resource busy`
- SSH alias: `kaggle` (zrok tunnel to the VM). VM env: /root/heretic, /tmp/<model>.log. Tunnel drops are intermittent - write scripts to a file locally, scp them over, launch with nohup + log files (never rely on long inline heredocs over ssh)
- Setting up a VM: source via tar over ssh (repo is private), then install `pip install -e "/root/heretic[tpu]"` (torch stack preinstalled: torch 2.8.0+cpu, torch_xla 2.8.0 - never pip-install torch)
- Study resume: stored snapshot must have `n_additional_trials > 0` in the jsonl user_attrs (patch with escaped quotes `\"n_additional_trials\":N`) before `--checkpoint-action=continue`, or Settings validation fails
- Sync the live study (`checkpoints/*.jsonl`) to `resume/` every few minutes during long runs

## Work Guidance

- Before changing run recipes, confirm the equivalent notebook (`notebooks/`) and `src/heretic/AGENTS.md` contracts stay consistent
- When a VM dies: re-tunnel, push `resume/*.jsonl` back to `/root/heretic/checkpoints/`, relaunch the same run script — study continues
- Eval/upload of the final model: `eval100.py` for the x/100 score, then model card + `huggingface-cli` upload

## Verification

- `smoke_test.py` on a fresh VM is the entry check (device:0 cores count, SPMD mode). Validated 10/10 PASS on 2026-09-24 fresh VM
- 1-trial E2E (`heretic --n-trials=1 ... --export-strategy=merge`) exits 0 with merged safetensors in save-directory; kill+resume procedure verified 2026-09-24
- Completed run: grep the run log for `Optimization finished` and exported_model/ presence

## Child DOX Index

- No child AGENTS.md files