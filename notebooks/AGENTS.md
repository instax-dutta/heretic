# notebooks/AGENTS.md

## Purpose

- Colab/Kaggle TPU notebooks: environment checks, debugging, and run orchestration for the heretic TPU port

## Ownership

- `heretic_colab_tpu.ipynb` - Colab TPU autodetection/run notebook
- `heretic_tpu_kaggle.ipynb` - Kaggle TPU v5e-8 run notebook: env setup, pip installs, run launch
- `tpu_env_check.ipynb` - TPU environment sanity checks (torch_xla device, core count)
- `tpu_debug_info.ipynb` - TPU debug/diagnostics notebook

## Local Contracts

- Notebooks must use the auto-detection path (no hard-coded --tpu-cores / --tpu-use-fsdp flags)
- Env vars: PJRT_DEVICE=TPU, XLA_USE_BF16=1, and XLA_USE_SPMD=1 set before any XLA client init (see `src/heretic/AGENTS.md`)
- These notebooks mirror what `tpu/run_*.sh` does in a headless VM; keep them in sync

## Work Guidance

- Prefer the headless scripts under `tpu/` for real runs; notebooks are for interactive/env-check sessions
- When changing the run recipe in a notebook, update the equivalent `tpu/run_*.sh` and note the sync requirement

## Verification

- `tpu_env_check.ipynb` is the verification surface; any notebook change should still pass its checks on a fresh TPU VM

## Child DOX Index

- No child AGENTS.md files