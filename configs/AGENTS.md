# configs/AGENTS.md

## Purpose

- TPU/FSDP runtime configuration files consumed by heretic CLI runs: torch-xla runtime settings and FSDP sharding specs used on Kaggle TPU v5e-8 VMs

## Ownership

- `config.tpu.toml` - TPU runtime overrides for heretic (dtype, quantization, device map, scorer/optimizer tuning)
- `fsdp_tpu_v5e_8.json` - FSDP sharding plan for 8-core TPU v5e (layer-class sharding spec)

## Local Contracts

- Configs are referenced by `--config` CLI flags or by doc in `tpu/run_*.sh`
- TPU runs must stay bfloat16, no quantization, `device-map=auto` (see root AGENTS.md)
- Configs are the tuning surface only; auto-detection flags live in `src/heretic` (see `src/heretic/AGENTS.md`)

## Work Guidance

- When a VM run needs different tuning, prefer editing `config.tpu.toml` or a dedicated config file over JSON flags in the run script
- Keep FSDP specs consistent with `fsdp_utils.py` defaults; document any divergence in the file header

## Verification

- Config changes affecting reproducibility require `tests/` runs (see root AGENTS.md, `tests/AGENTS.md`)

## Child DOX Index

- No child AGENTS.md files