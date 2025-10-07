# Phase 1: Hyperparameter Optimization

## Overview
Phase 1 uses Bayesian optimization (Optuna with TPE sampler) to find optimal hyperparameters for each task and method combination.

## Purpose
Find optimal hyperparameters efficiently using:
- **Optuna**: Bayesian optimization framework
- **TPE Sampler**: Tree-structured Parzen Estimator
- **Reduced Dataset**: Smaller samples for fast iteration
- **15 Trials per task/method**: Methodologically sound exploration

## Prerequisites
- Phase 0 completed (baselines established)
- Config file: `shared/config.yaml`
- Empty `analysis/` directory for optimal configs

## Execution

```bash
cd scripts/phase1
bash vm1.sh  # Runs Optuna optimization for all tasks
```

This will create 6 optimal config files in `analysis/`:
- `mrpc_full_finetune_optimal.yaml` / `mrpc_lora_optimal.yaml`
- `sst2_full_finetune_optimal.yaml` / `sst2_lora_optimal.yaml`
- `rte_full_finetune_optimal.yaml` / `rte_lora_optimal.yaml`

## Hyperparameters Optimized

### Full Fine-tuning
- Learning rate (1e-6 to 1e-4)
- Batch size (1, 2, 4)
- Warmup ratio (0.0 to 0.3)
- Weight decay (0.0 to 0.3)
- Number of epochs (2 to 6)

### LoRA
- All above, plus:
- LoRA rank (4, 8, 16, 32, 64)
- LoRA alpha (8, 16, 32, 64, 128)
- LoRA dropout (0.0, 0.05, 0.1)

## Outputs

### Optimal Configs (`analysis/*.yaml`)
After completion, optimal hyperparameters saved for each task/method combination (6 files total).

### Logs
- `logs/phase1_optuna/`
- Optuna study results
- Trial-by-trial performance

## Next Steps
- Verify all 6 optimal config files exist in `analysis/`
- Proceed to Phase 2 (Production Training with optimal configs)
