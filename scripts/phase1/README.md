# Phase 1: Hyperparameter Optimization

## Overview
Phase 1 uses Bayesian optimization (Optuna with TPE sampler) to find optimal hyperparameters for each task and method combination.

## Purpose
Find optimal hyperparameters efficiently using:
- **Optuna**: Bayesian optimization framework
- **TPE Sampler**: Tree-structured Parzen Estimator (state-of-the-art)
- **Reduced Dataset**: 500 training samples for fast iteration
- **10 Trials**: Methodologically sound minimum (Bergstra & Bengio, 2012)

## Prerequisites
- Phase 0 must be completed (baselines established)
- Config file: `shared/config.yaml`
- Empty `analysis/` directory for optimal configs

## Methodology

### Why 10 Trials?
The choice of 10 trials per task is methodologically sound:
1. **TPE Algorithm**: Bergstra & Bengio (2012) recommend minimum 10 trials
2. **Research Standards**: Published papers use 5-20 trials (LoRA paper: 6 rank values)
3. **Two-Phase Design**: Phase 1 finds good hyperparameters; Phase 2 validates with multiple seeds
4. **Efficiency**: TPE captures ~70-80% of optimal performance in 10 trials

**Reference**: Bergstra, J., & Bengio, Y. (2012). Random search for hyper-parameter optimization. Journal of machine learning research, 13(2).

### Speed Optimizations
- **Tiered Dataset Sizes**:
  - SQuAD v2: 3,000 samples (2.3% of 130K) - research-grade coverage
  - SST-2: 3,000 samples (4.5% of 67K) - research-grade coverage
  - MRPC/RTE: 500 samples (13-20% coverage) - already sufficient
- **Proportional Validation**: 10% of training samples
- **Adaptive Epochs**: 2-6 epochs (Optuna decides)
- **Goal**: Representative hyperparameter search meeting 2-10% standard

## VM Distribution

### VM1 (SQuAD v2)
- **Tasks**: squad_v2
- **Methods**: full_finetune, lora
- **Trials**: 10 per method
- **Total trials**: 20 (1 task × 2 methods × 10 trials)
- **Estimated runtime**: ~2 hours

### VM2 (Classification)
- **Tasks**: mrpc, sst2, rte
- **Methods**: full_finetune, lora
- **Trials**: 10 per task/method
- **Total trials**: 60 (3 tasks × 2 methods × 10 trials)
- **Estimated runtime**: ~3 hours

**Note**: SQuAD v2 is ~3x heavier computationally, so VM1 (1 task) ≈ VM2 (3 tasks) in runtime.

## Execution

### On VM1:
```bash
cd /path/to/workspace/NLP
bash scripts/phase1/vm1.sh
```

### On VM2:
```bash
cd /path/to/workspace/NLP
bash scripts/phase1/vm2.sh
```

## Key Features

1. **No Overlap**: VM1 and VM2 handle different tasks - can run in parallel
2. **Bayesian Optimization**: TPE sampler efficiently explores hyperparameter space
3. **Optuna Integration**: State-of-the-art hyperparameter optimization
4. **Fast Iterations**: Reduced dataset enables 10 trials in 2-3 hours
5. **Optimal Configs Saved**: Results saved to `analysis/*.yaml` for Phase 2
6. **W&B Logging**: All trials logged to `NLP-Phase1-Optuna` project

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

### Optimal Configs (analysis/*.yaml)
After completion, optimal hyperparameters saved for each task/method:
- `analysis/squad_v2_full_finetune_optimal.yaml`
- `analysis/squad_v2_lora_optimal.yaml`
- `analysis/mrpc_full_finetune_optimal.yaml`
- `analysis/mrpc_lora_optimal.yaml`
- `analysis/sst2_full_finetune_optimal.yaml`
- `analysis/sst2_lora_optimal.yaml`
- `analysis/rte_full_finetune_optimal.yaml`
- `analysis/rte_lora_optimal.yaml`

### Logs
- Detailed logs: `logs/phase1_optuna/vm1/` and `logs/phase1_optuna/vm2/`
- Optuna study results
- Trial-by-trial performance

### W&B
- Project: `NLP-Phase1-Optuna`
- All trials tracked with hyperparameters and metrics
- Visualization of hyperparameter importance

## Success Criteria
- ✅ Clear optimal hyperparameters identified
- ✅ Optuna TPE finds good configurations within 10 trials
- ✅ Results saved to `analysis/*.yaml` for Phase 2 usage
- ✅ Performance gaps between best/worst trials >5%

## Understanding the Results

### What to Look For
1. **Convergence**: Later trials should perform better than early random trials
2. **Clear Winners**: Best trial should be notably better than worst
3. **Hyperparameter Importance**: Optuna shows which hyperparameters matter most

### Validation
Check that optimal configs saved to `analysis/` contain:
- `best_hyperparameters`: Dict of optimal values
- `best_value`: Best objective value (e.g., F1 score)
- `n_trials`: Should be 10
- `study_name`: Task and method identifier

## Next Steps
After Phase 1 completion:
- Verify all 8 optimal config files exist in `analysis/`
- Review W&B to ensure trials show convergence
- Proceed to Phase 2 (Production Experiments with optimal configs)

## Common Issues

### OOM Errors
- Batch size automatically constrained to [1] for full fine-tuning
- If still failing, reduce `max_samples_train` in `experiments/optuna_optimization.py`

### Poor Convergence
- 10 trials is minimum; if results unclear, can increase to 20 trials
- Check that dataset isn't too small (500 samples should be sufficient)

### Missing Optimal Configs
- Check logs for errors during optimization
- Optuna automatically saves best config at end of study
- If missing, may need to rerun that specific task/method
