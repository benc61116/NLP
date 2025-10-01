# Phase 2: Production Experiments

## Overview
Phase 2 runs production experiments with optimal hyperparameters from Phase 1, using multiple seeds for statistical validity.

## Prerequisites
- Phase 1 must be completed
- Optimal hyperparameter files must exist in `analysis/`:
  - `squad_v2_full_finetune_optimal.yaml`
  - `squad_v2_lora_optimal.yaml`
  - `mrpc_full_finetune_optimal.yaml`
  - `mrpc_lora_optimal.yaml`
  - `sst2_full_finetune_optimal.yaml`
  - `sst2_lora_optimal.yaml`
  - `rte_full_finetune_optimal.yaml`
  - `rte_lora_optimal.yaml`

## VM Distribution

### VM1 (SQuAD v2)
- **Tasks**: squad_v2
- **Methods**: full_finetune, lora
- **Seeds**: 42, 1337, 2024
- **Total experiments**: 6 (1 task × 2 methods × 3 seeds)
- **Estimated runtime**: ~24-30 hours
  - Full FT: 6-8 hours/seed × 3 seeds = 18-24 hours
  - LoRA: 3-4 hours/seed × 3 seeds = 9-12 hours

### VM2 (Classification)
- **Tasks**: mrpc, sst2, rte
- **Methods**: full_finetune, lora
- **Seeds**: 42, 1337, 2024
- **Total experiments**: 18 (3 tasks × 2 methods × 3 seeds)
- **Estimated runtime**: ~20-24 hours
  - SST-2 (67K): Full FT 4-5 hrs/seed, LoRA 2-3 hrs/seed = 18-24 hours total
  - MRPC (3.7K): ~30 min/seed each method = 3 hours total
  - RTE (2.5K): ~20 min/seed each method = 2 hours total

## Execution

### On VM1:
```bash
cd /path/to/workspace/NLP
bash scripts/phase2/vm1.sh
```

### On VM2:
```bash
cd /path/to/workspace/NLP
bash scripts/phase2/vm2.sh
```

## Key Features

1. **No Overlap**: VM1 and VM2 handle different tasks - can run in parallel
2. **Optimal Hyperparameters**: Automatically loads from Phase 1 YAML files
3. **Multiple Seeds**: Runs each task/method with 3 seeds for statistical validity
4. **Speed Optimized**: Reduced eval sets during training (full dataset training preserved)
5. **Representation Extraction**: DISABLED during Phase 2 (extracted in Phase 3 for memory efficiency)
6. **W&B Logging**: All experiments logged to `NLP-Phase2` project
7. **Auto-cleanup**: GPU memory cleared between runs
8. **Error Handling**: Exits on first failure for quick debugging

## Outputs

- **Models**: `results/phase2/`
- **Logs**: `logs/phase2/vm1/` and `logs/phase2/vm2/`
- **W&B**: https://wandb.ai/galavny-tel-aviv-university/NLP-Phase2
- **Representations**: Automatically saved during training for Phase 3 drift analysis
