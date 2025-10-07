# Phase 2: Production Experiments

## Overview
Phase 2 runs production experiments with optimal hyperparameters from Phase 1, using multiple seeds for statistical validity.

## Prerequisites
- Phase 1 must be completed
- Optimal hyperparameter files must exist in `analysis/`:
  - `mrpc_full_finetune_optimal.yaml`
  - `mrpc_lora_optimal.yaml`
  - `sst2_full_finetune_optimal.yaml`
  - `sst2_lora_optimal.yaml`
  - `rte_full_finetune_optimal.yaml`
  - `rte_lora_optimal.yaml`

## Execution

### Classification Tasks (All 3 tasks: MRPC, SST-2, RTE)

**Full Fine-Tuning + LoRA:**
```bash
cd /path/to/workspace/NLP
bash scripts/phase2/vm2.sh
```

**LoRA Only (Re-run):**
```bash
cd /path/to/workspace/NLP
bash scripts/phase2/rerun_lora_only.sh
```

## Experiment Configuration

- **Tasks**: mrpc, sst2, rte
- **Methods**: full_finetune, lora
- **Seeds**: 42, 1337, 2024
- **Total experiments**: 18 (3 tasks × 2 methods × 3 seeds)

### Estimated Runtime (Single 24GB GPU)
- SST-2 (67K samples): Full FT 4-5 hrs/seed, LoRA 2-3 hrs/seed = 18-24 hours total
- MRPC (3.7K samples): ~30 min/seed each method = 3 hours total
- RTE (2.5K samples): ~20 min/seed each method = 2 hours total
- **Total**: ~20-30 hours for all experiments

## Key Features

1. **Optimal Hyperparameters**: Automatically loads from Phase 1 YAML files
2. **Multiple Seeds**: Runs each task/method with 3 seeds for statistical validity
3. **Speed Optimized**: Reduced eval sets during training (full dataset training preserved)
4. **Representation Extraction**: DISABLED during Phase 2 (extracted separately in Phase 3 for memory efficiency)
5. **W&B Logging**: All experiments logged to WandB projects
   - Full FT: `NLP-Phase2`
   - LoRA: `NLP-Phase2-LoRA-Rerun`
6. **Auto-cleanup**: GPU memory cleared between runs
7. **Error Handling**: Exits on first failure for quick debugging

## Outputs

- **Models**: 
  - Full FT: `results/full_finetune_model_{task}_seed{seed}/`
  - LoRA: `results/lora_adapter_{task}_seed{seed}/`
- **Logs**: `logs/phase2/`
- **W&B**: https://wandb.ai/galavny-tel-aviv-university/
  - Full FT: `/NLP-Phase2`
  - LoRA: `/NLP-Phase2-LoRA-Rerun`

## Notes

- All 18 final models (9 full FT + 9 LoRA) are available in WandB artifacts
- Use `scripts/download_wandb_models.py` to download trained models
- Representations are extracted separately in Phase 3 to manage memory efficiently
