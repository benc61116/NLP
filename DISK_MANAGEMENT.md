# Disk Space Management for NLP Project

## Overview

This project trains large language models that require significant disk space. To manage the limited 300GB disk on the VM, we've implemented a robust system with WandB backups and safe local cleanup.

---

## Architecture

### What Gets Saved Where:

| Data Type | Local Storage | WandB Backup | Can Delete Locally? |
|-----------|---------------|--------------|---------------------|
| **Model weights** | `results/*/final_model/` | ✅ As artifacts | ✅ YES (after upload) |
| **LoRA adapters** | `results/*/final_adapter/` | ✅ As artifacts | ✅ YES (after upload) |
| **Training metrics** | `wandb/` (cache) | ✅ In runs | ✅ YES (always) |
| **Logs** | `logs/` | ✅ In WandB runs | ✅ YES (always) |
| **Checkpoints** | `results/*/checkpoint-*` | ❌ No | ✅ YES (auto-cleaned) |
| **Base representations** | `base_representations/` | ✅ As artifacts | ✅ YES (after upload) |

---

## Automatic Safeguards

### During Training (Phase 2):

1. **Model Upload to WandB** (NEW):
   - Every trained model automatically uploaded as WandB artifact
   - Includes full fine-tune models (~2.5GB each)
   - Includes LoRA adapters (~5MB each)
   - Metadata: task, method, seed, hyperparameters

2. **Safe Cleanup**:
   - ✅ Auto-removes intermediate checkpoints
   - ✅ Warns if disk usage > 80%
   - ❌ **NEVER** touches `final_model/` or `final_adapter/`

3. **What Gets Cleaned Automatically**:
   - `results/*/checkpoint-*` (intermediate training checkpoints)
   - Old WandB run cache (`wandb/`)

---

## Manual Cleanup Commands

### After Phase 2 Completes:

```bash
# 1. Verify models are uploaded to WandB
python scripts/download_wandb_models.py --list --entity galavny-tel-aviv-university

# 2. Safe cleanup - remove local files that are backed up
rm -rf wandb/                    # WandB cache (safe)
rm -rf logs/                     # Log files (safe)
rm -rf results/phase0/           # Baseline models (safe)
rm -rf results/phase1/           # Optuna trials (safe)

# 3. Optional: Remove Phase 2 models after upload confirmation
# (Only do this if WandB upload succeeded and you need space)
find results/ -name "final_model" -type d -exec rm -rf {} +
find results/ -name "final_adapter" -type d -exec rm -rf {} +

# 4. Check disk space
df -h /
du -sh results/
```

### Aggressive Cleanup (If Disk Full):

```bash
# Remove everything except base_representations
# (Can re-download from WandB for Phase 3)
rm -rf results/full_finetune_*
rm -rf results/lora_finetune_*
rm -rf results/phase1/
rm -rf results/phase0/
rm -rf wandb/
rm -rf logs/

# Disk should now be ~50GB used
```

---

## Recovery: Download Models from WandB

### For Phase 3 Representation Extraction:

```bash
# List available models
python scripts/download_wandb_models.py \
    --list \
    --entity galavny-tel-aviv-university

# Download all Phase 2 models
python scripts/download_wandb_models.py \
    --all \
    --entity galavny-tel-aviv-university \
    --output-dir results/downloaded_models

# Download specific model
python scripts/download_wandb_models.py \
    --entity galavny-tel-aviv-university \
    --artifact lora_adapter_mrpc_seed42
```

### Automatic Download:

Phase 3 will **automatically download from WandB** if models aren't found locally:

```bash
# Phase 3 will auto-download missing models
bash scripts/phase3/vm2.sh
```

---

## Disk Space Estimates

### Phase 2 Training:

| Component | Size per Model | Total (18 models) |
|-----------|----------------|-------------------|
| Full FT models | ~2.5 GB | ~22.5 GB |
| LoRA adapters | ~5 MB | ~90 MB |
| Checkpoints (temp) | ~10 GB | Auto-cleaned |
| WandB cache | ~20 GB | Can delete |
| Logs | ~100 MB | Can delete |

**Total kept:** ~23 GB (models only)  
**Peak usage:** ~60-80 GB (during training)

### After Cleanup:

- With models local: ~25 GB
- Without models (WandB only): ~5 GB
- Base representations: ~48 GB (needed for Phase 3)

---

## Best Practices

### 1. Monitor Disk Usage:

```bash
# Check during training
watch -n 60 'df -h / && echo "" && du -sh results/'
```

### 2. After Each Phase:

```bash
# Phase 0 complete
rm -rf results/phase0/

# Phase 1 complete  
rm -rf results/phase1/
rm -rf wandb/

# Phase 2 complete (after WandB upload verified)
python scripts/download_wandb_models.py --list --entity galavny-tel-aviv-university
# If all 18 models show up, safe to clean:
rm -rf wandb/
rm -rf logs/
```

### 3. Emergency Disk Full:

```bash
# Quick cleanup
rm -rf wandb/ logs/
find results/ -name "checkpoint-*" -exec rm -rf {} +

# If still full, remove old experiments
rm -rf results/phase0/ results/phase1/

# Last resort: remove models (can re-download)
rm -rf results/full_finetune_*
rm -rf results/lora_finetune_*
```

---

## WandB Storage

All models are stored in:
- **Project**: `NLP-Phase2`
- **Entity**: `galavny-tel-aviv-university`
- **Type**: `model` artifacts

### Artifact Naming:

- Full fine-tune: `full_finetune_model_{task}_seed{seed}`
- LoRA: `lora_adapter_{task}_seed{seed}`

### Example:

- `full_finetune_model_mrpc_seed42`
- `lora_adapter_sst2_seed1337`

---

## Troubleshooting

### "No space left on device"

```bash
# 1. Check usage
df -h /
du -sh results/* | sort -h

# 2. Quick cleanup
rm -rf wandb/ logs/

# 3. Remove old experiments
rm -rf results/phase0/ results/phase1/

# 4. If still stuck
python scripts/download_wandb_models.py --list --entity galavny-tel-aviv-university
# Then delete local models
```

### "Model not found" in Phase 3

```bash
# Phase 3 will auto-download, but you can manually trigger:
python scripts/download_wandb_models.py --all --entity galavny-tel-aviv-university
```

### "WandB upload failed"

Check logs for specific model:
```bash
grep "WandB" logs/phase2/vm2/*.log
```

If upload failed, model is only local - **DO NOT DELETE**

---

## Summary

✅ **Safe to delete locally**:
- `wandb/` - WandB cache
- `logs/` - Log files  
- `results/phase0/`, `results/phase1/` - Old experiments
- `results/*/checkpoint-*` - Intermediate checkpoints
- `results/*/final_model/`, `results/*/final_adapter/` - **ONLY after verifying WandB upload**

❌ **Never delete**:
- `base_representations/` - Needed for Phase 3, not easy to regenerate
- Models **before** verifying WandB upload

🔄 **Can re-download from WandB**:
- All Phase 2 trained models
- Phase 3 will auto-download if needed

