# Full Fine-tuning Validation Guide

## Step 3 Validation Instructions Implementation

This guide provides comprehensive validation tools and procedures for full fine-tuning experiments according to the Step 3 requirements.

## 🔍 Validation Tools Available

### 1. Quick Status Check
```bash
python check_experiment_status.py
```
**Purpose**: Immediate overview of experiment status
- ✅ Local file system status
- ✅ W&B connectivity
- ✅ Recent run counts
- ✅ Basic representation extraction check

### 2. Red Flags Monitoring
```bash
python monitor_red_flags.py
```
**Purpose**: Detect critical issues and warning signs
- 🚨 Training loss instability
- 🚨 Performance below expectations
- 🚨 Missing/corrupted representation files
- 🚨 Checkpoint loading failures
- 🚨 Memory/OOM issues

### 3. Comprehensive Validation
```bash
python validate_full_finetune.py
```
**Purpose**: Complete validation suite
- 📊 W&B dashboard metrics analysis
- 🎯 Performance against expected ranges
- 🧠 Representation extraction integrity
- 💾 Checkpoint loading verification
- 📈 Training stability analysis

## 📊 Training Progress Monitoring

### W&B Dashboard Checklist

**Navigate to**: https://wandb.ai/galavny-tel-aviv-university/NLP

**Expected Metrics to Monitor**:
```yaml
Training Curves:
  - train_loss: Should decrease consistently
  - eval_loss: Should decrease, may plateau
  - learning_rate: Should follow scheduler (cosine/linear)

Performance Metrics:
  - eval_accuracy: Task-specific (see ranges below)
  - eval_f1: For MRPC and SQuAD v2
  - exact_match: For SQuAD v2

System Metrics:
  - gradient_norm_total: Should be stable (1-100 range)
  - cpu_memory_rss_mb: Monitor for leaks
  - gpu_0_memory_allocated_mb: Should be consistent

Training Dynamics:
  - epoch: Progress indicator
  - step: Should increment steadily
  - examples_per_second: Training speed
```

### Expected Loss Curves
```python
# Healthy loss patterns:
train_loss: 3.0 → 1.5 → 0.8 → 0.5  # Steady decrease
eval_loss:  3.2 → 1.8 → 1.2 → 1.0  # Follows train, may plateau

# Red flags:
train_loss: 5.0 → 8.0 → 12.0       # Increasing (gradient explosion)
eval_loss:  2.0 → 1.0 → 3.0 → 5.0  # Oscillating (instability)
```

## 🎯 Performance Validation

### Expected Performance Ranges

| Task | Metric | Expected Range | Baseline | SOTA |
|------|--------|----------------|----------|------|
| **MRPC** | Accuracy | **85-90%** | ~70% | ~92% |
| **SST-2** | Accuracy | **90-93%** | ~82% | ~95% |
| **RTE** | Accuracy | **65-75%** | ~55% | ~78% |
| **SQuAD v2** | F1 Score | **75-85%** | ~60% | ~88% |

### Performance Validation Commands
```bash
# Check current performance
python -c "
import wandb
api = wandb.Api()
runs = api.runs('galavny-tel-aviv-university/NLP')
for run in runs[:5]:
    if 'full_finetune' in run.tags:
        task = run.config.get('task_name', 'unknown')
        acc = run.summary.get('eval_accuracy', 'N/A')
        f1 = run.summary.get('eval_f1', 'N/A')
        print(f'{task}: Accuracy={acc}, F1={f1}')
"
```

### Red Flag Performance Indicators
```python
# Critical issues (🚨):
mrpc_accuracy < 0.75    # >10% below minimum
sst2_accuracy < 0.80    # >10% below minimum  
rte_accuracy < 0.55     # >10% below minimum
squad_f1 < 0.65         # >10% below minimum

# Warning signs (⚠️):
eval_loss > train_loss * 2.0  # Severe overfitting
gradient_norm > 1000          # Gradient explosion
final_train_loss > 5.0        # Poor convergence
```

## 🧠 Representation Extraction Check

### Expected File Structure
```
results/
└── representations/
    ├── base_pretrained_mrpc/
    │   └── step_000000/
    │       ├── layer_0.pt
    │       ├── layer_1.pt
    │       ├── ...
    │       └── metadata.json
    ├── full_finetune_mrpc/
    │   ├── step_000100/
    │   ├── step_000200/
    │   └── step_000300/
    └── [similar for sst2, rte, squad_v2]
```

### Validation Commands
```bash
# Check representation extraction intervals
find results/ -name "step_*" | sort | head -10

# Verify file sizes (should not be empty)
find results/ -name "*.pt" -size 0

# Test loading representations
python -c "
import torch
from pathlib import Path
files = list(Path('results').glob('**/step_*/layer_0.pt'))
if files:
    tensor = torch.load(files[0])
    print(f'Shape: {tensor.shape}, Non-zero: {tensor.numel() > 0}')
else:
    print('No representation files found')
"
```

### Expected Extraction Pattern
- **Frequency**: Every 100 training steps
- **Coverage**: All transformer layers (0-23 for Llama-2-1.3B)
- **File sizes**: 10-100 MB per layer per step
- **Base representations**: Must exist for all tasks

## 💾 Checkpoint Validation

### Expected Checkpoint Structure
```
results/full_ft_sst2_seed42/
├── final_model/
│   ├── config.json
│   ├── pytorch_model.bin
│   └── tokenizer_config.json
├── checkpoint-500/
├── checkpoint-1000/
└── logs/
```

### Checkpoint Loading Test
```bash
# Test checkpoint loading
python -c "
from transformers import AutoModelForCausalLM
import torch
from pathlib import Path

checkpoints = list(Path('results').glob('**/final_model'))
if checkpoints:
    model = AutoModelForCausalLM.from_pretrained(str(checkpoints[0]))
    
    # Test inference
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    output = model(input_ids)
    
    print(f'✓ Checkpoint loaded successfully')
    print(f'Output shape: {output.logits.shape}')
    print(f'No NaN values: {not torch.isnan(output.logits).any()}')
else:
    print('❌ No checkpoints found')
"
```

### Model Consistency Test
```bash
# Test model loading consistency
python -c "
from transformers import AutoModelForCausalLM
import torch
from pathlib import Path

checkpoints = list(Path('results').glob('**/final_model'))[:2]
if len(checkpoints) >= 2:
    model1 = AutoModelForCausalLM.from_pretrained(str(checkpoints[0]))
    model2 = AutoModelForCausalLM.from_pretrained(str(checkpoints[1]))
    
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    out1 = model1(input_ids).logits
    out2 = model2(input_ids).logits
    
    diff = torch.abs(out1 - out2).max().item()
    print(f'Max output difference: {diff}')
    print(f'Models consistent: {diff < 1e-4}')
else:
    print('Need at least 2 checkpoints for consistency test')
"
```

## 🚨 Red Flags to Watch For

### Critical Issues (Immediate Action Required)

1. **Training Loss Issues**:
   ```bash
   # Check for unstable training
   python monitor_red_flags.py | grep "training loss"
   ```
   - ❌ Loss increasing over time
   - ❌ Loss oscillating wildly
   - ❌ Loss stuck at high values (>5.0)
   - ❌ NaN or infinite loss values

2. **Performance Issues**:
   ```bash
   # Check performance ranges
   python validate_full_finetune.py | grep "Performance"
   ```
   - ❌ MRPC < 75% accuracy
   - ❌ SST-2 < 80% accuracy  
   - ❌ RTE < 55% accuracy
   - ❌ SQuAD v2 < 65% F1

3. **File Corruption**:
   ```bash
   # Check for corrupted files
   find results/ -name "*.pt" -size 0
   python monitor_red_flags.py | grep "corrupted"
   ```
   - ❌ Empty representation files
   - ❌ Cannot load .pt files
   - ❌ Missing metadata.json files

4. **Memory/OOM Issues**:
   ```bash
   # Check W&B for failed runs
   python check_experiment_status.py | grep "failed"
   ```
   - ❌ Runs failing with OOM
   - ❌ GPU memory >22GB consistently
   - ❌ Gradient norms >1000

### Warning Signs (Monitor Closely)

1. **Training Dynamics**:
   - ⚠️ Eval loss >> train loss (overfitting)
   - ⚠️ Very slow convergence
   - ⚠️ Learning rate too high/low

2. **System Resources**:
   - ⚠️ High memory usage trends
   - ⚠️ Slow training speed
   - ⚠️ Inconsistent step times

3. **Data Quality**:
   - ⚠️ Missing extraction intervals
   - ⚠️ Small representation files
   - ⚠️ Irregular checkpoint saves

## 🔧 Troubleshooting Common Issues

### Issue: Training Loss Not Decreasing
```bash
# Check hyperparameters
python -c "
import wandb
api = wandb.Api()
run = api.runs('galavny-tel-aviv-university/NLP')[0]
print(f'LR: {run.config.get(\"learning_rate\")}')
print(f'Batch size: {run.config.get(\"per_device_train_batch_size\")}')
print(f'Grad accum: {run.config.get(\"gradient_accumulation_steps\")}')
"

# Solutions:
# - Reduce learning rate by 2-5x
# - Increase warmup steps
# - Check gradient clipping
```

### Issue: Performance Below Expected
```bash
# Check training completion
python -c "
import wandb
api = wandb.Api()
for run in api.runs('galavny-tel-aviv-university/NLP')[:3]:
    if 'full_finetune' in run.tags:
        print(f'{run.name}: {run.state}, Steps: {run.summary.get(\"step\", 0)}')
"

# Solutions:
# - Ensure training completed full epochs
# - Check for early stopping issues
# - Verify data preprocessing
```

### Issue: Missing Representations
```bash
# Check extraction settings
grep -r "extract_representations_every_steps" shared/config.yaml

# Solutions:
# - Verify callback is registered
# - Check disk space
# - Ensure hooks are properly attached
```

### Issue: Checkpoint Loading Fails
```bash
# Check checkpoint directory structure
ls -la results/*/final_model/

# Solutions:
# - Ensure all required files present
# - Check file permissions
# - Verify model architecture consistency
```

## 📈 Validation Success Criteria

### ✅ Validation Passes When:

1. **Training Metrics**:
   - Loss curves show steady decrease
   - No NaN/infinite values
   - Reasonable convergence (eval loss plateaus)

2. **Performance Metrics**:
   - All tasks within expected ranges
   - Consistent across multiple seeds
   - No severe overfitting

3. **Representation Extraction**:
   - Files present for all steps (every 100)
   - All layers extracted (layer_0 to layer_N)
   - Base model representations available
   - No corrupted files

4. **Checkpoint Integrity**:
   - Models load without errors
   - Inference produces valid outputs
   - Consistent outputs across reloads
   - All required files present

5. **System Stability**:
   - No OOM errors
   - Reasonable memory usage
   - Stable gradient norms
   - Complete training runs

## 🚀 Next Steps After Validation

### If Validation Passes:
```bash
# Proceed to LoRA experiments
bash scripts/phase2a/vm1.sh  # LoRA training

# Start drift analysis preparation
python experiments/drift_analysis.py --prepare-base-representations
```

### If Validation Fails:
```bash
# Run red flags monitoring for specific issues
python monitor_red_flags.py

# Fix identified issues and re-run
# Check specific validation components
python validate_full_finetune.py --check-performance
```

## 📊 Monitoring Dashboard

**W&B Dashboard**: https://wandb.ai/galavny-tel-aviv-university/NLP

**Key Views to Monitor**:
1. **Runs Table**: Overview of all experiments
2. **Charts**: Loss curves and performance metrics  
3. **System Metrics**: GPU/CPU usage and memory
4. **Hyperparameters**: Learning rates and batch sizes
5. **Model Metrics**: Gradients and parameter statistics

**Automated Alerts**: Set up W&B alerts for:
- Training loss increases
- Performance drops below thresholds
- System resource issues
- Run failures

This validation framework ensures full fine-tuning experiments meet all Step 3 requirements and are ready for subsequent LoRA comparison phases.
