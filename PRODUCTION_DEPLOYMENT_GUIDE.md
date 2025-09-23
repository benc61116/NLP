# Production Deployment Guide - Phase 1 Parallel Execution

## ✅ READY FOR DEPLOYMENT

The codebase is production-ready for 3-VM parallel execution.

## 🚀 Phase 1 Deployment Steps

### Step 1: Pre-Flight Check (Run on VM1)
```bash
cd /home/benc6116/workspace/NLP

# 1. Verify all dependencies
pip install -r requirements.txt

# 2. Check HuggingFace authentication
python -c "
from transformers import AutoTokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-1.3b-hf')
    print('✅ HuggingFace authentication successful')
except Exception as e:
    print(f'❌ HuggingFace auth failed: {e}')
    print('Run: huggingface-cli login')
"

# 3. Download datasets (run once, then copy to other VMs)
python scripts/download_datasets.py

# 4. Test W&B connection
python -c "
import wandb
import os
os.environ['WANDB_PROJECT'] = 'NLP-Phase1-Training'
os.environ['WANDB_ENTITY'] = 'galavny-tel-aviv-university'
wandb.init(project='NLP-Phase1-Training', entity='galavny-tel-aviv-university')
print('✅ W&B connection successful')
wandb.finish()
"
```

### Step 2: Copy Codebase to All VMs
```bash
# Copy the entire workspace to VM2 and VM3
scp -r /home/benc6116/workspace/NLP user@vm2:/home/benc6116/workspace/
scp -r /home/benc6116/workspace/NLP user@vm3:/home/benc6116/workspace/
```

### Step 3: Launch Phase 1 (Parallel Execution)

**VM1** (SQuAD v2 Full FT + MRPC Full FT + MRPC LoRA):
```bash
cd /home/benc6116/workspace/NLP
tmux new-session -d -s phase1 './scripts/phase1/vm1.sh'
tmux attach -t phase1  # Optional: monitor in real-time
```

**VM2** (SQuAD v2 LoRA + SST-2 Full FT + SST-2 LoRA):
```bash
cd /home/benc6116/workspace/NLP
tmux new-session -d -s phase1 './scripts/phase1/vm2.sh'
tmux attach -t phase1  # Optional: monitor in real-time
```

**VM3** (RTE Full FT + RTE LoRA + All Baselines):
```bash
cd /home/benc6116/workspace/NLP
tmux new-session -d -s phase1 './scripts/phase1/vm3.sh'
tmux attach -t phase1  # Optional: monitor in real-time
```

## 📊 Monitoring & Expected Results

### W&B Dashboard Monitoring
**Primary Dashboard**: https://wandb.ai/galavny-tel-aviv-university/NLP-Phase1-Training

#### Expected Runs (Total: ~21 runs)

**VM1 Runs** (6 runs):
- `squad_v2_full_seed42`, `squad_v2_full_seed1337`, `squad_v2_full_seed2024`
- `mrpc_full_seed42`, `mrpc_full_seed1337`, `mrpc_full_seed2024`
- `mrpc_lora_seed42`, `mrpc_lora_seed1337`, `mrpc_lora_seed2024`

**VM2 Runs** (9 runs):
- `squad_v2_lora_seed42`, `squad_v2_lora_seed1337`, `squad_v2_lora_seed2024`
- `sst2_full_seed42`, `sst2_full_seed1337`, `sst2_full_seed2024`
- `sst2_lora_seed42`, `sst2_lora_seed1337`, `sst2_lora_seed2024`

**VM3 Runs** (6 runs + baselines):
- `rte_full_seed42`, `rte_full_seed1337`, `rte_full_seed2024`
- `rte_lora_seed42`, `rte_lora_seed1337`, `rte_lora_seed2024`
- `baseline_mrpc`, `baseline_squad_v2`, `baseline_sst2`, `baseline_rte`

### Key Metrics to Monitor

#### 1. Training Progress
- **Training Loss**: Should decrease consistently
- **Validation Loss**: Should plateau or decrease
- **Accuracy/F1**: Should improve over epochs
- **Learning Rate**: Should follow schedule

#### 2. LoRA-Specific Metrics
- **Parameter Efficiency**: ~0.3% trainable parameters
- **Adapter Weights**: `lora_adapters/*` metrics
- **Rank Utilization**: `lora_rank/*` metrics
- **Base Model Frozen**: `verification/base_params_frozen: true`

#### 3. Expected Performance Ranges

**MRPC (Paraphrase Detection)**:
- Baseline (Random): ~0.50 accuracy
- Baseline (Majority): ~0.68 accuracy
- Full Fine-tuning: 0.85-0.90 accuracy
- LoRA: 0.82-0.88 accuracy (≤3% drop target)

**SQuAD v2 (Question Answering)**:
- Baseline (Random): ~0.00 EM, ~0.10 F1
- Full Fine-tuning: 0.75-0.85 EM, 0.80-0.88 F1
- LoRA: 0.72-0.82 EM, 0.77-0.85 F1 (≤3% drop target)

**SST-2 (Sentiment Analysis)**:
- Baseline (Random): ~0.50 accuracy
- Baseline (Majority): ~0.51 accuracy
- Full Fine-tuning: 0.90-0.95 accuracy
- LoRA: 0.87-0.92 accuracy (≤3% drop target)

**RTE (Textual Entailment)**:
- Baseline (Random): ~0.50 accuracy
- Baseline (Majority): ~0.53 accuracy
- Full Fine-tuning: 0.65-0.75 accuracy
- LoRA: 0.62-0.72 accuracy (≤3% drop target)

## 🔍 Success Validation Checklist

### ✅ Phase 1 Success Criteria

**All VMs Running**:
- [ ] All 3 tmux sessions active and running
- [ ] No error messages in terminal outputs
- [ ] W&B runs appearing in dashboard

**Training Progress**:
- [ ] Loss decreasing for all experiments
- [ ] No NaN or infinite values in metrics
- [ ] Training completing without crashes

**LoRA Validation**:
- [ ] Parameter efficiency ~0.3% for all LoRA runs
- [ ] Base model parameters frozen (0 with gradients)
- [ ] Adapter weights showing reasonable values

**Performance Targets**:
- [ ] Full fine-tuning meets expected performance ranges
- [ ] LoRA within 3% of full fine-tuning performance
- [ ] All baselines completed successfully

## 🚨 Red Flags to Watch For

### Critical Issues
- **OOM Errors**: GPU memory exhaustion
- **Auth Failures**: HuggingFace token issues
- **W&B Upload Errors**: Network or permission issues
- **Data Loading Errors**: Dataset access problems

### Performance Issues
- **No Training Progress**: Loss not decreasing after 100 steps
- **Poor LoRA Performance**: >5% performance drop vs full FT
- **Parameter Leakage**: Base model parameters updating in LoRA

### Monitor Commands
```bash
# Check GPU usage
nvidia-smi

# Check tmux sessions
tmux list-sessions

# Check logs
tail -f logs/phase1/vm1/*.log

# Check W&B sync status
wandb status
```

## ⏱️ Expected Timing

**Phase 1 Duration**: 12-24 hours (depending on GPU specs)
- **VM1**: ~8-16 hours (heaviest load: SQuAD v2 full FT)
- **VM2**: ~10-18 hours (balanced load)
- **VM3**: ~6-12 hours (lightest load: RTE + baselines)

**Ready for Phase 2 When**:
- All W&B runs show "finished" status
- All VMs completed without errors
- Expected number of runs completed (~21 total)

## 🎯 Next Steps After Phase 1

1. **Verify Results**: Check all experiments completed successfully
2. **Performance Analysis**: Quick check that LoRA meets targets
3. **Phase 2 Preparation**: Update Phase 2 scripts if needed
4. **Launch Phase 2a**: Start parallel analysis on all 3 VMs

---

**Current Status**: ✅ READY FOR PRODUCTION DEPLOYMENT
**Last Updated**: $(date)
