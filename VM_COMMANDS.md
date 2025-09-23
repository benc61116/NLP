# 🚀 Phase 1 Execution Commands

## Prerequisites (Run on ALL VMs)

```bash
# 1. Clone repository
git clone https://github.com/benc61116/NLP.git
cd NLP

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download datasets
python scripts/download_datasets.py
```

**Note**: Log and result directories are created automatically by the scripts.

## Phase 1 Parallel Execution

### VM1 Commands
```bash
# Setup tmux session
tmux new-session -d -s phase1

# Run Phase 1 VM1 script
chmod +x scripts/phase1/vm1.sh
./scripts/phase1/vm1.sh

# Monitor progress
tmux attach -t phase1
```

**VM1 Tasks**: MRPC (Full + LoRA) + SST-2 (Full + LoRA)  
**Expected Time**: ~4 hours  
**W&B Group**: `mrpc_full_ft`, `mrpc_lora`, `sst2_full_ft`, `sst2_lora`

### VM2 Commands
```bash
# Setup tmux session
tmux new-session -d -s phase1

# Run Phase 1 VM2 script
chmod +x scripts/phase1/vm2.sh
./scripts/phase1/vm2.sh

# Monitor progress
tmux attach -t phase1
```

**VM2 Tasks**: SQuAD v2 (Full + LoRA) + RTE (Full + LoRA)  
**Expected Time**: ~4 hours  
**W&B Group**: `squad_v2_full_ft`, `squad_v2_lora`, `rte_full_ft`, `rte_lora`

### VM3 Commands
```bash
# Setup tmux session
tmux new-session -d -s phase1

# Run Phase 1 VM3 script
chmod +x scripts/phase1/vm3.sh
./scripts/phase1/vm3.sh

# Monitor progress
tmux attach -t phase1
```

**VM3 Tasks**: All Baselines (MRPC, SST-2, RTE, SQuAD v2) + Base Model Representations  
**Expected Time**: ~2 hours  
**W&B Group**: `baselines_all_tasks`, `base_representations`

## Monitoring

### W&B Dashboard
- **Phase 1 Project**: https://wandb.ai/galavny-tel-aviv-university/NLP-Phase1-Training
- **Real-time metrics**: Loss curves, memory usage, training progress

### Terminal Monitoring
```bash
# Check tmux sessions
tmux list-sessions

# Attach to session
tmux attach -t phase1

# Detach from session (Ctrl+B, then D)

# Check logs
tail -f logs/phase1/vm1/*.log
tail -f logs/phase1/vm2/*.log  
tail -f logs/phase1/vm3/*.log
```

### Progress Indicators
Each VM script shows:
- ✅ Task completion status
- ⚡ Current experiment running
- 📊 W&B run links
- ⏰ Timestamps for tracking

## Expected Results

### Performance Metrics
- **Training Time**: 1.15s/step average
- **GPU Memory**: ~15GB during training (fits L4's 22GB)
- **Total Phase 1**: ~4 hours (parallel execution)

### Output Structure
```
results/
├── full_finetune_YYYYMMDD_HHMMSS/
│   ├── mrpc_seed42/
│   ├── squad_v2_seed42/
│   └── ...
├── lora_YYYYMMDD_HHMMSS/
│   ├── mrpc_seed42/
│   ├── sst2_seed42/
│   └── ...
├── baselines_YYYYMMDD_HHMMSS/
└── base_model_representations/
```

## Troubleshooting

### Common Issues
1. **OOM Errors**: Scripts include GPU memory clearing
2. **Connection Issues**: W&B login persists across sessions
3. **Resume**: Checkpoint manager handles interruptions automatically

### Manual Intervention
```bash
# Clear GPU memory if needed
python -c "import torch; torch.cuda.empty_cache(); print('GPU cleared')"

# Check experiment progress
python -c "
from shared.checkpoint_utils import CheckpointManager
cm = CheckpointManager('results')
progress = cm._load_progress()
for k, v in progress.items():
    print(f'{k}: {v[\"status\"]}')
"
```

## Success Criteria
- All VM scripts complete without errors
- W&B shows all planned experiments
- Each task has 3 seeds (42, 1337, 2024) completed
- Base model representations extracted for all tasks
- Ready to proceed to Phase 2a

---
**Total Expected Time**: ~4 hours for complete Phase 1  
**Next Step**: Phase 2a parallel analysis (separate instructions)
