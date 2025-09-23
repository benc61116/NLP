# Simple VM Commands for Phase 1 Parallel Execution

## Prerequisites (One-time setup)
Run these commands on each VM before starting:

```bash
cd /home/benc6116/workspace/NLP
pip install -r requirements.txt
python scripts/download_datasets.py  # Only run once, then copy data/ to other VMs
```

## Phase 1 Execution Commands

### VM1 (SQuAD v2 Full FT + MRPC Full FT + MRPC LoRA)
```bash
cd /home/benc6116/workspace/NLP
tmux new-session -d -s phase1 './scripts/phase1/vm1.sh'
tmux attach -t phase1  # Optional: monitor progress
```

### VM2 (SQuAD v2 LoRA + SST-2 Full FT + SST-2 LoRA)  
```bash
cd /home/benc6116/workspace/NLP
tmux new-session -d -s phase1 './scripts/phase1/vm2.sh'
tmux attach -t phase1  # Optional: monitor progress
```

### VM3 (RTE Full FT + RTE LoRA + All Baselines)
```bash
cd /home/benc6116/workspace/NLP
tmux new-session -d -s phase1 './scripts/phase1/vm3.sh'
tmux attach -t phase1  # Optional: monitor progress
```

## Monitoring Commands

### Check tmux sessions
```bash
tmux list-sessions
```

### Monitor logs in real-time
```bash
# VM1
tail -f logs/phase1/vm1/*.log

# VM2  
tail -f logs/phase1/vm2/*.log

# VM3
tail -f logs/phase1/vm3/*.log
```

### Check W&B Dashboard
Visit: https://wandb.ai/galavny-tel-aviv-university/NLP-Phase1-Training

### Detach from tmux (keep running)
Press: `Ctrl+B` then `D`

### Reattach to tmux session
```bash
tmux attach -t phase1
```

## Expected Results
- **Total runs**: ~21 experiments across all VMs
- **Duration**: 12-24 hours  
- **Model**: TinyLlama-1.1B (no authentication needed)
- **Projects**: All results go to `NLP-Phase1-Training` in W&B

## When Phase 1 Complete
All tmux sessions will exit and you'll see "Phase 1 complete" messages. Check W&B dashboard to verify all runs finished successfully.
