#!/bin/bash
# Re-run VM2 LoRA with fixed rank=8 for consistency

set -e  # Exit on error

echo "🔄 RE-RUNNING VM2 LORA WITH FIXED RANK=8"
echo "========================================"
echo "Reason: Need consistent LoRA rank across all tasks for fair comparison"
echo "Tasks: MRPC, SST-2, RTE (10 trials each)"
echo "Expected runtime: ~1 hour"
echo "========================================"

# Setup environment
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$WORKSPACE_DIR"

# Create log directory
mkdir -p logs/phase1_optuna/vm2_rerun

# Cleanup function
cleanup_memory() {
    echo ""
    echo "🧹 Cleaning GPU and CPU memory..."
    python -c "
import torch
import gc

# Python garbage collection
gc.collect()

# CUDA cleanup
if torch.cuda.is_available():
    torch.cuda.empty_cache()  # Clear CUDA cache
    torch.cuda.synchronize()  # Sync CUDA operations
    torch.cuda.ipc_collect()  # Clean IPC
    
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    free = total - reserved
    
    print(f'✓ GPU cleanup complete:')
    print(f'  • Allocated: {allocated:.2f}GB')
    print(f'  • Reserved: {reserved:.2f}GB')
    print(f'  • Free: {free:.2f}GB / {total:.2f}GB total')
else:
    print('⚠ No CUDA available')
"
    echo ""
}

echo ""
echo "⚡ [1/3] MRPC LoRA Optimization (10 trials, rank=8)"
if python experiments/optuna_optimization.py \
    --task mrpc \
    --method lora \
    --n-trials 10 \
    --wandb-project NLP-Phase1-Optuna \
    --output-file analysis/mrpc_lora_optimal.yaml \
    > logs/phase1_optuna/vm2_rerun/mrpc_lora_optuna.log 2>&1; then
    echo "✅ MRPC LoRA optimization completed (rank=8)"
    cleanup_memory
else
    echo "❌ MRPC LoRA optimization FAILED"
    exit 1
fi

echo "⚡ [2/3] SST-2 LoRA Optimization (10 trials, rank=8)"
if python experiments/optuna_optimization.py \
    --task sst2 \
    --method lora \
    --n-trials 10 \
    --wandb-project NLP-Phase1-Optuna \
    --output-file analysis/sst2_lora_optimal.yaml \
    > logs/phase1_optuna/vm2_rerun/sst2_lora_optuna.log 2>&1; then
    echo "✅ SST-2 LoRA optimization completed (rank=8)"
    cleanup_memory
else
    echo "❌ SST-2 LoRA optimization FAILED"
    exit 1
fi

echo "⚡ [3/3] RTE LoRA Optimization (10 trials, rank=8)"
if python experiments/optuna_optimization.py \
    --task rte \
    --method lora \
    --n-trials 10 \
    --wandb-project NLP-Phase1-Optuna \
    --output-file analysis/rte_lora_optimal.yaml \
    > logs/phase1_optuna/vm2_rerun/rte_lora_optuna.log 2>&1; then
    echo "✅ RTE LoRA optimization completed (rank=8)"
    cleanup_memory
else
    echo "❌ RTE LoRA optimization FAILED"
    exit 1
fi

echo ""
echo "🎉 VM2 LoRA re-run completed successfully!"
echo "All tasks now use consistent rank=8 for fair comparison"
echo ""
