#!/bin/bash
# Phase 2 - VM1: SQuAD v2 PRODUCTION EXPERIMENTS
# Run with optimal hyperparameters from Phase 1, using 3 seeds for statistical validity
set -e  # Exit on error

echo "🚀 PHASE 2 - VM1: SQuAD v2 PRODUCTION EXPERIMENTS"
echo "============================================================================"
echo "Production experiments with optimal hyperparameters from Phase 1:"
echo "1. SQuAD v2: Full fine-tuning × 3 seeds (42, 1337, 2024)"
echo "2. SQuAD v2: LoRA × 3 seeds (42, 1337, 2024)"
echo "3. Representation extraction for drift analysis"
echo "4. Total: 6 experiments (1 task × 2 methods × 3 seeds)"
echo "Expected runtime: ~23 hours"
echo "============================================================================"

# Setup environment
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$WORKSPACE_DIR"
export PYTHONPATH="$WORKSPACE_DIR:$PYTHONPATH"

export WANDB_PROJECT=NLP-Phase2
export WANDB_ENTITY=galavny-tel-aviv-university

echo "🔧 Running on workspace: $WORKSPACE_DIR"

# Clear GPU memory cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Function to clean GPU and CPU memory between runs
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

# Initial cleanup
cleanup_memory

# Create directories
mkdir -p logs/phase2/vm1
mkdir -p results/phase2

echo "📅 Started at: $(date)"
echo ""

# Wait for Phase 1 to complete and optimal configs to be available
echo "⏳ Checking for Phase 1 optimal hyperparameters..."
REQUIRED_FILES=(
    "analysis/squad_v2_full_finetune_optimal.yaml"
    "analysis/squad_v2_lora_optimal.yaml"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        echo "❌ ERROR: $file not found!"
        echo "   Phase 1 must complete before running Phase 2"
        exit 1
    fi
done
echo "✅ All optimal hyperparameter files found"
echo ""

# ============================================================================
# PHASE 2A: SQUAD v2 FULL FINE-TUNING (3 seeds)
# ============================================================================
echo "🔬 PHASE 2A: SQuAD v2 FULL FINE-TUNING"
echo "Running with optimal hyperparameters from Phase 1 × 3 seeds"
echo "------------------------------------------------------------"

SEEDS=(42 1337 2024)
TASK="squad_v2"
METHOD="full_finetune"

for SEED in "${SEEDS[@]}"; do
    echo "⚡ [Full FT - Seed $SEED] SQuAD v2 Production Experiment"
    
    # Load optimal hyperparameters from Phase 1
    OPTIMAL_CONFIG="analysis/${TASK}_${METHOD}_optimal.yaml"
    HYPERPARAMS=$(python -c "
import yaml
with open('$OPTIMAL_CONFIG') as f:
    config = yaml.safe_load(f)
    hp = config.get('best_hyperparameters', {})
    args = []
    if 'learning_rate' in hp:
        args.append(f'--learning-rate {hp[\"learning_rate\"]}')
    if 'per_device_train_batch_size' in hp:
        args.append(f'--batch-size {hp[\"per_device_train_batch_size\"]}')
    if 'warmup_ratio' in hp:
        args.append(f'--warmup-ratio {hp[\"warmup_ratio\"]}')
    if 'weight_decay' in hp:
        args.append(f'--weight-decay {hp[\"weight_decay\"]}')
    if 'num_train_epochs' in hp:
        args.append(f'--epochs {hp[\"num_train_epochs\"]}')
    print(' '.join(args))
")
    
    if python experiments/full_finetune.py \
        --task "$TASK" \
        --seed "$SEED" \
        $HYPERPARAMS \
        > "logs/phase2/vm1/${TASK}_${METHOD}_seed${SEED}.log" 2>&1; then
        echo "✅ SQuAD v2 full fine-tuning (seed $SEED) completed"
    else
        echo "❌ SQuAD v2 full fine-tuning (seed $SEED) FAILED"
        exit 1
    fi
    
    # Clear GPU memory between runs
        cleanup_memory
    echo ""
done

echo "✅ All SQuAD v2 full fine-tuning experiments completed (3 seeds)"
echo ""

# ============================================================================
# PHASE 2B: SQUAD v2 LORA (3 seeds)
# ============================================================================
echo "🔬 PHASE 2B: SQuAD v2 LoRA"
echo "Running with optimal hyperparameters from Phase 1 × 3 seeds"
echo "------------------------------------------------------------"

METHOD="lora"

for SEED in "${SEEDS[@]}"; do
    echo "⚡ [LoRA - Seed $SEED] SQuAD v2 Production Experiment"
    
    # Load optimal hyperparameters from Phase 1
    OPTIMAL_CONFIG="analysis/${TASK}_${METHOD}_optimal.yaml"
    HYPERPARAMS=$(python -c "
import yaml
with open('$OPTIMAL_CONFIG') as f:
    config = yaml.safe_load(f)
    hp = config.get('best_hyperparameters', {})
    args = []
    if 'learning_rate' in hp:
        args.append(f'--learning-rate {hp[\"learning_rate\"]}')
    if 'per_device_train_batch_size' in hp:
        args.append(f'--batch-size {hp[\"per_device_train_batch_size\"]}')
    if 'warmup_ratio' in hp:
        args.append(f'--warmup-ratio {hp[\"warmup_ratio\"]}')
    if 'weight_decay' in hp:
        args.append(f'--weight-decay {hp[\"weight_decay\"]}')
    if 'num_train_epochs' in hp:
        args.append(f'--epochs {hp[\"num_train_epochs\"]}')
    if 'lora_r' in hp:
        args.append(f'--lora-r {hp[\"lora_r\"]}')
    if 'lora_alpha' in hp:
        args.append(f'--lora-alpha {hp[\"lora_alpha\"]}')
    if 'lora_dropout' in hp:
        args.append(f'--lora-dropout {hp[\"lora_dropout\"]}')
    print(' '.join(args))
")
    
    if python experiments/lora_finetune.py \
        --task "$TASK" \
        --seed "$SEED" \
        $HYPERPARAMS \
        > "logs/phase2/vm1/${TASK}_${METHOD}_seed${SEED}.log" 2>&1; then
        echo "✅ SQuAD v2 LoRA (seed $SEED) completed"
    else
        echo "❌ SQuAD v2 LoRA (seed $SEED) FAILED"
        exit 1
    fi
    
    # Clear GPU memory between runs
        cleanup_memory
    echo ""
done

echo "✅ All SQuAD v2 LoRA experiments completed (3 seeds)"
echo ""

# ============================================================================
# FINAL SUMMARY
# ============================================================================
echo "============================================================================"
echo "🎉 PHASE 2 - VM1 COMPLETE!"
echo "============================================================================"
echo "Completed experiments:"
echo "  - SQuAD v2 Full Fine-tuning: 3 seeds ✅"
echo "  - SQuAD v2 LoRA: 3 seeds ✅"
echo "  - Total: 6 experiments"
echo ""
echo "Results saved to:"
echo "  - Models: results/phase2/"
echo "  - Logs: logs/phase2/vm1/"
echo "  - W&B: https://wandb.ai/$WANDB_ENTITY/$WANDB_PROJECT"
echo ""
echo "📅 Finished at: $(date)"
echo "============================================================================"
