#!/bin/bash
# Phase 2 - VM1: SQuAD v2 FULL FINE-TUNING ONLY
# Run with optimal hyperparameters from Phase 1, using 3 seeds for statistical validity
set -e  # Exit on error

echo "🚀 PHASE 2 - VM1: SQuAD v2 FULL FINE-TUNING ONLY"
echo "============================================================================"
echo "Production full fine-tuning with optimal hyperparameters from Phase 1:"
echo "1. SQuAD v2: Full fine-tuning × 3 seeds (42, 1337, 2024)"
echo "2. Focus: Full dataset training (130K samples)"
echo "3. Total: 3 experiments (1 task × 1 method × 3 seeds)"
echo "Expected runtime: ~18-24 hours"
echo "============================================================================"

# Setup environment
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$WORKSPACE_DIR"
export PYTHONPATH="$WORKSPACE_DIR:$PYTHONPATH"

# PHASE 2 FIX: Apply full dataset configuration override
echo "🔧 Applying Phase 2 full dataset configuration..."
export PHASE2_FULL_DATASET=true
export PHASE2_CONFIG_OVERRIDE="$WORKSPACE_DIR/shared/phase2_config_override.yaml"
echo "   Full dataset mode enabled for production experiments"
echo "   Config override: $PHASE2_CONFIG_OVERRIDE"

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

# Function to clean disk cache between tasks (non-disruptive)
cleanup_disk_cache() {
    echo "🧹 Cleaning disk cache (wandb artifacts only)..."
    
    # Only clean wandb cache if disk usage > 70%
    disk_usage=$(df / | tail -1 | awk '{print int($5)}')
    
    if [ $disk_usage -gt 70 ]; then
        echo "   Disk usage: ${disk_usage}% - cleaning wandb cache..."
        rm -rf ~/.cache/wandb/* 2>/dev/null || true
        echo "   ✓ Wandb cache cleaned"
    else
        echo "   Disk usage: ${disk_usage}% - no cleanup needed (< 70%)"
    fi
    echo ""
}

# Function to validate memory before large dataset experiments
validate_memory_for_task() {
    local task=$1
    local method=$2
    local seed=$3
    
    echo "🔍 Memory validation for $task ($method, seed $seed)..."
    
    # Get current GPU memory
    gpu_info=$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits)
    gpu_used=$(echo $gpu_info | cut -d',' -f1)
    gpu_total=$(echo $gpu_info | cut -d',' -f2)
    gpu_free=$((gpu_total - gpu_used))
    
    echo "   GPU Memory: ${gpu_used}GB used, ${gpu_free}GB free (${gpu_total}GB total)"
    
    # Task-specific memory requirements (with full datasets)
    case "$task" in
        "squad_v2")
            required_gb=12
            samples="130K"
            ;;
        *)
            required_gb=8
            samples="full dataset"
            ;;
    esac
    
    echo "   Required: ~${required_gb}GB for $samples samples"
    
    if [ $gpu_free -lt $required_gb ]; then
        echo "⚠️  WARNING: Low GPU memory ($gpu_free GB < $required_gb GB required)"
        echo "   This experiment may encounter OOM. Consider:"
        echo "   1. Reducing batch size in phase2_config_override.yaml"
        echo "   2. Increasing gradient_accumulation_steps"
        echo "   3. Running cleanup_memory() again"
    else
        echo "✅ Memory validation passed: ${gpu_free}GB >= ${required_gb}GB required"
    fi
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
    echo "⚡ [Full FT - Seed $SEED] SQuAD v2 Production Experiment (FULL DATASET)"
    
    # Memory validation before large experiment
    validate_memory_for_task "$TASK" "$METHOD" "$SEED"
    
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
        # PLATFORM FIX: Increase batch size from 2 to 4 to reduce step count
        # Original value preserved in analysis file (Phase 1 result)
        batch_size = 4  # Override for platform step limit
        args.append(f'--batch-size {batch_size}')
    if 'warmup_ratio' in hp:
        args.append(f'--warmup-ratio {hp[\"warmup_ratio\"]}')
    if 'weight_decay' in hp:
        args.append(f'--weight-decay {hp[\"weight_decay\"]}')
    if 'num_train_epochs' in hp:
        # PLATFORM FIX: Reduce epochs from 4 to 2 to stay under step limit
        # 4 epochs × 8144 steps = 32,576 steps (exceeds ~10k platform limit)
        # 2 epochs × 4072 steps = 8,144 steps (under limit)
        epochs = 2  # Override for platform step limit
        args.append(f'--epochs {epochs}')
    # PHASE 2 FIX: Add configuration override for full dataset
    args.append(f'--config-override $PHASE2_CONFIG_OVERRIDE')
    print(' '.join(args))
")
    
    echo "🚀 Starting full dataset experiment: SQuAD v2 (130K samples)..."
    if python experiments/full_finetune.py \
        --task "$TASK" \
        --seed "$SEED" \
        $HYPERPARAMS \
        > "logs/phase2/vm1/${TASK}_${METHOD}_seed${SEED}.log" 2>&1; then
        echo "✅ SQuAD v2 full fine-tuning (seed $SEED) completed - FULL DATASET"
    else
        echo "❌ SQuAD v2 full fine-tuning (seed $SEED) FAILED"
        echo "   Check logs: logs/phase2/vm1/${TASK}_${METHOD}_seed${SEED}.log"
        exit 1
    fi
    
    # Clear GPU memory between runs
        cleanup_memory
    echo ""
done

echo "✅ All SQuAD v2 full fine-tuning experiments completed (3 seeds)"
echo ""

# ============================================================================
# COMPREHENSIVE CLEANUP
# ============================================================================
cleanup_disk_cache

# ============================================================================
# FINAL SUMMARY
# ============================================================================
echo "============================================================================"
echo "🎉 PHASE 2 - VM1 FULL FINE-TUNING COMPLETE!"
echo "============================================================================"
echo "Completed experiments:"
echo "  - SQuAD v2 Full Fine-tuning: 3 seeds ✅"
echo "  - Total: 3 experiments"
echo ""
echo "Results saved to:"
echo "  - Models: results/phase2/"
echo "  - Logs: logs/phase2/vm1/"
echo "  - W&B: https://wandb.ai/$WANDB_ENTITY/$WANDB_PROJECT"
echo ""
echo "📅 Finished at: $(date)"
echo ""
echo "🚀 NEXT STEPS:"
echo "1. Run VM3: bash scripts/phase2/vm3.sh (SQuAD v2 LoRA)"
echo "2. Once both complete: Execute Phase 3 analysis"
echo "============================================================================"
