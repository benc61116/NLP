#!/bin/bash
# Phase 2 - VM3: SQuAD v2 LoRA ONLY
# Run with optimal hyperparameters from Phase 1, using 3 seeds for statistical validity
set -e  # Exit on error

echo "🚀 PHASE 2 - VM3: SQuAD v2 LoRA ONLY"
echo "============================================================================"
echo "Production LoRA experiments with optimal hyperparameters from Phase 1:"
echo "1. SQuAD v2: LoRA × 3 seeds (42, 1337, 2024)"
echo "2. Focus: Parameter-efficient fine-tuning (130K samples)"
echo "3. Total: 3 experiments (1 task × 1 method × 3 seeds)"
echo "Expected runtime: ~9-12 hours"
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
    
    # LoRA-specific memory requirements
    case "$method" in
        "lora")
            required_gb=8
            samples="130K (LoRA - more memory efficient)"
            ;;
        *)
            required_gb=8
            samples="130K samples"
            ;;
    esac
    
    echo "   Required: ~${required_gb}GB for $samples"
    
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

# Function for comprehensive cleanup
cleanup_disk_cache() {
    echo "🧹 Comprehensive cleanup (disk cache, wandb, etc.)..."
    python -c "
import subprocess
import shutil
from pathlib import Path

# Clean wandb cache
try:
    result = subprocess.run(['wandb', 'artifact', 'cache', 'cleanup', '2GB'], 
                          capture_output=True, text=True, timeout=60)
    if result.returncode == 0:
        print('✓ Cleaned wandb cache')
    else:
        print(f'⚠ Wandb cache cleanup warning: {result.stderr[:100]}')
except Exception as e:
    print(f'⚠ Could not clean wandb cache: {e}')

# Report disk usage
total, used, free = shutil.disk_usage('/')
usage_percent = (used / total) * 100
print(f'💾 Disk usage: {usage_percent:.1f}% ({used//(1024**3)}GB used, {free//(1024**3)}GB free)')
"
    echo ""
}

# Initial cleanup
cleanup_memory

# Create directories
mkdir -p logs/phase2/vm3
mkdir -p results/phase2

echo "📅 Started at: $(date)"
echo ""

# Wait for Phase 1 to complete and optimal configs to be available
echo "⏳ Checking for Phase 1 optimal hyperparameters..."
REQUIRED_FILES=(
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
# PHASE 2B: SQUAD v2 LoRA (3 seeds)
# ============================================================================
echo "🔬 PHASE 2B: SQuAD v2 LoRA"
echo "Running with optimal hyperparameters from Phase 1 × 3 seeds"
echo "------------------------------------------------------------"

SEEDS=(42 1337 2024)
TASK="squad_v2"
METHOD="lora"

for SEED in "${SEEDS[@]}"; do
    echo "⚡ [LoRA - Seed $SEED] SQuAD v2 Production Experiment (FULL DATASET)"
    
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
    # PHASE 2 FIX: Add configuration override for full dataset
    args.append(f'--config-override $PHASE2_CONFIG_OVERRIDE')
    print(' '.join(args))
")
    
    echo "🚀 Starting full dataset experiment: SQuAD v2 LoRA (130K samples)..."
    if python experiments/lora_finetune.py \
        --task "$TASK" \
        --seed "$SEED" \
        $HYPERPARAMS \
        > "logs/phase2/vm3/${TASK}_${METHOD}_seed${SEED}.log" 2>&1; then
        echo "✅ SQuAD v2 LoRA (seed $SEED) completed - FULL DATASET"
    else
        echo "❌ SQuAD v2 LoRA (seed $SEED) FAILED"
        echo "   Check logs: logs/phase2/vm3/${TASK}_${METHOD}_seed${SEED}.log"
        exit 1
    fi
    
    # Clear GPU memory between runs
    cleanup_memory
    echo ""
done

# ============================================================================
# COMPREHENSIVE CLEANUP
# ============================================================================
cleanup_disk_cache

echo "✅ All SQuAD v2 LoRA experiments completed (3 seeds)"
echo ""

# ============================================================================
# FINAL SUMMARY
# ============================================================================
echo "============================================================================"
echo "🎉 PHASE 2 - VM3 LORA COMPLETE!"
echo "============================================================================"
echo "Completed experiments:"
echo "  - SQuAD v2 LoRA: 3 seeds ✅"
echo "  - Total: 3 experiments"
echo ""
echo "Results saved to:"
echo "  - Models: results/phase2/"
echo "  - Logs: logs/phase2/vm3/"
echo "  - W&B: https://wandb.ai/$WANDB_ENTITY/$WANDB_PROJECT"
echo ""
echo "📅 Finished at: $(date)"
echo ""
echo "🚀 NEXT STEPS:"
echo "1. Ensure VM1 completes: bash scripts/phase2/vm1.sh (SQuAD v2 Full FT)"
echo "2. Once both complete: Execute Phase 3 analysis"
echo "3. Combine results for comprehensive LoRA vs Full FT comparison"
echo "============================================================================"
