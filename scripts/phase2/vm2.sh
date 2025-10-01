#!/bin/bash
# Phase 2 - VM2: CLASSIFICATION TASKS PRODUCTION EXPERIMENTS
# Run with optimal hyperparameters from Phase 1, using 3 seeds for statistical validity
set -e  # Exit on error

echo "🚀 PHASE 2 - VM2: CLASSIFICATION TASKS PRODUCTION EXPERIMENTS"
echo "============================================================================"
echo "Production experiments with optimal hyperparameters from Phase 1:"
echo "1. MRPC, SST-2, RTE: Full fine-tuning × 3 seeds (42, 1337, 2024)"
echo "2. MRPC, SST-2, RTE: LoRA × 3 seeds (42, 1337, 2024)"
echo "3. Representation extraction for drift analysis"
echo "4. Total: 18 experiments (3 tasks × 2 methods × 3 seeds)"
echo "Expected runtime: ~20 hours"
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
echo "   Full dataset mode enabled for SST-2 (67K), MRPC (3.7K), RTE (2.5K)"
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
    
    # Task-specific memory requirements (with full datasets)
    case "$task" in
        "sst2")
            required_gb=10
            samples="67K"
            ;;
        "mrpc"|"rte")
            required_gb=6
            samples="3.7K/2.5K"
            ;;
        *)
            required_gb=8
            samples="full dataset"
            ;;
    esac
    
    echo "   Required: ~${required_gb}GB for $samples samples"
    
    if [ $gpu_free -lt $required_gb ]; then
        echo "⚠️  WARNING: Low GPU memory ($gpu_free GB < $required_gb GB required)"
        echo "   This experiment may encounter OOM. Consider reducing batch size."
    else
        echo "✅ Memory validation passed: ${gpu_free}GB >= ${required_gb}GB required"
    fi
    echo ""
}

# Initial cleanup
cleanup_memory

# Create directories
mkdir -p logs/phase2/vm2
mkdir -p results/phase2

echo "📅 Started at: $(date)"
echo ""

# Wait for Phase 1 to complete and optimal configs to be available
echo "⏳ Checking for Phase 1 optimal hyperparameters..."
REQUIRED_FILES=(
    "analysis/mrpc_full_finetune_optimal.yaml"
    "analysis/mrpc_lora_optimal.yaml"
    "analysis/sst2_full_finetune_optimal.yaml"
    "analysis/sst2_lora_optimal.yaml"
    "analysis/rte_full_finetune_optimal.yaml"
    "analysis/rte_lora_optimal.yaml"
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
# PHASE 2A: CLASSIFICATION FULL FINE-TUNING (3 tasks × 3 seeds)
# ============================================================================
echo "🔬 PHASE 2A: CLASSIFICATION FULL FINE-TUNING"
echo "Running with optimal hyperparameters from Phase 1 × 3 seeds"
echo "------------------------------------------------------------"

SEEDS=(42 1337 2024)
TASKS=(mrpc sst2 rte)
METHOD="full_finetune"

for TASK in "${TASKS[@]}"; do
    echo ""
    echo "📋 Task: $TASK (Full Fine-tuning)"
    echo "------------------------------------------------------------"
    
    for SEED in "${SEEDS[@]}"; do
        echo "⚡ [Full FT - Seed $SEED] $TASK Production Experiment (FULL DATASET)"
        
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
    # PHASE 2 FIX: Add configuration override for full dataset
    args.append(f'--config-override $PHASE2_CONFIG_OVERRIDE')
    print(' '.join(args))
")
        
        # Show dataset size information
        case "$TASK" in
            "sst2") echo "   🚀 Starting full dataset: SST-2 (67K samples)..." ;;
            "mrpc") echo "   🚀 Starting full dataset: MRPC (3.7K samples)..." ;;
            "rte") echo "   🚀 Starting full dataset: RTE (2.5K samples)..." ;;
        esac
        
        if python experiments/full_finetune.py \
            --task "$TASK" \
            --seed "$SEED" \
            $HYPERPARAMS \
            > "logs/phase2/vm2/${TASK}_${METHOD}_seed${SEED}.log" 2>&1; then
            echo "✅ $TASK full fine-tuning (seed $SEED) completed - FULL DATASET"
        else
            echo "❌ $TASK full fine-tuning (seed $SEED) FAILED"
            echo "   Check logs: logs/phase2/vm2/${TASK}_${METHOD}_seed${SEED}.log"
            exit 1
        fi
        
        # Clear GPU memory between runs
        cleanup_memory
    done
    
    echo "✅ $TASK full fine-tuning completed (3 seeds)"
done

echo ""
echo "✅ All classification full fine-tuning experiments completed (9 experiments)"
echo ""

# ============================================================================
# PHASE 2B: CLASSIFICATION LORA (3 tasks × 3 seeds)
# ============================================================================
echo "🔬 PHASE 2B: CLASSIFICATION LoRA"
echo "Running with optimal hyperparameters from Phase 1 × 3 seeds"
echo "------------------------------------------------------------"

METHOD="lora"

for TASK in "${TASKS[@]}"; do
    echo ""
    echo "📋 Task: $TASK (LoRA)"
    echo "------------------------------------------------------------"
    
    for SEED in "${SEEDS[@]}"; do
        echo "⚡ [LoRA - Seed $SEED] $TASK Production Experiment (FULL DATASET)"
        
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
        
        # Show dataset size information
        case "$TASK" in
            "sst2") echo "   🚀 Starting full dataset: SST-2 LoRA (67K samples)..." ;;
            "mrpc") echo "   🚀 Starting full dataset: MRPC LoRA (3.7K samples)..." ;;
            "rte") echo "   🚀 Starting full dataset: RTE LoRA (2.5K samples)..." ;;
        esac
        
        if python experiments/lora_finetune.py \
            --task "$TASK" \
            --seed "$SEED" \
            $HYPERPARAMS \
            > "logs/phase2/vm2/${TASK}_${METHOD}_seed${SEED}.log" 2>&1; then
            echo "✅ $TASK LoRA (seed $SEED) completed - FULL DATASET"
        else
            echo "❌ $TASK LoRA (seed $SEED) FAILED"
            echo "   Check logs: logs/phase2/vm2/${TASK}_${METHOD}_seed${SEED}.log"
            exit 1
        fi
        
        # Clear GPU memory between runs
        cleanup_memory
    done
    
    echo "✅ $TASK LoRA completed (3 seeds)"
done

echo ""
echo "✅ All classification LoRA experiments completed (9 experiments)"
echo ""

# ============================================================================
# FINAL SUMMARY
# ============================================================================
echo "============================================================================"
echo "🎉 PHASE 2 - VM2 COMPLETE!"
echo "============================================================================"
echo "Completed experiments:"
echo "  - MRPC Full Fine-tuning: 3 seeds ✅"
echo "  - MRPC LoRA: 3 seeds ✅"
echo "  - SST-2 Full Fine-tuning: 3 seeds ✅"
echo "  - SST-2 LoRA: 3 seeds ✅"
echo "  - RTE Full Fine-tuning: 3 seeds ✅"
echo "  - RTE LoRA: 3 seeds ✅"
echo "  - Total: 18 experiments"
echo ""
echo "Results saved to:"
echo "  - Models: results/phase2/"
echo "  - Logs: logs/phase2/vm2/"
echo "  - W&B: https://wandb.ai/$WANDB_ENTITY/$WANDB_PROJECT"
echo ""
echo "📅 Finished at: $(date)"
echo "============================================================================"
