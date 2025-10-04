#!/bin/bash
# Phase 2 - LoRA ONLY Re-run (after bug fix)
# Run LoRA experiments with the gradient checkpointing fix
set -e  # Exit on error

echo "🚀 PHASE 2 - LoRA RE-RUN (Bug Fix Applied)"
echo "============================================================================"
echo "Re-running LoRA experiments with enable_input_require_grads() fix"
echo "Tasks: MRPC, SST-2, RTE × 3 seeds (42, 1337, 2024)"
echo "Total: 9 experiments"
echo "Expected runtime: ~25-30 hours (RTE: 2h, MRPC: 3h, SST-2: 18-24h)"
echo "============================================================================"

# Setup environment
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$WORKSPACE_DIR"
export PYTHONPATH="$WORKSPACE_DIR:$PYTHONPATH"

# PHASE 2 configuration
echo "🔧 Applying Phase 2 full dataset configuration..."
export PHASE2_FULL_DATASET=true
export PHASE2_CONFIG_OVERRIDE="$WORKSPACE_DIR/shared/phase2_config_override.yaml"
echo "   Full dataset mode enabled for SST-2 (67K), MRPC (3.7K), RTE (2.5K)"
echo "   Config override: $PHASE2_CONFIG_OVERRIDE"

export WANDB_PROJECT=NLP-Phase2-LoRA-Rerun
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
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.ipc_collect()
    
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
    
    # Task-specific memory requirements
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
mkdir -p logs/phase2/lora_rerun
mkdir -p results/phase2_lora_rerun

echo "📅 Started at: $(date)"
echo ""

# Check for Phase 1 optimal configs
echo "⏳ Checking for Phase 1 optimal hyperparameters..."
REQUIRED_FILES=(
    "analysis/mrpc_lora_optimal.yaml"
    "analysis/sst2_lora_optimal.yaml"
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
# PHASE 2: CLASSIFICATION LORA (3 tasks × 3 seeds)
# ============================================================================
echo "🔬 PHASE 2: CLASSIFICATION LoRA (WITH BUG FIX)"
echo "Running with enable_input_require_grads() fix applied"
echo "------------------------------------------------------------"

SEEDS=(42 1337 2024)
TASKS=(mrpc sst2 rte)
METHOD="lora"

for TASK in "${TASKS[@]}"; do
    echo ""
    echo "📋 Task: $TASK (LoRA)"
    echo "------------------------------------------------------------"
    
    for SEED in "${SEEDS[@]}"; do
        echo "⚡ [LoRA - Seed $SEED] $TASK Production Experiment (FULL DATASET, BUG FIXED)"
        
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
            "sst2") echo "   🚀 Starting full dataset: SST-2 LoRA (67K samples) with gradient fix..." ;;
            "mrpc") echo "   🚀 Starting full dataset: MRPC LoRA (3.7K samples) with gradient fix..." ;;
            "rte") echo "   🚀 Starting full dataset: RTE LoRA (2.5K samples) with gradient fix..." ;;
        esac
        
        if python experiments/lora_finetune.py \
            --task "$TASK" \
            --seed "$SEED" \
            $HYPERPARAMS \
            > "logs/phase2/lora_rerun/${TASK}_${METHOD}_seed${SEED}.log" 2>&1; then
            echo "✅ $TASK LoRA (seed $SEED) completed - FULL DATASET - BUG FIXED"
            
            # Quick validation: Check if LoRA_B is non-zero
            echo "   🔍 Validating LoRA_B weights are non-zero..."
            python -c "
from safetensors import safe_open
import glob

# Find the adapter file
adapter_files = glob.glob('results/lora_finetune_*/lora_${TASK}_*_seed${SEED}/final_adapter/adapter_model.safetensors')
if adapter_files:
    adapter_path = adapter_files[-1]  # Get most recent
    with safe_open(adapter_path, framework='pt') as f:
        # Check first LoRA_B
        for key in f.keys():
            if 'lora_B' in key:
                tensor = f.get_tensor(key)
                abs_max = tensor.float().abs().max().item()
                if abs_max > 1e-6:
                    print(f'   ✅ LoRA_B is NON-ZERO (abs_max={abs_max:.6f}) - Fix is working!')
                else:
                    print(f'   ❌ WARNING: LoRA_B is still ZERO - Fix may not be working!')
                break
" || echo "   ⚠️  Could not validate LoRA_B (file not found or error)"
            
        else
            echo "❌ $TASK LoRA (seed $SEED) FAILED"
            echo "   Check logs: logs/phase2/lora_rerun/${TASK}_${METHOD}_seed${SEED}.log"
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
echo "🎉 PHASE 2 - LoRA RE-RUN COMPLETE!"
echo "============================================================================"
echo "Completed experiments:"
echo "  - MRPC LoRA: 3 seeds ✅"
echo "  - SST-2 LoRA: 3 seeds ✅"
echo "  - RTE LoRA: 3 seeds ✅"
echo "  - Total: 9 experiments (with enable_input_require_grads() fix)"
echo ""
echo "Results saved to:"
echo "  - Models: results/"
echo "  - Logs: logs/phase2/lora_rerun/"
echo "  - W&B: https://wandb.ai/$WANDB_ENTITY/$WANDB_PROJECT"
echo ""
echo "📋 NEXT STEPS:"
echo "  1. Run Phase 3: Extract representations from NEW LoRA models"
echo "  2. Run Phase 4: Re-run drift analysis with REAL LoRA representations"
echo "  3. Compare results to see if zero-variance was real or a bug artifact"
echo ""
echo "📅 Finished at: $(date)"
echo "============================================================================"

