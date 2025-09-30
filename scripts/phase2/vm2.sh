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

export WANDB_PROJECT=NLP-Phase2
export WANDB_ENTITY=galavny-tel-aviv-university

echo "🔧 Running on workspace: $WORKSPACE_DIR"

# Clear GPU memory cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -c "
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    print(f'GPU memory cleared. Available: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB')
else:
    print('No CUDA available')
"

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
        echo "⚡ [Full FT - Seed $SEED] $TASK Production Experiment"
        
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
            > "logs/phase2/vm2/${TASK}_${METHOD}_seed${SEED}.log" 2>&1; then
            echo "✅ $TASK full fine-tuning (seed $SEED) completed"
        else
            echo "❌ $TASK full fine-tuning (seed $SEED) FAILED"
            exit 1
        fi
        
        # Clear GPU memory between runs
        python -c "import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None"
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
        echo "⚡ [LoRA - Seed $SEED] $TASK Production Experiment"
        
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
            > "logs/phase2/vm2/${TASK}_${METHOD}_seed${SEED}.log" 2>&1; then
            echo "✅ $TASK LoRA (seed $SEED) completed"
        else
            echo "❌ $TASK LoRA (seed $SEED) FAILED"
            exit 1
        fi
        
        # Clear GPU memory between runs
        python -c "import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None"
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
