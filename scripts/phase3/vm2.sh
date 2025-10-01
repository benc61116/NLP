#!/bin/bash
# Phase 3 - VM2: Extract Representations from Classification Models
# Runs post-training representation extraction for MRPC, SST-2, RTE

set -e

echo "🔬 PHASE 3 - VM2: CLASSIFICATION REPRESENTATION EXTRACTION"
echo "============================================================================"
echo "Extracting representations from saved Phase 2 classification models"
echo "Tasks: MRPC, SST-2, RTE"
echo "Methods: full_finetune, lora"
echo "Seeds: 42, 1337, 2024"
echo "Expected runtime: ~6-8 hours"
echo "============================================================================"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$WORKSPACE_DIR"
export PYTHONPATH="$WORKSPACE_DIR:$PYTHONPATH"

export WANDB_PROJECT=NLP-Phase3-Representations
export WANDB_ENTITY=galavny-tel-aviv-university

echo "🔧 Running on workspace: $WORKSPACE_DIR"
echo ""

# Create output directories
mkdir -p results/phase3_representations
mkdir -p logs/phase3/vm2

TASKS=(mrpc sst2 rte)
METHODS=(full_finetune lora)
SEEDS=(42 1337 2024)

# Function to clean GPU memory between extractions
cleanup_memory() {
    echo "🧹 Cleaning GPU memory..."
    python -c "
import torch
import gc
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated() / 1e9
    print(f'✓ GPU cleanup: {allocated:.2f}GB allocated')
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

echo "📅 Started at: $(date)"
echo ""

# ============================================================================
# CLASSIFICATION REPRESENTATION EXTRACTION
# ============================================================================

for TASK in "${TASKS[@]}"; do
    echo ""
    echo "📋 Task: $TASK"
    echo "============================================================================"
    
    # Show task information
    case "$TASK" in
        "mrpc")
            echo "   MRPC: Paraphrase detection (3.7K samples)"
            ;;
        "sst2")
            echo "   SST-2: Sentiment analysis (67K samples)"
            ;;
        "rte")
            echo "   RTE: Textual entailment (2.5K samples)"
            ;;
    esac
    echo ""
    
    for METHOD in "${METHODS[@]}"; do
        echo "📊 Method: $METHOD"
        echo "------------------------------------------------------------"
        
        for SEED in "${SEEDS[@]}"; do
            echo "⚡ [$METHOD - Seed $SEED] Extracting $TASK representations..."
            
            # Clean GPU before each extraction
            cleanup_memory
            
            if python scripts/phase3/extract_representations.py \
                --task "$TASK" \
                --method "$METHOD" \
                --seed "$SEED" \
                --output-dir results/phase3_representations \
                > "logs/phase3/vm2/${TASK}_${METHOD}_seed${SEED}.log" 2>&1; then
                echo "✅ $TASK $METHOD (seed $SEED) extraction completed"
            else
                echo "❌ $TASK $METHOD (seed $SEED) extraction FAILED"
                echo "   Check logs: logs/phase3/vm2/${TASK}_${METHOD}_seed${SEED}.log"
                exit 1
            fi
        done
        
        echo "✅ $TASK $METHOD completed (3 seeds)"
        echo ""
    done
    
    echo "✅ $TASK extraction completed (all methods, all seeds)"
    echo ""
done

echo ""
echo "============================================================================"
echo "✅ Phase 3 VM2 Complete!"
echo "📅 Finished at: $(date)"
echo "============================================================================"
echo ""
echo "Completed extractions:"
echo "  - MRPC: full_finetune + lora × 3 seeds ✅"
echo "  - SST-2: full_finetune + lora × 3 seeds ✅"
echo "  - RTE: full_finetune + lora × 3 seeds ✅"
echo "  - Total: 18 representation extractions"
echo ""
echo "Results saved to: results/phase3_representations/"
echo "Logs saved to: logs/phase3/vm2/"
echo ""
echo "Next steps:"
echo "1. Run drift analysis: python scripts/phase3/analyze_drift.py"
echo "2. Generate visualizations: python scripts/phase3/visualize_drift.py"
echo "3. Statistical hypothesis testing"
echo "============================================================================"
