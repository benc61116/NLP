#!/bin/bash
# Phase 3 - Re-extract LoRA Representations Only
# Fixes the bug where full_finetune representations were uploaded to LoRA artifacts

set -e

echo "🔬 PHASE 3 - RE-EXTRACTING LORA REPRESENTATIONS"
echo "============================================================================"
echo "Extracting LoRA representations ONLY (full_finetune already complete)"
echo "Tasks: MRPC, SST-2, RTE"
echo "Seeds: 42, 1337, 2024"
echo "Expected runtime: ~20 minutes"
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
mkdir -p results/phase3_representations/representations
mkdir -p logs/phase3/lora_reextract

TASKS=(mrpc sst2 rte)
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

echo "📅 Started at: $(date)"
echo ""

# ============================================================================
# LORA REPRESENTATION EXTRACTION
# ============================================================================

TOTAL=0
SUCCESS=0
FAILED=0

for TASK in "${TASKS[@]}"; do
    echo ""
    echo "📋 Task: $TASK"
    echo "============================================================================"
    
    for SEED in "${SEEDS[@]}"; do
        TOTAL=$((TOTAL + 1))
        echo "⚡ [LoRA - Seed $SEED] Extracting $TASK representations..."
        
        # Clean GPU before each extraction
        cleanup_memory
        
        if python scripts/phase3/extract_representations.py \
            --task "$TASK" \
            --method lora \
            --seed "$SEED" \
            --output-dir results/phase3_representations \
            > "logs/phase3/lora_reextract/${TASK}_lora_seed${SEED}.log" 2>&1; then
            echo "✅ $TASK LoRA (seed $SEED) extraction completed"
            SUCCESS=$((SUCCESS + 1))
        else
            echo "❌ $TASK LoRA (seed $SEED) extraction FAILED"
            echo "   Check logs: logs/phase3/lora_reextract/${TASK}_lora_seed${SEED}.log"
            FAILED=$((FAILED + 1))
        fi
    done
    
    echo "✅ $TASK LoRA extraction completed (3 seeds)"
    echo ""
done

echo ""
echo "============================================================================"
echo "✅ LoRA Re-extraction Complete!"
echo "📅 Finished at: $(date)"
echo "============================================================================"
echo ""
echo "Summary:"
echo "  ✅ Successful: $SUCCESS/$TOTAL"
echo "  ❌ Failed: $FAILED/$TOTAL"
echo ""
echo "Results saved to: results/phase3_representations/representations/"
echo "Logs saved to: logs/phase3/lora_reextract/"
echo ""

if [ $FAILED -eq 0 ]; then
    echo "🎉 All LoRA representations extracted successfully!"
    echo ""
    echo "Next steps:"
    echo "1. Verify local files: ls results/phase3_representations/representations/lora_*"
    echo "2. Run drift analysis: python scripts/phase3/analyze_drift.py --task all"
    echo "============================================================================"
    exit 0
else
    echo "⚠️  Some extractions failed. Check logs for details."
    exit 1
fi

