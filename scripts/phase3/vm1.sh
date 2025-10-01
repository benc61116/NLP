#!/bin/bash
# Phase 3 - VM1: Extract Representations from SQuAD v2 Models
# Runs post-training representation extraction for drift analysis

set -e

echo "🔬 PHASE 3 - VM1: SQUAD v2 REPRESENTATION EXTRACTION"
echo "============================================================================"
echo "Extracting representations from saved Phase 2 models"
echo "This enables drift analysis without memory constraints during training"
echo "Expected runtime: ~4-6 hours"
echo "============================================================================"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$WORKSPACE_DIR"
export PYTHONPATH="$WORKSPACE_DIR:$PYTHONPATH"

export WANDB_PROJECT=NLP-Phase3-Representations
export WANDB_ENTITY=galavny-tel-aviv-university

echo "🔧 Running on workspace: $WORKSPACE_DIR"
echo ""

# Create output directory
mkdir -p results/phase3_representations
mkdir -p logs/phase3

TASK="squad_v2"
SEEDS=(42 1337 2024)

# ============================================================================
# SQUAD v2 FULL FINE-TUNING REPRESENTATION EXTRACTION
# ============================================================================

echo "📊 Extracting representations: SQuAD v2 Full Fine-tuning"
echo "------------------------------------------------------------"

for SEED in "${SEEDS[@]}"; do
    echo ""
    echo "⚡ [Full FT - Seed $SEED] Extracting representations..."
    
    python scripts/phase3/extract_representations.py \
        --task $TASK \
        --method full_finetune \
        --seed $SEED \
        --output-dir results/phase3_representations \
        > logs/phase3/${TASK}_full_finetune_seed${SEED}.log 2>&1
    
    echo "✅ Completed seed $SEED"
done

echo ""

# ============================================================================
# SQUAD v2 LORA REPRESENTATION EXTRACTION
# ============================================================================

echo "📊 Extracting representations: SQuAD v2 LoRA"
echo "------------------------------------------------------------"

for SEED in "${SEEDS[@]}"; do
    echo ""
    echo "⚡ [LoRA - Seed $SEED] Extracting representations..."
    
    python scripts/phase3/extract_representations.py \
        --task $TASK \
        --method lora \
        --seed $SEED \
        --output-dir results/phase3_representations \
        > logs/phase3/${TASK}_lora_seed${SEED}.log 2>&1
    
    echo "✅ Completed seed $SEED"
done

echo ""
echo "============================================================================"
echo "✅ Phase 3 VM1 Complete!"
echo "📅 Finished at: $(date)"
echo "============================================================================"
echo ""
echo "Next steps:"
echo "1. Run Phase 3 VM2 script for classification tasks"
echo "2. Run drift analysis scripts to compute CKA/cosine similarity"
echo "3. Generate visualizations and statistical tests"
echo "============================================================================"

