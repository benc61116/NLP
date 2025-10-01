#!/bin/bash
# Test VM1 SQuAD v2 with 15min timeout to validate OOM fixes
set -e

echo "🧪 TESTING VM1 SQUAD v2 MEMORY FIXES (15min timeout)"
echo "============================================================================"
echo "This will run 1 trial each of Full FT and LoRA with timeout to validate"
echo "that all OOM issues are resolved before committing to full Phase 1 run."
echo "============================================================================"
echo ""

# Setup
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$WORKSPACE_DIR"
export PYTHONPATH="$WORKSPACE_DIR:$PYTHONPATH"

export WANDB_PROJECT=NLP-Phase1-Optuna-Test
export WANDB_ENTITY=galavny-tel-aviv-university

echo "🔧 Running on workspace: $WORKSPACE_DIR"
echo ""

# Create log directory
mkdir -p logs/test

# Memory cleanup function
cleanup_memory() {
    echo ""
    echo "🧹 Cleaning GPU memory..."
    python -c "
import torch
import gc
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.ipc_collect()
    print('✓ GPU cleanup complete')
"
    echo ""
}

cleanup_memory

echo "📅 Test started at: $(date)"
echo ""

# Test 1: Full Fine-tuning
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "TEST 1/2: SQuAD v2 Full Fine-tuning (1 trial, 15min timeout)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

timeout 900 python experiments/optuna_optimization.py \
    --task squad_v2 \
    --method full_finetune \
    --n-trials 1 \
    --wandb-project NLP-Phase1-Optuna-Test \
    --output-file analysis/test_squad_v2_full_finetune.yaml \
    > logs/test/squad_v2_full_test.log 2>&1

if [ $? -eq 0 ]; then
    echo "✅ Full fine-tuning test completed successfully (no OOM!)"
elif [ $? -eq 124 ]; then
    echo "⏱️  Test timed out (reached 15min limit - this is OK, just testing OOM)"
else
    echo "❌ Full fine-tuning test FAILED (check logs/test/squad_v2_full_test.log)"
    exit 1
fi

cleanup_memory
sleep 5

# Test 2: LoRA
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "TEST 2/2: SQuAD v2 LoRA (1 trial, 15min timeout)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

timeout 900 python experiments/optuna_optimization.py \
    --task squad_v2 \
    --method lora \
    --n-trials 1 \
    --wandb-project NLP-Phase1-Optuna-Test \
    --output-file analysis/test_squad_v2_lora.yaml \
    > logs/test/squad_v2_lora_test.log 2>&1

if [ $? -eq 0 ]; then
    echo "✅ LoRA test completed successfully (no OOM!)"
elif [ $? -eq 124 ]; then
    echo "⏱️  Test timed out (reached 15min limit - this is OK, just testing OOM)"
else
    echo "❌ LoRA test FAILED (check logs/test/squad_v2_lora_test.log)"
    exit 1
fi

cleanup_memory

echo ""
echo "============================================================================"
echo "🎉 ALL TESTS PASSED! No OOM detected in 15min window"
echo "============================================================================"
echo "📅 Test finished at: $(date)"
echo ""
echo "✅ VM1 is ready for full Phase 1 run!"
echo ""
echo "To view logs:"
echo "  - Full FT: tail -f logs/test/squad_v2_full_test.log"
echo "  - LoRA:    tail -f logs/test/squad_v2_lora_test.log"
echo ""
echo "To clean up test files:"
echo "  - rm -f analysis/test_*.yaml"
echo "  - rm -f logs/test/*.log"
echo "============================================================================"

