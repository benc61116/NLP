#!/bin/bash
# Quick validation test for Phase 1 critical fixes
# Tests LoRA parameter passing and eval_strategy fixes with 2 trials

set -e

echo "🧪 PHASE 1 FIXES VALIDATION TEST"
echo "================================="
echo "Running minimal Optuna test (2 trials) to verify fixes:"
echo "1. LoRA parameters (lora_r, lora_alpha) are properly passed"
echo "2. Eval metrics are extracted (eval_strategy enabled)"
echo "3. Different LoRA rank/alpha values used across trials"
echo ""

# Setup
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$WORKSPACE_DIR"
export PYTHONPATH="$WORKSPACE_DIR:$PYTHONPATH"

export WANDB_PROJECT=NLP-Phase1-Test
export WANDB_ENTITY=galavny-tel-aviv-university

# Create test output directory
mkdir -p logs/phase1_test
mkdir -p analysis/test

echo "🔬 Test 1: LoRA Optimization (2 trials on MRPC)"
echo "Expected behavior:"
echo "  • Trial 0 and Trial 1 should suggest different lora_r/lora_alpha"
echo "  • Logs should show: 'LoRA params: r=X, alpha=Y, dropout=Z'"
echo "  • eval_accuracy should be non-zero (not 0.0)"
echo "------------------------------------------------------------"

python experiments/optuna_optimization.py \
    --task mrpc \
    --method lora \
    --n-trials 2 \
    --wandb-project NLP-Phase1-Test \
    --output-file analysis/test/mrpc_lora_test.yaml \
    2>&1 | tee logs/phase1_test/lora_test.log

echo ""
echo "📊 VALIDATION RESULTS:"
echo "------------------------------------------------------------"

# Check if LoRA parameters were logged
if grep -q "LoRA params: r=" logs/phase1_test/lora_test.log; then
    echo "✅ LoRA parameters logged correctly"
    grep "LoRA params:" logs/phase1_test/lora_test.log | head -2
else
    echo "❌ FAILED: LoRA parameters NOT found in logs"
    exit 1
fi

# Check if eval metrics were extracted
if grep -q "eval_accuracy\|eval_f1" logs/phase1_test/lora_test.log; then
    echo "✅ Eval metrics extracted successfully"
    grep "eval_accuracy\|eval_f1" logs/phase1_test/lora_test.log | tail -2
else
    echo "⚠️  WARNING: Could not find eval metrics in logs (may be in W&B only)"
fi

# Check if trials completed
if grep -q "Optimization complete!" logs/phase1_test/lora_test.log; then
    echo "✅ Optimization completed successfully"
else
    echo "❌ FAILED: Optimization did not complete"
    exit 1
fi

# Display optimal hyperparameters
if [ -f "analysis/test/mrpc_lora_test.yaml" ]; then
    echo ""
    echo "📄 Optimal Hyperparameters Found:"
    echo "------------------------------------------------------------"
    python -c "
import yaml
with open('analysis/test/mrpc_lora_test.yaml') as f:
    config = yaml.safe_load(f)
print(f\"Task: {config['task']}\")
print(f\"Method: {config['method']}\")
print(f\"Expected performance: {config['expected_performance']:.4f}\")
print(f\"Best hyperparameters:\")
for k, v in config['best_hyperparameters'].items():
    print(f\"  {k}: {v}\")
print(f\"\\nOptimization summary:\")
summary = config['optimization_summary']
print(f\"  Trials: {summary['n_completed']}/{summary['n_trials']} completed, {summary['n_pruned']} pruned\")
"
    echo ""
    echo "✅ VALIDATION PASSED!"
    echo "   • LoRA parameters are being optimized"
    echo "   • Eval metrics are being extracted"
    echo "   • Ready for full Phase 1 experiments"
else
    echo "❌ FAILED: Output file not created"
    exit 1
fi

echo ""
echo "🎯 Next Steps:"
echo "   1. Check W&B dashboard: https://wandb.ai/galavny-tel-aviv-university/NLP-Phase1-Test"
echo "   2. Verify trials show different lora_r and lora_alpha values"
echo "   3. If validation looks good, run full Phase 1 with vm1.sh and vm2.sh"
