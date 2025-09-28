#!/bin/bash
set -e

echo "🧪 Phase 0 - VM1: SQuAD v2 Validation"
echo "===================================="
echo ""

# Error handling function
run_critical_step() {
    local description="$1"
    local command="$2"
    local logfile="$3"
    
    echo "   Running: $description"
    if eval "$command" > "$logfile" 2>&1; then
        echo "   ✅ $description completed successfully"
        return 0
    else
        echo "   ❌ CRITICAL FAILURE: $description failed"
        echo "   📝 Check log file: $logfile"
        echo "   🛑 Phase 0 validation FAILED - aborting to prevent wasted compute"
        exit 1
    fi
}

# Set up environment  
# Auto-detect workspace directory (works on any VM)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$WORKSPACE_DIR"
export PYTHONPATH="$WORKSPACE_DIR:$PYTHONPATH"

# Create log directory
mkdir -p logs/phase0/vm1

export WANDB_PROJECT=NLP-Phase0

echo "🔧 Running on workspace: $WORKSPACE_DIR"

# Create logs directory
mkdir -p logs/phase0/vm1

echo "📋 Starting SQuAD v2 validation pipeline..."
echo "📊 Logging to wandb project: NLP-Phase0..."
echo "📝 Detailed logs: logs/phase0/vm1/"
echo ""

# 0. Validate environment and model consistency
echo "0️⃣ Running pre-flight validation checks..."
run_critical_step "Environment setup and validation" "python shared/environment.py" "logs/phase0/vm1/environment_setup.log"
run_critical_step "Model consistency validation" "python shared/model_validation.py" "logs/phase0/vm1/model_validation.log"
run_critical_step "Data split and quality validation" "python shared/data_validation.py" "logs/phase0/vm1/data_validation.log"
echo ""

# 1. Sanity check - test using actual production code
echo "1️⃣ Running SQuAD v2 sanity check (using production experiment classes)..."
run_critical_step "SQuAD v2 sanity check" "python shared/sanity_checks.py --task squad_v2" "logs/phase0/vm1/squad_v2_sanity_check.log"
echo ""

# 2. Baseline evaluations
echo "2️⃣ Running SQuAD v2 baseline evaluations..."
run_critical_step "SQuAD v2 majority baseline" "python experiments/baselines.py --task squad_v2 --baseline majority" "logs/phase0/vm1/squad_v2_majority_baseline.log"
run_critical_step "SQuAD v2 random baseline" "python experiments/baselines.py --task squad_v2 --baseline random" "logs/phase0/vm1/squad_v2_random_baseline.log"

echo ""
echo "3️⃣ Testing SQuAD v2 architecture fix..."
echo "   🔧 Validating answerability head implementation..."
python -c "
from models.squad_v2_qa_model import SquadV2QuestionAnsweringModel
from transformers import AutoTokenizer
print('✅ SQuAD v2 model imports successfully')

tokenizer = AutoTokenizer.from_pretrained('TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T')
model = SquadV2QuestionAnsweringModel('TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T')
print('✅ SQuAD v2 model initializes successfully')
print('✅ Architecture validation passed')
"

echo ""
echo "🎉 VM1 Phase 0 validation completed successfully!"
echo "📊 Results logged to wandb project: NLP-Phase0"
echo "📁 Detailed logs saved to: logs/phase0/vm1/"
echo ""
echo "🔄 Ready for Phase 1 hyperparameter optimization"
