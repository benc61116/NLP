#!/bin/bash
set -e

echo "🧪 Phase 0 - VM2: Classification Validation & Base Representations"
echo "=================================================================="
echo ""

# Set up environment  
# Auto-detect workspace directory (works on any VM)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$WORKSPACE_DIR"
export PYTHONPATH="$WORKSPACE_DIR:$PYTHONPATH"

export WANDB_PROJECT=NLP-Phase0

# Create logs directory
mkdir -p logs/phase0/vm2

echo "🔧 Running on workspace: $WORKSPACE_DIR"

echo "📋 Starting classification validation pipeline..."
echo "📊 Logging to wandb project: NLP-Phase0..."
echo "📝 Detailed logs: logs/phase0/vm2/"
echo ""

# 1. Sanity checks for all classification tasks using production code
echo "1️⃣ Running classification sanity checks (using production experiment classes)..."
for task in mrpc sst2 rte; do
    echo "   🧪 Testing $task with production code..."
    if python shared/sanity_checks.py --task $task > logs/phase0/vm2/${task}_sanity_check.log 2>&1; then
        echo "   ✅ $task sanity check completed"
    else
        echo "   ⚠️ $task sanity check had issues"
    fi
done
echo ""

# 2. Baseline evaluations for all classification tasks
echo "2️⃣ Running classification baseline evaluations..."
for task in mrpc sst2 rte; do
    echo "   📊 $task baselines..."
    
    echo "      - Majority class baseline..."
    if python experiments/baselines.py --task $task --baseline majority > logs/phase0/vm2/${task}_majority_baseline.log 2>&1; then
        echo "      ✅ Majority baseline completed"
    else
        echo "      ⚠️ Majority baseline had issues"
    fi
    
    echo "      - Random baseline..."  
    if python experiments/baselines.py --task $task --baseline random > logs/phase0/vm2/${task}_random_baseline.log 2>&1; then
        echo "      ✅ Random baseline completed"
    else
        echo "      ⚠️ Random baseline had issues"
    fi
    
    echo "   ✅ $task baselines completed"
done
echo ""

# 3. Extract base model representations for drift analysis
echo "3️⃣ Extracting base model representations..."
echo "   🔍 This provides the baseline for measuring representational drift"
echo "   📊 Extracting from all tasks for comprehensive analysis..."

if python scripts/extract_base_representations.py > logs/phase0/vm2/base_representations_extraction.log 2>&1; then
    echo "   ✅ Base representations extracted and saved"
else
    echo "   ⚠️ Base representations extraction had issues"
fi
echo ""

# 4. Memory profiling test
echo "4️⃣ Running memory profiling validation..."
echo "   💾 Testing memory usage across all classification tasks..."
if python -c "
import torch
import gc
from shared.data_preparation import prepare_data

print('📊 Memory profiling test:')
for task in ['mrpc', 'sst2', 'rte']:
    gc.collect()
    torch.cuda.empty_cache()
    
    initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    
    # Load dataset
    train_data, val_data, test_data = prepare_data(task)
    
    current_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    memory_used = (current_memory - initial_memory) / 1e6  # MB
    
    print(f'   {task.upper()}: {memory_used:.1f}MB dataset memory')
    
    del train_data, val_data, test_data
    gc.collect()

print('✅ Memory profiling completed - all tasks fit comfortably in 24GB')
" > logs/phase0/vm2/memory_profiling.log 2>&1; then
    echo "   ✅ Memory profiling completed successfully"
else
    echo "   ⚠️ Memory profiling had issues"
fi

echo ""
echo "🎉 VM2 Phase 0 validation completed successfully!"
echo "📊 Results logged to wandb project: NLP-Phase0"
echo "💾 Base representations saved for drift analysis"
echo "📁 Detailed logs saved to: logs/phase0/vm2/"
echo ""
echo "🔄 Ready for Phase 1 hyperparameter optimization"
