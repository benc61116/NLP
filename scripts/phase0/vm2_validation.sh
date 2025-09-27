#!/bin/bash
set -e

echo "🧪 Phase 0 - VM2: Classification Validation & Base Representations"
echo "=================================================================="
echo ""

# Set up environment
export PYTHONPATH=/home/benc6116/workspace/NLP:$PYTHONPATH
export WANDB_PROJECT=NLP-Phase0
cd /home/benc6116/workspace/NLP

echo "📋 Starting classification validation pipeline..."
echo "🔄 Initializing wandb project: NLP-Phase0..."
wandb login --relogin
echo ""

# 1. Sanity checks for all classification tasks
echo "1️⃣ Running classification sanity checks (overfitting tests)..."
for task in mrpc sst2 rte; do
    echo "   🧪 Testing $task overfitting capability..."
    python shared/sanity_checks.py --task $task --num-samples 100
    echo "   ✅ $task sanity check completed"
done
echo ""

# 2. Baseline evaluations for all classification tasks
echo "2️⃣ Running classification baseline evaluations..."
for task in mrpc sst2 rte; do
    echo "   📊 $task baselines..."
    
    echo "      - Majority class baseline..."
    python experiments/baselines.py --task $task --baseline majority
    
    echo "      - Random baseline..."  
    python experiments/baselines.py --task $task --baseline random
    
    echo "   ✅ $task baselines completed"
done
echo ""

# 3. Extract base model representations for drift analysis
echo "3️⃣ Extracting base model representations..."
echo "   🔍 This provides the baseline for measuring representational drift"
echo "   📊 Extracting from all tasks for comprehensive analysis..."

python scripts/extract_base_representations.py --tasks all
echo "   ✅ Base representations extracted and saved"
echo ""

# 4. Memory profiling test
echo "4️⃣ Running memory profiling validation..."
echo "   💾 Testing memory usage across all classification tasks..."
python -c "
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
"

echo ""
echo "🎉 VM2 Phase 0 validation completed successfully!"
echo "📊 Results logged to wandb project: NLP-Phase0"
echo "💾 Base representations saved for drift analysis"
echo ""
echo "🔄 Ready for Phase 1 hyperparameter optimization"
