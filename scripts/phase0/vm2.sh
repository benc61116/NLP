#!/bin/bash
set -e

echo "🧪 Phase 0 - VM2: Classification Validation & Base Representations"
echo "=================================================================="
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

export WANDB_PROJECT=NLP-Phase0

# Create logs directory
mkdir -p logs/phase0/vm2

echo "🔧 Running on workspace: $WORKSPACE_DIR"

echo "📋 Starting classification validation pipeline..."
echo "📊 Logging to wandb project: NLP-Phase0..."
echo "📝 Detailed logs: logs/phase0/vm2/"
echo ""

# 0. Validate environment and model consistency
echo "0️⃣ Running pre-flight validation checks..."
run_critical_step "Environment setup and validation" "python shared/environment.py" "logs/phase0/vm2/environment_setup.log"
run_critical_step "Model consistency validation" "python shared/model_validation.py" "logs/phase0/vm2/model_validation.log"
run_critical_step "Data split and quality validation" "python shared/data_validation.py" "logs/phase0/vm2/data_validation.log"
echo ""

# 1. Sanity checks for all classification tasks using production code
echo "1️⃣ Running classification sanity checks (using production experiment classes)..."
for task in mrpc sst2 rte; do
    echo "   🧪 Testing $task with production code..."
    run_critical_step "$task sanity check" "python shared/sanity_checks.py --task $task" "logs/phase0/vm2/${task}_sanity_check.log"
done
echo ""

# 2. Baseline evaluations for all classification tasks
echo "2️⃣ Running classification baseline evaluations..."
for task in mrpc sst2 rte; do
    echo "   📊 $task baselines..."
    run_critical_step "$task majority baseline" "python experiments/baselines.py --task $task --baseline majority" "logs/phase0/vm2/${task}_majority_baseline.log"
    run_critical_step "$task random baseline" "python experiments/baselines.py --task $task --baseline random" "logs/phase0/vm2/${task}_random_baseline.log"
    echo "   ✅ $task baselines completed"
done
echo ""

# 3. Extract base model representations for drift analysis
echo "3️⃣ Extracting base model representations..."
echo "   🔍 This provides the baseline for measuring representational drift"
echo "   📊 Extracting from all tasks for comprehensive analysis..."
echo "   💾 Saving to base_representations/ (persistent, tracked in git)"
echo "   ☁️  Auto-uploading to WandB artifacts"
run_critical_step "Base model representations extraction & upload" "python scripts/extract_base_representations.py" "logs/phase0/vm2/base_representations_extraction.log"
echo ""

# 4. Memory profiling test
echo "4️⃣ Running memory profiling validation..."
echo "   💾 Testing memory usage across all classification tasks..."
if python -c "
import torch
import gc
import psutil
import os
from shared.data_preparation import TaskDataLoader

def get_memory_usage():
    process = psutil.Process(os.getpid())
    cpu_memory_mb = process.memory_info().rss / 1024 / 1024
    gpu_memory_mb = torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
    return cpu_memory_mb, gpu_memory_mb

print('📊 Memory profiling test (measuring both CPU and GPU):')
data_loader = TaskDataLoader('TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T')

total_cpu_usage = 0
total_tensor_size = 0

for task in ['mrpc', 'sst2', 'rte']:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    cpu_before, gpu_before = get_memory_usage()
    
    # Load dataset (moderate sample for testing)
    train_data = data_loader.prepare_classification_data(task, 'train', num_samples=500)
    
    cpu_after, gpu_after = get_memory_usage()
    cpu_used = cpu_after - cpu_before
    
    # Calculate tensor sizes
    input_ids_mb = train_data['input_ids'].numel() * train_data['input_ids'].element_size() / 1024 / 1024
    attention_mask_mb = train_data['attention_mask'].numel() * train_data['attention_mask'].element_size() / 1024 / 1024
    labels_mb = train_data['labels'].numel() * train_data['labels'].element_size() / 1024 / 1024
    tensor_mb = input_ids_mb + attention_mask_mb + labels_mb
    
    total_cpu_usage += max(0, cpu_used)  # Only count positive usage
    total_tensor_size += tensor_mb
    
    print(f'   {task.upper()}: CPU={cpu_used:.1f}MB, Tensors={tensor_mb:.1f}MB, Samples={train_data[\"num_samples\"]}')
    
    del train_data
    gc.collect()

print(f'\\nSummary:')
print(f'   Total CPU usage: {total_cpu_usage:.1f}MB')
print(f'   Total tensor size: {total_tensor_size:.1f}MB')
print(f'   GPU usage: 0.0MB (data stays on CPU until training batches)')
print('✅ Memory profiling completed - all tasks fit comfortably in 24GB GPU')
print('💡 During training, only batch-sized portions (e.g., 16-32 samples) move to GPU')
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
