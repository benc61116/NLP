#!/bin/bash
# Phase 1 - VM3: Analysis & Baselines (All Baselines + Base Model Representations)
set -e  # Exit on error

echo "Starting Phase 1 on VM3: Analysis & Baselines (All Baselines + Base Model Representations)..."

# Setup environment
export WANDB_PROJECT=NLP-Phase1-Training
export WANDB_ENTITY=galavny-tel-aviv-university

# Clear GPU memory cache to ensure maximum available memory
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

# Create logs directory
mkdir -p logs/phase1/vm3

# HuggingFace authentication check
echo "Checking TinyLlama model access..."
python -c "
from transformers import AutoTokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained('TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T')
    print('✅ TinyLlama model accessible')
except Exception as e:
    print(f'❌ TinyLlama access failed: {e}')
    print('Please check internet connection')
    exit(1)
" 2>&1 | tee logs/phase1/vm3/auth_check.log

echo "✅ TinyLlama model check passed! Starting Phase 1 experiments..."
echo "📅 Started at: $(date)"
echo ""

# Baseline experiments for all tasks
echo "🔬 [1/5] Running All Baseline Experiments"

echo "  ⚡ $(date +'%H:%M') - Running MRPC baselines..."
python experiments/baselines.py --task mrpc --baseline majority > logs/phase1/vm3/mrpc_majority.log 2>&1
python experiments/baselines.py --task mrpc --baseline random > logs/phase1/vm3/mrpc_random.log 2>&1
python experiments/baselines.py --task mrpc --baseline sota > logs/phase1/vm3/mrpc_sota.log 2>&1
echo "  ✅ $(date +'%H:%M') - MRPC baselines complete"

echo "  ⚡ $(date +'%H:%M') - Running SST-2 baselines..."
python experiments/baselines.py --task sst2 --baseline majority > logs/phase1/vm3/sst2_majority.log 2>&1
python experiments/baselines.py --task sst2 --baseline random > logs/phase1/vm3/sst2_random.log 2>&1
python experiments/baselines.py --task sst2 --baseline sota > logs/phase1/vm3/sst2_sota.log 2>&1
echo "  ✅ $(date +'%H:%M') - SST-2 baselines complete"

echo "  ⚡ $(date +'%H:%M') - Running RTE baselines..."
python experiments/baselines.py --task rte --baseline majority > logs/phase1/vm3/rte_majority.log 2>&1
python experiments/baselines.py --task rte --baseline random > logs/phase1/vm3/rte_random.log 2>&1
python experiments/baselines.py --task rte --baseline sota > logs/phase1/vm3/rte_sota.log 2>&1
echo "  ✅ $(date +'%H:%M') - RTE baselines complete"

echo "  ⚡ $(date +'%H:%M') - Running SQuAD v2 baselines..."
python experiments/baselines.py --task squad_v2 --baseline majority > logs/phase1/vm3/squad_v2_majority.log 2>&1
python experiments/baselines.py --task squad_v2 --baseline random > logs/phase1/vm3/squad_v2_random.log 2>&1
python experiments/baselines.py --task squad_v2 --baseline sota > logs/phase1/vm3/squad_v2_sota.log 2>&1
echo "  ✅ $(date +'%H:%M') - SQuAD v2 baselines complete"

echo "🎯 [1/5] All Baseline Experiments COMPLETE"
echo ""

# Base model representation extraction for drift analysis
echo "🔬 [2/5] Base Model Representation Extraction"

echo "  ⚡ $(date +'%H:%M') - Extracting base representations for all tasks..."
python scripts/extract_base_representations.py > logs/phase1/vm3/base_repr_all.log 2>&1
echo "  ✅ $(date +'%H:%M') - All base representations complete"

echo "🎯 [2/5] Base Model Representation Extraction COMPLETE"

echo ""
echo "🎉 VM3 PHASE 1 COMPLETE! $(date)"
echo "📊 W&B Dashboard: https://wandb.ai/galavny-tel-aviv-university/NLP-Phase1-Training"
echo "⏳ Ready for Phase 2a when all VMs complete"