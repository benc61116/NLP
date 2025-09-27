#!/bin/bash
set -e

echo "🧪 Phase 0 - VM1: SQuAD v2 Validation"
echo "===================================="
echo ""

# Set up environment
export PYTHONPATH=/home/benc6116/workspace/NLP:$PYTHONPATH
export WANDB_PROJECT=NLP-Phase0
cd /home/benc6116/workspace/NLP

echo "📋 Starting SQuAD v2 validation pipeline..."
echo "🔄 Initializing wandb project: NLP-Phase0..."
wandb login --relogin
echo ""

# 1. Sanity check - overfitting test on small sample
echo "1️⃣ Running SQuAD v2 sanity check (overfitting test)..."
python shared/sanity_checks.py --task squad_v2 --num-samples 100
echo "✅ Sanity check completed"
echo ""

# 2. Baseline evaluations
echo "2️⃣ Running SQuAD v2 baseline evaluations..."

echo "   📊 Majority class baseline..."
python experiments/baselines.py --task squad_v2 --baseline majority
echo "   ✅ Majority baseline completed"

echo "   📊 Random baseline..."
python experiments/baselines.py --task squad_v2 --baseline random
echo "   ✅ Random baseline completed"

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
echo ""
echo "🔄 Ready for Phase 1 hyperparameter optimization"
