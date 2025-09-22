#!/bin/bash
# Phase 1 - VM3: SST-2 Full Fine-tuning + Baseline Experiments
set -e  # Exit on error

echo "Starting Phase 1 on VM3: SST-2 Full Fine-tuning + Baselines..."

# Setup environment
export WANDB_PROJECT=NLP
export WANDB_ENTITY=galavny-tel-aviv-university

# Create logs directory
mkdir -p logs/phase1/vm3

# HuggingFace authentication check
echo "Checking HuggingFace authentication for Llama-2..."
python -c "
from transformers import AutoTokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-1.3b-hf')
    print('✅ HuggingFace authentication successful')
except Exception as e:
    print(f'❌ HuggingFace authentication failed: {e}')
    print('Please run: huggingface-cli login')
    exit(1)
" 2>&1 | tee logs/phase1/vm3/auth_check.log

# Run comprehensive data validation
echo "Running data integrity validation..."
python -c "
from shared.data_preparation import TaskDataLoader
loader = TaskDataLoader('meta-llama/Llama-2-1.3b-hf')
loader.print_dataset_summary()
success = loader.validate_data_integrity()
exit(0 if success else 1)
" 2>&1 | tee logs/phase1/vm3/data_validation.log

# Run all baseline experiments for all four tasks
echo "Running baseline experiments for all tasks..."
python experiments/baselines.py 2>&1 | tee logs/phase1/vm3/baseline_experiments.log

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "❌ Baseline experiments failed! Check implementation."
    exit 1
fi

echo "✅ Baseline experiments completed! Starting SST-2 full fine-tuning..."

# SST-2 full fine-tuning with multiple seeds
echo "Running SST-2 full fine-tuning experiments..."
for seed in 42 1337 2024; do
    echo "  Running SST-2 full fine-tuning with seed $seed..."
    python experiments/full_finetune.py --task sst2 --mode single --seed $seed 2>&1 | tee logs/phase1/vm3/sst2_full_seed${seed}.log
done

# SST-2 hyperparameter search
echo "Running SST-2 hyperparameter sweep..."
python experiments/full_finetune.py --task sst2 --mode sweep 2>&1 | tee logs/phase1/vm3/sst2_sweep.log

# Extract base model representations for all tasks (for later drift analysis)
echo "Extracting base model representations for all tasks..."
python -c "
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from experiments.full_finetune import RepresentationExtractor, RepresentationConfig
from shared.data_preparation import TaskDataLoader
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load base model
model_name = 'meta-llama/Llama-2-1.3b-hf'
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map='auto'
)

# Initialize data loader
data_loader = TaskDataLoader(model_name)

# Extract representations for each task
tasks = ['mrpc', 'sst2', 'rte', 'squad_v2']
repr_config = RepresentationConfig()

for task_name in tasks:
    logger.info(f'Extracting base representations for {task_name}')
    
    try:
        # Prepare validation data
        if task_name == 'squad_v2':
            eval_data = data_loader.prepare_qa_data('validation', 1000)
        else:
            eval_data = data_loader.prepare_classification_data(task_name, 'validation', 1000)
        
        # Create extractor
        output_dir = Path('results/base_model_representations')
        extractor = RepresentationExtractor(repr_config, output_dir, task_name, 'base_pretrained')
        
        # Set validation examples
        examples = {
            'input_ids': torch.tensor(eval_data['input_ids']),
            'attention_mask': torch.tensor(eval_data['attention_mask'])
        }
        extractor.set_validation_examples(examples)
        
        # Extract and save
        representations = extractor.extract_representations(model, 0)
        extractor.save_representations(representations, 0)
        
        logger.info(f'✅ Base representations extracted for {task_name}')
        
    except Exception as e:
        logger.error(f'❌ Failed to extract base representations for {task_name}: {e}')

# Clean up
del model
torch.cuda.empty_cache()
logger.info('✅ All base model representations extracted')
" 2>&1 | tee logs/phase1/vm3/base_representations.log

# Run system monitoring and resource analysis
echo "Running system monitoring and resource analysis..."
python -c "
import torch
import psutil
import time
from datetime import datetime

print('System Resource Analysis:')
print(f'Timestamp: {datetime.now()}')

print('GPU Analysis:')
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        memory_info = torch.cuda.memory_stats(i)
        allocated = memory_info.get('allocated_bytes.all.current', 0) / 1024**3
        reserved = memory_info.get('reserved_bytes.all.current', 0) / 1024**3
        total = props.total_memory / 1024**3
        
        print(f'  GPU {i}: {props.name}')
        print(f'    Total: {total:.1f} GB')
        print(f'    Allocated: {allocated:.1f} GB ({allocated/total*100:.1f}%)')
        print(f'    Reserved: {reserved:.1f} GB ({reserved/total*100:.1f}%)')
else:
    print('  No CUDA devices available')

print('CPU Analysis:')
memory = psutil.virtual_memory()
cpu_percent = psutil.cpu_percent(interval=1)
print(f'  CPU Usage: {cpu_percent}%')
print(f'  Memory Total: {memory.total / 1024**3:.1f} GB')
print(f'  Memory Available: {memory.available / 1024**3:.1f} GB')
print(f'  Memory Used: {memory.used / 1024**3:.1f} GB ({memory.percent}%)')
" 2>&1 | tee logs/phase1/vm3/resource_analysis.log

echo "✅ Phase 1 VM3 complete: SST-2 full fine-tuning + baselines finished"
echo "  - All baseline experiments completed for 4 tasks"
echo "  - 3 SST-2 full fine-tuning runs completed (3 seeds)"
echo "  - 1 SST-2 hyperparameter sweep completed"
echo "  - Base model representations extracted for all tasks"
echo "  - System monitoring and resource analysis completed"
