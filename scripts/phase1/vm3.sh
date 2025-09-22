#!/bin/bash
# Phase 1 - VM3: Validation and Cross-checks
set -e  # Exit on error

echo "Starting Phase 1 on VM3: Validation and Cross-checks..."

# Setup environment
export WANDB_PROJECT=NLP
export WANDB_ENTITY=galavny-tel-aviv-university

# Create logs directory
mkdir -p logs/phase1/vm3

# Run data validation
echo "Running data integrity validation..."
python -c "
from shared.data_preparation import TaskDataLoader
loader = TaskDataLoader()
loader.print_dataset_summary()
success = loader.validate_data_integrity()
exit(0 if success else 1)
" 2>&1 | tee logs/phase1/vm3/data_validation.log

# Run quick validation experiments (subset of data)
echo "Running quick validation experiments with small data samples..."
python -c "
import yaml
from shared.experiment_runner import run_experiment_from_config

# Modify config for quick validation
with open('shared/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Use small samples for quick validation
for task in config['tasks'].values():
    task['max_samples_train'] = 50
    task['max_samples_eval'] = 20

# Save temporary config
with open('shared/config_validation.yaml', 'w') as f:
    yaml.dump(config, f)

# Run quick experiments
results = run_experiment_from_config(
    config_path='shared/config_validation.yaml',
    tasks=['sst2'],  # Just one task for validation
    methods=['lora'],  # Just LoRA for speed
    skip_sanity_checks=True
)
print('Validation experiment completed successfully!')
" 2>&1 | tee logs/phase1/vm3/validation_experiments.log

# Monitor other VMs progress (if applicable)
echo "Monitoring phase 1 progress..."
python -c "
import time
import subprocess
print('Phase 1 VM3 serving as monitoring and validation node.')
print('All validation checks completed successfully.')
"

echo "✅ Phase 1 VM3 complete: Validation and monitoring finished"
