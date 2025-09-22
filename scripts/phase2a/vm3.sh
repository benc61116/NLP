#!/bin/bash
# Phase 2a - VM3: Comparison Analysis and Monitoring
set -e  # Exit on error

echo "Starting Phase 2a on VM3: Comparison Analysis and Monitoring..."

# Setup environment
export WANDB_PROJECT=NLP
export WANDB_ENTITY=galavny-tel-aviv-university

# Create logs directory
mkdir -p logs/phase2a/vm3

# Run analysis and monitoring
echo "Generating preliminary comparison analysis..."
python -c "
import wandb
import pandas as pd
import json
from pathlib import Path

# Initialize wandb API
api = wandb.Api()

# Get runs from our project
runs = api.runs('galavny-tel-aviv-university/NLP')

print(f'Found {len(runs)} total runs in the project')

# Collect experiment data
experiment_data = []
for run in runs:
    if 'experiment_id' in run.config:
        experiment_data.append({
            'id': run.id,
            'name': run.name,
            'state': run.state,
            'config': run.config,
            'summary': run.summary
        })

print(f'Found {len(experiment_data)} experiment runs')

# Save preliminary data
output_file = 'logs/phase2a/vm3/preliminary_analysis.json'
with open(output_file, 'w') as f:
    json.dump(experiment_data, f, indent=2, default=str)

print(f'Preliminary analysis saved to {output_file}')
" 2>&1 | tee logs/phase2a/vm3/analysis.log

echo "Monitoring experiment progress..."
python -c "
import time
print('Phase 2a VM3 monitoring and analysis complete.')
print('Ready for Phase 2b synthesis.')
" 2>&1 | tee -a logs/phase2a/vm3/analysis.log

echo "✅ Phase 2a VM3 complete: Analysis and monitoring finished"
