#!/bin/bash
# Phase 2a - VM1: MRPC + RTE Drift Analysis + Correlation Analysis Prep (Balanced Load)
set -e  # Exit on error

echo "Starting Phase 2a on VM1: MRPC + RTE Drift Analysis + Correlation Analysis Prep..."

# Setup environment
export WANDB_PROJECT=NLP-Phase2-Analysis
export WANDB_ENTITY=galavny-tel-aviv-university

# Create logs directory
mkdir -p logs/phase2a/vm1

# Check that Phase 1 is complete
echo "Verifying Phase 1 completion..."
python -c "
import wandb
import sys

api = wandb.Api()
runs = list(api.runs('galavny-tel-aviv-university/NLP'))

# Check for required Phase 1 runs
required_runs = ['mrpc_full_finetune', 'mrpc_lora', 'rte_full_finetune', 'rte_lora']
completed_runs = [run.name for run in runs if run.state == 'finished']

missing_runs = [run for run in required_runs if run not in completed_runs]
if missing_runs:
    print(f'❌ Phase 1 not complete. Missing runs: {missing_runs}')
    sys.exit(1)
else:
    print('✅ Phase 1 verification passed')
" 2>&1 | tee logs/phase2a/vm1/phase1_verification.log

# MRPC drift analysis
echo "Running MRPC representational drift analysis..."
python experiments/drift_analysis.py --task mrpc --comparison_tasks mrpc --analysis_type classification 2>&1 | tee logs/phase2a/vm1/mrpc_drift_analysis.log

# RTE drift analysis
echo "Running RTE representational drift analysis..."
python experiments/drift_analysis.py --task rte --comparison_tasks rte --analysis_type classification 2>&1 | tee logs/phase2a/vm1/rte_drift_analysis.log

# Correlation analysis preparation
echo "Preparing correlation analysis data..."
python -c "
import pandas as pd
import json
import wandb
from pathlib import Path

# Load experimental results for correlation analysis
api = wandb.Api()
runs = list(api.runs('galavny-tel-aviv-university/NLP'))

# Prepare correlation data structure
correlation_data = {
    'mrpc': {'full_ft': {}, 'lora': {}},
    'rte': {'full_ft': {}, 'lora': {}}
}

# Extract performance metrics
for run in runs:
    if run.state == 'finished' and any(task in run.name for task in ['mrpc', 'rte']):
        task_name = 'mrpc' if 'mrpc' in run.name else 'rte'
        method = 'full_ft' if 'full' in run.name else 'lora'
        
        correlation_data[task_name][method][run.id] = {
            'accuracy': run.summary.get('eval_accuracy', 0),
            'loss': run.summary.get('eval_loss', float('inf')),
            'runtime': run.summary.get('train_runtime', 0)
        }

# Save for Phase 2b analysis
output_file = 'results/correlation_prep/mrpc_rte_performance_data.json'
Path('results/correlation_prep').mkdir(parents=True, exist_ok=True)
with open(output_file, 'w') as f:
    json.dump(correlation_data, f, indent=2)

print(f'✅ Correlation analysis data prepared: {output_file}')
" 2>&1 | tee logs/phase2a/vm1/correlation_prep.log

echo "✅ Phase 2a VM1 complete: MRPC + RTE drift analysis + correlation prep finished"
echo "  - MRPC representational drift analysis completed"
echo "  - RTE representational drift analysis completed" 
echo "  - Correlation analysis data prepared for Phase 2b"
echo "  - Ready for Phase 2b statistical synthesis"
