#!/bin/bash
# Phase 2a - VM2: SQuAD v2 Drift Analysis + Deployment Benchmarking (Balanced Load)
set -e  # Exit on error

echo "Starting Phase 2a on VM2: SQuAD v2 Drift Analysis + Deployment Benchmarking..."

# Setup environment
export WANDB_PROJECT=NLP-Phase2-Analysis
export WANDB_ENTITY=galavny-tel-aviv-university

# Create logs directory
mkdir -p logs/phase2a/vm2

# Check that Phase 1 is complete
echo "Verifying Phase 1 completion..."
python -c "
import wandb
import sys

api = wandb.Api()
runs = list(api.runs('galavny-tel-aviv-university/NLP'))

# Check for required Phase 1 runs
required_runs = ['squad_v2_full_finetune', 'squad_v2_lora']
completed_runs = [run.name for run in runs if run.state == 'finished']

missing_runs = [run for run in required_runs if run not in completed_runs]
if missing_runs:
    print(f'❌ Phase 1 not complete. Missing runs: {missing_runs}')
    sys.exit(1)
else:
    print('✅ Phase 1 verification passed')
" 2>&1 | tee logs/phase2a/vm2/phase1_verification.log

# SQuAD v2 drift analysis
echo "Running SQuAD v2 representational drift analysis..."
python experiments/drift_analysis.py --task squad_v2 --comparison_tasks squad_v2 --analysis_type qa 2>&1 | tee logs/phase2a/vm2/squad_v2_drift_analysis.log

# vLLM deployment benchmarking
echo "Running vLLM deployment overhead benchmarking..."
python experiments/deployment_benchmark.py --tasks squad_v2 --test_modes merged,unmerged --batch_sizes 1,4,8,16 2>&1 | tee logs/phase2a/vm2/deployment_benchmark.log

# Memory and performance analysis
echo "Running comprehensive deployment analysis..."
python -c "
import torch
import time
import json
from pathlib import Path
import psutil

# System resource analysis for deployment
deployment_analysis = {
    'system_info': {
        'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'gpu_memory': [],
        'cpu_cores': psutil.cpu_count(),
        'system_memory_gb': psutil.virtual_memory().total / (1024**3)
    },
    'timestamp': time.time()
}

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        deployment_analysis['system_info']['gpu_memory'].append({
            'gpu_id': i,
            'name': props.name,
            'total_memory_gb': props.total_memory / (1024**3)
        })

# Save deployment system info
output_file = 'results/deployment_analysis/system_info.json'
Path('results/deployment_analysis').mkdir(parents=True, exist_ok=True)
with open(output_file, 'w') as f:
    json.dump(deployment_analysis, f, indent=2)

print(f'✅ Deployment system analysis saved: {output_file}')
" 2>&1 | tee logs/phase2a/vm2/deployment_analysis.log

echo "✅ Phase 2a VM2 complete: SQuAD v2 drift analysis + deployment benchmarking finished"
echo "  - SQuAD v2 representational drift analysis completed"
echo "  - vLLM deployment overhead benchmarking completed"
echo "  - System performance analysis for deployment completed"
echo "  - Ready for Phase 2b statistical synthesis"
