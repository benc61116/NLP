#!/bin/bash
# Phase 2a - VM3: SST-2 Drift Analysis + Visualization Preparation (Balanced Load)
set -e  # Exit on error

echo "Starting Phase 2a on VM3: SST-2 Drift Analysis + Visualization Preparation..."

# Setup environment
export WANDB_PROJECT=NLP
export WANDB_ENTITY=galavny-tel-aviv-university

# Create logs directory
mkdir -p logs/phase2a/vm3

# Check that Phase 1 is complete
echo "Verifying Phase 1 completion..."
python -c "
import wandb
import sys

api = wandb.Api()
runs = list(api.runs('galavny-tel-aviv-university/NLP'))

# Check for required Phase 1 runs
required_runs = ['sst2_full_finetune', 'sst2_lora']
completed_runs = [run.name for run in runs if run.state == 'finished']

missing_runs = [run for run in required_runs if run not in completed_runs]
if missing_runs:
    print(f'❌ Phase 1 not complete. Missing runs: {missing_runs}')
    sys.exit(1)
else:
    print('✅ Phase 1 verification passed')
" 2>&1 | tee logs/phase2a/vm3/phase1_verification.log

# SST-2 drift analysis
echo "Running SST-2 representational drift analysis..."
python experiments/drift_analysis.py --task sst2 --comparison_tasks sst2 --analysis_type classification 2>&1 | tee logs/phase2a/vm3/sst2_drift_analysis.log

# Visualization preparation
echo "Preparing visualization data for all experiments..."
python -c "
import wandb
import pandas as pd
import json
import numpy as np
from pathlib import Path

# Initialize wandb API
api = wandb.Api()
runs = list(api.runs('galavny-tel-aviv-university/NLP'))

# Collect all experimental data for visualization
visualization_data = {
    'experiments': {},
    'summary_stats': {},
    'drift_metrics': {},
    'performance_metrics': {}
}

# Process completed runs
for run in runs:
    if run.state == 'finished':
        experiment_info = {
            'id': run.id,
            'name': run.name,
            'config': dict(run.config),
            'summary': dict(run.summary),
            'created_at': run.created_at.isoformat() if run.created_at else None,
            'runtime': run.summary.get('train_runtime', 0)
        }
        visualization_data['experiments'][run.id] = experiment_info
        
        # Extract performance metrics
        if any(task in run.name for task in ['mrpc', 'sst2', 'rte', 'squad_v2']):
            task_name = next(task for task in ['mrpc', 'sst2', 'rte', 'squad_v2'] if task in run.name)
            method = 'full_ft' if 'full' in run.name else 'lora'
            
            if task_name not in visualization_data['performance_metrics']:
                visualization_data['performance_metrics'][task_name] = {}
            if method not in visualization_data['performance_metrics'][task_name]:
                visualization_data['performance_metrics'][task_name][method] = []
                
            metrics = {
                'accuracy': run.summary.get('eval_accuracy', 0),
                'loss': run.summary.get('eval_loss', float('inf')),
                'f1': run.summary.get('eval_f1', 0),
                'runtime': run.summary.get('train_runtime', 0)
            }
            visualization_data['performance_metrics'][task_name][method].append(metrics)

# Calculate summary statistics
tasks = ['mrpc', 'sst2', 'rte', 'squad_v2']
methods = ['full_ft', 'lora']

for task in tasks:
    if task in visualization_data['performance_metrics']:
        visualization_data['summary_stats'][task] = {}
        for method in methods:
            if method in visualization_data['performance_metrics'][task]:
                metrics_list = visualization_data['performance_metrics'][task][method]
                if metrics_list:
                    accuracies = [m['accuracy'] for m in metrics_list]
                    visualization_data['summary_stats'][task][method] = {
                        'mean_accuracy': np.mean(accuracies),
                        'std_accuracy': np.std(accuracies),
                        'count': len(accuracies)
                    }

# Save visualization data
output_dir = Path('results/visualization_prep')
output_dir.mkdir(parents=True, exist_ok=True)

with open(output_dir / 'experiment_data.json', 'w') as f:
    json.dump(visualization_data, f, indent=2, default=str)

# Prepare plot configurations
plot_configs = {
    'performance_comparison': {
        'type': 'bar_chart',
        'title': 'LoRA vs Full Fine-tuning Performance Comparison',
        'x_axis': 'Tasks',
        'y_axis': 'Accuracy',
        'data_source': 'summary_stats'
    },
    'drift_analysis_plots': {
        'type': 'line_plot',
        'title': 'Representational Drift by Layer',
        'x_axis': 'Transformer Layer',
        'y_axis': 'CKA Distance',
        'data_source': 'drift_metrics'
    },
    'deployment_overhead': {
        'type': 'scatter_plot',
        'title': 'Deployment Overhead vs Performance',
        'x_axis': 'Inference Latency (ms)',
        'y_axis': 'Model Accuracy',
        'data_source': 'deployment_analysis'
    }
}

with open(output_dir / 'plot_configs.json', 'w') as f:
    json.dump(plot_configs, f, indent=2)

print(f'✅ Visualization data prepared in {output_dir}')
print(f'  - {len(visualization_data[\"experiments\"])} experiments processed')
print(f'  - Summary statistics calculated for {len(visualization_data[\"summary_stats\"])} tasks')
print(f'  - Plot configurations saved for Phase 2b')
" 2>&1 | tee logs/phase2a/vm3/visualization_prep.log

# Generate preliminary figures for validation
echo "Generating preliminary validation figures..."
python -c "
import matplotlib.pyplot as plt
import json
import numpy as np
from pathlib import Path

# Load visualization data
with open('results/visualization_prep/experiment_data.json', 'r') as f:
    viz_data = json.load(f)

# Create a simple performance comparison plot
fig, ax = plt.subplots(figsize=(10, 6))

tasks = []
full_ft_scores = []
lora_scores = []

for task, stats in viz_data['summary_stats'].items():
    if 'full_ft' in stats and 'lora' in stats:
        tasks.append(task.upper())
        full_ft_scores.append(stats['full_ft']['mean_accuracy'])
        lora_scores.append(stats['lora']['mean_accuracy'])

if tasks:
    x = np.arange(len(tasks))
    width = 0.35
    
    ax.bar(x - width/2, full_ft_scores, width, label='Full Fine-tuning', alpha=0.8)
    ax.bar(x + width/2, lora_scores, width, label='LoRA', alpha=0.8)
    
    ax.set_xlabel('Tasks')
    ax.set_ylabel('Mean Accuracy')
    ax.set_title('Preliminary Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(tasks)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Save preliminary plot
    output_dir = Path('results/visualization_prep/preliminary_plots')
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / 'performance_comparison_preview.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f'✅ Preliminary performance plot saved: {output_dir}/performance_comparison_preview.png')
else:
    print('⚠️ No data available for preliminary plots yet')
" 2>&1 | tee logs/phase2a/vm3/preliminary_plots.log

echo "✅ Phase 2a VM3 complete: SST-2 drift analysis + visualization preparation finished"
echo "  - SST-2 representational drift analysis completed"
echo "  - Visualization data prepared for all experiments"
echo "  - Plot configurations set up for Phase 2b"
echo "  - Preliminary validation figures generated"
echo "  - Ready for Phase 2b statistical synthesis"
