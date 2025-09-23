#!/bin/bash
# Phase 2b - VM1: Final Analysis and Synthesis
set -e  # Exit on error

echo "Starting Phase 2b on VM1: Final Analysis and Synthesis..."

# Setup environment
export WANDB_PROJECT=NLP-Phase2-Analysis
export WANDB_ENTITY=galavny-tel-aviv-university

# Create logs directory
mkdir -p logs/phase2b/vm1

# Generate comprehensive analysis
echo "Generating comprehensive analysis..."
python -c "
import wandb
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

# Initialize wandb API
api = wandb.Api()
runs = api.runs('galavny-tel-aviv-university/NLP')

print('Collecting all experimental results...')

# Collect results
results = []
for run in runs:
    if run.state == 'finished' and 'task_name' in run.config and 'method' in run.config:
        result = {
            'task': run.config.get('task_name'),
            'method': run.config.get('method'),
            'train_loss': run.summary.get('train_loss', None),
            'eval_loss': run.summary.get('eval_loss', None),
            'eval_accuracy': run.summary.get('eval_accuracy', None),
            'eval_f1': run.summary.get('eval_f1', None),
            'runtime': run.summary.get('train_runtime', None),
            'run_id': run.id,
            'run_name': run.name
        }
        results.append(result)

if not results:
    print('No completed experiments found. Results may still be running.')
    exit(0)

# Create DataFrame
df = pd.DataFrame(results)
print(f'Found {len(df)} completed experiment results')

# Basic statistics
print('\nBasic Statistics:')
print(df.groupby(['task', 'method']).agg({
    'eval_loss': ['mean', 'std'],
    'runtime': ['mean', 'std']
}).round(4))

# Save results
output_dir = Path('logs/phase2b/vm1')
output_dir.mkdir(parents=True, exist_ok=True)

df.to_csv(output_dir / 'all_results.csv', index=False)

# Generate comparison plots
plt.figure(figsize=(12, 8))

# Plot 1: Eval loss comparison
plt.subplot(2, 2, 1)
if 'eval_loss' in df.columns and df['eval_loss'].notna().any():
    pivot_loss = df.pivot(index='task', columns='method', values='eval_loss')
    sns.heatmap(pivot_loss, annot=True, fmt='.3f', cmap='viridis_r')
    plt.title('Evaluation Loss by Task and Method')

# Plot 2: Runtime comparison
plt.subplot(2, 2, 2)
if 'runtime' in df.columns and df['runtime'].notna().any():
    pivot_time = df.pivot(index='task', columns='method', values='runtime')
    sns.heatmap(pivot_time, annot=True, fmt='.1f', cmap='plasma')
    plt.title('Training Runtime (seconds)')

# Plot 3: Performance comparison
plt.subplot(2, 2, 3)
metric_col = 'eval_accuracy' if 'eval_accuracy' in df.columns else 'eval_f1'
if metric_col in df.columns and df[metric_col].notna().any():
    df_clean = df.dropna(subset=[metric_col])
    sns.barplot(data=df_clean, x='task', y=metric_col, hue='method')
    plt.title(f'{metric_col.replace(\"_\", \" \").title()} Comparison')
    plt.xticks(rotation=45)

# Plot 4: Method comparison across tasks
plt.subplot(2, 2, 4)
if 'eval_loss' in df.columns and df['eval_loss'].notna().any():
    df_clean = df.dropna(subset=['eval_loss'])
    sns.boxplot(data=df_clean, x='method', y='eval_loss')
    plt.title('Eval Loss Distribution by Method')

plt.tight_layout()
plt.savefig(output_dir / 'comparison_plots.png', dpi=300, bbox_inches='tight')
plt.close()

# Generate summary report
summary = {
    'total_experiments': len(df),
    'tasks_completed': df['task'].nunique(),
    'methods_tested': df['method'].nunique(),
    'avg_performance_by_method': df.groupby('method')['eval_loss'].mean().to_dict(),
    'avg_runtime_by_method': df.groupby('method')['runtime'].mean().to_dict()
}

with open(output_dir / 'experiment_summary.json', 'w') as f:
    json.dump(summary, f, indent=2, default=str)

print(f'\nAnalysis complete! Results saved to {output_dir}')
print(f'- CSV results: all_results.csv')
print(f'- Plots: comparison_plots.png')  
print(f'- Summary: experiment_summary.json')

# Print key findings
print('\n=== KEY FINDINGS ===')
if len(df) > 0:
    best_lora = df[df['method'] == 'lora']['eval_loss'].min() if 'lora' in df['method'].values else 'N/A'
    best_full = df[df['method'] == 'full']['eval_loss'].min() if 'full' in df['method'].values else 'N/A'
    print(f'Best LoRA performance (eval_loss): {best_lora}')
    print(f'Best Full FT performance (eval_loss): {best_full}')
    
    if 'lora' in df['method'].values and 'full' in df['method'].values:
        lora_avg_time = df[df['method'] == 'lora']['runtime'].mean()
        full_avg_time = df[df['method'] == 'full']['runtime'].mean()
        print(f'Average LoRA training time: {lora_avg_time:.1f}s')
        print(f'Average Full FT training time: {full_avg_time:.1f}s')
        if lora_avg_time and full_avg_time:
            speedup = full_avg_time / lora_avg_time
            print(f'LoRA speedup: {speedup:.2f}x')

print('\n=== EXPERIMENT COMPLETE ===')
" 2>&1 | tee logs/phase2b/vm1/final_analysis.log

echo "Generating reproducibility report..."
python -c "
import yaml
import json
from pathlib import Path

# Load config
with open('shared/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Generate reproducibility report
report = {
    'experiment_details': {
        'model': config['model']['name'],
        'tasks': list(config['tasks'].keys()),
        'methods': ['lora', 'full'],
        'seed': config['reproducibility']['seed'],
        'deterministic': config['reproducibility']['deterministic']
    },
    'hardware_requirements': {
        'gpu_memory': '24GB (recommended)',
        'framework_versions': {
            'torch': '2.1.0',
            'transformers': '4.35.0',
            'peft': '0.6.0'
        }
    },
    'reproduction_steps': [
        'pip install -r requirements.txt',
        'python scripts/download_datasets.py',
        'bash scripts/phase1/vm1.sh  # Run sanity checks',
        'bash scripts/phase2a/vm1.sh  # LoRA experiments',
        'bash scripts/phase2a/vm2.sh  # Full FT experiments',
        'bash scripts/phase2b/vm1.sh  # Analysis'
    ]
}

output_file = 'logs/phase2b/vm1/reproducibility_report.json'
with open(output_file, 'w') as f:
    json.dump(report, f, indent=2)

print(f'Reproducibility report saved to {output_file}')
" 2>&1 | tee -a logs/phase2b/vm1/final_analysis.log

echo "✅ Phase 2b VM1 complete: Final analysis and synthesis finished!"
echo "🎉 ALL EXPERIMENTS COMPLETED SUCCESSFULLY!"
echo ""
echo "Results available in:"
echo "  - logs/phase2b/vm1/all_results.csv"
echo "  - logs/phase2b/vm1/comparison_plots.png"
echo "  - logs/phase2b/vm1/experiment_summary.json"
echo "  - W&B Dashboard: https://wandb.ai/galavny-tel-aviv-university/NLP"
