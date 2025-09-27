#!/usr/bin/env python3
"""
Hyperparameter Sweep Analysis Tool
Analyzes W&B sweep results to identify optimal hyperparameters for each task/method combination.
This implements the critical missing piece of the sweep-first methodology.
"""

import os
import sys
import argparse
import yaml
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import wandb

def setup_wandb(project: str = "NLP-Phase1-Training", entity: str = "galavny-tel-aviv-university"):
    """Initialize W&B API connection"""
    try:
        wandb.login(key=os.environ.get('WANDB_API_KEY'))
        api = wandb.Api()
        return api
    except Exception as e:
        print(f"❌ Failed to connect to W&B: {e}")
        print("Please ensure WANDB_API_KEY is set or run 'wandb login'")
        sys.exit(1)

def get_sweep_runs(api: wandb.Api, project: str, entity: str, sweep_id: Optional[str] = None) -> Dict[str, List]:
    """Retrieve all sweep runs from W&B"""
    print(f"🔍 Fetching sweep runs from {entity}/{project}...")
    
    if sweep_id:
        # Get specific sweep
        sweep = api.sweep(f"{entity}/{project}/{sweep_id}")
        runs = list(sweep.runs)
        sweeps_data = {sweep_id: runs}
    else:
        # Get all sweeps in project
        sweeps = api.project(f"{entity}/{project}").sweeps()
        sweeps_data = {}
        
        for sweep in sweeps:
            sweep_runs = list(sweep.runs)
            if sweep_runs:  # Only include sweeps with runs
                sweeps_data[sweep.id] = sweep_runs
                print(f"  📊 Found sweep {sweep.id}: {len(sweep_runs)} runs")
    
    total_runs = sum(len(runs) for runs in sweeps_data.values())
    print(f"✅ Retrieved {len(sweeps_data)} sweeps with {total_runs} total runs")
    return sweeps_data

def parse_run_config(run) -> Dict[str, Any]:
    """Extract task, method, and hyperparameters from run"""
    config = {}
    
    # Extract from run config first (preferred), then fall back to name parsing
    run_config = run.config
    
    # Determine task
    if 'task_name' in run_config:
        config['task'] = run_config['task_name']
    else:
        # Fall back to name parsing
        name = run.name.lower()
        if 'mrpc' in name:
            config['task'] = 'mrpc'
        elif 'sst2' in name or 'sst-2' in name:
            config['task'] = 'sst2'
        elif 'rte' in name:
            config['task'] = 'rte'
        elif 'squad' in name:
            config['task'] = 'squad_v2'
        else:
            config['task'] = 'unknown'
    
    # Determine method
    if 'method' in run_config:
        config['method'] = run_config['method']
    else:
        # Fall back to name parsing
        name = run.name.lower()
        if 'lora' in name:
            config['method'] = 'lora'
        elif any(x in name for x in ['full', 'ft', 'finetune']):
            config['method'] = 'full_finetune'
        else:
            config['method'] = 'unknown'
    
    # Extract hyperparameters from run config
    
    # Common hyperparameters
    config['learning_rate'] = run_config.get('learning_rate', None)
    config['per_device_train_batch_size'] = run_config.get('per_device_train_batch_size', None)
    config['warmup_ratio'] = run_config.get('warmup_ratio', None)
    config['num_train_epochs'] = run_config.get('num_train_epochs', 3)
    config['seed'] = run_config.get('seed', None)
    
    # LoRA-specific parameters
    if config['method'] == 'lora':
        config['lora_r'] = run_config.get('lora_r', run_config.get('r', None))
        config['lora_alpha'] = run_config.get('lora_alpha', run_config.get('alpha', None))
        config['lora_dropout'] = run_config.get('lora_dropout', run_config.get('dropout', None))
    
    return config

def extract_run_metrics(run) -> Dict[str, float]:
    """Extract final metrics from run"""
    metrics = {}
    
    # Get final metrics from run summary
    summary = run.summary
    
    # Common metrics
    metrics['final_eval_loss'] = summary.get('eval_loss', summary.get('final_eval_loss', float('inf')))
    metrics['final_eval_accuracy'] = summary.get('eval_accuracy', summary.get('final_eval_accuracy', 0.0))
    metrics['final_eval_f1'] = summary.get('eval_f1', summary.get('eval_f1_binary', summary.get('final_eval_f1', 0.0)))
    
    # Training metrics
    metrics['train_loss'] = summary.get('train_loss', float('inf'))
    metrics['train_runtime'] = summary.get('train_runtime', 0.0)
    
    # Run status
    metrics['state'] = run.state
    metrics['duration'] = (run.summary.get('_runtime', 0) or 0)
    
    return metrics

def analyze_task_method_combination(runs: List, task: str, method: str) -> Dict[str, Any]:
    """Analyze all runs for a specific task/method combination"""
    print(f"📈 Analyzing {task} + {method}...")
    
    # Filter runs for this task/method
    filtered_runs = []
    for run in runs:
        config = parse_run_config(run)
        if config['task'] == task and config['method'] == method:
            metrics = extract_run_metrics(run)
            if metrics['state'] == 'finished':  # Only successful runs
                filtered_runs.append({
                    'run': run,
                    'config': config,
                    'metrics': metrics
                })
    
    if not filtered_runs:
        print(f"  ⚠️  No successful runs found for {task} + {method}")
        return None
    
    print(f"  📊 Found {len(filtered_runs)} successful runs")
    
    # Determine best metric for this task
    if task == 'squad_v2':
        primary_metric = 'final_eval_f1'
        higher_is_better = True
    else:
        primary_metric = 'final_eval_accuracy'  
        higher_is_better = True
    
    # Find best run based on primary metric
    best_run_data = None
    best_metric_value = float('-inf') if higher_is_better else float('inf')
    
    for run_data in filtered_runs:
        metric_value = run_data['metrics'][primary_metric]
        
        if higher_is_better and metric_value > best_metric_value:
            best_metric_value = metric_value
            best_run_data = run_data
        elif not higher_is_better and metric_value < best_metric_value:
            best_metric_value = metric_value
            best_run_data = run_data
    
    if not best_run_data:
        print(f"  ❌ Could not identify best run for {task} + {method}")
        return None
    
    # Extract optimal hyperparameters
    optimal_config = best_run_data['config']
    best_metrics = best_run_data['metrics']
    
    print(f"  ✅ Best run: {best_run_data['run'].name}")
    print(f"     {primary_metric}: {best_metric_value:.4f}")
    print(f"     Learning rate: {optimal_config['learning_rate']}")
    print(f"     Batch size: {optimal_config['per_device_train_batch_size']}")
    print(f"     Warmup ratio: {optimal_config['warmup_ratio']}")
    
    return {
        'task': task,
        'method': method,
        'best_run_id': best_run_data['run'].id,
        'best_run_name': best_run_data['run'].name,
        'optimal_hyperparameters': {
            'learning_rate': optimal_config['learning_rate'],
            'per_device_train_batch_size': optimal_config['per_device_train_batch_size'],
            'warmup_ratio': optimal_config['warmup_ratio'],
            'num_train_epochs': optimal_config['num_train_epochs'],
            **({
                'lora_r': optimal_config['lora_r'],
                'lora_alpha': optimal_config['lora_alpha'],
                'lora_dropout': optimal_config['lora_dropout']
            } if method == 'lora' else {})
        },
        'best_metrics': {
            'eval_loss': best_metrics['final_eval_loss'],
            'eval_accuracy': best_metrics['final_eval_accuracy'],
            'eval_f1': best_metrics['final_eval_f1'],
            'train_loss': best_metrics['train_loss'],
            'train_runtime': best_metrics['train_runtime']
        },
        'num_runs_analyzed': len(filtered_runs),
        'metric_used': primary_metric,
        'metric_value': best_metric_value
    }

def generate_optimal_config(analysis_results: List[Dict]) -> Dict[str, Any]:
    """Generate optimal_hyperparameters.yaml from analysis results"""
    config = {
        'metadata': {
            'generated_at': pd.Timestamp.now().isoformat(),
            'description': 'Optimal hyperparameters identified from W&B sweep analysis',
            'methodology': 'Best performing configuration per task/method combination'
        },
        'optimal_hyperparameters': {}
    }
    
    for result in analysis_results:
        if result is None:
            continue
            
        task = result['task']
        method = result['method']
        
        # Create nested structure
        if task not in config['optimal_hyperparameters']:
            config['optimal_hyperparameters'][task] = {}
        
        config['optimal_hyperparameters'][task][method] = {
            'hyperparameters': result['optimal_hyperparameters'],
            'expected_performance': result['best_metrics'],
            'source_run': result['best_run_name'],
            'runs_analyzed': result['num_runs_analyzed'],
            'selection_metric': f"{result['metric_used']}={result['metric_value']:.4f}"
        }
    
    return config

def main():
    parser = argparse.ArgumentParser(description='Analyze hyperparameter sweep results')
    parser.add_argument('--project', default='NLP-Phase1-Training', help='W&B project name')
    parser.add_argument('--entity', default='galavny-tel-aviv-university', help='W&B entity name')
    parser.add_argument('--sweep-id', help='Specific sweep ID to analyze (optional)')
    parser.add_argument('--export-optimal-configs', action='store_true', 
                       help='Export optimal hyperparameters to YAML file')
    parser.add_argument('--output-dir', default='./analysis', help='Output directory for results')
    parser.add_argument('--tasks', nargs='+', default=['mrpc', 'sst2', 'rte', 'squad_v2'],
                       help='Tasks to analyze')
    parser.add_argument('--methods', nargs='+', default=['full_finetune', 'lora'],
                       help='Methods to analyze')
    
    args = parser.parse_args()
    
    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("🚀 Hyperparameter Sweep Analysis")
    print("=" * 50)
    
    # Connect to W&B
    api = setup_wandb(args.project, args.entity)
    
    # Get sweep runs
    sweeps_data = get_sweep_runs(api, args.project, args.entity, args.sweep_id)
    
    if not sweeps_data:
        print("❌ No sweeps found!")
        sys.exit(1)
    
    # Combine all runs from all sweeps
    all_runs = []
    for sweep_id, runs in sweeps_data.items():
        all_runs.extend(runs)
    
    print(f"\n🔬 Analyzing {len(all_runs)} total runs...")
    print("=" * 50)
    
    # Analyze each task/method combination
    analysis_results = []
    for task in args.tasks:
        for method in args.methods:
            result = analyze_task_method_combination(all_runs, task, method)
            if result:
                analysis_results.append(result)
    
    # Export results
    if args.export_optimal_configs and analysis_results:
        print(f"\n📄 Exporting optimal configurations...")
        
        # Generate optimal config
        optimal_config = generate_optimal_config(analysis_results)
        
        # Save to YAML
        config_path = output_dir / 'optimal_hyperparameters.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(optimal_config, f, default_flow_style=False, indent=2)
        
        print(f"✅ Saved optimal hyperparameters to: {config_path}")
        
        # Save detailed analysis
        analysis_path = output_dir / 'sweep_analysis_detailed.json'
        with open(analysis_path, 'w') as f:
            json.dump(analysis_results, f, indent=2, default=str)
        
        print(f"✅ Saved detailed analysis to: {analysis_path}")
        
        # Print summary
        print(f"\n📋 OPTIMAL HYPERPARAMETERS SUMMARY")
        print("=" * 50)
        for result in analysis_results:
            task = result['task']
            method = result['method']
            lr = result['optimal_hyperparameters']['learning_rate']
            batch_size = result['optimal_hyperparameters']['per_device_train_batch_size']
            warmup = result['optimal_hyperparameters']['warmup_ratio']
            metric_val = result['metric_value']
            
            print(f"{task:8} | {method:12} | LR: {lr:8} | BS: {batch_size:2} | WU: {warmup:.2f} | Perf: {metric_val:.3f}")
    
    else:
        print(f"\n⚠️  No results to export (use --export-optimal-configs to save)")
    
    print(f"\n🎉 Analysis complete! Found optimal configs for {len(analysis_results)} task/method combinations")

if __name__ == "__main__":
    main()
