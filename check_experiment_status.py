#!/usr/bin/env python3
"""Quick status check for full fine-tuning experiments."""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
import wandb

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def check_file_system_status():
    """Check local file system for experiment artifacts."""
    logger.info("Checking local file system status...")
    
    results_dir = Path("results")
    status = {
        'results_dir_exists': results_dir.exists(),
        'experiment_dirs': [],
        'representation_dirs': [],
        'checkpoint_dirs': [],
        'log_dirs': []
    }
    
    if results_dir.exists():
        # Find experiment directories
        for item in results_dir.iterdir():
            if item.is_dir():
                if "full_finetune" in item.name or "full_ft" in item.name:
                    status['experiment_dirs'].append(str(item))
                elif "representations" in item.name:
                    status['representation_dirs'].append(str(item))
        
        # Find checkpoint directories
        checkpoint_patterns = ["**/final_model", "**/pytorch_model.bin", "**/model.safetensors"]
        for pattern in checkpoint_patterns:
            status['checkpoint_dirs'].extend([str(p) for p in results_dir.glob(pattern)])
        
        # Find representation directories
        repr_dirs = list(results_dir.glob("**/representations"))
        status['representation_dirs'].extend([str(p) for p in repr_dirs])
    
    # Check logs directory
    logs_dir = Path("logs")
    if logs_dir.exists():
        for phase_dir in logs_dir.iterdir():
            if phase_dir.is_dir():
                status['log_dirs'].append(str(phase_dir))
    
    # Print status
    print("\n📁 FILE SYSTEM STATUS")
    print("=" * 50)
    print(f"Results directory: {'✓' if status['results_dir_exists'] else '✗'}")
    print(f"Experiment directories: {len(status['experiment_dirs'])}")
    print(f"Checkpoint directories: {len(status['checkpoint_dirs'])}")
    print(f"Representation directories: {len(status['representation_dirs'])}")
    print(f"Log directories: {len(status['log_dirs'])}")
    
    if status['experiment_dirs']:
        print("\nExperiment directories found:")
        for exp_dir in status['experiment_dirs'][:5]:  # Show first 5
            print(f"  - {exp_dir}")
    
    return status


def check_wandb_status():
    """Check W&B for recent runs."""
    logger.info("Checking W&B status...")
    
    status = {
        'wandb_accessible': False,
        'recent_runs': [],
        'full_finetune_runs': [],
        'running_runs': [],
        'failed_runs': []
    }
    
    try:
        # Set up W&B
        os.environ.setdefault('WANDB_PROJECT', 'NLP')
        default_entity = os.environ.get('WANDB_ENTITY', 'galavny-tel-aviv-university')
        os.environ.setdefault('WANDB_ENTITY', default_entity)
        
        api = wandb.Api()
        project_name = f"{wandb.api.default_entity or default_entity}/NLP"
        
        runs = list(api.runs(project_name))[:20]  # Get recent 20 runs
        status['wandb_accessible'] = True
        status['recent_runs'] = runs
        
        # Categorize runs
        for run in runs:
            run_info = {
                'name': run.name,
                'state': run.state,
                'created_at': run.created_at,
                'tags': run.tags
            }
            
            if "full_finetune" in run.tags or "full_ft" in run.name:
                status['full_finetune_runs'].append(run_info)
            
            if run.state == "running":
                status['running_runs'].append(run_info)
            elif run.state == "failed" or run.state == "crashed":
                status['failed_runs'].append(run_info)
        
        # Print status
        print("\n📊 WANDB STATUS")
        print("=" * 50)
        print(f"W&B accessible: ✓")
        print(f"Recent runs: {len(status['recent_runs'])}")
        print(f"Full fine-tuning runs: {len(status['full_finetune_runs'])}")
        print(f"Currently running: {len(status['running_runs'])}")
        print(f"Failed runs: {len(status['failed_runs'])}")
        
        if status['full_finetune_runs']:
            print("\nFull fine-tuning runs:")
            for run in status['full_finetune_runs'][:5]:
                print(f"  - {run['name']} ({run['state']}) - {run['created_at']}")
        
        if status['running_runs']:
            print("\nCurrently running:")
            for run in status['running_runs']:
                print(f"  - {run['name']}")
        
        if status['failed_runs']:
            print("\nFailed runs:")
            for run in status['failed_runs'][:3]:
                print(f"  - {run['name']} ({run['state']})")
    
    except Exception as e:
        print(f"\n📊 WANDB STATUS: ❌ Error accessing W&B: {e}")
        status['wandb_accessible'] = False
    
    return status


def check_performance_expectations():
    """Check if any completed runs meet performance expectations."""
    logger.info("Checking performance against expectations...")
    
    expectations = {
        'mrpc': {'min': 0.85, 'max': 0.90, 'metric': 'accuracy'},
        'sst2': {'min': 0.90, 'max': 0.93, 'metric': 'accuracy'},
        'rte': {'min': 0.65, 'max': 0.75, 'metric': 'accuracy'},
        'squad_v2': {'min': 0.75, 'max': 0.85, 'metric': 'f1'}
    }
    
    performance_status = {
        'tasks_with_results': {},
        'meeting_expectations': {},
        'performance_issues': []
    }
    
    try:
        api = wandb.Api()
        default_entity = os.environ.get('WANDB_ENTITY', 'galavny-tel-aviv-university')
        project_name = f"{wandb.api.default_entity or default_entity}/NLP"
        runs = list(api.runs(project_name))[:50]
        
        # Group runs by task
        task_results = {}
        for run in runs:
            if "full_finetune" in run.tags or "full_ft" in run.name:
                task_name = None
                if hasattr(run, 'config') and 'task_name' in run.config:
                    task_name = run.config['task_name']
                elif any(task in run.name for task in expectations.keys()):
                    for task in expectations.keys():
                        if task in run.name:
                            task_name = task
                            break
                
                if task_name and task_name in expectations:
                    if task_name not in task_results:
                        task_results[task_name] = []
                    
                    # Get performance metric
                    summary = dict(run.summary)
                    if task_name == 'squad_v2':
                        perf_value = summary.get('eval_f1', summary.get('f1'))
                    else:
                        perf_value = summary.get('eval_accuracy', summary.get('accuracy'))
                    
                    if perf_value is not None:
                        task_results[task_name].append({
                            'run_name': run.name,
                            'performance': perf_value,
                            'state': run.state
                        })
        
        # Analyze performance
        print("\n🎯 PERFORMANCE STATUS")
        print("=" * 50)
        
        for task_name, expected in expectations.items():
            if task_name in task_results:
                results = task_results[task_name]
                performance_status['tasks_with_results'][task_name] = len(results)
                
                completed_results = [r for r in results if r['state'] == 'finished']
                if completed_results:
                    perfs = [r['performance'] for r in completed_results]
                    avg_perf = sum(perfs) / len(perfs)
                    best_perf = max(perfs)
                    
                    within_range = expected['min'] <= best_perf <= expected['max']
                    performance_status['meeting_expectations'][task_name] = within_range
                    
                    status_icon = "✓" if within_range else "⚠"
                    print(f"{task_name}: {status_icon} Best: {best_perf:.3f}, Avg: {avg_perf:.3f} "
                          f"(Expected: {expected['min']:.2f}-{expected['max']:.2f})")
                    
                    if not within_range:
                        if best_perf < expected['min']:
                            performance_status['performance_issues'].append(
                                f"{task_name}: Performance below expected minimum ({best_perf:.3f} < {expected['min']:.2f})"
                            )
                        elif best_perf > expected['max']:
                            print(f"  Note: {task_name} exceeds maximum expected (which is good!)")
                else:
                    print(f"{task_name}: ⏳ No completed runs yet ({len(results)} running/failed)")
            else:
                print(f"{task_name}: ❌ No runs found")
    
    except Exception as e:
        print(f"❌ Error checking performance: {e}")
    
    return performance_status


def check_representation_extraction():
    """Check representation extraction status."""
    logger.info("Checking representation extraction...")
    
    status = {
        'base_representations_found': False,
        'training_representations_found': False,
        'extraction_intervals_correct': True,
        'tasks_with_representations': []
    }
    
    repr_dirs = list(Path("results").glob("**/representations"))
    
    print("\n🧠 REPRESENTATION EXTRACTION STATUS")
    print("=" * 50)
    
    if not repr_dirs:
        print("❌ No representation directories found")
        return status
    
    for repr_dir in repr_dirs:
        print(f"Found representation directory: {repr_dir}")
        
        # Check subdirectories
        for subdir in repr_dir.iterdir():
            if subdir.is_dir():
                print(f"  - {subdir.name}")
                
                if 'base_pretrained' in subdir.name:
                    status['base_representations_found'] = True
                    print(f"    ✓ Base model representations found")
                
                # Check step directories
                step_dirs = [d for d in subdir.iterdir() if d.is_dir() and d.name.startswith('step_')]
                if step_dirs:
                    step_numbers = []
                    for step_dir in step_dirs:
                        try:
                            step_num = int(step_dir.name.split('_')[1])
                            step_numbers.append(step_num)
                        except ValueError:
                            continue
                    
                    if step_numbers:
                        status['training_representations_found'] = True
                        step_numbers.sort()
                        print(f"    ✓ {len(step_numbers)} extraction steps: {step_numbers[:5]}..." if len(step_numbers) > 5 else f"    ✓ {len(step_numbers)} extraction steps: {step_numbers}")
                        
                        # Check intervals
                        if len(step_numbers) > 1:
                            intervals = [step_numbers[i+1] - step_numbers[i] for i in range(len(step_numbers)-1)]
                            if not all(interval == 100 for interval in intervals[:-1]):  # Exclude last
                                print(f"    ⚠ Irregular intervals detected: {intervals}")
                                status['extraction_intervals_correct'] = False
                
                # Check for task name
                for task in ['mrpc', 'sst2', 'rte', 'squad_v2']:
                    if task in subdir.name and task not in status['tasks_with_representations']:
                        status['tasks_with_representations'].append(task)
    
    print(f"\nSummary:")
    print(f"Base representations: {'✓' if status['base_representations_found'] else '❌'}")
    print(f"Training representations: {'✓' if status['training_representations_found'] else '❌'}")
    print(f"Extraction intervals: {'✓' if status['extraction_intervals_correct'] else '⚠'}")
    print(f"Tasks with representations: {status['tasks_with_representations']}")
    
    return status


def main():
    """Main status check function."""
    print("🔍 FULL FINE-TUNING EXPERIMENT STATUS CHECK")
    print("=" * 60)
    print(f"Timestamp: {datetime.now()}")
    
    # Run all checks
    file_status = check_file_system_status()
    wandb_status = check_wandb_status()
    performance_status = check_performance_expectations()
    representation_status = check_representation_extraction()
    
    # Overall summary
    print("\n📋 OVERALL SUMMARY")
    print("=" * 50)
    
    checks = [
        ("Local files present", file_status['results_dir_exists']),
        ("W&B accessible", wandb_status['wandb_accessible']),
        ("Full FT runs found", len(wandb_status['full_finetune_runs']) > 0),
        ("Base representations", representation_status['base_representations_found']),
        ("Training representations", representation_status['training_representations_found']),
    ]
    
    all_good = True
    for check_name, status in checks:
        status_icon = "✓" if status else "❌"
        print(f"{check_name}: {status_icon}")
        if not status:
            all_good = False
    
    if all_good:
        print("\n🎉 Basic setup looks good!")
    else:
        print("\n⚠ Some components need attention")
    
    # Next steps
    print("\n🚀 NEXT STEPS")
    print("=" * 50)
    
    if not wandb_status['wandb_accessible']:
        print("1. Set up W&B authentication: wandb login")
    elif len(wandb_status['full_finetune_runs']) == 0:
        print("1. Run full fine-tuning experiments: bash scripts/phase1/vm1.sh")
    elif len(wandb_status['running_runs']) > 0:
        print("1. Wait for current experiments to complete")
        print("2. Monitor progress in W&B dashboard")
    else:
        print("1. Run comprehensive validation: python validate_full_finetune.py")
        print("2. Check performance against expected ranges")
    
    print("📊 Monitor: https://wandb.ai/galavny-tel-aviv-university/NLP")


if __name__ == "__main__":
    main()
