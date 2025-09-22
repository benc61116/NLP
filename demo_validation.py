#!/usr/bin/env python3
"""Demonstration of validation process for full fine-tuning experiments."""

import os
import sys
import torch
import json
import logging
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_mock_experiment_data():
    """Create mock experiment data to demonstrate validation."""
    logger.info("Creating mock experiment data for validation demonstration...")
    
    # Create results directory structure
    results_dir = Path("results/demo_validation")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Create mock experiment directories
    experiments = [
        "full_ft_sst2_seed42",
        "full_ft_mrpc_seed42", 
        "full_ft_rte_seed1337",
        "full_ft_squad_v2_seed2024"
    ]
    
    for exp_name in experiments:
        exp_dir = results_dir / exp_name
        exp_dir.mkdir(exist_ok=True)
        
        # Create final model directory
        model_dir = exp_dir / "final_model"
        model_dir.mkdir(exist_ok=True)
        
        # Create mock model files
        with open(model_dir / "config.json", 'w') as f:
            json.dump({
                "model_type": "llama",
                "vocab_size": 32000,
                "hidden_size": 2048,
                "num_hidden_layers": 24
            }, f)
        
        # Create mock model weights (small dummy file)
        torch.save(torch.randn(100, 100), model_dir / "pytorch_model.bin")
    
    # Create mock representations
    repr_dir = results_dir / "representations"
    repr_dir.mkdir(exist_ok=True)
    
    tasks = ["sst2", "mrpc", "rte", "squad_v2"]
    
    for task in tasks:
        # Base model representations
        base_dir = repr_dir / f"base_pretrained_{task}"
        base_step_dir = base_dir / "step_000000"
        base_step_dir.mkdir(parents=True, exist_ok=True)
        
        # Create mock representation tensors
        for layer in range(3):  # Just 3 layers for demo
            layer_tensor = torch.randn(100, 512)  # Mock representation
            torch.save(layer_tensor, base_step_dir / f"layer_{layer}.pt")
        
        # Create metadata
        metadata = {
            "step": 0,
            "task_name": task,
            "method": "base_pretrained",
            "timestamp": datetime.now().isoformat(),
            "num_samples": 100,
            "layer_names": [f"layer_{i}" for i in range(3)],
            "tensor_shapes": {"layer_0": [100, 512], "layer_1": [100, 512], "layer_2": [100, 512]}
        }
        
        with open(base_step_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Fine-tuned model representations
        ft_dir = repr_dir / f"full_finetune_{task}"
        
        # Create multiple steps (every 100)
        for step in [100, 200, 300, 400]:
            step_dir = ft_dir / f"step_{step:06d}"
            step_dir.mkdir(parents=True, exist_ok=True)
            
            # Create mock representation tensors
            for layer in range(3):
                # Slightly different from base to simulate drift
                layer_tensor = torch.randn(100, 512) * 0.9 + torch.randn(100, 512) * 0.1
                torch.save(layer_tensor, step_dir / f"layer_{layer}.pt")
            
            # Create metadata
            metadata = {
                "step": step,
                "task_name": task,
                "method": "full_finetune",
                "timestamp": datetime.now().isoformat(),
                "num_samples": 100,
                "layer_names": [f"layer_{i}" for i in range(3)],
                "tensor_shapes": {"layer_0": [100, 512], "layer_1": [100, 512], "layer_2": [100, 512]}
            }
            
            with open(step_dir / "metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
    
    logger.info("✓ Mock experiment data created successfully")
    return results_dir


def create_mock_wandb_data():
    """Create mock W&B run data for validation."""
    logger.info("Creating mock W&B run data...")
    
    # Mock performance data that meets Step 3 requirements
    mock_runs = [
        {
            'name': 'full_ft_sst2_seed42',
            'state': 'finished',
            'tags': ['full_finetune', 'sst2'],
            'config': {'task_name': 'sst2', 'learning_rate': 2e-5, 'seed': 42},
            'summary': {
                'train_loss': 0.12,
                'eval_loss': 0.28,
                'eval_accuracy': 0.915,  # Within 90-93% range
                'gradient_norm_total': 12.5,
                'cpu_memory_rss_mb': 1024,
                'gpu_0_memory_allocated_mb': 8192,
                'step': 500
            }
        },
        {
            'name': 'full_ft_mrpc_seed42',
            'state': 'finished', 
            'tags': ['full_finetune', 'mrpc'],
            'config': {'task_name': 'mrpc', 'learning_rate': 1e-5, 'seed': 42},
            'summary': {
                'train_loss': 0.08,
                'eval_loss': 0.32,
                'eval_accuracy': 0.87,  # Within 85-90% range
                'eval_f1': 0.89,
                'gradient_norm_total': 8.3,
                'cpu_memory_rss_mb': 1156,
                'gpu_0_memory_allocated_mb': 7899,
                'step': 400
            }
        },
        {
            'name': 'full_ft_rte_seed1337',
            'state': 'finished',
            'tags': ['full_finetune', 'rte'],
            'config': {'task_name': 'rte', 'learning_rate': 2e-5, 'seed': 1337},
            'summary': {
                'train_loss': 0.15,
                'eval_loss': 0.45,
                'eval_accuracy': 0.71,  # Within 65-75% range
                'gradient_norm_total': 15.2,
                'cpu_memory_rss_mb': 1089,
                'gpu_0_memory_allocated_mb': 8034,
                'step': 350
            }
        },
        {
            'name': 'full_ft_squad_v2_seed2024',
            'state': 'finished',
            'tags': ['full_finetune', 'squad_v2'],
            'config': {'task_name': 'squad_v2', 'learning_rate': 1e-5, 'seed': 2024},
            'summary': {
                'train_loss': 0.22,
                'eval_loss': 0.58,
                'eval_f1': 0.81,  # Within 75-85% range
                'exact_match': 0.73,
                'gradient_norm_total': 22.1,
                'cpu_memory_rss_mb': 1234,
                'gpu_0_memory_allocated_mb': 9876,
                'step': 600
            }
        },
        # Add a problematic run to demonstrate red flag detection
        {
            'name': 'full_ft_sst2_seed999_failed',
            'state': 'failed',
            'tags': ['full_finetune', 'sst2'],
            'config': {'task_name': 'sst2', 'learning_rate': 1e-3, 'seed': 999},  # High LR
            'summary': {
                'train_loss': 8.45,  # High loss - red flag
                'eval_loss': 12.33,  # Very high loss - red flag
                'eval_accuracy': 0.52,  # Below threshold - red flag
                'gradient_norm_total': 1500.0,  # Gradient explosion - red flag
                'cpu_memory_rss_mb': 2048,
                'step': 150
            }
        }
    ]
    
    logger.info("✓ Mock W&B run data created")
    return mock_runs


def demonstrate_training_progress_monitoring(runs_data: List[Dict]):
    """Demonstrate training progress monitoring validation."""
    logger.info("\n" + "="*60)
    logger.info("DEMONSTRATING: Training Progress Monitoring")
    logger.info("="*60)
    
    print("\n📊 TRAINING METRICS ANALYSIS")
    print("-" * 50)
    
    for run in runs_data:
        name = run['name']
        state = run['state']
        summary = run['summary']
        
        train_loss = summary.get('train_loss', 'N/A')
        eval_loss = summary.get('eval_loss', 'N/A')
        grad_norm = summary.get('gradient_norm_total', 'N/A')
        
        status_icon = "✓" if state == "finished" else "❌" if state == "failed" else "⏳"
        print(f"{status_icon} {name}")
        print(f"    Train Loss: {train_loss}")
        print(f"    Eval Loss: {eval_loss}")
        print(f"    Gradient Norm: {grad_norm}")
        print(f"    State: {state}")
        
        # Check for red flags
        if isinstance(train_loss, (int, float)) and train_loss > 5.0:
            print(f"    🚨 RED FLAG: High training loss")
        if isinstance(grad_norm, (int, float)) and grad_norm > 1000:
            print(f"    🚨 RED FLAG: Gradient explosion")
        
        print()


def demonstrate_performance_validation(runs_data: List[Dict]):
    """Demonstrate performance validation against expected ranges."""
    logger.info("\n" + "="*60)
    logger.info("DEMONSTRATING: Performance Validation")
    logger.info("="*60)
    
    # Expected ranges from Step 3 requirements
    thresholds = {
        'mrpc': {'min_accuracy': 0.85, 'max_accuracy': 0.90, 'metric': 'accuracy'},
        'sst2': {'min_accuracy': 0.90, 'max_accuracy': 0.93, 'metric': 'accuracy'},
        'rte': {'min_accuracy': 0.65, 'max_accuracy': 0.75, 'metric': 'accuracy'},
        'squad_v2': {'min_f1': 0.75, 'max_f1': 0.85, 'metric': 'f1'}
    }
    
    print("\n🎯 PERFORMANCE AGAINST EXPECTED RANGES")
    print("-" * 50)
    
    for task, threshold in thresholds.items():
        print(f"\n{task.upper()}:")
        
        task_runs = [run for run in runs_data if run['config'].get('task_name') == task]
        
        if not task_runs:
            print(f"  ❌ No runs found")
            continue
        
        for run in task_runs:
            summary = run['summary']
            
            if task == 'squad_v2':
                performance = summary.get('eval_f1')
                min_thresh = threshold['min_f1']
                max_thresh = threshold['max_f1']
                metric_name = 'F1'
            else:
                performance = summary.get('eval_accuracy')
                min_thresh = threshold['min_accuracy']
                max_thresh = threshold['max_accuracy']
                metric_name = 'Accuracy'
            
            if performance is not None:
                if min_thresh <= performance <= max_thresh:
                    status = "✓ PASS"
                elif performance < min_thresh:
                    status = "🚨 BELOW RANGE"
                else:
                    status = "✅ ABOVE RANGE"
                
                print(f"  {run['name']}: {metric_name}={performance:.3f} ({status})")
                print(f"    Expected: {min_thresh:.2f}-{max_thresh:.2f}")
            else:
                print(f"  {run['name']}: No {metric_name.lower()} data")


def demonstrate_representation_validation(results_dir: Path):
    """Demonstrate representation extraction validation."""
    logger.info("\n" + "="*60)
    logger.info("DEMONSTRATING: Representation Extraction Validation")
    logger.info("="*60)
    
    print("\n🧠 REPRESENTATION EXTRACTION CHECK")
    print("-" * 50)
    
    repr_dir = results_dir / "representations"
    
    if not repr_dir.exists():
        print("❌ No representations directory found")
        return
    
    tasks_checked = []
    base_representations_found = False
    training_representations_found = False
    
    for task_dir in repr_dir.iterdir():
        if not task_dir.is_dir():
            continue
        
        task_name = task_dir.name
        print(f"\n📁 {task_name}")
        
        if 'base_pretrained' in task_name:
            base_representations_found = True
            print(f"  ✓ Base model representations found")
        
        # Check step directories
        step_dirs = sorted([d for d in task_dir.iterdir() if d.is_dir() and d.name.startswith('step_')])
        
        if step_dirs:
            training_representations_found = True
            step_numbers = []
            for step_dir in step_dirs:
                try:
                    step_num = int(step_dir.name.split('_')[1])
                    step_numbers.append(step_num)
                except ValueError:
                    continue
            
            print(f"  ✓ {len(step_numbers)} extraction steps: {step_numbers}")
            
            # Check extraction intervals
            if len(step_numbers) > 1:
                intervals = [step_numbers[i+1] - step_numbers[i] for i in range(len(step_numbers)-1)]
                all_100 = all(interval == 100 for interval in intervals)
                print(f"  {'✓' if all_100 else '⚠'} Extraction intervals: {intervals}")
            
            # Check file integrity
            sample_step = step_dirs[0]
            pt_files = list(sample_step.glob("*.pt"))
            metadata_files = list(sample_step.glob("metadata.json"))
            
            print(f"  ✓ {len(pt_files)} layer files per step")
            print(f"  {'✓' if metadata_files else '❌'} Metadata files present")
            
            # Test loading a file
            if pt_files:
                try:
                    tensor = torch.load(pt_files[0], map_location='cpu')
                    print(f"  ✓ File loading test: shape {tensor.shape}")
                except Exception as e:
                    print(f"  ❌ File loading failed: {e}")
        
        task_name_clean = task_name.replace('base_pretrained_', '').replace('full_finetune_', '')
        if task_name_clean not in tasks_checked:
            tasks_checked.append(task_name_clean)
    
    print(f"\n📋 SUMMARY:")
    print(f"  Base representations: {'✓' if base_representations_found else '❌'}")
    print(f"  Training representations: {'✓' if training_representations_found else '❌'}")
    print(f"  Tasks with representations: {tasks_checked}")


def demonstrate_checkpoint_validation(results_dir: Path):
    """Demonstrate checkpoint validation."""
    logger.info("\n" + "="*60)
    logger.info("DEMONSTRATING: Checkpoint Validation")
    logger.info("="*60)
    
    print("\n💾 CHECKPOINT VALIDATION")
    print("-" * 50)
    
    # Find checkpoint directories
    checkpoint_dirs = list(results_dir.glob("**/final_model"))
    
    if not checkpoint_dirs:
        print("❌ No checkpoints found")
        return
    
    print(f"✓ Found {len(checkpoint_dirs)} checkpoints")
    
    for checkpoint_dir in checkpoint_dirs:
        print(f"\n📁 {checkpoint_dir.parent.name}/final_model")
        
        # Check required files
        required_files = ['config.json', 'pytorch_model.bin']
        for file_name in required_files:
            file_path = checkpoint_dir / file_name
            if file_path.exists():
                size = file_path.stat().st_size
                print(f"  ✓ {file_name} ({size:,} bytes)")
            else:
                print(f"  ❌ {file_name} missing")
        
        # Test loading (simplified)
        try:
            config_path = checkpoint_dir / 'config.json'
            if config_path.exists():
                with open(config_path) as f:
                    config = json.load(f)
                print(f"  ✓ Config loadable: {config.get('model_type', 'unknown')} model")
            
            model_path = checkpoint_dir / 'pytorch_model.bin'
            if model_path.exists():
                # Just check it's a valid tensor file (don't load full model)
                try:
                    torch.load(model_path, map_location='cpu')
                    print(f"  ✓ Model weights loadable")
                except:
                    print(f"  ❌ Model weights corrupted")
        
        except Exception as e:
            print(f"  ❌ Loading test failed: {e}")


def demonstrate_red_flags_detection(runs_data: List[Dict]):
    """Demonstrate red flags detection."""
    logger.info("\n" + "="*60)
    logger.info("DEMONSTRATING: Red Flags Detection")
    logger.info("="*60)
    
    print("\n🚨 RED FLAGS ANALYSIS")
    print("-" * 50)
    
    red_flags = []
    warnings = []
    
    for run in runs_data:
        name = run['name']
        state = run['state']
        summary = run['summary']
        
        # Check for failed runs
        if state in ['failed', 'crashed']:
            red_flags.append(f"Run failed: {name}")
        
        # Check training stability
        train_loss = summary.get('train_loss')
        if isinstance(train_loss, (int, float)):
            if train_loss > 5.0:
                red_flags.append(f"High training loss: {name} ({train_loss:.3f})")
            elif train_loss > 2.0:
                warnings.append(f"Elevated training loss: {name} ({train_loss:.3f})")
        
        # Check gradient norms
        grad_norm = summary.get('gradient_norm_total')
        if isinstance(grad_norm, (int, float)):
            if grad_norm > 1000:
                red_flags.append(f"Gradient explosion: {name} ({grad_norm:.1f})")
            elif grad_norm > 100:
                warnings.append(f"High gradient norm: {name} ({grad_norm:.1f})")
        
        # Check performance
        accuracy = summary.get('eval_accuracy')
        if isinstance(accuracy, (int, float)):
            task = run['config'].get('task_name')
            if task == 'sst2' and accuracy < 0.80:
                red_flags.append(f"Low SST-2 accuracy: {name} ({accuracy:.3f})")
            elif task == 'mrpc' and accuracy < 0.75:
                red_flags.append(f"Low MRPC accuracy: {name} ({accuracy:.3f})")
    
    print(f"🚨 Critical Issues: {len(red_flags)}")
    for flag in red_flags:
        print(f"  - {flag}")
    
    print(f"\n⚠️  Warnings: {len(warnings)}")
    for warning in warnings:
        print(f"  - {warning}")
    
    if not red_flags and not warnings:
        print("✅ No issues detected - all systems normal")


def main():
    """Main demonstration function."""
    print("🔍 FULL FINE-TUNING VALIDATION DEMONSTRATION")
    print("=" * 60)
    print("This demonstrates the Step 3 validation process")
    print("with mock data that shows expected behaviors.")
    
    # Create mock data
    results_dir = create_mock_experiment_data()
    runs_data = create_mock_wandb_data()
    
    # Run validation demonstrations
    demonstrate_training_progress_monitoring(runs_data)
    demonstrate_performance_validation(runs_data)
    demonstrate_representation_validation(results_dir)
    demonstrate_checkpoint_validation(results_dir)
    demonstrate_red_flags_detection(runs_data)
    
    # Final summary
    logger.info("\n" + "="*60)
    logger.info("VALIDATION DEMONSTRATION COMPLETE")
    logger.info("="*60)
    
    print("\n📋 VALIDATION TOOLS SUMMARY:")
    print("1. check_experiment_status.py - Quick status overview")
    print("2. monitor_red_flags.py - Detect critical issues")
    print("3. validate_full_finetune.py - Comprehensive validation")
    print("4. VALIDATION_GUIDE.md - Complete validation procedures")
    
    print("\n🎯 STEP 3 REQUIREMENTS COVERAGE:")
    print("✓ Training progress monitoring (W&B dashboard)")
    print("✓ Performance validation (expected ranges)")
    print("✓ Representation extraction check")
    print("✓ Checkpoint validation")
    print("✓ Red flags detection")
    
    print("\n🚀 NEXT STEPS:")
    print("1. Run actual experiments: bash scripts/phase1/vm*.sh")
    print("2. Use validation tools to monitor progress")
    print("3. Address any red flags or warnings")
    print("4. Proceed to LoRA experiments once validation passes")


if __name__ == "__main__":
    main()
