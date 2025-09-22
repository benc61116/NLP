#!/usr/bin/env python3
"""Comprehensive validation script for full fine-tuning experiments according to Step 3 requirements."""

import os
import sys
import torch
import json
import wandb
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FullFinetuneValidator:
    """Comprehensive validator for full fine-tuning experiments."""
    
    def __init__(self, results_dir: str = "results", wandb_project: str = "NLP"):
        self.results_dir = Path(results_dir)
        self.wandb_project = wandb_project
        self.validation_results = {}
        
        # Expected performance ranges
        self.performance_thresholds = {
            'mrpc': {'min_accuracy': 0.85, 'max_accuracy': 0.90, 'metric': 'accuracy'},
            'sst2': {'min_accuracy': 0.90, 'max_accuracy': 0.93, 'metric': 'accuracy'},
            'rte': {'min_accuracy': 0.65, 'max_accuracy': 0.75, 'metric': 'accuracy'},
            'squad_v2': {'min_f1': 0.75, 'max_f1': 0.85, 'metric': 'f1'}
        }
    
    def validate_wandb_monitoring(self) -> Dict[str, Any]:
        """Validate W&B dashboard monitoring and metrics."""
        logger.info("Validating W&B monitoring...")
        
        validation_results = {
            'wandb_accessible': False,
            'recent_runs_found': False,
            'required_metrics_logged': False,
            'runs_data': [],
            'missing_metrics': [],
            'issues': []
        }
        
        try:
            # Initialize W&B API
            api = wandb.Api()
            
            # Check if project exists and is accessible
            try:
                runs = api.runs(f"{wandb.api.default_entity}/{self.wandb_project}")
                validation_results['wandb_accessible'] = True
                logger.info("✓ W&B project accessible")
            except Exception as e:
                validation_results['issues'].append(f"Cannot access W&B project: {e}")
                return validation_results
            
            # Check for recent full fine-tuning runs
            recent_runs = []
            cutoff_time = datetime.now() - timedelta(days=7)  # Last 7 days
            
            for run in runs:
                if run.state == "finished" or run.state == "running":
                    if "full_finetune" in run.tags or "full_ft" in run.name:
                        run_data = {
                            'name': run.name,
                            'state': run.state,
                            'created_at': run.created_at,
                            'tags': run.tags,
                            'config': dict(run.config),
                            'summary': dict(run.summary)
                        }
                        recent_runs.append(run_data)
                        
                        if len(recent_runs) >= 10:  # Limit to recent runs
                            break
            
            validation_results['runs_data'] = recent_runs
            validation_results['recent_runs_found'] = len(recent_runs) > 0
            
            if recent_runs:
                logger.info(f"✓ Found {len(recent_runs)} recent full fine-tuning runs")
                
                # Check required metrics in runs
                required_metrics = [
                    'train_loss', 'eval_loss', 'learning_rate',
                    'gradient_norm_total', 'cpu_memory_rss_mb'
                ]
                
                metrics_found = set()
                for run in recent_runs[:3]:  # Check first 3 runs
                    for metric in required_metrics:
                        if metric in run['summary']:
                            metrics_found.add(metric)
                
                missing_metrics = set(required_metrics) - metrics_found
                validation_results['missing_metrics'] = list(missing_metrics)
                validation_results['required_metrics_logged'] = len(missing_metrics) == 0
                
                if missing_metrics:
                    logger.warning(f"⚠ Missing metrics in W&B: {missing_metrics}")
                else:
                    logger.info("✓ All required metrics found in W&B runs")
            else:
                validation_results['issues'].append("No recent full fine-tuning runs found in W&B")
        
        except Exception as e:
            validation_results['issues'].append(f"W&B validation error: {e}")
            logger.error(f"❌ W&B validation failed: {e}")
        
        return validation_results
    
    def validate_performance_metrics(self, runs_data: List[Dict]) -> Dict[str, Any]:
        """Validate performance metrics against expected ranges."""
        logger.info("Validating performance metrics...")
        
        validation_results = {
            'tasks_validated': {},
            'performance_issues': [],
            'within_expected_ranges': True
        }
        
        # Group runs by task
        task_runs = {}
        for run in runs_data:
            config = run.get('config', {})
            task_name = config.get('task_name', 'unknown')
            
            if task_name in self.performance_thresholds:
                if task_name not in task_runs:
                    task_runs[task_name] = []
                task_runs[task_name].append(run)
        
        # Validate each task's performance
        for task_name, task_thresholds in self.performance_thresholds.items():
            if task_name not in task_runs:
                validation_results['performance_issues'].append(f"No runs found for task: {task_name}")
                validation_results['within_expected_ranges'] = False
                continue
            
            task_validation = {
                'runs_count': len(task_runs[task_name]),
                'performance_values': [],
                'within_range_count': 0,
                'issues': []
            }
            
            for run in task_runs[task_name]:
                summary = run.get('summary', {})
                
                # Get the appropriate metric
                if task_name == 'squad_v2':
                    metric_value = summary.get('eval_f1', summary.get('f1'))
                    min_threshold = task_thresholds['min_f1']
                    max_threshold = task_thresholds['max_f1']
                else:
                    metric_value = summary.get('eval_accuracy', summary.get('accuracy'))
                    min_threshold = task_thresholds['min_accuracy']
                    max_threshold = task_thresholds['max_accuracy']
                
                if metric_value is not None:
                    task_validation['performance_values'].append(metric_value)
                    
                    if min_threshold <= metric_value <= max_threshold:
                        task_validation['within_range_count'] += 1
                    else:
                        issue_msg = f"Performance out of range: {metric_value:.3f} (expected {min_threshold:.2f}-{max_threshold:.2f})"
                        task_validation['issues'].append(issue_msg)
                        validation_results['performance_issues'].append(f"{task_name}: {issue_msg}")
                        validation_results['within_expected_ranges'] = False
            
            # Calculate statistics
            if task_validation['performance_values']:
                values = task_validation['performance_values']
                task_validation['mean_performance'] = np.mean(values)
                task_validation['std_performance'] = np.std(values)
                task_validation['min_performance'] = np.min(values)
                task_validation['max_performance'] = np.max(values)
                
                logger.info(f"✓ {task_name}: {len(values)} runs, "
                           f"mean {task_validation['mean_performance']:.3f} ± {task_validation['std_performance']:.3f}")
            else:
                task_validation['issues'].append("No performance metrics found in runs")
                validation_results['performance_issues'].append(f"{task_name}: No performance metrics found")
            
            validation_results['tasks_validated'][task_name] = task_validation
        
        return validation_results
    
    def validate_representation_extraction(self) -> Dict[str, Any]:
        """Validate representation extraction files and integrity."""
        logger.info("Validating representation extraction...")
        
        validation_results = {
            'representation_dirs_found': False,
            'extraction_intervals_correct': True,
            'file_integrity_good': True,
            'base_representations_found': False,
            'issues': [],
            'tasks_checked': {}
        }
        
        # Look for representation directories
        repr_dirs = list(self.results_dir.glob("**/representations"))
        if not repr_dirs:
            validation_results['issues'].append("No representation directories found")
            return validation_results
        
        validation_results['representation_dirs_found'] = True
        logger.info(f"✓ Found {len(repr_dirs)} representation directories")
        
        # Check each representation directory
        for repr_dir in repr_dirs:
            # Look for task-specific subdirectories
            task_dirs = [d for d in repr_dir.iterdir() if d.is_dir()]
            
            for task_dir in task_dirs:
                task_name = task_dir.name
                if 'base_pretrained' in task_name:
                    validation_results['base_representations_found'] = True
                
                task_validation = {
                    'step_dirs_found': [],
                    'file_counts': {},
                    'file_sizes': {},
                    'metadata_files': [],
                    'issues': []
                }
                
                # Check step directories
                step_dirs = sorted([d for d in task_dir.iterdir() if d.is_dir() and d.name.startswith('step_')])
                
                if step_dirs:
                    # Extract step numbers
                    step_numbers = []
                    for step_dir in step_dirs:
                        try:
                            step_num = int(step_dir.name.split('_')[1])
                            step_numbers.append(step_num)
                            task_validation['step_dirs_found'].append(step_num)
                        except ValueError:
                            continue
                    
                    # Check if extraction intervals are correct (every 100 steps)
                    if len(step_numbers) > 1:
                        intervals = np.diff(sorted(step_numbers))
                        if not all(interval == 100 for interval in intervals[:-1]):  # Exclude last interval
                            task_validation['issues'].append("Extraction intervals not consistent with 100 steps")
                            validation_results['extraction_intervals_correct'] = False
                    
                    # Check file integrity in each step directory
                    for step_dir in step_dirs[:3]:  # Check first 3 for efficiency
                        pt_files = list(step_dir.glob("*.pt"))
                        metadata_files = list(step_dir.glob("metadata.json"))
                        
                        task_validation['file_counts'][step_dir.name] = len(pt_files)
                        
                        # Check file sizes
                        total_size = 0
                        for pt_file in pt_files:
                            try:
                                size = pt_file.stat().st_size
                                total_size += size
                                
                                # Try to load the tensor to check integrity
                                tensor = torch.load(pt_file, map_location='cpu')
                                if tensor.numel() == 0:
                                    task_validation['issues'].append(f"Empty tensor in {pt_file}")
                                    validation_results['file_integrity_good'] = False
                            except Exception as e:
                                task_validation['issues'].append(f"Cannot load {pt_file}: {e}")
                                validation_results['file_integrity_good'] = False
                        
                        task_validation['file_sizes'][step_dir.name] = total_size / (1024 * 1024)  # MB
                        
                        # Check metadata
                        if metadata_files:
                            try:
                                with open(metadata_files[0]) as f:
                                    metadata = json.load(f)
                                task_validation['metadata_files'].append(metadata)
                            except Exception as e:
                                task_validation['issues'].append(f"Cannot load metadata: {e}")
                else:
                    task_validation['issues'].append("No step directories found")
                
                validation_results['tasks_checked'][task_name] = task_validation
                
                if task_validation['issues']:
                    logger.warning(f"⚠ Issues with {task_name}: {task_validation['issues']}")
                else:
                    logger.info(f"✓ {task_name}: {len(task_validation['step_dirs_found'])} extractions, "
                               f"avg {np.mean(list(task_validation['file_sizes'].values())):.1f} MB per step")
        
        return validation_results
    
    def validate_checkpoint_integrity(self) -> Dict[str, Any]:
        """Validate model checkpoints can be loaded and are consistent."""
        logger.info("Validating checkpoint integrity...")
        
        validation_results = {
            'checkpoints_found': False,
            'loading_successful': True,
            'output_consistency': True,
            'checkpoint_paths': [],
            'issues': []
        }
        
        # Find checkpoint directories
        checkpoint_dirs = []
        for pattern in ["**/final_model", "**/pytorch_model.bin", "**/model.safetensors"]:
            checkpoint_dirs.extend(list(self.results_dir.glob(pattern)))
        
        # Also look for full fine-tuning specific directories
        ft_dirs = list(self.results_dir.glob("**/full_ft_*"))
        for ft_dir in ft_dirs:
            if ft_dir.is_dir():
                model_files = list(ft_dir.glob("final_model"))
                checkpoint_dirs.extend(model_files)
        
        if not checkpoint_dirs:
            validation_results['issues'].append("No model checkpoints found")
            return validation_results
        
        validation_results['checkpoints_found'] = True
        validation_results['checkpoint_paths'] = [str(p) for p in checkpoint_dirs[:5]]  # Limit for testing
        
        logger.info(f"✓ Found {len(checkpoint_dirs)} checkpoint locations")
        
        # Test loading a few checkpoints
        test_inputs = None
        original_outputs = {}
        
        for i, checkpoint_path in enumerate(checkpoint_dirs[:3]):  # Test first 3
            try:
                # Try to load the model
                if checkpoint_path.is_dir():
                    model = AutoModelForCausalLM.from_pretrained(str(checkpoint_path))
                    logger.info(f"✓ Successfully loaded checkpoint: {checkpoint_path}")
                    
                    # Test inference consistency
                    if test_inputs is None:
                        # Create test inputs
                        try:
                            tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
                            if tokenizer.pad_token is None:
                                tokenizer.pad_token = tokenizer.eos_token
                            test_inputs = tokenizer("Hello, this is a test.", return_tensors="pt")
                        except:
                            test_inputs = {
                                'input_ids': torch.tensor([[1, 2, 3, 4, 5]]),
                                'attention_mask': torch.tensor([[1, 1, 1, 1, 1]])
                            }
                    
                    # Test inference
                    model.eval()
                    with torch.no_grad():
                        outputs = model(**test_inputs)
                        original_outputs[str(checkpoint_path)] = outputs.logits[:, :5]  # Save first 5 tokens
                    
                    # Clean up
                    del model
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    
                else:
                    validation_results['issues'].append(f"Checkpoint path is not a directory: {checkpoint_path}")
                    validation_results['loading_successful'] = False
            
            except Exception as e:
                validation_results['issues'].append(f"Cannot load checkpoint {checkpoint_path}: {e}")
                validation_results['loading_successful'] = False
                logger.error(f"❌ Failed to load checkpoint {checkpoint_path}: {e}")
        
        # Check output consistency between reloads
        if len(original_outputs) > 1:
            output_values = list(original_outputs.values())
            for i in range(1, len(output_values)):
                diff = torch.abs(output_values[0] - output_values[i]).max().item()
                if diff > 1e-5:  # Allow small numerical differences
                    validation_results['issues'].append(f"Output inconsistency detected: max diff {diff}")
                    validation_results['output_consistency'] = False
        
        return validation_results
    
    def validate_training_stability(self, runs_data: List[Dict]) -> Dict[str, Any]:
        """Check for training stability issues and red flags."""
        logger.info("Validating training stability...")
        
        validation_results = {
            'stable_training': True,
            'loss_decreasing': True,
            'no_memory_errors': True,
            'issues': [],
            'warning_signs': []
        }
        
        for run in runs_data[:5]:  # Check first 5 runs
            run_name = run.get('name', 'unknown')
            summary = run.get('summary', {})
            
            # Check final losses
            train_loss = summary.get('train_loss')
            eval_loss = summary.get('eval_loss')
            
            if train_loss is not None:
                if train_loss > 10.0:  # Very high final loss
                    validation_results['issues'].append(f"{run_name}: High final train loss: {train_loss:.3f}")
                    validation_results['stable_training'] = False
                elif train_loss > 5.0:
                    validation_results['warning_signs'].append(f"{run_name}: Elevated train loss: {train_loss:.3f}")
            
            if eval_loss is not None:
                if eval_loss > 10.0:  # Very high final eval loss
                    validation_results['issues'].append(f"{run_name}: High final eval loss: {eval_loss:.3f}")
                    validation_results['stable_training'] = False
            
            # Check for loss progression (if available)
            if train_loss is not None and eval_loss is not None:
                if eval_loss > train_loss * 2:  # Significant overfitting
                    validation_results['warning_signs'].append(f"{run_name}: Possible overfitting (eval/train loss ratio: {eval_loss/train_loss:.2f})")
            
            # Check run state for errors
            run_state = run.get('state')
            if run_state == 'failed' or run_state == 'crashed':
                validation_results['issues'].append(f"{run_name}: Run failed or crashed")
                validation_results['stable_training'] = False
        
        return validation_results
    
    def generate_validation_report(self) -> str:
        """Generate comprehensive validation report."""
        report = []
        report.append("="*80)
        report.append("FULL FINE-TUNING VALIDATION REPORT")
        report.append("="*80)
        report.append(f"Generated: {datetime.now()}")
        report.append("")
        
        # Overall status
        all_checks_passed = True
        for check_results in self.validation_results.values():
            if isinstance(check_results, dict) and check_results.get('issues'):
                all_checks_passed = False
                break
        
        status = "✅ PASSED" if all_checks_passed else "❌ FAILED"
        report.append(f"Overall Status: {status}")
        report.append("")
        
        # Individual checks
        for check_name, results in self.validation_results.items():
            report.append(f"## {check_name.upper()}")
            report.append("-" * 40)
            
            if isinstance(results, dict):
                # Summary status
                has_issues = bool(results.get('issues', []))
                check_status = "❌ FAILED" if has_issues else "✅ PASSED"
                report.append(f"Status: {check_status}")
                
                # Key metrics
                if 'runs_data' in results and results['runs_data']:
                    report.append(f"Runs found: {len(results['runs_data'])}")
                
                if 'tasks_validated' in results:
                    for task, task_data in results['tasks_validated'].items():
                        if 'mean_performance' in task_data:
                            report.append(f"{task}: {task_data['mean_performance']:.3f} ± {task_data['std_performance']:.3f}")
                
                # Issues
                if results.get('issues'):
                    report.append("Issues:")
                    for issue in results['issues']:
                        report.append(f"  - {issue}")
                
                # Warning signs
                if results.get('warning_signs'):
                    report.append("Warnings:")
                    for warning in results['warning_signs']:
                        report.append(f"  - {warning}")
            
            report.append("")
        
        return "\n".join(report)
    
    def run_full_validation(self) -> bool:
        """Run complete validation suite."""
        logger.info("Starting comprehensive full fine-tuning validation...")
        
        # 1. Validate W&B monitoring
        self.validation_results['wandb_monitoring'] = self.validate_wandb_monitoring()
        
        # 2. Validate performance metrics
        runs_data = self.validation_results['wandb_monitoring'].get('runs_data', [])
        self.validation_results['performance_metrics'] = self.validate_performance_metrics(runs_data)
        
        # 3. Validate representation extraction
        self.validation_results['representation_extraction'] = self.validate_representation_extraction()
        
        # 4. Validate checkpoint integrity
        self.validation_results['checkpoint_integrity'] = self.validate_checkpoint_integrity()
        
        # 5. Validate training stability
        self.validation_results['training_stability'] = self.validate_training_stability(runs_data)
        
        # Generate and save report
        report = self.generate_validation_report()
        
        # Save report
        report_path = self.results_dir / "validation_report.txt"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(report)
        logger.info(f"Validation report saved to: {report_path}")
        
        # Return overall success
        return all(
            not results.get('issues', []) 
            for results in self.validation_results.values() 
            if isinstance(results, dict)
        )


def main():
    """Main validation function."""
    logger.info("Starting full fine-tuning validation according to Step 3 requirements...")
    
    # Set up environment
    os.environ.setdefault('WANDB_PROJECT', 'NLP')
    os.environ.setdefault('WANDB_ENTITY', 'galavny-tel-aviv-university')
    
    # Initialize validator
    validator = FullFinetuneValidator()
    
    # Run validation
    success = validator.run_full_validation()
    
    if success:
        logger.info("🎉 ALL VALIDATION CHECKS PASSED!")
        logger.info("Full fine-tuning experiments are working correctly")
    else:
        logger.error("❌ VALIDATION FAILED!")
        logger.error("Some components need attention before proceeding")
    
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
