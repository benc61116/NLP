#!/usr/bin/env python3
"""Red flags monitoring for full fine-tuning experiments as specified in Step 3 validation."""

import os
import sys
import torch
import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import wandb

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RedFlagMonitor:
    """Monitor for red flags in full fine-tuning experiments."""
    
    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.red_flags = []
        self.warnings = []
        
        # Performance thresholds from Step 3 requirements
        self.performance_thresholds = {
            'mrpc': {'min_accuracy': 0.85, 'max_accuracy': 0.90},
            'sst2': {'min_accuracy': 0.90, 'max_accuracy': 0.93},
            'rte': {'min_accuracy': 0.65, 'max_accuracy': 0.75},
            'squad_v2': {'min_f1': 0.75, 'max_f1': 0.85}
        }
    
    def check_training_loss_stability(self, runs_data: List[Dict]) -> List[str]:
        """Check for unstable or non-decreasing training loss."""
        logger.info("Checking training loss stability...")
        issues = []
        
        for run in runs_data:
            run_name = run.get('name', 'unknown')
            summary = run.get('summary', {})
            
            # Check final training loss
            train_loss = summary.get('train_loss')
            if train_loss is not None:
                if train_loss > 10.0:
                    issues.append(f"🚨 {run_name}: Very high final training loss: {train_loss:.3f}")
                elif train_loss > 5.0:
                    self.warnings.append(f"⚠ {run_name}: High training loss: {train_loss:.3f}")
                elif train_loss < 0.001:
                    self.warnings.append(f"⚠ {run_name}: Suspiciously low training loss (possible overfitting): {train_loss:.6f}")
            
            # Check evaluation loss
            eval_loss = summary.get('eval_loss')
            if eval_loss is not None and train_loss is not None:
                loss_ratio = eval_loss / train_loss
                if loss_ratio > 3.0:
                    issues.append(f"🚨 {run_name}: Large eval/train loss gap (ratio: {loss_ratio:.2f}) - possible overfitting")
                elif loss_ratio > 2.0:
                    self.warnings.append(f"⚠ {run_name}: Moderate eval/train loss gap (ratio: {loss_ratio:.2f})")
            
            # Check for NaN or infinite losses
            for loss_type in ['train_loss', 'eval_loss']:
                loss_value = summary.get(loss_type)
                if loss_value is not None:
                    if np.isnan(loss_value) or np.isinf(loss_value):
                        issues.append(f"🚨 {run_name}: Invalid {loss_type}: {loss_value}")
        
        return issues
    
    def check_performance_expectations(self, runs_data: List[Dict]) -> List[str]:
        """Check if performance is much lower than expected ranges."""
        logger.info("Checking performance against expected ranges...")
        issues = []
        
        # Group runs by task
        task_runs = {}
        for run in runs_data:
            config = run.get('config', {})
            task_name = config.get('task_name')
            
            if task_name in self.performance_thresholds:
                if task_name not in task_runs:
                    task_runs[task_name] = []
                task_runs[task_name].append(run)
        
        for task_name, task_thresholds in self.performance_thresholds.items():
            if task_name not in task_runs:
                continue
            
            for run in task_runs[task_name]:
                run_name = run.get('name', 'unknown')
                summary = run.get('summary', {})
                
                # Get appropriate metric
                if task_name == 'squad_v2':
                    metric_value = summary.get('eval_f1', summary.get('f1'))
                    min_threshold = task_thresholds['min_f1']
                    metric_name = 'F1'
                else:
                    metric_value = summary.get('eval_accuracy', summary.get('accuracy'))
                    min_threshold = task_thresholds['min_accuracy']
                    metric_name = 'Accuracy'
                
                if metric_value is not None:
                    # Check for very low performance (red flag)
                    very_low_threshold = min_threshold * 0.7  # 30% below minimum
                    low_threshold = min_threshold * 0.85      # 15% below minimum
                    
                    if metric_value < very_low_threshold:
                        issues.append(f"🚨 {run_name} ({task_name}): Very low {metric_name}: {metric_value:.3f} "
                                    f"(expected ≥{min_threshold:.2f})")
                    elif metric_value < low_threshold:
                        self.warnings.append(f"⚠ {run_name} ({task_name}): Low {metric_name}: {metric_value:.3f} "
                                           f"(expected ≥{min_threshold:.2f})")
        
        return issues
    
    def check_representation_files(self) -> List[str]:
        """Check for missing or corrupted representation files."""
        logger.info("Checking representation files...")
        issues = []
        
        repr_dirs = list(self.results_dir.glob("**/representations"))
        if not repr_dirs:
            issues.append("🚨 No representation directories found - extraction may have failed")
            return issues
        
        for repr_dir in repr_dirs:
            for task_dir in repr_dir.iterdir():
                if not task_dir.is_dir():
                    continue
                
                task_name = task_dir.name
                step_dirs = [d for d in task_dir.iterdir() if d.is_dir() and d.name.startswith('step_')]
                
                if not step_dirs:
                    issues.append(f"🚨 {task_name}: No step directories found in representations")
                    continue
                
                # Check file integrity in a sample of step directories
                for step_dir in step_dirs[:3]:  # Check first 3
                    pt_files = list(step_dir.glob("*.pt"))
                    
                    if not pt_files:
                        issues.append(f"🚨 {task_name}/{step_dir.name}: No .pt files found")
                        continue
                    
                    # Check for empty or corrupted files
                    total_size = 0
                    corrupt_files = 0
                    
                    for pt_file in pt_files:
                        try:
                            file_size = pt_file.stat().st_size
                            total_size += file_size
                            
                            if file_size == 0:
                                issues.append(f"🚨 {task_name}: Empty representation file: {pt_file.name}")
                                corrupt_files += 1
                            elif file_size < 1000:  # Very small file
                                self.warnings.append(f"⚠ {task_name}: Suspiciously small file: {pt_file.name} ({file_size} bytes)")
                            
                            # Try to load tensor
                            tensor = torch.load(pt_file, map_location='cpu')
                            if tensor.numel() == 0:
                                issues.append(f"🚨 {task_name}: Empty tensor in {pt_file.name}")
                                corrupt_files += 1
                        
                        except Exception as e:
                            issues.append(f"🚨 {task_name}: Cannot load {pt_file.name}: {e}")
                            corrupt_files += 1
                    
                    # Check metadata
                    metadata_files = list(step_dir.glob("metadata.json"))
                    if not metadata_files:
                        self.warnings.append(f"⚠ {task_name}/{step_dir.name}: No metadata.json found")
                    else:
                        try:
                            with open(metadata_files[0]) as f:
                                metadata = json.load(f)
                            
                            # Check metadata consistency
                            if 'tensor_shapes' not in metadata:
                                self.warnings.append(f"⚠ {task_name}: Metadata missing tensor_shapes")
                        except Exception as e:
                            issues.append(f"🚨 {task_name}: Cannot load metadata: {e}")
        
        return issues
    
    def check_checkpoint_loading(self) -> List[str]:
        """Check for checkpoint loading failures."""
        logger.info("Checking checkpoint loading...")
        issues = []
        
        # Find checkpoint directories
        checkpoint_patterns = ["**/final_model", "**/pytorch_model.bin"]
        checkpoint_dirs = []
        
        for pattern in checkpoint_patterns:
            checkpoint_dirs.extend(list(self.results_dir.glob(pattern)))
        
        if not checkpoint_dirs:
            issues.append("🚨 No model checkpoints found")
            return issues
        
        # Test loading a few checkpoints
        test_count = min(3, len(checkpoint_dirs))
        loading_failures = 0
        
        for checkpoint_path in checkpoint_dirs[:test_count]:
            try:
                if checkpoint_path.is_dir():
                    # Try to load model
                    from transformers import AutoModelForCausalLM
                    model = AutoModelForCausalLM.from_pretrained(str(checkpoint_path))
                    
                    # Quick inference test
                    test_input = torch.tensor([[1, 2, 3, 4, 5]])
                    with torch.no_grad():
                        output = model(test_input)
                    
                    # Check for NaN outputs
                    if torch.isnan(output.logits).any():
                        issues.append(f"🚨 Checkpoint produces NaN outputs: {checkpoint_path}")
                    
                    del model
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
            except Exception as e:
                issues.append(f"🚨 Cannot load checkpoint {checkpoint_path}: {e}")
                loading_failures += 1
        
        if loading_failures == test_count:
            issues.append(f"🚨 All tested checkpoints ({test_count}) failed to load")
        elif loading_failures > 0:
            self.warnings.append(f"⚠ {loading_failures}/{test_count} checkpoints failed to load")
        
        return issues
    
    def check_memory_and_oom_issues(self, runs_data: List[Dict]) -> List[str]:
        """Check for memory errors and OOM issues."""
        logger.info("Checking for memory and OOM issues...")
        issues = []
        
        for run in runs_data:
            run_name = run.get('name', 'unknown')
            run_state = run.get('state')
            summary = run.get('summary', {})
            
            # Check for failed runs (potential OOM)
            if run_state in ['failed', 'crashed']:
                issues.append(f"🚨 {run_name}: Run failed or crashed (possible OOM or memory error)")
            
            # Check memory usage if available
            max_memory_mb = summary.get('gpu_0_memory_max_allocated_mb')
            if max_memory_mb is not None:
                if max_memory_mb > 22000:  # Close to 24GB limit
                    self.warnings.append(f"⚠ {run_name}: High GPU memory usage: {max_memory_mb:.0f} MB")
            
            # Check for gradient issues
            grad_norm = summary.get('gradient_norm_total')
            if grad_norm is not None:
                if grad_norm > 1000:
                    issues.append(f"🚨 {run_name}: Very high gradient norm: {grad_norm:.2f} (possible gradient explosion)")
                elif grad_norm < 1e-8:
                    issues.append(f"🚨 {run_name}: Very low gradient norm: {grad_norm:.2e} (possible vanishing gradients)")
        
        return issues
    
    def check_wandb_connectivity(self, runs_data: List[Dict] = None) -> List[str]:
        """Check W&B connectivity and logging issues."""
        logger.info("Checking W&B connectivity...")
        issues = []
        
        try:
            # Test W&B connection
            api = wandb.Api()
            
            # Try to access the project
            default_entity = os.environ.get('WANDB_ENTITY', 'galavny-tel-aviv-university')
            project_name = f"{wandb.api.default_entity or default_entity}/NLP"
            runs = list(api.runs(project_name))
            
            # Count runs in different states
            recent_runs = []
            for run in runs:
                if len(recent_runs) >= 20:  # Limit check
                    break
                recent_runs.append(run)
            
            if not recent_runs:
                self.warnings.append("⚠ No runs found in W&B project")
            
            # Check for incomplete logging
            incomplete_runs = []
            for run in recent_runs[:10]:  # Check recent 10
                summary = dict(run.summary)
                if not summary:
                    incomplete_runs.append(run.name)
            
            if incomplete_runs:
                self.warnings.append(f"⚠ Runs with incomplete logging: {len(incomplete_runs)}")
        
        except Exception as e:
            issues.append(f"🚨 W&B connectivity issue: {e}")
        
        return issues
    
    def generate_red_flag_report(self) -> str:
        """Generate comprehensive red flag report."""
        report = []
        report.append("🚨 RED FLAGS MONITORING REPORT")
        report.append("=" * 60)
        report.append(f"Generated: {datetime.now()}")
        report.append("")
        
        total_red_flags = len(self.red_flags)
        total_warnings = len(self.warnings)
        
        if total_red_flags == 0 and total_warnings == 0:
            report.append("✅ NO RED FLAGS OR WARNINGS DETECTED")
            report.append("All systems appear to be functioning normally.")
        else:
            report.append(f"Summary: {total_red_flags} red flags, {total_warnings} warnings")
        
        if self.red_flags:
            report.append("")
            report.append("🚨 CRITICAL ISSUES (RED FLAGS)")
            report.append("-" * 40)
            for flag in self.red_flags:
                report.append(f"  {flag}")
        
        if self.warnings:
            report.append("")
            report.append("⚠ WARNINGS")
            report.append("-" * 40)
            for warning in self.warnings:
                report.append(f"  {warning}")
        
        report.append("")
        report.append("📋 MONITORING CHECKLIST")
        report.append("-" * 40)
        report.append("✓ Training loss stability")
        report.append("✓ Performance expectations")
        report.append("✓ Representation file integrity")
        report.append("✓ Checkpoint loading")
        report.append("✓ Memory and OOM issues")
        report.append("✓ W&B connectivity")
        
        return "\n".join(report)
    
    def run_monitoring(self, runs_data: Optional[List[Dict]] = None) -> Tuple[int, int]:
        """Run complete red flag monitoring."""
        logger.info("Starting red flag monitoring...")
        
        # Get runs data if not provided
        if runs_data is None:
            try:
                api = wandb.Api()
                default_entity = os.environ.get('WANDB_ENTITY', 'galavny-tel-aviv-university')
                project_name = f"{wandb.api.default_entity or default_entity}/NLP"
                runs = list(api.runs(project_name))
                runs_data = []
                for run in runs:
                    if len(runs_data) >= 20:
                        break
                    if "full_finetune" in run.tags or "full_ft" in run.name:
                        runs_data.append({
                            'name': run.name,
                            'state': run.state,
                            'config': dict(run.config),
                            'summary': dict(run.summary)
                        })
            except Exception as e:
                logger.warning(f"Cannot access W&B runs: {e}")
                runs_data = []
        
        # Run all checks
        checks = [
            self.check_training_loss_stability,
            self.check_performance_expectations,
            self.check_memory_and_oom_issues,
            self.check_wandb_connectivity
        ]
        
        for check_func in checks:
            try:
                issues = check_func(runs_data)
                self.red_flags.extend(issues)
            except Exception as e:
                self.red_flags.append(f"🚨 Monitoring check failed: {check_func.__name__}: {e}")
        
        # File-based checks
        file_checks = [
            self.check_representation_files,
            self.check_checkpoint_loading
        ]
        
        for check_func in file_checks:
            try:
                issues = check_func()
                self.red_flags.extend(issues)
            except Exception as e:
                self.red_flags.append(f"🚨 File check failed: {check_func.__name__}: {e}")
        
        return len(self.red_flags), len(self.warnings)


def main():
    """Main monitoring function."""
    logger.info("Starting red flag monitoring for full fine-tuning experiments...")
    
    # Set up environment
    os.environ.setdefault('WANDB_PROJECT', 'NLP')
    default_entity = os.environ.get('WANDB_ENTITY', 'galavny-tel-aviv-university')
    os.environ.setdefault('WANDB_ENTITY', default_entity)
    
    # Initialize monitor
    monitor = RedFlagMonitor()
    
    # Run monitoring
    red_flags_count, warnings_count = monitor.run_monitoring()
    
    # Generate and display report
    report = monitor.generate_red_flag_report()
    print(report)
    
    # Save report
    report_path = Path("results") / "red_flags_report.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w') as f:
        f.write(report)
    
    logger.info(f"Red flags report saved to: {report_path}")
    
    # Exit code based on issues found
    if red_flags_count > 0:
        logger.error(f"❌ {red_flags_count} critical issues detected!")
        return 1
    elif warnings_count > 0:
        logger.warning(f"⚠ {warnings_count} warnings detected")
        return 0
    else:
        logger.info("✅ No issues detected")
        return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
