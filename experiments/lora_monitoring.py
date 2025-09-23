#!/usr/bin/env python3
"""
LoRA Training Monitoring and Red Flag Detection
Real-time monitoring during production LoRA experiments
"""

import os
import sys
import torch
import wandb
import numpy as np
import json
import warnings
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class LoRATrainingMonitor:
    """Monitors LoRA training for red flags and performance issues."""
    
    def __init__(self, model: torch.nn.Module, log_interval: int = 50):
        self.model = model
        self.log_interval = log_interval
        self.step_count = 0
        self.red_flags = []
        self.warnings = []
        self.performance_history = []
        
        # Initialize parameter tracking
        self.baseline_param_count = self._count_parameters()
        self.baseline_frozen_check = self._verify_base_model_frozen()
        
        print(f"🔍 LoRA Training Monitor initialized")
        print(f"   Baseline trainable parameters: {self.baseline_param_count['trainable']:,}")
        print(f"   Baseline frozen check: {'✅' if self.baseline_frozen_check['all_frozen'] else '🚨'}")
    
    def _count_parameters(self) -> Dict[str, int]:
        """Count model parameters."""
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        lora_trainable = 0
        base_trainable = 0
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if any(key in name for key in ['lora_', 'adapter_']):
                    lora_trainable += param.numel()
                else:
                    base_trainable += param.numel()
        
        return {
            'total': total,
            'trainable': trainable,
            'lora_trainable': lora_trainable,
            'base_trainable': base_trainable,
            'trainable_ratio': trainable / total if total > 0 else 0
        }
    
    def _verify_base_model_frozen(self) -> Dict[str, Any]:
        """Verify base model parameters remain frozen."""
        base_params_with_grad = []
        lora_params_with_grad = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if any(key in name for key in ['lora_', 'adapter_']):
                    lora_params_with_grad.append(name)
                else:
                    base_params_with_grad.append(name)
        
        return {
            'all_frozen': len(base_params_with_grad) == 0,
            'base_params_with_grad': base_params_with_grad,
            'lora_params_with_grad': len(lora_params_with_grad),
            'unfrozen_count': len(base_params_with_grad)
        }
    
    def _check_gradient_health(self) -> Dict[str, Any]:
        """Check gradient health and detect issues."""
        grad_norms = []
        zero_grad_params = []
        exploding_grad_params = []
        lora_grad_norms = []
        
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_norms.append(grad_norm)
                
                if any(key in name for key in ['lora_', 'adapter_']):
                    lora_grad_norms.append(grad_norm)
                
                if grad_norm == 0:
                    zero_grad_params.append(name)
                elif grad_norm > 10.0:  # Threshold for exploding gradients
                    exploding_grad_params.append((name, grad_norm))
        
        return {
            'total_grad_norm': np.sum([g**2 for g in grad_norms])**0.5 if grad_norms else 0,
            'mean_grad_norm': np.mean(grad_norms) if grad_norms else 0,
            'max_grad_norm': np.max(grad_norms) if grad_norms else 0,
            'lora_mean_grad_norm': np.mean(lora_grad_norms) if lora_grad_norms else 0,
            'zero_grad_count': len(zero_grad_params),
            'exploding_grad_count': len(exploding_grad_params),
            'exploding_grads': exploding_grad_params[:5],  # First 5
            'gradient_health': len(exploding_grad_params) == 0 and np.mean(grad_norms) > 0 if grad_norms else False
        }
    
    def _analyze_adapter_weights(self) -> Dict[str, Any]:
        """Analyze adapter weight magnitudes and distributions."""
        try:
            from models.lora_utils_simple import LoRAAnalyzer
            analyzer = LoRAAnalyzer(self.model)
            
            adapter_stats = analyzer.compute_adapter_statistics()
            rank_stats = analyzer.analyze_rank_utilization()
            
            # Check for weight issues
            weight_issues = []
            if adapter_stats:
                # Check for zero weights (only after initial training steps)
                if step > 50:  # Allow proper initialization time
                    if abs(adapter_stats.get('adapter_weight_mean_A', 0)) < 1e-8:
                        weight_issues.append("LoRA_A weights stuck at zero after training")
                    if abs(adapter_stats.get('adapter_weight_mean_B', 0)) < 1e-8:
                        weight_issues.append("LoRA_B weights stuck at zero after training")
                
                # Check for exploding weights
                if abs(adapter_stats.get('adapter_weight_mean_A', 0)) > 5.0:
                    weight_issues.append("LoRA_A weights too large")
                if abs(adapter_stats.get('adapter_weight_mean_B', 0)) > 5.0:
                    weight_issues.append("LoRA_B weights too large")
                
                # Check adapter norms
                max_norm = adapter_stats.get('adapter_norm_max', 0)
                if max_norm > 50.0:
                    weight_issues.append("Adapter norms too large")
            
            return {
                'adapter_stats': adapter_stats,
                'rank_stats': rank_stats,
                'weight_issues': weight_issues,
                'weights_healthy': len(weight_issues) == 0,
                'num_adapters': adapter_stats.get('num_adapters', 0) if adapter_stats else 0
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'weights_healthy': False
            }
    
    def check_step(self, step: int, loss: float, learning_rate: float, 
                  logs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Check training step for red flags and issues."""
        self.step_count = step
        
        # Parameter verification
        param_count = self._count_parameters()
        frozen_check = self._verify_base_model_frozen()
        gradient_health = self._check_gradient_health()
        adapter_analysis = self._analyze_adapter_weights()
        
        # Detect red flags
        step_red_flags = []
        step_warnings = []
        
        # Red flag: Base model not frozen
        if not frozen_check['all_frozen']:
            step_red_flags.append(f"Step {step}: Base model parameters unfrozen ({frozen_check['unfrozen_count']} params)")
        
        # Red flag: Parameter ratio changed significantly
        ratio_change = abs(param_count['trainable_ratio'] - self.baseline_param_count['trainable_ratio'])
        if ratio_change > 0.001:  # More than 0.1% change
            step_red_flags.append(f"Step {step}: Trainable parameter ratio changed by {ratio_change:.6f}")
        
        # Red flag: Gradient issues
        if not gradient_health['gradient_health']:
            if gradient_health['exploding_grad_count'] > 0:
                step_red_flags.append(f"Step {step}: Exploding gradients detected ({gradient_health['exploding_grad_count']} params)")
            elif gradient_health['mean_grad_norm'] == 0:
                step_red_flags.append(f"Step {step}: No gradients detected")
        
        # Red flag: Adapter weight issues
        if not adapter_analysis['weights_healthy']:
            weight_issues = adapter_analysis.get('weight_issues', [])
            for issue in weight_issues[:3]:  # First 3 issues
                step_red_flags.append(f"Step {step}: {issue}")
        
        # Warnings: Performance issues
        if loss > 100:  # Very high loss
            step_warnings.append(f"Step {step}: Very high loss ({loss:.4f})")
        
        if gradient_health['total_grad_norm'] > 1.0:
            step_warnings.append(f"Step {step}: High gradient norm ({gradient_health['total_grad_norm']:.4f})")
        
        # Store flags and warnings
        self.red_flags.extend(step_red_flags)
        self.warnings.extend(step_warnings)
        
        # Performance history
        performance_entry = {
            'step': step,
            'loss': loss,
            'learning_rate': learning_rate,
            'trainable_ratio': param_count['trainable_ratio'],
            'gradient_norm': gradient_health['total_grad_norm'],
            'adapter_count': adapter_analysis.get('num_adapters', 0),
            'red_flags': len(step_red_flags),
            'warnings': len(step_warnings)
        }
        self.performance_history.append(performance_entry)
        
        # Log to W&B if available
        if wandb.run is not None and step % self.log_interval == 0:
            self._log_to_wandb(step, param_count, frozen_check, gradient_health, adapter_analysis)
        
        # Print alerts
        if step_red_flags:
            print(f"🚨 RED FLAGS at step {step}:")
            for flag in step_red_flags:
                print(f"   - {flag}")
        
        if step_warnings:
            print(f"⚠️  WARNINGS at step {step}:")
            for warning in step_warnings:
                print(f"   - {warning}")
        
        return {
            'step': step,
            'red_flags': step_red_flags,
            'warnings': step_warnings,
            'param_count': param_count,
            'frozen_check': frozen_check,
            'gradient_health': gradient_health,
            'adapter_analysis': adapter_analysis,
            'total_red_flags': len(self.red_flags),
            'total_warnings': len(self.warnings)
        }
    
    def _log_to_wandb(self, step: int, param_count: Dict, frozen_check: Dict, 
                     gradient_health: Dict, adapter_analysis: Dict):
        """Log LoRA-specific metrics to W&B."""
        
        # Parameter efficiency metrics
        wandb.log({
            'lora_monitoring/trainable_parameters': param_count['trainable'],
            'lora_monitoring/trainable_ratio': param_count['trainable_ratio'],
            'lora_monitoring/lora_trainable': param_count['lora_trainable'],
            'lora_monitoring/base_trainable': param_count['base_trainable'],
        }, step=step)
        
        # Frozen verification
        wandb.log({
            'lora_verification/base_model_frozen': frozen_check['all_frozen'],
            'lora_verification/unfrozen_param_count': frozen_check['unfrozen_count'],
            'lora_verification/lora_params_with_grad': frozen_check['lora_params_with_grad'],
        }, step=step)
        
        # Gradient health
        wandb.log({
            'lora_gradients/total_norm': gradient_health['total_grad_norm'],
            'lora_gradients/mean_norm': gradient_health['mean_grad_norm'],
            'lora_gradients/max_norm': gradient_health['max_grad_norm'],
            'lora_gradients/lora_mean_norm': gradient_health['lora_mean_grad_norm'],
            'lora_gradients/zero_grad_count': gradient_health['zero_grad_count'],
            'lora_gradients/exploding_grad_count': gradient_health['exploding_grad_count'],
            'lora_gradients/gradient_health': gradient_health['gradient_health'],
        }, step=step)
        
        # Adapter analysis
        if adapter_analysis.get('adapter_stats'):
            stats = adapter_analysis['adapter_stats']
            wandb.log({
                'lora_adapters/num_adapters': stats.get('num_adapters', 0),
                'lora_adapters/weight_mean_A': stats.get('adapter_weight_mean_A', 0),
                'lora_adapters/weight_std_A': stats.get('adapter_weight_std_A', 0),
                'lora_adapters/weight_mean_B': stats.get('adapter_weight_mean_B', 0),
                'lora_adapters/weight_std_B': stats.get('adapter_weight_std_B', 0),
                'lora_adapters/norm_mean': stats.get('adapter_norm_mean', 0),
                'lora_adapters/norm_max': stats.get('adapter_norm_max', 0),
            }, step=step)
        
        if adapter_analysis.get('rank_stats'):
            rank_stats = adapter_analysis['rank_stats']
            wandb.log({
                'lora_rank/utilization_mean': rank_stats.get('rank_utilization_mean', 0),
                'lora_rank/utilization_std': rank_stats.get('rank_utilization_std', 0),
                'lora_rank/utilization_max': rank_stats.get('rank_utilization_max', 0),
            }, step=step)
        
        # Red flags and warnings
        wandb.log({
            'lora_monitoring/total_red_flags': len(self.red_flags),
            'lora_monitoring/total_warnings': len(self.warnings),
            'lora_monitoring/weights_healthy': adapter_analysis.get('weights_healthy', False),
        }, step=step)
    
    def get_summary_report(self) -> Dict[str, Any]:
        """Generate summary report of training monitoring."""
        return {
            'total_steps_monitored': self.step_count,
            'total_red_flags': len(self.red_flags),
            'total_warnings': len(self.warnings),
            'red_flags': self.red_flags,
            'warnings': self.warnings,
            'performance_history': self.performance_history,
            'baseline_params': self.baseline_param_count,
            'final_validation': self._verify_base_model_frozen(),
            'monitoring_summary': {
                'baseline_frozen': self.baseline_frozen_check['all_frozen'],
                'final_frozen': self._verify_base_model_frozen()['all_frozen'],
                'training_stable': len(self.red_flags) == 0,
                'recommended_action': 'Continue' if len(self.red_flags) == 0 else 'Investigate red flags'
            }
        }
    
    def save_monitoring_report(self, filepath: str):
        """Save monitoring report to file."""
        report = self.get_summary_report()
        report['timestamp'] = datetime.now().isoformat()
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"📊 Monitoring report saved to: {filepath}")


class LoRAPerformanceComparator:
    """Compares LoRA performance against full fine-tuning targets."""
    
    def __init__(self, target_metrics: Dict[str, float]):
        """
        Initialize with target metrics from full fine-tuning.
        
        Args:
            target_metrics: Dict with keys like 'accuracy', 'f1', 'loss' etc.
        """
        self.target_metrics = target_metrics
        self.tolerance = 0.03  # 3% tolerance as specified
        
    def compare_performance(self, lora_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Compare LoRA performance against targets."""
        comparison = {}
        
        for metric_name, target_value in self.target_metrics.items():
            if metric_name in lora_metrics:
                lora_value = lora_metrics[metric_name]
                
                # For accuracy/F1, higher is better
                if metric_name in ['accuracy', 'f1', 'eval_accuracy', 'eval_f1']:
                    diff = target_value - lora_value
                    within_tolerance = diff <= self.tolerance
                    performance_gap = diff / target_value if target_value > 0 else 0
                
                # For loss, lower is better
                elif 'loss' in metric_name:
                    diff = lora_value - target_value
                    within_tolerance = diff <= self.tolerance * target_value if target_value > 0 else lora_value <= self.tolerance
                    performance_gap = diff / target_value if target_value > 0 else 0
                
                else:
                    # Generic comparison
                    diff = abs(target_value - lora_value)
                    within_tolerance = diff <= self.tolerance * abs(target_value) if target_value != 0 else diff <= self.tolerance
                    performance_gap = diff / abs(target_value) if target_value != 0 else diff
                
                comparison[metric_name] = {
                    'target': target_value,
                    'lora': lora_value,
                    'difference': diff,
                    'performance_gap_percent': performance_gap * 100,
                    'within_tolerance': within_tolerance,
                    'status': '✅' if within_tolerance else '❌'
                }
        
        # Overall assessment
        all_within_tolerance = all(comp['within_tolerance'] for comp in comparison.values())
        max_gap = max((comp['performance_gap_percent'] for comp in comparison.values()), default=0)
        
        comparison['overall'] = {
            'all_within_tolerance': all_within_tolerance,
            'max_performance_gap_percent': max_gap,
            'status': '✅ PASSED' if all_within_tolerance else '❌ FAILED',
            'recommendation': 'LoRA ready for production' if all_within_tolerance else 'Adjust LoRA hyperparameters'
        }
        
        return comparison


def create_monitoring_dashboard_config() -> Dict[str, Any]:
    """Create W&B dashboard configuration for LoRA monitoring."""
    return {
        "sections": [
            {
                "name": "LoRA Parameter Efficiency",
                "panels": [
                    {"type": "line", "metric": "lora_monitoring/trainable_ratio", "title": "Trainable Parameter Ratio"},
                    {"type": "line", "metric": "lora_monitoring/trainable_parameters", "title": "Trainable Parameters"},
                    {"type": "scalar", "metric": "lora_verification/base_model_frozen", "title": "Base Model Frozen"},
                ]
            },
            {
                "name": "LoRA Adapter Analysis", 
                "panels": [
                    {"type": "line", "metric": "lora_adapters/weight_mean_A", "title": "LoRA A Weight Mean"},
                    {"type": "line", "metric": "lora_adapters/weight_mean_B", "title": "LoRA B Weight Mean"},
                    {"type": "line", "metric": "lora_adapters/norm_mean", "title": "Adapter Norm Mean"},
                    {"type": "line", "metric": "lora_rank/utilization_mean", "title": "Rank Utilization"},
                ]
            },
            {
                "name": "LoRA Training Health",
                "panels": [
                    {"type": "line", "metric": "lora_gradients/total_norm", "title": "Total Gradient Norm"},
                    {"type": "scalar", "metric": "lora_gradients/gradient_health", "title": "Gradient Health"},
                    {"type": "line", "metric": "lora_monitoring/total_red_flags", "title": "Red Flags Count"},
                    {"type": "line", "metric": "lora_monitoring/total_warnings", "title": "Warnings Count"},
                ]
            }
        ]
    }


def main():
    """Example usage of LoRA monitoring."""
    print("🔍 LoRA Monitoring and Red Flag Detection")
    print("This module provides real-time monitoring during LoRA training.")
    print("\nKey features:")
    print("- Real-time parameter efficiency tracking")
    print("- Base model frozen verification")
    print("- Gradient health monitoring")
    print("- Adapter weight analysis")
    print("- Red flag detection")
    print("- W&B metrics logging")
    print("- Performance comparison vs full fine-tuning")
    
    # Create example monitoring dashboard config
    dashboard_config = create_monitoring_dashboard_config()
    
    with open("lora_monitoring_dashboard.json", "w") as f:
        json.dump(dashboard_config, f, indent=2)
    
    print(f"\n📊 W&B dashboard config saved to: lora_monitoring_dashboard.json")
    print("\nTo use in production:")
    print("1. Import LoRATrainingMonitor in your training script")
    print("2. Initialize monitor with your LoRA model")
    print("3. Call monitor.check_step() at each training step")
    print("4. Review monitor.get_summary_report() after training")


if __name__ == "__main__":
    main()
