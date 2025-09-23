#!/usr/bin/env python3
"""Custom trainer utilities for LoRA research project."""

import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional, List, Tuple
from transformers import Trainer, TrainerCallback, AutoModelForCausalLM
import logging
import numpy as np
import time
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class ParameterEfficiencyTracker:
    """Tracks parameter efficiency metrics for LoRA vs full fine-tuning."""
    
    def __init__(self, model: torch.nn.Module, method: str):
        self.model = model
        self.method = method
        self.total_params = self.count_total_parameters()
        self.trainable_params = self.count_trainable_parameters()
        
    def count_total_parameters(self) -> int:
        """Count total parameters in the model."""
        return sum(p.numel() for p in self.model.parameters())
    
    def count_trainable_parameters(self) -> int:
        """Count trainable parameters in the model."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def get_efficiency_metrics(self) -> Dict[str, float]:
        """Get parameter efficiency metrics."""
        trainable_ratio = self.trainable_params / self.total_params if self.total_params > 0 else 0
        
        return {
            'total_parameters': self.total_params,
            'trainable_parameters': self.trainable_params,
            'trainable_parameter_ratio': trainable_ratio,
            'efficiency_score': 1.0 / trainable_ratio if trainable_ratio > 0 else float('inf'),
            'method': self.method
        }
    
    def log_parameter_info(self):
        """Log parameter information."""
        metrics = self.get_efficiency_metrics()
        logger.info(f"Parameter Efficiency ({self.method}):")
        logger.info(f"  Total parameters: {metrics['total_parameters']:,}")
        logger.info(f"  Trainable parameters: {metrics['trainable_parameters']:,}")
        logger.info(f"  Trainable ratio: {metrics['trainable_parameter_ratio']:.4f}")


class LoRAAnalyzer:
    """Analyzes LoRA adapter weights and utilization with enhanced PEFT support."""
    
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.adapter_modules = self._find_adapter_modules()
    
    def _find_adapter_modules(self) -> List[Tuple[str, torch.nn.Module]]:
        """Find LoRA adapter modules in the model (PEFT compatible)."""
        adapter_modules = []
        
        # Handle PEFT models
        if hasattr(self.model, 'peft_config'):
            # This is a PEFT model, traverse differently
            for name, module in self.model.named_modules():
                # PEFT LoRA modules have different attribute names
                if (hasattr(module, 'lora_A') and hasattr(module, 'lora_B')) or \
                   (hasattr(module, 'adapter_A') and hasattr(module, 'adapter_B')):
                    adapter_modules.append((name, module))
        else:
            # Standard traversal for custom LoRA implementations
            for name, module in self.model.named_modules():
                if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                    adapter_modules.append((name, module))
        
        logger.debug(f"Found {len(adapter_modules)} LoRA adapter modules")
        return adapter_modules
    
    def compute_adapter_statistics(self) -> Dict[str, float]:
        """Compute comprehensive statistics about LoRA adapters."""
        if not self.adapter_modules:
            logger.warning("No LoRA adapter modules found for statistics computation")
            return {}
        
        all_weights_A = []
        all_weights_B = []
        adapter_norms = []
        adapter_magnitudes = []
        singular_values_all = []
        
        for name, module in self.adapter_modules:
            try:
                # Get LoRA weights (handle different naming conventions)
                lora_A = getattr(module, 'lora_A', getattr(module, 'adapter_A', None))
                lora_B = getattr(module, 'lora_B', getattr(module, 'adapter_B', None))
                
                if lora_A is None or lora_B is None:
                    continue
                
                weight_A = lora_A.weight.data.flatten()
                weight_B = lora_B.weight.data.flatten()
                
                all_weights_A.extend(weight_A.cpu().numpy())
                all_weights_B.extend(weight_B.cpu().numpy())
                
                # Compute adapter norm (||B @ A||_F)
                adapter_weight = lora_B.weight @ lora_A.weight
                adapter_norm = torch.norm(adapter_weight, p='fro').item()
                adapter_norms.append(adapter_norm)
                
                # Compute adapter magnitude (max singular value)
                try:
                    U, S, V = torch.svd(adapter_weight.float())
                    if len(S) > 0:
                        adapter_magnitudes.append(S[0].item())
                        singular_values_all.extend(S.cpu().numpy())
                except:
                    # SVD failed, skip this adapter for magnitude computation
                    pass
                
            except Exception as e:
                logger.warning(f"Failed to process adapter {name}: {e}")
                continue
        
        if not all_weights_A:
            return {}
        
        stats = {
            'adapter_weight_mean_A': np.mean(all_weights_A),
            'adapter_weight_std_A': np.std(all_weights_A),
            'adapter_weight_mean_B': np.mean(all_weights_B),
            'adapter_weight_std_B': np.std(all_weights_B),
            'adapter_norm_mean': np.mean(adapter_norms) if adapter_norms else 0.0,
            'adapter_norm_std': np.std(adapter_norms) if adapter_norms else 0.0,
            'adapter_norm_max': np.max(adapter_norms) if adapter_norms else 0.0,
            'adapter_norm_min': np.min(adapter_norms) if adapter_norms else 0.0,
            'num_adapters': len(self.adapter_modules)
        }
        
        if adapter_magnitudes:
            stats.update({
                'adapter_magnitude_mean': np.mean(adapter_magnitudes),
                'adapter_magnitude_std': np.std(adapter_magnitudes),
                'adapter_magnitude_max': np.max(adapter_magnitudes)
            })
        
        if singular_values_all:
            stats.update({
                'singular_value_mean': np.mean(singular_values_all),
                'singular_value_std': np.std(singular_values_all),
                'effective_rank_estimate': np.sum(np.array(singular_values_all) > 1e-6)
            })
        
        return stats
    
    def analyze_rank_utilization(self) -> Dict[str, float]:
        """Analyze how well the LoRA rank is utilized."""
        if not self.adapter_modules:
            return {}
        
        rank_utilizations = []
        effective_ranks = []
        condition_numbers = []
        
        for name, module in self.adapter_modules:
            try:
                # Get LoRA weights
                lora_A = getattr(module, 'lora_A', getattr(module, 'adapter_A', None))
                lora_B = getattr(module, 'lora_B', getattr(module, 'adapter_B', None))
                
                if lora_A is None or lora_B is None:
                    continue
                
                # Compute effective rank using SVD
                adapter_weight = lora_B.weight @ lora_A.weight
                
                try:
                    U, S, V = torch.svd(adapter_weight.float())
                    
                    # Compute rank utilization (ratio of significant singular values)
                    threshold = 1e-6
                    effective_rank = torch.sum(S > threshold).item()
                    theoretical_rank = min(adapter_weight.shape)
                    
                    rank_utilization = effective_rank / theoretical_rank if theoretical_rank > 0 else 0
                    rank_utilizations.append(rank_utilization)
                    effective_ranks.append(effective_rank)
                    
                    # Compute condition number
                    if len(S) > 1 and S[-1] > threshold:
                        condition_number = (S[0] / S[-1]).item()
                        condition_numbers.append(condition_number)
                    
                except Exception as svd_error:
                    logger.debug(f"SVD failed for adapter {name}: {svd_error}")
                    continue
                    
            except Exception as e:
                logger.warning(f"Failed to analyze rank utilization for adapter {name}: {e}")
                continue
        
        if not rank_utilizations:
            return {}
        
        stats = {
            'rank_utilization_mean': np.mean(rank_utilizations),
            'rank_utilization_std': np.std(rank_utilizations),
            'rank_utilization_min': np.min(rank_utilizations),
            'rank_utilization_max': np.max(rank_utilizations)
        }
        
        if effective_ranks:
            stats.update({
                'effective_rank_mean': np.mean(effective_ranks),
                'effective_rank_std': np.std(effective_ranks),
                'effective_rank_max': np.max(effective_ranks)
            })
        
        if condition_numbers:
            stats.update({
                'condition_number_mean': np.mean(condition_numbers),
                'condition_number_std': np.std(condition_numbers),
                'condition_number_max': np.max(condition_numbers)
            })
        
        return stats
    
    def get_adapter_weight_distribution(self) -> Dict[str, np.ndarray]:
        """Get weight distributions for visualization and analysis."""
        if not self.adapter_modules:
            return {}
        
        distributions = {'A_weights': [], 'B_weights': []}
        
        for name, module in self.adapter_modules:
            try:
                lora_A = getattr(module, 'lora_A', getattr(module, 'adapter_A', None))
                lora_B = getattr(module, 'lora_B', getattr(module, 'adapter_B', None))
                
                if lora_A is not None and lora_B is not None:
                    distributions['A_weights'].extend(lora_A.weight.data.flatten().cpu().numpy())
                    distributions['B_weights'].extend(lora_B.weight.data.flatten().cpu().numpy())
            
            except Exception as e:
                logger.warning(f"Failed to extract weight distribution for {name}: {e}")
                continue
        
        return {k: np.array(v) for k, v in distributions.items() if v}


class ModelMerger:
    """Utilities for merging LoRA adapters into base models with enhanced PEFT support."""
    
    @staticmethod
    def merge_lora_weights(model: torch.nn.Module) -> torch.nn.Module:
        """Merge LoRA weights into the base model using PEFT functionality."""
        try:
            # Check if this is a PEFT model with merge capability
            if hasattr(model, 'merge_and_unload'):
                logger.info("Merging LoRA weights using PEFT merge_and_unload...")
                merged_model = model.merge_and_unload()
                logger.info("✓ LoRA weights merged using PEFT")
                return merged_model
            
            elif hasattr(model, 'merge_adapter'):
                logger.info("Merging LoRA weights using PEFT merge_adapter...")
                model.merge_adapter()
                logger.info("✓ LoRA weights merged in-place using PEFT")
                return model
            
            else:
                logger.warning("No PEFT merge functionality available")
                logger.info("Attempting manual LoRA merge...")
                return ModelMerger._manual_lora_merge(model)
                
        except Exception as e:
            logger.error(f"Failed to merge LoRA weights: {e}")
            logger.info("Falling back to original model")
            return model
    
    @staticmethod
    def _manual_lora_merge(model: torch.nn.Module) -> torch.nn.Module:
        """Manual LoRA merge implementation as fallback."""
        logger.warning("Manual LoRA merge not implemented, returning original model")
        # TODO: Implement manual merge if needed
        return model
    
    @staticmethod
    def test_merge_equivalence(base_model: Optional[torch.nn.Module], 
                             adapter_model: torch.nn.Module,
                             merged_model: torch.nn.Module,
                             test_input: torch.Tensor,
                             tolerance: float = 1e-5) -> Dict[str, float]:
        """Test that merged model produces equivalent outputs to adapter model."""
        try:
            adapter_model.eval()
            merged_model.eval()
            
            device = next(adapter_model.parameters()).device
            if test_input.device != device:
                test_input = test_input.to(device)
            
            with torch.no_grad():
                # Get outputs from both models
                adapter_output = adapter_model(test_input)
                merged_output = merged_model(test_input)
                
                # Extract logits or raw output
                if hasattr(adapter_output, 'logits') and hasattr(merged_output, 'logits'):
                    adapter_logits = adapter_output.logits
                    merged_logits = merged_output.logits
                elif hasattr(adapter_output, 'last_hidden_state') and hasattr(merged_output, 'last_hidden_state'):
                    adapter_logits = adapter_output.last_hidden_state
                    merged_logits = merged_output.last_hidden_state
                else:
                    adapter_logits = adapter_output if torch.is_tensor(adapter_output) else adapter_output[0]
                    merged_logits = merged_output if torch.is_tensor(merged_output) else merged_output[0]
                
                # Ensure same shape
                if adapter_logits.shape != merged_logits.shape:
                    logger.warning(f"Shape mismatch: adapter {adapter_logits.shape} vs merged {merged_logits.shape}")
                    return {
                        'max_absolute_difference': float('inf'),
                        'mean_absolute_difference': float('inf'),
                        'relative_difference': float('inf'),
                        'equivalence_check': False,
                        'error': 'Shape mismatch'
                    }
                
                # Compute differences
                abs_diff = torch.abs(adapter_logits - merged_logits)
                max_diff = torch.max(abs_diff).item()
                mean_diff = torch.mean(abs_diff).item()
                
                # Relative difference
                adapter_magnitude = torch.mean(torch.abs(adapter_logits)).item()
                relative_diff = mean_diff / adapter_magnitude if adapter_magnitude > 0 else float('inf')
                
                # Element-wise relative differences for detailed analysis
                rel_diff_elementwise = abs_diff / (torch.abs(adapter_logits) + 1e-8)
                max_rel_diff = torch.max(rel_diff_elementwise).item()
                
                # Equivalence check
                equivalence_check = max_diff < tolerance
                
                return {
                    'max_absolute_difference': max_diff,
                    'mean_absolute_difference': mean_diff,
                    'relative_difference': relative_diff,
                    'max_relative_difference': max_rel_diff,
                    'equivalence_check': equivalence_check,
                    'tolerance_used': tolerance,
                    'adapter_magnitude': adapter_magnitude,
                    'num_elements': adapter_logits.numel()
                }
                
        except Exception as e:
            logger.error(f"Error in merge equivalence test: {e}")
            return {
                'max_absolute_difference': float('inf'),
                'mean_absolute_difference': float('inf'),
                'relative_difference': float('inf'),
                'equivalence_check': False,
                'error': str(e)
            }
    
    @staticmethod
    def benchmark_inference_speed(adapter_model: torch.nn.Module,
                                merged_model: torch.nn.Module,
                                test_inputs: List[torch.Tensor],
                                num_warmup: int = 5,
                                num_runs: int = 20) -> Dict[str, float]:
        """Benchmark inference speed of adapter vs merged model."""
        try:
            adapter_model.eval()
            merged_model.eval()
            
            device = next(adapter_model.parameters()).device
            
            # Prepare test inputs
            test_inputs_device = []
            for inp in test_inputs[:num_runs]:
                if inp.device != device:
                    inp = inp.to(device)
                test_inputs_device.append(inp)
            
            # Warmup
            logger.info("Warming up models for benchmarking...")
            with torch.no_grad():
                for i in range(min(num_warmup, len(test_inputs_device))):
                    _ = adapter_model(test_inputs_device[i])
                    _ = merged_model(test_inputs_device[i])
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            
            # Benchmark adapter model
            logger.info("Benchmarking adapter model...")
            adapter_times = []
            with torch.no_grad():
                for inp in test_inputs_device:
                    start_time = time.time()
                    _ = adapter_model(inp)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    adapter_times.append(time.time() - start_time)
            
            # Benchmark merged model
            logger.info("Benchmarking merged model...")
            merged_times = []
            with torch.no_grad():
                for inp in test_inputs_device:
                    start_time = time.time()
                    _ = merged_model(inp)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    merged_times.append(time.time() - start_time)
            
            # Compute statistics
            adapter_mean = np.mean(adapter_times)
            adapter_std = np.std(adapter_times)
            merged_mean = np.mean(merged_times)
            merged_std = np.std(merged_times)
            
            speedup = adapter_mean / merged_mean if merged_mean > 0 else 0
            
            return {
                'adapter_inference_time_ms': adapter_mean * 1000,
                'adapter_inference_std_ms': adapter_std * 1000,
                'merged_inference_time_ms': merged_mean * 1000,
                'merged_inference_std_ms': merged_std * 1000,
                'speedup_ratio': speedup,
                'num_benchmark_runs': len(adapter_times),
                'adapter_faster': speedup < 1.0
            }
            
        except Exception as e:
            logger.error(f"Error in inference speed benchmark: {e}")
            return {'error': str(e)}


class TrainingEfficiencyMonitor:
    """Monitors training efficiency metrics like speed and memory usage."""
    
    def __init__(self):
        self.step_times = []
        self.memory_usage = []
        self.start_time = None
    
    def start_timing(self):
        """Start timing a training step."""
        import time
        self.start_time = time.time()
    
    def end_timing(self):
        """End timing a training step."""
        if self.start_time is not None:
            import time
            step_time = time.time() - self.start_time
            self.step_times.append(step_time)
            self.start_time = None
    
    def record_memory_usage(self):
        """Record current GPU memory usage."""
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            memory_reserved = torch.cuda.memory_reserved() / 1024**3   # GB
            self.memory_usage.append({
                'allocated_gb': memory_allocated,
                'reserved_gb': memory_reserved
            })
    
    def get_efficiency_metrics(self) -> Dict[str, float]:
        """Get training efficiency metrics."""
        if not self.step_times:
            return {}
        
        import numpy as np
        
        metrics = {
            'steps_per_second': 1.0 / np.mean(self.step_times) if self.step_times else 0,
            'mean_step_time': np.mean(self.step_times),
            'std_step_time': np.std(self.step_times),
            'total_steps': len(self.step_times)
        }
        
        if self.memory_usage:
            allocated = [m['allocated_gb'] for m in self.memory_usage]
            reserved = [m['reserved_gb'] for m in self.memory_usage]
            
            metrics.update({
                'mean_memory_allocated_gb': np.mean(allocated),
                'max_memory_allocated_gb': np.max(allocated),
                'mean_memory_reserved_gb': np.mean(reserved),
                'max_memory_reserved_gb': np.max(reserved)
            })
        
        return metrics


class LoRAParameterEfficiencyAnalyzer:
    """Specialized analyzer for LoRA parameter efficiency with detailed breakdown."""
    
    def __init__(self, model: torch.nn.Module, base_model: Optional[torch.nn.Module] = None):
        self.model = model
        self.base_model = base_model
        self.lora_analyzer = LoRAAnalyzer(model)
        
    def get_detailed_parameter_breakdown(self) -> Dict[str, Any]:
        """Get detailed breakdown of parameter usage in LoRA model."""
        breakdown = {
            'base_model': {'total': 0, 'trainable': 0, 'frozen': 0},
            'lora_adapters': {'total': 0, 'trainable': 0, 'by_module': {}},
            'other': {'total': 0, 'trainable': 0}
        }
        
        for name, param in self.model.named_parameters():
            param_count = param.numel()
            is_trainable = param.requires_grad
            
            if 'lora_A' in name or 'lora_B' in name or 'adapter_A' in name or 'adapter_B' in name:
                # LoRA adapter parameters
                breakdown['lora_adapters']['total'] += param_count
                if is_trainable:
                    breakdown['lora_adapters']['trainable'] += param_count
                
                # Break down by module
                module_name = name.split('.lora_')[0] if '.lora_' in name else name.split('.adapter_')[0]
                if module_name not in breakdown['lora_adapters']['by_module']:
                    breakdown['lora_adapters']['by_module'][module_name] = {'total': 0, 'trainable': 0}
                
                breakdown['lora_adapters']['by_module'][module_name]['total'] += param_count
                if is_trainable:
                    breakdown['lora_adapters']['by_module'][module_name]['trainable'] += param_count
                    
            elif any(x in name for x in ['lora', 'adapter']) and 'scaling' not in name:
                # Other LoRA-related parameters (scaling factors, etc.)
                breakdown['other']['total'] += param_count
                if is_trainable:
                    breakdown['other']['trainable'] += param_count
            else:
                # Base model parameters
                breakdown['base_model']['total'] += param_count
                if is_trainable:
                    breakdown['base_model']['trainable'] += param_count
                else:
                    breakdown['base_model']['frozen'] += param_count
        
        # Calculate summary statistics
        total_params = sum(breakdown[category]['total'] for category in breakdown)
        total_trainable = sum(breakdown[category]['trainable'] for category in breakdown)
        
        breakdown['summary'] = {
            'total_parameters': total_params,
            'total_trainable': total_trainable,
            'trainable_ratio': total_trainable / total_params if total_params > 0 else 0,
            'lora_efficiency': breakdown['lora_adapters']['trainable'] / total_params if total_params > 0 else 0,
            'base_model_frozen_ratio': breakdown['base_model']['frozen'] / breakdown['base_model']['total'] if breakdown['base_model']['total'] > 0 else 0
        }
        
        return breakdown
    
    def verify_lora_efficiency_target(self, target_ratio: float = 0.003) -> Dict[str, Any]:
        """Verify that LoRA achieves target parameter efficiency (~0.3%)."""
        breakdown = self.get_detailed_parameter_breakdown()
        
        actual_ratio = breakdown['summary']['trainable_ratio']
        efficiency_score = target_ratio / actual_ratio if actual_ratio > 0 else float('inf')
        
        # Check if we're within acceptable range (±0.2%)
        within_target = abs(actual_ratio - target_ratio) <= 0.002
        
        return {
            'target_ratio': target_ratio,
            'actual_ratio': actual_ratio,
            'efficiency_score': efficiency_score,
            'within_target': within_target,
            'efficiency_grade': self._grade_efficiency(actual_ratio, target_ratio),
            'recommendations': self._get_efficiency_recommendations(breakdown)
        }
    
    def _grade_efficiency(self, actual_ratio: float, target_ratio: float) -> str:
        """Grade the parameter efficiency."""
        if actual_ratio <= target_ratio * 0.8:
            return "Excellent"
        elif actual_ratio <= target_ratio * 1.2:
            return "Good"
        elif actual_ratio <= target_ratio * 1.5:
            return "Acceptable"
        elif actual_ratio <= target_ratio * 2.0:
            return "Poor"
        else:
            return "Very Poor"
    
    def _get_efficiency_recommendations(self, breakdown: Dict[str, Any]) -> List[str]:
        """Get recommendations for improving parameter efficiency."""
        recommendations = []
        
        # Check if base model is properly frozen
        if breakdown['base_model']['trainable'] > 0:
            recommendations.append("Base model parameters are not fully frozen")
        
        # Check LoRA adapter efficiency
        lora_ratio = breakdown['summary']['lora_efficiency']
        if lora_ratio > 0.005:  # > 0.5%
            recommendations.append("Consider reducing LoRA rank or target fewer modules")
        elif lora_ratio < 0.001:  # < 0.1%
            recommendations.append("LoRA adaptation might be too constrained; consider increasing rank")
        
        # Check for unexpected trainable parameters
        if breakdown['other']['trainable'] > 0:
            recommendations.append("Unexpected trainable parameters detected outside LoRA adapters")
        
        return recommendations


class CheckpointValidator:
    """Validates that model checkpoints can be loaded correctly."""
    
    @staticmethod
    def validate_checkpoint(checkpoint_path: str, 
                          original_model: torch.nn.Module,
                          test_input: torch.Tensor) -> Dict[str, Any]:
        """Validate that a checkpoint can be loaded and produces consistent outputs."""
        try:
            from transformers import AutoModelForCausalLM
            
            # Load checkpoint
            loaded_model = AutoModelForCausalLM.from_pretrained(checkpoint_path)
            loaded_model.eval()
            original_model.eval()
            
            # Test outputs
            with torch.no_grad():
                original_output = original_model(test_input)
                loaded_output = loaded_model(test_input)
                
                # Compare outputs
                if hasattr(original_output, 'logits') and hasattr(loaded_output, 'logits'):
                    diff = torch.mean(torch.abs(original_output.logits - loaded_output.logits)).item()
                else:
                    diff = torch.mean(torch.abs(original_output - loaded_output)).item()
                
                return {
                    'checkpoint_valid': True,
                    'output_difference': diff,
                    'consistent_outputs': diff < 1e-4,
                    'checkpoint_path': checkpoint_path
                }
        
        except Exception as e:
            return {
                'checkpoint_valid': False,
                'error': str(e),
                'checkpoint_path': checkpoint_path
            }


class LoRAValidationSuite:
    """Comprehensive validation suite for LoRA experiments as required in the specifications."""
    
    def __init__(self, model: torch.nn.Module, tokenizer, eval_dataset):
        self.model = model
        self.tokenizer = tokenizer
        self.eval_dataset = eval_dataset
        self.parameter_analyzer = LoRAParameterEfficiencyAnalyzer(model)
        self.lora_analyzer = LoRAAnalyzer(model)
        
    def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run all validation checks as specified in requirements."""
        logger.info("Running comprehensive LoRA validation suite...")
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'validations': {}
        }
        
        # 1. Verify base model is frozen
        results['validations']['base_model_frozen'] = self._verify_base_model_frozen()
        
        # 2. Verify parameter efficiency target (~0.3%)
        results['validations']['parameter_efficiency'] = self._verify_parameter_efficiency()
        
        # 3. Test LoRA merge equivalence
        results['validations']['merge_equivalence'] = self._test_merge_equivalence()
        
        # 4. Verify LoRA-specific metrics tracking
        results['validations']['lora_metrics'] = self._verify_lora_metrics()
        
        # 5. Test adapter saving and loading
        results['validations']['adapter_persistence'] = self._test_adapter_persistence()
        
        # Overall validation result
        all_passed = all(
            validation.get('passed', False) 
            for validation in results['validations'].values()
        )
        
        results['overall_validation'] = {
            'passed': all_passed,
            'grade': 'PASS' if all_passed else 'FAIL',
            'summary': self._generate_validation_summary(results['validations'])
        }
        
        logger.info(f"LoRA validation complete: {'PASS' if all_passed else 'FAIL'}")
        
        return results
    
    def _verify_base_model_frozen(self) -> Dict[str, Any]:
        """Verify that base model parameters are properly frozen."""
        base_params_with_grad = 0
        total_base_params = 0
        unfrozen_params = []
        
        for name, param in self.model.named_parameters():
            if not any(x in name for x in ['lora_', 'adapter_']):
                total_base_params += 1
                if param.requires_grad:
                    base_params_with_grad += 1
                    unfrozen_params.append(name)
        
        passed = base_params_with_grad == 0
        
        return {
            'passed': passed,
            'base_params_with_grad': base_params_with_grad,
            'total_base_params': total_base_params,
            'unfrozen_param_names': unfrozen_params[:10],  # Limit to first 10
            'message': f"{'✓' if passed else '✗'} Base model frozen check"
        }
    
    def _verify_parameter_efficiency(self) -> Dict[str, Any]:
        """Verify parameter efficiency meets target (~0.3%)."""
        efficiency_result = self.parameter_analyzer.verify_lora_efficiency_target()
        
        passed = efficiency_result['within_target']
        
        return {
            'passed': passed,
            'target_ratio': efficiency_result['target_ratio'],
            'actual_ratio': efficiency_result['actual_ratio'],
            'efficiency_grade': efficiency_result['efficiency_grade'],
            'recommendations': efficiency_result['recommendations'],
            'message': f"{'✓' if passed else '✗'} Parameter efficiency: {efficiency_result['actual_ratio']:.6f} (target: {efficiency_result['target_ratio']:.3f})"
        }
    
    def _test_merge_equivalence(self) -> Dict[str, Any]:
        """Test LoRA merge equivalence with detailed validation."""
        try:
            # Create test input from evaluation dataset
            test_examples = self.eval_dataset.select(range(min(5, len(self.eval_dataset))))
            test_input = torch.tensor([ex['input_ids'] for ex in test_examples]).to(self.model.device)
            
            # Test merge
            merged_model = ModelMerger.merge_lora_weights(self.model)
            
            # Test equivalence
            equivalence_results = ModelMerger.test_merge_equivalence(
                base_model=None,
                adapter_model=self.model,
                merged_model=merged_model,
                test_input=test_input
            )
            
            passed = equivalence_results.get('equivalence_check', False)
            
            return {
                'passed': passed,
                'max_diff': equivalence_results.get('max_absolute_difference', float('inf')),
                'mean_diff': equivalence_results.get('mean_absolute_difference', float('inf')),
                'relative_diff': equivalence_results.get('relative_difference', float('inf')),
                'tolerance': equivalence_results.get('tolerance_used', 1e-5),
                'message': f"{'✓' if passed else '✗'} LoRA merge equivalence test"
            }
            
        except Exception as e:
            return {
                'passed': False,
                'error': str(e),
                'message': f"✗ LoRA merge equivalence test failed: {e}"
            }
    
    def _verify_lora_metrics(self) -> Dict[str, Any]:
        """Verify LoRA-specific metrics can be computed."""
        try:
            # Test adapter statistics
            adapter_stats = self.lora_analyzer.compute_adapter_statistics()
            
            # Test rank utilization
            rank_stats = self.lora_analyzer.analyze_rank_utilization()
            
            passed = bool(adapter_stats) and bool(rank_stats)
            
            return {
                'passed': passed,
                'num_adapters': adapter_stats.get('num_adapters', 0),
                'has_rank_utilization': bool(rank_stats),
                'metrics_computed': list(adapter_stats.keys()),
                'message': f"{'✓' if passed else '✗'} LoRA metrics computation"
            }
            
        except Exception as e:
            return {
                'passed': False,
                'error': str(e),
                'message': f"✗ LoRA metrics computation failed: {e}"
            }
    
    def _test_adapter_persistence(self) -> Dict[str, Any]:
        """Test that LoRA adapters can be saved and loaded."""
        try:
            import tempfile
            
            # Create temporary directory
            with tempfile.TemporaryDirectory() as temp_dir:
                adapter_path = Path(temp_dir) / "test_adapter"
                
                # Save adapter
                self.model.save_pretrained(str(adapter_path))
                
                # Check if files exist
                config_exists = (adapter_path / "adapter_config.json").exists()
                weights_exists = (adapter_path / "adapter_model.bin").exists() or \
                               (adapter_path / "adapter_model.safetensors").exists()
                
                passed = config_exists and weights_exists
                
                return {
                    'passed': passed,
                    'config_exists': config_exists,
                    'weights_exists': weights_exists,
                    'adapter_path': str(adapter_path),
                    'message': f"{'✓' if passed else '✗'} LoRA adapter persistence test"
                }
                
        except Exception as e:
            return {
                'passed': False,
                'error': str(e),
                'message': f"✗ LoRA adapter persistence test failed: {e}"
            }
    
    def _generate_validation_summary(self, validations: Dict[str, Any]) -> str:
        """Generate a summary of validation results."""
        passed_count = sum(1 for v in validations.values() if v.get('passed', False))
        total_count = len(validations)
        
        summary = f"LoRA Validation Results: {passed_count}/{total_count} tests passed\n"
        
        for test_name, result in validations.items():
            status = "PASS" if result.get('passed', False) else "FAIL"
            summary += f"  - {test_name}: {status}\n"
        
        return summary


class AdapterSwitchingBenchmark:
    """Benchmarks adapter switching performance for deployment analysis."""
    
    def __init__(self, base_model: torch.nn.Module):
        self.base_model = base_model
        self.loaded_adapters = {}
    
    def load_adapter(self, adapter_name: str, adapter_path: str):
        """Load an adapter for switching benchmarks."""
        try:
            from peft import PeftModel
            adapter_model = PeftModel.from_pretrained(self.base_model, adapter_path)
            self.loaded_adapters[adapter_name] = adapter_model
            logger.info(f"Loaded adapter: {adapter_name}")
        except Exception as e:
            logger.error(f"Failed to load adapter {adapter_name}: {e}")
    
    def benchmark_switching(self, test_inputs: List[torch.Tensor], 
                          adapter_sequence: List[str]) -> Dict[str, float]:
        """Benchmark the overhead of switching between adapters."""
        import time
        
        if not self.loaded_adapters:
            return {'error': 'No adapters loaded'}
        
        switch_times = []
        inference_times = []
        
        for i, (test_input, adapter_name) in enumerate(zip(test_inputs, adapter_sequence)):
            if adapter_name not in self.loaded_adapters:
                continue
            
            # Time adapter switching
            start_switch = time.time()
            current_model = self.loaded_adapters[adapter_name]
            current_model.eval()
            switch_time = time.time() - start_switch
            switch_times.append(switch_time)
            
            # Time inference
            start_inference = time.time()
            with torch.no_grad():
                _ = current_model(test_input)
            inference_time = time.time() - start_inference
            inference_times.append(inference_time)
        
        if not switch_times:
            return {'error': 'No valid adapter switches'}
        
        import numpy as np
        
        return {
            'mean_switch_time_ms': np.mean(switch_times) * 1000,
            'std_switch_time_ms': np.std(switch_times) * 1000,
            'mean_inference_time_ms': np.mean(inference_times) * 1000,
            'std_inference_time_ms': np.std(inference_times) * 1000,
            'total_switches': len(switch_times),
            'switching_overhead_percent': (np.mean(switch_times) / np.mean(inference_times)) * 100
        }
