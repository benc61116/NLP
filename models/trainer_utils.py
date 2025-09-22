#!/usr/bin/env python3
"""Custom trainer utilities for LoRA research project."""

import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional, List, Tuple
from transformers import Trainer, TrainerCallback
import logging

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
    """Analyzes LoRA adapter weights and utilization."""
    
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.adapter_modules = self._find_adapter_modules()
    
    def _find_adapter_modules(self) -> List[torch.nn.Module]:
        """Find LoRA adapter modules in the model."""
        adapter_modules = []
        for name, module in self.model.named_modules():
            if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                adapter_modules.append((name, module))
        return adapter_modules
    
    def compute_adapter_statistics(self) -> Dict[str, float]:
        """Compute statistics about LoRA adapters."""
        if not self.adapter_modules:
            return {}
        
        all_weights_A = []
        all_weights_B = []
        adapter_norms = []
        
        for name, module in self.adapter_modules:
            if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                weight_A = module.lora_A.weight.data.flatten()
                weight_B = module.lora_B.weight.data.flatten()
                
                all_weights_A.extend(weight_A.cpu().numpy())
                all_weights_B.extend(weight_B.cpu().numpy())
                
                # Compute adapter norm (||B @ A||_F)
                adapter_weight = module.lora_B.weight @ module.lora_A.weight
                adapter_norm = torch.norm(adapter_weight, p='fro').item()
                adapter_norms.append(adapter_norm)
        
        if not all_weights_A:
            return {}
        
        import numpy as np
        
        return {
            'adapter_weight_mean_A': np.mean(all_weights_A),
            'adapter_weight_std_A': np.std(all_weights_A),
            'adapter_weight_mean_B': np.mean(all_weights_B),
            'adapter_weight_std_B': np.std(all_weights_B),
            'adapter_norm_mean': np.mean(adapter_norms),
            'adapter_norm_std': np.std(adapter_norms),
            'adapter_norm_max': np.max(adapter_norms),
            'num_adapters': len(self.adapter_modules)
        }
    
    def analyze_rank_utilization(self) -> Dict[str, float]:
        """Analyze how well the LoRA rank is utilized."""
        if not self.adapter_modules:
            return {}
        
        rank_utilizations = []
        
        for name, module in self.adapter_modules:
            if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                # Compute effective rank using SVD
                adapter_weight = module.lora_B.weight @ module.lora_A.weight
                U, S, V = torch.svd(adapter_weight.float())
                
                # Compute rank utilization (ratio of non-zero singular values)
                threshold = 1e-6
                effective_rank = torch.sum(S > threshold).item()
                max_rank = min(adapter_weight.shape)
                
                rank_utilization = effective_rank / max_rank if max_rank > 0 else 0
                rank_utilizations.append(rank_utilization)
        
        if not rank_utilizations:
            return {}
        
        import numpy as np
        
        return {
            'rank_utilization_mean': np.mean(rank_utilizations),
            'rank_utilization_std': np.std(rank_utilizations),
            'rank_utilization_min': np.min(rank_utilizations),
            'rank_utilization_max': np.max(rank_utilizations)
        }


class ModelMerger:
    """Utilities for merging LoRA adapters into base models."""
    
    @staticmethod
    def merge_lora_weights(model: torch.nn.Module) -> torch.nn.Module:
        """Merge LoRA weights into the base model."""
        try:
            # Use PEFT's merge functionality if available
            if hasattr(model, 'merge_and_unload'):
                merged_model = model.merge_and_unload()
                logger.info("✓ LoRA weights merged using PEFT")
                return merged_model
            else:
                logger.warning("PEFT merge not available, manual merge required")
                return model
        except Exception as e:
            logger.error(f"Failed to merge LoRA weights: {e}")
            return model
    
    @staticmethod
    def test_merge_equivalence(base_model: torch.nn.Module, 
                             adapter_model: torch.nn.Module,
                             merged_model: torch.nn.Module,
                             test_input: torch.Tensor) -> Dict[str, float]:
        """Test that merged model produces equivalent outputs to adapter model."""
        base_model.eval()
        adapter_model.eval()
        merged_model.eval()
        
        with torch.no_grad():
            adapter_output = adapter_model(test_input)
            merged_output = merged_model(test_input)
            
            # Compare logits
            if hasattr(adapter_output, 'logits') and hasattr(merged_output, 'logits'):
                adapter_logits = adapter_output.logits
                merged_logits = merged_output.logits
            else:
                adapter_logits = adapter_output
                merged_logits = merged_output
            
            # Compute differences
            max_diff = torch.max(torch.abs(adapter_logits - merged_logits)).item()
            mean_diff = torch.mean(torch.abs(adapter_logits - merged_logits)).item()
            relative_diff = mean_diff / torch.mean(torch.abs(adapter_logits)).item()
            
            return {
                'max_absolute_difference': max_diff,
                'mean_absolute_difference': mean_diff,
                'relative_difference': relative_diff,
                'equivalence_check': max_diff < 1e-5  # Threshold for equivalence
            }


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
