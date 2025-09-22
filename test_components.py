#!/usr/bin/env python3
"""Simplified components for testing without transformers training dependencies."""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import json
import psutil
import os


@dataclass
class SimpleRepresentationConfig:
    """Configuration for representation extraction."""
    extract_every_steps: int = 100
    save_layers: List[int] = field(default_factory=lambda: list(range(12)))  # Default for small models
    max_validation_samples: int = 1000
    save_attention: bool = True
    save_mlp: bool = True
    memory_map: bool = True


class SimpleRepresentationExtractor:
    """Simplified version of RepresentationExtractor for testing."""
    
    def __init__(self, config: SimpleRepresentationConfig, output_dir: Path, task_name: str, method: str):
        self.config = config
        self.output_dir = output_dir / "representations" / f"{method}_{task_name}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.task_name = task_name
        self.method = method
        self.validation_examples = None
        
    def set_validation_examples(self, examples: Dict[str, torch.Tensor]):
        """Set validation examples for consistent representation extraction."""
        max_samples = min(self.config.max_validation_samples, len(examples['input_ids']))
        
        self.validation_examples = {
            'input_ids': examples['input_ids'][:max_samples],
            'attention_mask': examples['attention_mask'][:max_samples],
        }
        if 'labels' in examples:
            self.validation_examples['labels'] = examples['labels'][:max_samples]
    
    def extract_representations(self, model: nn.Module, step: int) -> Dict[str, torch.Tensor]:
        """Extract representations from the model."""
        if self.validation_examples is None:
            return {}
        
        model.eval()
        representations = {}
        layer_outputs = {}
        hooks = []
        
        def create_hook(layer_name):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    layer_outputs[layer_name] = output[0].detach().cpu()
                else:
                    layer_outputs[layer_name] = output.detach().cpu()
            return hook
        
        try:
            # Register hooks for transformer layers
            if hasattr(model, 'transformer'):  # GPT-style models
                layers = model.transformer.h
            elif hasattr(model, 'model') and hasattr(model.model, 'layers'):  # Llama-style
                layers = model.model.layers
            elif hasattr(model, 'bert'):  # BERT-style
                layers = model.bert.encoder.layer
            else:
                layers = []
            
            for i, layer in enumerate(layers[:len(self.config.save_layers)]):
                if i in self.config.save_layers:
                    hook = layer.register_forward_hook(create_hook(f'layer_{i}'))
                    hooks.append(hook)
            
            # Forward pass
            with torch.no_grad():
                input_ids = self.validation_examples['input_ids']
                attention_mask = self.validation_examples['attention_mask']
                
                if input_ids.device != next(model.parameters()).device:
                    input_ids = input_ids.to(next(model.parameters()).device)
                    attention_mask = attention_mask.to(next(model.parameters()).device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                
                # Store extracted representations
                for layer_name, layer_output in layer_outputs.items():
                    representations[layer_name] = layer_output
                
                # Store final hidden states if available
                if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                    representations['final_hidden_states'] = outputs.hidden_states[-1].detach().cpu()
                elif hasattr(outputs, 'last_hidden_state'):
                    representations['final_hidden_states'] = outputs.last_hidden_state.detach().cpu()
        
        finally:
            for hook in hooks:
                hook.remove()
            model.train()
        
        return representations
    
    def save_representations(self, representations: Dict[str, torch.Tensor], step: int):
        """Save representations to disk."""
        if not representations:
            return
        
        step_dir = self.output_dir / f"step_{step:06d}"
        step_dir.mkdir(exist_ok=True)
        
        for layer_name, tensor in representations.items():
            file_path = step_dir / f"{layer_name}.pt"
            torch.save(tensor, file_path)
        
        # Save metadata
        metadata = {
            'step': step,
            'task_name': self.task_name,
            'method': self.method,
            'num_samples': len(self.validation_examples['input_ids']),
            'layer_names': list(representations.keys()),
            'tensor_shapes': {name: list(tensor.shape) for name, tensor in representations.items()}
        }
        
        with open(step_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)


class SimpleGradientStatsMonitor:
    """Simplified gradient statistics monitor."""
    
    def __init__(self, log_every_steps: int = 50):
        self.log_every_steps = log_every_steps
        self.gradient_stats = []
    
    def compute_gradient_stats(self, model: nn.Module) -> Dict[str, float]:
        """Compute gradient statistics for the model."""
        total_norm = 0.0
        param_count = 0
        grad_norms = []
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.data.norm(2).item()
                grad_norms.append(grad_norm)
                total_norm += grad_norm ** 2
                param_count += param.numel()
        
        total_norm = total_norm ** 0.5
        
        if grad_norms:
            stats = {
                'gradient_norm_total': total_norm,
                'gradient_norm_mean': np.mean(grad_norms),
                'gradient_norm_std': np.std(grad_norms),
                'gradient_norm_max': np.max(grad_norms),
                'gradient_norm_min': np.min(grad_norms),
                'num_parameters_with_gradients': len(grad_norms),
                'total_parameters': param_count,
            }
        else:
            stats = {
                'gradient_norm_total': 0.0,
                'gradient_norm_mean': 0.0,
                'gradient_norm_std': 0.0,
                'gradient_norm_max': 0.0,
                'gradient_norm_min': 0.0,
                'num_parameters_with_gradients': 0,
                'total_parameters': param_count,
            }
        
        return stats


class SimpleMemoryProfiler:
    """Simplified memory profiler."""
    
    def __init__(self):
        self.memory_stats = []
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        stats = {}
        
        # CPU memory
        process = psutil.Process(os.getpid())
        cpu_memory = process.memory_info()
        stats['cpu_memory_rss_mb'] = cpu_memory.rss / 1024 / 1024
        stats['cpu_memory_vms_mb'] = cpu_memory.vms / 1024 / 1024
        stats['cpu_memory_percent'] = process.memory_percent()
        
        # GPU memory
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                try:
                    memory_info = torch.cuda.memory_stats(i)
                    stats[f'gpu_{i}_memory_allocated_mb'] = memory_info.get('allocated_bytes.all.current', 0) / 1024 / 1024
                    stats[f'gpu_{i}_memory_reserved_mb'] = memory_info.get('reserved_bytes.all.current', 0) / 1024 / 1024
                    stats[f'gpu_{i}_memory_max_allocated_mb'] = memory_info.get('allocated_bytes.all.peak', 0) / 1024 / 1024
                except:
                    # If GPU stats fail, use basic memory info
                    allocated = torch.cuda.memory_allocated(i) / 1024 / 1024
                    reserved = torch.cuda.memory_reserved(i) / 1024 / 1024
                    stats[f'gpu_{i}_memory_allocated_mb'] = allocated
                    stats[f'gpu_{i}_memory_reserved_mb'] = reserved
        
        return stats
