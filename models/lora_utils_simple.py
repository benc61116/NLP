#!/usr/bin/env python3
"""Simplified LoRA utilities for validation without problematic imports."""

import torch
import numpy as np
from typing import Dict, Any, List, Tuple, Optional


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


class LoRAAnalyzer:
    """Analyzes LoRA adapter weights and utilization."""
    
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.adapter_modules = self._find_adapter_modules()
    
    def _find_adapter_modules(self) -> List[Tuple[str, torch.nn.Module]]:
        """Find LoRA adapter modules in the model."""
        adapter_modules = []
        
        for name, module in self.model.named_modules():
            # Check for direct LoRA attributes
            if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                adapter_modules.append((name, module))
            # Check for PEFT module structure
            elif hasattr(module, 'base_layer') and hasattr(module, 'lora_A'):
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
            try:
                # Try to access PEFT LoRA weights
                lora_A = None
                lora_B = None
                
                if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                    # Check if they are ModuleDict (PEFT structure)
                    if hasattr(module.lora_A, 'default'):
                        lora_A = module.lora_A.default
                        lora_B = module.lora_B.default
                    elif hasattr(module.lora_A, 'weight'):
                        lora_A = module.lora_A
                        lora_B = module.lora_B
                    else:
                        # Try to get first item from ModuleDict
                        lora_A_items = list(module.lora_A.children())
                        lora_B_items = list(module.lora_B.children())
                        if lora_A_items and lora_B_items:
                            lora_A = lora_A_items[0]
                            lora_B = lora_B_items[0]
                
                if lora_A is not None and lora_B is not None and hasattr(lora_A, 'weight') and hasattr(lora_B, 'weight'):
                    weight_A = lora_A.weight.data.flatten()
                    weight_B = lora_B.weight.data.flatten()
                    
                    all_weights_A.extend(weight_A.cpu().numpy())
                    all_weights_B.extend(weight_B.cpu().numpy())
                    
                    # Compute adapter norm (||B @ A||_F)
                    adapter_weight = lora_B.weight @ lora_A.weight
                    adapter_norm = torch.norm(adapter_weight, p='fro').item()
                    adapter_norms.append(adapter_norm)
                
            except Exception as e:
                # Skip this module if we can't access the weights
                continue
        
        if not all_weights_A:
            return {}
        
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
            try:
                # Try to access PEFT LoRA weights
                lora_A = None
                lora_B = None
                
                if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                    # Check if they are ModuleDict (PEFT structure)
                    if hasattr(module.lora_A, 'default'):
                        lora_A = module.lora_A.default
                        lora_B = module.lora_B.default
                    elif hasattr(module.lora_A, 'weight'):
                        lora_A = module.lora_A
                        lora_B = module.lora_B
                    else:
                        # Try to get first item from ModuleDict
                        lora_A_items = list(module.lora_A.children())
                        lora_B_items = list(module.lora_B.children())
                        if lora_A_items and lora_B_items:
                            lora_A = lora_A_items[0]
                            lora_B = lora_B_items[0]
                
                if lora_A is not None and lora_B is not None and hasattr(lora_A, 'weight') and hasattr(lora_B, 'weight'):
                    # Compute effective rank using SVD
                    adapter_weight = lora_B.weight @ lora_A.weight
                    try:
                        U, S, V = torch.svd(adapter_weight.float())
                        
                        # Compute rank utilization (ratio of non-zero singular values)
                        threshold = 1e-6
                        effective_rank = torch.sum(S > threshold).item()
                        max_rank = min(adapter_weight.shape)
                        
                        rank_utilization = effective_rank / max_rank if max_rank > 0 else 0
                        rank_utilizations.append(rank_utilization)
                    except:
                        # SVD failed, skip this adapter
                        continue
                        
            except Exception as e:
                # Skip this module if we can't access the weights
                continue
        
        if not rank_utilizations:
            return {}
        
        return {
            'rank_utilization_mean': np.mean(rank_utilizations),
            'rank_utilization_std': np.std(rank_utilizations),
            'rank_utilization_min': np.min(rank_utilizations),
            'rank_utilization_max': np.max(rank_utilizations)
        }
