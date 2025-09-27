#!/usr/bin/env python3
"""Sanity check utilities for validating experimental setup."""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType
import wandb
import numpy as np
from typing import Dict, List, Optional, Tuple
import time
import os
try:
    from .data_preparation import TaskDataLoader
except ImportError:
    # For direct execution, use absolute import
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from shared.data_preparation import TaskDataLoader
import yaml
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SanityCheckError(Exception):
    """Custom exception for sanity check failures."""
    pass


class ModelSanityChecker:
    """Comprehensive sanity checks for model training setup."""
    
    def __init__(self, config_path: str = "shared/config.yaml"):
        """Initialize sanity checker with configuration.
        
        Args:
            config_path: Path to the configuration YAML file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model_name = self.config['model']['name']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self.data_loader = None
        
        # Sanity check configuration
        self.sanity_config = self.config['sanity_check']
        self.num_samples = self.sanity_config['num_samples']
        self.max_epochs = self.sanity_config['max_epochs']
        self.expected_loss_threshold = self.sanity_config['expected_loss_threshold']
        
        logger.info(f"Initialized sanity checker for {self.model_name}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Sanity check samples: {self.num_samples}")
    
    def initialize_components(self):
        """Initialize tokenizer, model, and data loader."""
        logger.info("Initializing components...")
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initialize data loader
        self.data_loader = TaskDataLoader(self.model_name)
        
        logger.info("✓ Components initialized successfully")
    
    def check_model_loading(self) -> bool:
        """Test if model can be loaded correctly."""
        logger.info("Testing model loading...")
        
        try:
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                device_map="auto" if self.device.type == "cuda" else None
            )
            
            # Check model parameters
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            logger.info(f"✓ Model loaded successfully")
            logger.info(f"  Total parameters: {total_params:,}")
            logger.info(f"  Trainable parameters: {trainable_params:,}")
            
            return True
            
        except Exception as e:
            logger.error(f"✗ Model loading failed: {e}")
            return False
    
    def check_lora_setup(self) -> bool:
        """Test LoRA configuration and model modification."""
        logger.info("Testing LoRA setup...")
        
        try:
            # Create LoRA config
            lora_config = LoraConfig(
                r=self.config['lora']['r'],
                lora_alpha=self.config['lora']['alpha'],
                target_modules=self.config['lora']['target_modules'],
                lora_dropout=self.config['lora']['dropout'],
                bias=self.config['lora']['bias'],
                task_type=TaskType.CAUSAL_LM
            )
            
            # Apply LoRA to model
            lora_model = get_peft_model(self.model, lora_config)
            
            # Check parameter counts
            total_params = sum(p.numel() for p in lora_model.parameters())
            trainable_params = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
            
            logger.info(f"✓ LoRA setup successful")
            logger.info(f"  Total parameters: {total_params:,}")
            logger.info(f"  Trainable parameters: {trainable_params:,}")
            logger.info(f"  Trainable ratio: {trainable_params/total_params:.4f}")
            
            # Verify that only LoRA parameters are trainable
            lora_param_count = 0
            for name, param in lora_model.named_parameters():
                if param.requires_grad and "lora" in name:
                    lora_param_count += param.numel()
            
            if lora_param_count != trainable_params:
                logger.warning(f"Mismatch in LoRA parameters: {lora_param_count} vs {trainable_params}")
            
            return True
            
        except Exception as e:
            logger.error(f"✗ LoRA setup failed: {e}")
            return False
    
    def check_data_loading(self) -> bool:
        """Test data loading for all tasks."""
        logger.info("Testing data loading...")
        
        try:
            # Test loading samples from all tasks
            samples = self.data_loader.get_all_task_samples(
                num_samples_per_task=self.num_samples,
                split="train"
            )
            
            if len(samples) != 4:
                raise SanityCheckError(f"Expected 4 tasks, got {len(samples)}")
            
            # Verify each task's data format
            for task_name, data in samples.items():
                if "input_ids" not in data or "attention_mask" not in data:
                    raise SanityCheckError(f"Missing required keys in {task_name}")
                
                expected_shape = (self.num_samples,)
                if data["input_ids"].shape[0] != self.num_samples:
                    raise SanityCheckError(f"Wrong batch size for {task_name}: {data['input_ids'].shape[0]}")
                
                logger.info(f"  ✓ {task_name}: {data['input_ids'].shape}")
            
            logger.info("✓ Data loading successful for all tasks")
            return True
            
        except Exception as e:
            logger.error(f"✗ Data loading failed: {e}")
            return False
    
    def check_gradient_flow(self, method: str = "lora") -> bool:
        """Test gradient flow during training."""
        logger.info(f"Testing gradient flow for {method}...")
        
        try:
            # Prepare model
            if method == "lora":
                lora_config = LoraConfig(
                    r=self.config['lora']['r'],
                    lora_alpha=self.config['lora']['alpha'],
                    target_modules=self.config['lora']['target_modules'],
                    lora_dropout=self.config['lora']['dropout'],
                    bias=self.config['lora']['bias'],
                    task_type=TaskType.CAUSAL_LM
                )
                model = get_peft_model(self.model, lora_config)
            else:
                model = self.model
                # For full fine-tuning, enable gradients for all parameters
                for param in model.parameters():
                    param.requires_grad = True
            
            model.train()
            
            # Get small data sample
            sample_data = self.data_loader.prepare_classification_data("sst2", "train", 5)
            input_ids = sample_data["input_ids"].to(self.device)
            attention_mask = sample_data["attention_mask"].to(self.device)
            labels = sample_data["labels"].to(self.device)
            
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            
            # Check gradients
            grad_params = 0
            total_grad_norm = 0.0
            
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    grad_params += 1
                    total_grad_norm += param.grad.norm().item()
            
            if grad_params == 0:
                raise SanityCheckError("No gradients found")
            
            if total_grad_norm == 0:
                raise SanityCheckError("All gradients are zero")
            
            logger.info(f"✓ Gradient flow successful for {method}")
            logger.info(f"  Parameters with gradients: {grad_params}")
            logger.info(f"  Total gradient norm: {total_grad_norm:.6f}")
            
            return True
            
        except Exception as e:
            logger.error(f"✗ Gradient flow test failed for {method}: {e}")
            return False
    
    def check_overfitting_capability(self, task_name: str = "sst2", method: str = "lora") -> bool:
        """Test if model can overfit a small dataset."""
        logger.info(f"Testing overfitting capability on {task_name} with {method}...")
        
        try:
            # Prepare model
            if method == "lora":
                lora_config = LoraConfig(
                    r=self.config['lora']['r'],
                    lora_alpha=self.config['lora']['alpha'],
                    target_modules=self.config['lora']['target_modules'],
                    lora_dropout=0.0,  # Disable dropout for overfitting
                    bias=self.config['lora']['bias'],
                    task_type=TaskType.CAUSAL_LM
                )
                model = get_peft_model(self.model, lora_config)
            else:
                model = self.model
                for param in model.parameters():
                    param.requires_grad = True
            
            model.train()
            
            # Get small training sample
            sample_data = self.data_loader.prepare_classification_data(task_name, "train", self.num_samples)
            
            # Simple training loop
            optimizer = torch.optim.AdamW(
                model.parameters(), 
                lr=self.config['training']['learning_rate'] * self.sanity_config['learning_rate_multiplier']
            )
            
            initial_loss = None
            final_loss = None
            
            for epoch in range(self.max_epochs):
                # Forward pass
                input_ids = sample_data["input_ids"].to(self.device)
                attention_mask = sample_data["attention_mask"].to(self.device)
                labels = sample_data["labels"].to(self.device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                
                if epoch == 0:
                    initial_loss = loss.item()
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                logger.info(f"  Epoch {epoch + 1}: Loss = {loss.item():.6f}")
                
                # Check for overfitting
                if loss.item() < self.expected_loss_threshold:
                    final_loss = loss.item()
                    logger.info(f"✓ Successfully overfitted at epoch {epoch + 1}")
                    break
            else:
                final_loss = loss.item()
                if final_loss >= self.expected_loss_threshold:
                    logger.warning(f"Model may not have overfitted sufficiently: final loss {final_loss:.6f}")
            
            # Verify loss reduction
            loss_reduction = initial_loss - final_loss
            if loss_reduction <= 0:
                raise SanityCheckError(f"Loss did not decrease: {initial_loss:.6f} -> {final_loss:.6f}")
            
            logger.info(f"✓ Overfitting test successful for {task_name} with {method}")
            logger.info(f"  Initial loss: {initial_loss:.6f}")
            logger.info(f"  Final loss: {final_loss:.6f}")
            logger.info(f"  Loss reduction: {loss_reduction:.6f}")
            
            return True
            
        except Exception as e:
            logger.error(f"✗ Overfitting test failed for {task_name} with {method}: {e}")
            return False
    
    def check_wandb_logging(self) -> bool:
        """Test Weights & Biases logging functionality."""
        logger.info("Testing W&B logging...")
        
        try:
            # Initialize W&B with test run
            wandb.init(
                project=self.config['wandb']['project'],
                entity=self.config['wandb']['entity'],
                name="sanity-check-test",
                tags=["sanity-check", "test"],
                config={
                    "model": self.model_name,
                    "test_type": "sanity_check",
                    "timestamp": time.time()
                }
            )
            
            # Log some test metrics
            test_metrics = {
                "test_loss": 0.5,
                "test_accuracy": 0.8,
                "test_step": 1
            }
            
            wandb.log(test_metrics)
            
            # Log a simple plot
            x = np.linspace(0, 10, 100)
            y = np.sin(x)
            
            wandb.log({
                "test_plot": wandb.plot.line_series(
                    xs=x,
                    ys=[y],
                    keys=["sin(x)"],
                    title="Test Plot",
                    xname="x"
                )
            })
            
            # Finish the run
            wandb.finish()
            
            logger.info("✓ W&B logging successful")
            return True
            
        except Exception as e:
            logger.error(f"✗ W&B logging failed: {e}")
            return False
    
    def check_reproducibility(self, task_name: str = "sst2", method: str = "lora", num_runs: int = 2) -> bool:
        """Test reproducibility with fixed seeds."""
        logger.info(f"Testing reproducibility with {num_runs} runs...")
        
        try:
            results = []
            
            for run_idx in range(num_runs):
                logger.info(f"  Run {run_idx + 1}/{num_runs}")
                
                # Set seeds
                torch.manual_seed(self.config['reproducibility']['seed'])
                np.random.seed(self.config['reproducibility']['seed'])
                
                # Prepare model
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                    device_map="auto" if self.device.type == "cuda" else None
                )
                
                if method == "lora":
                    lora_config = LoraConfig(
                        r=self.config['lora']['r'],
                        lora_alpha=self.config['lora']['alpha'],
                        target_modules=self.config['lora']['target_modules'],
                        lora_dropout=0.0,
                        bias=self.config['lora']['bias'],
                        task_type=TaskType.CAUSAL_LM
                    )
                    model = get_peft_model(model, lora_config)
                
                model.train()
                
                # Get data (with same seed)
                sample_data = self.data_loader.prepare_classification_data(task_name, "train", self.num_samples)
                
                # Single training step
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
                
                input_ids = sample_data["input_ids"].to(self.device)
                attention_mask = sample_data["attention_mask"].to(self.device)
                labels = sample_data["labels"].to(self.device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Record results
                results.append(loss.item())
                logger.info(f"    Loss: {loss.item():.8f}")
            
            # Check if results are identical
            if len(set(f"{x:.8f}" for x in results)) == 1:
                logger.info("✓ Reproducibility test passed - identical results")
                return True
            else:
                # Check if results are very close (allowing for minor numerical differences)
                max_diff = max(results) - min(results)
                if max_diff < 1e-6:
                    logger.info(f"✓ Reproducibility test passed - results very close (max diff: {max_diff:.2e})")
                    return True
                else:
                    logger.warning(f"Reproducibility test questionable - max diff: {max_diff:.2e}")
                    logger.warning(f"Results: {results}")
                    return False
            
        except Exception as e:
            logger.error(f"✗ Reproducibility test failed: {e}")
            return False
    
    def run_comprehensive_sanity_checks(self) -> Dict[str, bool]:
        """Run all sanity checks and return results."""
        logger.info("Running comprehensive sanity checks...")
        logger.info("=" * 60)
        
        # Initialize components
        self.initialize_components()
        
        results = {}
        
        # Basic checks
        results["model_loading"] = self.check_model_loading()
        results["lora_setup"] = self.check_lora_setup()
        results["data_loading"] = self.check_data_loading()
        
        # Gradient flow checks
        results["gradient_flow_lora"] = self.check_gradient_flow("lora")
        results["gradient_flow_full"] = self.check_gradient_flow("full")
        
        # Overfitting checks
        results["overfitting_lora"] = self.check_overfitting_capability("sst2", "lora")
        results["overfitting_full"] = self.check_overfitting_capability("sst2", "full")
        
        # Infrastructure checks
        results["wandb_logging"] = self.check_wandb_logging()
        results["reproducibility"] = self.check_reproducibility()
        
        # Summary
        logger.info("=" * 60)
        logger.info("SANITY CHECK SUMMARY")
        logger.info("=" * 60)
        
        passed = 0
        total = len(results)
        
        for check_name, passed_check in results.items():
            status = "✓ PASS" if passed_check else "✗ FAIL"
            logger.info(f"{check_name:<25}: {status}")
            if passed_check:
                passed += 1
        
        logger.info("=" * 60)
        logger.info(f"Overall: {passed}/{total} checks passed")
        
        if passed == total:
            logger.info("🎉 All sanity checks passed! Setup is ready for experiments.")
        else:
            logger.error(f"❌ {total - passed} sanity checks failed. Please fix issues before proceeding.")
        
        return results


def run_sanity_checks(config_path: str = "shared/config.yaml") -> bool:
    """Main function to run sanity checks.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        True if all checks pass, False otherwise
    """
    checker = ModelSanityChecker(config_path)
    results = checker.run_comprehensive_sanity_checks()
    return all(results.values())


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run sanity checks for NLP experiments")
    parser.add_argument("--task", type=str, help="Specific task to test overfitting on")
    parser.add_argument("--num-samples", type=int, help="Number of samples for overfitting test")
    parser.add_argument("--config", default="shared/config.yaml", help="Config file path")
    
    args = parser.parse_args()
    
    if args.task:
        # Run specific task overfitting test
        checker = ModelSanityChecker(args.config)
        checker.initialize_components()
        
        # Override num_samples if provided
        if args.num_samples:
            checker.num_samples = args.num_samples
            
        print(f"Running overfitting test for {args.task}...")
        success = checker.check_overfitting_capability(args.task, "lora")
        if success:
            print(f"✅ {args.task} overfitting test passed")
        else:
            print(f"⚠️ {args.task} overfitting test had issues")
        exit(0 if success else 1)
    else:
        # Run comprehensive sanity checks
        success = run_sanity_checks()
        exit(0 if success else 1)
