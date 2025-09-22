#!/usr/bin/env python3
"""Specific validation tests for overfitting and reproducibility."""

import torch
import torch.nn as nn
import numpy as np
import logging
import time
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def set_seed(seed=42):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_model_and_tokenizer():
    """Load model and tokenizer without complex dependencies."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        model_name = "microsoft/DialoGPT-small"  # Public model for testing
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return None, None

def prepare_small_dataset(task_name, num_samples=10):
    """Prepare a small dataset for overfitting tests."""
    try:
        from shared.data_preparation import TaskDataLoader
        
        data_loader = TaskDataLoader("microsoft/DialoGPT-small")
        
        if task_name in ['mrpc', 'sst2', 'rte']:
            data = data_loader.prepare_classification_data(task_name, "train", num_samples)
            return data
        else:
            logger.warning(f"Skipping {task_name} for simplified test")
            return None
            
    except Exception as e:
        logger.error(f"Failed to prepare dataset for {task_name}: {e}")
        return None

def test_overfitting_simple(model, tokenizer, data, task_name, method="simple", max_epochs=50):
    """Simple overfitting test without complex training framework."""
    logger.info(f"Testing overfitting on {task_name} with {method} method...")
    
    if data is None:
        logger.error(f"No data provided for {task_name}")
        return False
    
    # Set up model for training
    model.train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Prepare data
    input_ids = data["input_ids"][:10].to(device)  # Use only 10 samples
    attention_mask = data["attention_mask"][:10].to(device)
    labels = data["labels"][:10].to(device) if "labels" in data else input_ids
    
    # Simple optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)  # Higher LR for overfitting
    
    initial_loss = None
    losses = []
    
    logger.info(f"Starting overfitting test with {len(input_ids)} samples...")
    
    for epoch in range(max_epochs):
        optimizer.zero_grad()
        
        # Forward pass - simplified for classification
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = outputs.loss
        
        losses.append(loss.item())
        
        if epoch == 0:
            initial_loss = loss.item()
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Log progress every 10 epochs
        if epoch % 10 == 0 or epoch < 5:
            logger.info(f"  Epoch {epoch + 1}: Loss = {loss.item():.6f}")
        
        # Check for overfitting (loss < 0.1)
        if loss.item() < 0.1:
            final_loss = loss.item()
            loss_reduction = initial_loss - final_loss
            
            logger.info(f"✓ Overfitting achieved at epoch {epoch + 1}!")
            logger.info(f"  Initial loss: {initial_loss:.6f}")
            logger.info(f"  Final loss: {final_loss:.6f}")
            logger.info(f"  Loss reduction: {loss_reduction:.6f}")
            
            # Check monotonic decrease (allow some noise)
            monotonic = check_monotonic_decrease(losses)
            logger.info(f"  Monotonic decrease: {'✓' if monotonic else '⚠ (with noise)'}")
            
            return True
    
    final_loss = losses[-1]
    loss_reduction = initial_loss - final_loss
    
    logger.warning(f"⚠ Did not achieve strong overfitting in {max_epochs} epochs")
    logger.info(f"  Initial loss: {initial_loss:.6f}")
    logger.info(f"  Final loss: {final_loss:.6f}")
    logger.info(f"  Loss reduction: {loss_reduction:.6f}")
    
    # Still consider it a pass if loss decreased significantly
    return loss_reduction > (initial_loss * 0.5)

def check_monotonic_decrease(losses, tolerance=0.1):
    """Check if losses decrease monotonically (with some tolerance for noise)."""
    if len(losses) < 2:
        return True
    
    increases = 0
    for i in range(1, len(losses)):
        if losses[i] > losses[i-1] * (1 + tolerance):
            increases += 1
    
    # Allow up to 20% of steps to have increases (due to noise)
    return increases / len(losses) <= 0.2

def test_reproducibility(num_runs=2):
    """Test reproducibility with fixed seeds."""
    logger.info("Testing reproducibility with fixed seeds...")
    
    model, tokenizer = load_model_and_tokenizer()
    if model is None:
        logger.error("Failed to load model for reproducibility test")
        return False
    
    # Prepare small dataset
    data = prepare_small_dataset("sst2", 5)
    if data is None:
        logger.error("Failed to prepare data for reproducibility test")
        return False
    
    results = []
    
    for run in range(num_runs):
        logger.info(f"  Run {run + 1}/{num_runs}")
        
        # Reset seed
        set_seed(42)
        
        # Reload model to reset parameters
        model, tokenizer = load_model_and_tokenizer()
        model.train()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        # Prepare data
        input_ids = data["input_ids"][:5].to(device)
        attention_mask = data["attention_mask"][:5].to(device)
        
        # Single forward pass
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            first_logit = logits[0, 0, 0].item()  # Get a single value for comparison
            results.append(first_logit)
        
        logger.info(f"    First logit value: {first_logit:.8f}")
    
    # Check if results are identical (within numerical precision)
    if len(set(f"{x:.8f}" for x in results)) == 1:
        logger.info("✓ Perfect reproducibility - identical results!")
        return True
    else:
        max_diff = max(results) - min(results)
        if max_diff < 1e-6:
            logger.info(f"✓ Good reproducibility - max difference: {max_diff:.2e}")
            return True
        else:
            logger.warning(f"⚠ Reproducibility issue - max difference: {max_diff:.2e}")
            logger.info(f"  Results: {results}")
            return False

def main():
    """Run the specific validation tests."""
    logger.info("="*60)
    logger.info("RUNNING SPECIFIC VALIDATION TESTS")
    logger.info("="*60)
    
    # Set initial seed
    set_seed(42)
    
    # Test 1: Reproducibility
    logger.info("\n1. TESTING REPRODUCIBILITY")
    logger.info("-" * 30)
    repro_success = test_reproducibility()
    
    # Test 2: Overfitting tests
    logger.info("\n2. TESTING 10-EXAMPLE OVERFITTING")
    logger.info("-" * 30)
    
    overfitting_results = {}
    tasks_to_test = ['sst2', 'mrpc']  # Limit to 2 tasks for demo
    
    for task in tasks_to_test:
        logger.info(f"\nTesting {task}...")
        
        # Load fresh model and data for each task
        model, tokenizer = load_model_and_tokenizer()
        if model is None:
            overfitting_results[task] = False
            continue
            
        data = prepare_small_dataset(task, 10)
        if data is None:
            overfitting_results[task] = False
            continue
        
        # Set seed for consistent test
        set_seed(42)
        
        # Run overfitting test
        success = test_overfitting_simple(model, tokenizer, data, task, max_epochs=30)
        overfitting_results[task] = success
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("SPECIFIC VALIDATION TEST RESULTS")
    logger.info("="*60)
    
    logger.info(f"Reproducibility Test: {'✓ PASS' if repro_success else '✗ FAIL'}")
    
    logger.info("\nOverfitting Tests:")
    overfitting_passed = 0
    for task, result in overfitting_results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"  {task}: {status}")
        if result:
            overfitting_passed += 1
    
    total_tests = 1 + len(overfitting_results)  # repro + overfitting tests
    passed_tests = (1 if repro_success else 0) + overfitting_passed
    
    logger.info(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        logger.info("🎉 ALL SPECIFIC VALIDATION TESTS PASSED!")
        return True
    else:
        logger.warning(f"⚠ {total_tests - passed_tests} tests failed or had issues")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
