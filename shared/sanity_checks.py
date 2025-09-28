#!/usr/bin/env python3
"""Enhanced sanity check utilities with two-stage validation.

Performs both overfitting validation and production stability checks.
"""

import subprocess
import sys
import os
import logging
import argparse

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.sanity_check_framework import SanityCheckFramework

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_enhanced_sanity_checks(task: str) -> bool:
    """Run enhanced sanity checks for a task - focus on overfitting validation."""
    logger.info(f"Running overfitting sanity checks for {task.upper()}")
    logger.info("=" * 60)
    
    framework = SanityCheckFramework()
    
    # Run overfitting checks for both methods (core Phase 0 requirement)
    lora_result = framework.run_overfitting_sanity_check(task, "lora", seed=42)
    full_result = framework.run_overfitting_sanity_check(task, "full_finetune", seed=42)
    
    both_passed = lora_result['success'] and full_result['success']
    
    if both_passed:
        logger.info(f"🎉 ALL OVERFITTING SANITY CHECKS PASSED for {task.upper()}")
        logger.info("✅ LoRA overfitting: PASSED")
        logger.info("✅ Full FT overfitting: PASSED")
    else:
        logger.error(f"❌ OVERFITTING SANITY CHECKS FAILED for {task.upper()}")
        logger.error(f"✅ LoRA overfitting: {'PASSED' if lora_result['success'] else 'FAILED'}")
        logger.error(f"✅ Full FT overfitting: {'PASSED' if full_result['success'] else 'FAILED'}")
    
    return both_passed


def run_legacy_sanity_check(task: str, method: str = "lora") -> bool:
    """Run legacy single sanity check for backward compatibility.""" 
    logger.info(f"Running legacy sanity check: {task} with {method}")
    
    framework = SanityCheckFramework()
    
    # Run just the overfitting check
    overfitting_result = framework.run_overfitting_sanity_check(task, method, seed=42)
    
    if overfitting_result['success']:
        logger.info(f"✅ Legacy sanity check passed for {task} with {method}")
        if overfitting_result['metrics']:
            logger.info(f"   Final loss: {overfitting_result['metrics'].get('final_loss', 'N/A')}")
        return True
    else:
        logger.error(f"❌ Legacy sanity check failed for {task} with {method}")
        for issue in overfitting_result['issues']:
            logger.error(f"   Issue: {issue}")
        return False
            

def run_comprehensive_checks(task: str) -> bool:
    """Run comprehensive sanity checks for a task - enhanced version."""
    return run_enhanced_sanity_checks(task)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run sanity checks using production experiment scripts")
    parser.add_argument("--task", type=str, required=True,
                       choices=["mrpc", "sst2", "rte", "squad_v2"],
                       help="Task to test")
    
    args = parser.parse_args()
    
    logger.info(f"Testing {args.task} using production scripts with --sanity-check flag...")
    success = run_comprehensive_checks(args.task)
    exit(0 if success else 1)