#!/usr/bin/env python3
"""Sanity check utilities using production experiment scripts.

Uses actual production scripts with --sanity-check flag to ensure consistency.
"""

import subprocess
import sys
import os
import logging
import argparse

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_sanity_check(task: str, method: str = "lora") -> bool:
    """Run sanity check using production scripts."""
    logger.info(f"Testing {task} with {method} using production script")
    
    try:
        # Determine script to use
        if method == "lora":
            script = "experiments/lora_finetune.py"
        else:  # full_finetune
            script = "experiments/full_finetune.py"
        
        # Build command with sanity check flag
        cmd = [
            sys.executable, script,
            "--task", task,
            "--mode", "single",
            "--seed", "42",
            "--sanity-check"
        ]
        
        logger.info(f"Running: {' '.join(cmd)}")
        
        # Run the production script in sanity check mode
        result = subprocess.run(
            cmd,
            cwd=os.getcwd(),
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout for sanity check
        )
        
        if result.returncode == 0:
            logger.info(f"✅ Sanity check passed for {task} with {method}")
            # Show last few lines of output
            output_lines = result.stdout.strip().split('\n')
            for line in output_lines[-3:]:
                if line.strip():
                    logger.info(f"   {line}")
            return True
        else:
            logger.error(f"❌ Sanity check failed for {task} with {method}")
            logger.error(f"stdout: {result.stdout}")
            logger.error(f"stderr: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error(f"❌ Sanity check timed out for {task} with {method}")
        return False
    except Exception as e:
        logger.error(f"❌ Sanity check failed for {task} with {method}: {e}")
        return False


def run_comprehensive_checks(task: str) -> bool:
    """Run comprehensive sanity checks for a task."""
    logger.info(f"Running comprehensive sanity checks for {task.upper()}")
    logger.info("=" * 60)
    
    results = {}
    
    # Test both methods using production scripts
    results["lora"] = run_sanity_check(task, "lora")
    results["full_finetune"] = run_sanity_check(task, "full_finetune")
    
    # Summary
    logger.info("=" * 60)
    logger.info(f"SANITY CHECK SUMMARY FOR {task.upper()}")
    logger.info("=" * 60)
    
    passed = 0
    total = len(results)
    
    for method, passed_check in results.items():
        status = "✓ PASS" if passed_check else "✗ FAIL"
        logger.info(f"{method:<15}: {status}")
        if passed_check:
            passed += 1
    
    logger.info("=" * 60)
    logger.info(f"Overall: {passed}/{total} checks passed")
    
    if passed == total:
        logger.info(f"🎉 All sanity checks passed for {task}! Production setup is working.")
    else:
        logger.error(f"❌ {total - passed} sanity checks failed for {task}. Fix issues before running experiments.")
    
    return passed == total


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run sanity checks using production experiment scripts")
    parser.add_argument("--task", type=str, required=True,
                       choices=["mrpc", "sst2", "rte", "squad_v2"],
                       help="Task to test")
    
    args = parser.parse_args()
    
    logger.info(f"Testing {args.task} using production scripts with --sanity-check flag...")
    success = run_comprehensive_checks(args.task)
    exit(0 if success else 1)