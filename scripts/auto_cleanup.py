#!/usr/bin/env python3
"""
Automatic Cleanup Script for Continuous Experiments

This script can be integrated into the experiment pipeline to automatically
clean up representations after each completed task, preventing disk space issues.
"""

import os
import sys
import yaml
from pathlib import Path
from cleanup_experiment import cleanup_experiment, get_directory_size, format_size

def check_disk_space(threshold_percent=80):
    """Check if disk usage is above threshold."""
    import shutil
    total, used, free = shutil.disk_usage('/')
    usage_percent = (used / total) * 100
    return usage_percent, usage_percent > threshold_percent

def auto_cleanup_after_task(results_dir="results", task_name=None, keep_final=True):
    """Automatically clean up representations after a task completes."""
    
    print(f"\n🧹 AUTO-CLEANUP: Task '{task_name}' completed")
    print("=" * 50)
    
    # Check disk usage
    usage_percent, needs_cleanup = check_disk_space(threshold_percent=75)
    print(f"💾 Current disk usage: {usage_percent:.1f}%")
    
    # Always clean wandb cache (accumulates rapidly)
    print("🧹 Cleaning wandb temporary directories...")
    try:
        from cleanup_experiment import cleanup_wandb_cache
        wandb_freed = cleanup_wandb_cache()
        print(f"✅ Wandb cleanup freed: {format_size(wandb_freed)}")
    except Exception as e:
        print(f"⚠️  Wandb cleanup failed: {e}")
        wandb_freed = 0
    
    # Re-check disk usage after wandb cleanup
    usage_percent, needs_cleanup = check_disk_space(threshold_percent=75)
    print(f"💾 Disk usage after wandb cleanup: {usage_percent:.1f}%")
    
    if not needs_cleanup:
        print("✅ Disk usage is healthy after wandb cleanup")
        return
    
    print("⚠️  Disk usage still above 75%, cleaning experiment data...")
    
    # Find the most recent experiment directory
    results_path = Path(results_dir)
    if not results_path.exists():
        print(f"Results directory not found: {results_dir}")
        return
    
    # Get all experiment directories sorted by modification time (newest first)
    experiments = []
    for item in results_path.iterdir():
        if item.is_dir() and "finetune_" in item.name:
            experiments.append((item.stat().st_mtime, item))
    
    experiments.sort(reverse=True)  # Newest first
    
    if not experiments:
        print("No experiment directories found")
        return
    
    # Clean up the most recent experiment (current one)
    _, most_recent = experiments[0]
    print(f"🎯 Cleaning most recent experiment: {most_recent.name}")
    
    # Clean representations but keep final step
    freed_space = cleanup_experiment(str(most_recent), mode="representations", keep_final=keep_final)
    
    # Check if we need to clean older experiments too
    usage_percent_after, still_needs_cleanup = check_disk_space(threshold_percent=70)
    print(f"💾 Disk usage after cleanup: {usage_percent_after:.1f}%")
    
    if still_needs_cleanup and len(experiments) > 1:
        print("Still above 70%, cleaning older experiments...")
        
        # Clean older experiments (keep final representations)
        for _, old_experiment in experiments[1:]:
            if usage_percent_after <= 60:  # Stop when we get to reasonable usage
                break
                
            print(f"🗂️  Cleaning older experiment: {old_experiment.name}")
            additional_freed = cleanup_experiment(str(old_experiment), mode="representations", keep_final=True)
            freed_space += additional_freed
            
            usage_percent_after, _ = check_disk_space()
            print(f"💾 Disk usage now: {usage_percent_after:.1f}%")
    
    total_freed = wandb_freed + freed_space
    print(f"\n✅ Auto-cleanup complete! Total freed: {format_size(total_freed)}")
    print(f"💾 Final disk usage: {usage_percent_after:.1f}%")

def integrate_with_experiment_callback():
    """Integration point for experiment scripts."""
    
    # This can be called at the end of each experiment
    # Usage: python auto_cleanup.py --task squad_v2 --results-dir results
    
    import argparse
    parser = argparse.ArgumentParser(description="Auto-cleanup after experiment")
    parser.add_argument("--task", type=str, required=True, help="Completed task name")
    parser.add_argument("--results-dir", type=str, default="results", help="Results directory")
    parser.add_argument("--no-keep-final", action="store_true", help="Don't keep final representations")
    
    args = parser.parse_args()
    
    keep_final = not args.no_keep_final
    auto_cleanup_after_task(args.results_dir, args.task, keep_final)

if __name__ == "__main__":
    integrate_with_experiment_callback()
