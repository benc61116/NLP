#!/usr/bin/env python3
"""
Disk Space Cleanup Script for NLP Experiments

This script helps manage disk space by cleaning up completed experiments.
It can remove representations, checkpoints, or entire experiment directories.
"""

import os
import shutil
import argparse
from pathlib import Path
import subprocess

def get_directory_size(path):
    """Get directory size in bytes."""
    result = subprocess.run(['du', '-sb', path], capture_output=True, text=True)
    if result.returncode == 0:
        return int(result.stdout.split()[0])
    return 0

def format_size(bytes_size):
    """Format bytes as human readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.1f}{unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.1f}PB"

def cleanup_representations(experiment_dir, keep_final=True):
    """Remove representation files, optionally keeping final step."""
    repr_dir = Path(experiment_dir) / "representations"
    if not repr_dir.exists():
        print(f"No representations directory found in {experiment_dir}")
        return 0
    
    total_freed = 0
    step_dirs = sorted([d for d in repr_dir.iterdir() if d.is_dir()])
    
    # Find the highest step number if keeping final
    final_step_dir = None
    if keep_final and step_dirs:
        # Look for step directories
        step_numbers = []
        for step_dir in step_dirs:
            for task_dir in step_dir.iterdir():
                if task_dir.is_dir():
                    for step_subdir in task_dir.iterdir():
                        if step_subdir.name.startswith("step_"):
                            try:
                                step_num = int(step_subdir.name.split("_")[1])
                                step_numbers.append((step_num, step_subdir))
                            except (ValueError, IndexError):
                                continue
        
        if step_numbers:
            final_step_num, final_step_dir = max(step_numbers, key=lambda x: x[0])
            print(f"Keeping final representations from step {final_step_num}")
    
    # Remove representation directories
    for task_dir in step_dirs:
        if task_dir.is_dir():
            for step_subdir in task_dir.rglob("step_*"):
                if step_subdir.is_dir() and step_subdir != final_step_dir:
                    size = get_directory_size(str(step_subdir))
                    shutil.rmtree(step_subdir)
                    total_freed += size
                    print(f"Removed {step_subdir}: {format_size(size)}")
    
    return total_freed

def cleanup_checkpoints(experiment_dir, keep_final=True):
    """Remove model checkpoints, optionally keeping final checkpoint."""
    experiment_path = Path(experiment_dir)
    total_freed = 0
    
    # Find all checkpoint directories
    checkpoint_dirs = []
    for item in experiment_path.rglob("checkpoint-*"):
        if item.is_dir():
            try:
                step_num = int(item.name.split("-")[1])
                checkpoint_dirs.append((step_num, item))
            except (ValueError, IndexError):
                continue
    
    if not checkpoint_dirs:
        print(f"No checkpoints found in {experiment_dir}")
        return 0
    
    # Sort by step number
    checkpoint_dirs.sort(key=lambda x: x[0])
    
    # Keep final checkpoint if requested
    if keep_final and checkpoint_dirs:
        final_step, final_dir = checkpoint_dirs[-1]
        checkpoint_dirs = checkpoint_dirs[:-1]
        print(f"Keeping final checkpoint: {final_dir} (step {final_step})")
    
    # Remove other checkpoints
    for step_num, checkpoint_dir in checkpoint_dirs:
        size = get_directory_size(str(checkpoint_dir))
        shutil.rmtree(checkpoint_dir)
        total_freed += size
        print(f"Removed checkpoint-{step_num}: {format_size(size)}")
    
    return total_freed

def cleanup_experiment(experiment_dir, mode="representations", keep_final=True):
    """Clean up an experiment directory."""
    experiment_path = Path(experiment_dir)
    if not experiment_path.exists():
        print(f"Experiment directory does not exist: {experiment_dir}")
        return 0
    
    print(f"\n🧹 Cleaning {experiment_dir} (mode: {mode})")
    print("=" * 60)
    
    initial_size = get_directory_size(str(experiment_path))
    print(f"Initial size: {format_size(initial_size)}")
    
    total_freed = 0
    
    if mode == "representations":
        total_freed += cleanup_representations(experiment_dir, keep_final)
    elif mode == "checkpoints":
        total_freed += cleanup_checkpoints(experiment_dir, keep_final)
    elif mode == "both":
        total_freed += cleanup_representations(experiment_dir, keep_final)
        total_freed += cleanup_checkpoints(experiment_dir, keep_final)
    elif mode == "all":
        # Remove entire experiment directory
        total_freed = initial_size
        shutil.rmtree(experiment_path)
        print(f"Removed entire experiment: {format_size(total_freed)}")
        return total_freed
    
    final_size = get_directory_size(str(experiment_path))
    print(f"Final size: {format_size(final_size)}")
    print(f"Space freed: {format_size(total_freed)}")
    
    return total_freed

def list_experiments(results_dir):
    """List all experiments with their sizes."""
    results_path = Path(results_dir)
    if not results_path.exists():
        print(f"Results directory does not exist: {results_dir}")
        return
    
    print("\n📊 EXPERIMENT DIRECTORY SIZES")
    print("=" * 60)
    
    experiments = []
    for item in results_path.iterdir():
        if item.is_dir():
            size = get_directory_size(str(item))
            experiments.append((size, item.name, item))
    
    # Sort by size (largest first)
    experiments.sort(reverse=True)
    
    total_size = 0
    for size, name, path in experiments:
        total_size += size
        print(f"{format_size(size):>8} - {name}")
    
    print("-" * 60)
    print(f"{format_size(total_size):>8} - TOTAL")

def cleanup_wandb_cache():
    """Clean up wandb temporary directories and cache."""
    print("\n🧹 CLEANING WANDB CACHE AND TEMP DIRECTORIES")
    print("=" * 50)
    
    total_freed = 0
    
    # 1. Clean up /var/tmp wandb directories
    import glob
    wandb_temp_dirs = glob.glob("/var/tmp/*wandb*")
    
    if wandb_temp_dirs:
        print(f"Found {len(wandb_temp_dirs)} wandb temp directories in /var/tmp/")
        for temp_dir in wandb_temp_dirs:
            try:
                size = get_directory_size(temp_dir)
                shutil.rmtree(temp_dir)
                total_freed += size
                print(f"Removed {temp_dir}: {format_size(size)}")
            except Exception as e:
                print(f"Could not remove {temp_dir}: {e}")
    
    # 2. Clean up ~/.cache/wandb (keep recent)
    cache_dir = Path.home() / ".cache" / "wandb"
    if cache_dir.exists():
        # Keep only last 7 days of cache
        import time
        cutoff_time = time.time() - (7 * 24 * 3600)  # 7 days ago
        
        for item in cache_dir.iterdir():
            try:
                if item.stat().st_mtime < cutoff_time:
                    size = get_directory_size(str(item))
                    shutil.rmtree(item)
                    total_freed += size
                    print(f"Removed old cache: {item.name}: {format_size(size)}")
            except Exception as e:
                print(f"Could not remove cache {item}: {e}")
    
    # 3. Clean up old artifacts (keep recent)
    artifacts_dir = Path.home() / ".local" / "share" / "wandb" / "artifacts"
    if artifacts_dir.exists():
        # Keep only last 30 days of artifacts
        import time
        cutoff_time = time.time() - (30 * 24 * 3600)  # 30 days ago
        
        for item in artifacts_dir.iterdir():
            try:
                if item.stat().st_mtime < cutoff_time:
                    size = get_directory_size(str(item))
                    shutil.rmtree(item)
                    total_freed += size
                    print(f"Removed old artifacts: {item.name}: {format_size(size)}")
            except Exception as e:
                print(f"Could not remove artifacts {item}: {e}")
    
    print(f"\n✅ Wandb cleanup complete! Total freed: {format_size(total_freed)}")
    return total_freed

def main():
    parser = argparse.ArgumentParser(description="Clean up NLP experiment data")
    parser.add_argument("--list", action="store_true", help="List all experiments with sizes")
    parser.add_argument("--experiment", type=str, help="Experiment directory to clean")
    parser.add_argument("--mode", choices=["representations", "checkpoints", "both", "all"], 
                       default="representations", help="What to clean")
    parser.add_argument("--no-keep-final", action="store_true", help="Don't keep final checkpoint/representations")
    parser.add_argument("--results-dir", type=str, default="results", help="Results directory path")
    parser.add_argument("--wandb-cache", action="store_true", help="Clean wandb cache and temp directories")
    
    args = parser.parse_args()
    
    if args.wandb_cache:
        cleanup_wandb_cache()
        return
    
    if args.list:
        list_experiments(args.results_dir)
        return
    
    if not args.experiment:
        print("Please specify --experiment or use --list to see available experiments")
        return
    
    keep_final = not args.no_keep_final
    total_freed = cleanup_experiment(args.experiment, args.mode, keep_final)
    
    print(f"\n✅ Cleanup complete! Total space freed: {format_size(total_freed)}")

if __name__ == "__main__":
    main()
