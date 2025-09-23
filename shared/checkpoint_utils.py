#!/usr/bin/env python3
"""Checkpoint utilities for resuming interrupted training."""

import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
import torch

logger = logging.getLogger(__name__)

class CheckpointManager:
    """Manages training checkpoints and resume functionality."""
    
    def __init__(self, base_output_dir: str):
        self.base_output_dir = Path(base_output_dir)
        self.progress_file = self.base_output_dir / "experiment_progress.json"
        
    def save_experiment_progress(self, task_name: str, method: str, seed: int, 
                               status: str, checkpoint_path: Optional[str] = None):
        """Save progress for an experiment."""
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing progress
        progress = self.load_experiment_progress()
        
        experiment_key = f"{task_name}_{method}_seed{seed}"
        progress[experiment_key] = {
            "task_name": task_name,
            "method": method,
            "seed": seed,
            "status": status,  # 'started', 'completed', 'failed'
            "checkpoint_path": checkpoint_path,
            "last_updated": datetime.now().isoformat()
        }
        
        # Save updated progress
        with open(self.progress_file, 'w') as f:
            json.dump(progress, f, indent=2)
            
        logger.info(f"Saved progress: {experiment_key} -> {status}")
    
    def load_experiment_progress(self) -> Dict[str, Any]:
        """Load experiment progress from file."""
        if not self.progress_file.exists():
            return {}
        
        try:
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            logger.warning("Could not load progress file, starting fresh")
            return {}
    
    def is_experiment_completed(self, task_name: str, method: str, seed: int) -> bool:
        """Check if an experiment was already completed."""
        progress = self.load_experiment_progress()
        experiment_key = f"{task_name}_{method}_seed{seed}"
        
        return (experiment_key in progress and 
                progress[experiment_key].get("status") == "completed")
    
    def find_latest_checkpoint(self, task_name: str, method: str, seed: int) -> Optional[str]:
        """Find the latest checkpoint for an experiment."""
        progress = self.load_experiment_progress()
        experiment_key = f"{task_name}_{method}_seed{seed}"
        
        if experiment_key in progress:
            checkpoint_path = progress[experiment_key].get("checkpoint_path")
            if checkpoint_path and Path(checkpoint_path).exists():
                return checkpoint_path
        
        # Fallback: search for checkpoints in expected locations
        possible_dirs = [
            self.base_output_dir / f"{method}_{task_name}_seed{seed}",
            self.base_output_dir / f"full_ft_{task_name}_seed{seed}",
            self.base_output_dir / f"lora_{task_name}_seed{seed}"
        ]
        
        for exp_dir in possible_dirs:
            if exp_dir.exists():
                # Look for checkpoint subdirectories
                checkpoint_dirs = [d for d in exp_dir.iterdir() 
                                 if d.is_dir() and d.name.startswith("checkpoint-")]
                if checkpoint_dirs:
                    # Return the latest checkpoint (highest number)
                    latest = max(checkpoint_dirs, 
                               key=lambda x: int(x.name.split("-")[1]) if x.name.split("-")[1].isdigit() else 0)
                    return str(latest)
        
        return None
    
    def get_resume_info(self, task_name: str, method: str, seed: int) -> Dict[str, Any]:
        """Get resume information for an experiment."""
        if self.is_experiment_completed(task_name, method, seed):
            return {
                "should_skip": True,
                "reason": "experiment_completed",
                "checkpoint_path": None
            }
        
        checkpoint_path = self.find_latest_checkpoint(task_name, method, seed)
        if checkpoint_path:
            return {
                "should_skip": False,
                "should_resume": True,
                "checkpoint_path": checkpoint_path,
                "reason": "checkpoint_found"
            }
        
        return {
            "should_skip": False,
            "should_resume": False,
            "checkpoint_path": None,
            "reason": "no_checkpoint"
        }
    
    def cleanup_failed_checkpoints(self, task_name: str, method: str, seed: int):
        """Clean up checkpoints from failed runs."""
        progress = self.load_experiment_progress()
        experiment_key = f"{task_name}_{method}_seed{seed}"
        
        if (experiment_key in progress and 
            progress[experiment_key].get("status") == "failed"):
            
            checkpoint_path = progress[experiment_key].get("checkpoint_path")
            if checkpoint_path and Path(checkpoint_path).exists():
                logger.info(f"Cleaning up failed checkpoint: {checkpoint_path}")
                # Could implement cleanup logic here if needed
            
            # Reset the progress entry
            progress[experiment_key]["status"] = "reset"
            with open(self.progress_file, 'w') as f:
                json.dump(progress, f, indent=2)
