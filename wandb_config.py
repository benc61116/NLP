#!/usr/bin/env python3
"""
Wandb Configuration Utilities

This module provides utilities to configure wandb to use project-specific
temporary directories and prevent cache buildup in system temp folders.
"""

import os
import shutil
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def setup_wandb_directories():
    """Set up project-specific wandb directories to prevent system cache pollution."""
    
    # Create project-specific wandb directories
    wandb_cache_dir = Path.cwd() / 'wandb_cache'
    wandb_temp_dir = Path.cwd() / 'wandb_temp'
    
    # Ensure directories exist
    wandb_cache_dir.mkdir(exist_ok=True)
    wandb_temp_dir.mkdir(exist_ok=True)
    
    # Configure environment variables for wandb
    os.environ['WANDB_CACHE_DIR'] = str(wandb_cache_dir)
    os.environ['WANDB_TMPDIR'] = str(wandb_temp_dir)
    
    logger.info(f"📁 Wandb cache directory: {wandb_cache_dir}")
    logger.info(f"📁 Wandb temp directory: {wandb_temp_dir}")
    
    return wandb_cache_dir, wandb_temp_dir

def cleanup_wandb_temp_dirs():
    """Clean up project-specific wandb temporary directories."""
    
    cleaned_size = 0
    
    for temp_dir_name in ['wandb_temp', 'wandb_cache']:
        temp_path = Path(temp_dir_name)
        if temp_path.exists():
            for item in temp_path.iterdir():
                try:
                    if item.is_dir():
                        size = sum(f.stat().st_size for f in item.rglob('*') if f.is_file())
                        shutil.rmtree(item)
                        cleaned_size += size
                        logger.debug(f"Removed wandb temp dir: {item}")
                    else:
                        size = item.stat().st_size
                        item.unlink()
                        cleaned_size += size
                        logger.debug(f"Removed wandb temp file: {item}")
                except Exception as e:
                    logger.debug(f"Could not remove {item}: {e}")
    
    if cleaned_size > 0:
        size_mb = cleaned_size / (1024 * 1024)
        logger.info(f"🧹 Cleaned {size_mb:.1f}MB from wandb temp directories")
    
    return cleaned_size

def get_wandb_settings():
    """Get optimized wandb settings to prevent cache buildup."""
    
    import wandb
    
    return wandb.Settings(
        _disable_stats=True,
        _disable_meta=True,
        _tmp_dir=str(Path.cwd() / 'wandb_temp')
    )

def initialize_wandb_with_cleanup(project, name, config, mode="online"):
    """Initialize wandb with proper temp directory management."""
    
    import wandb
    
    # Set up directories
    setup_wandb_directories()
    
    # Get optimized settings
    settings = get_wandb_settings()
    
    if mode == "offline":
        settings.update({"mode": "offline"})
    
    try:
        wandb.init(
            project=project,
            name=name,
            config=config,
            settings=settings
        )
        logger.info(f"✅ Wandb initialized: {name}")
        return True
        
    except Exception as e:
        logger.warning(f"Wandb initialization failed: {e}")
        # Fallback to offline mode
        try:
            wandb.init(
                project=project,
                name=name,
                config=config,
                mode="offline",
                settings=wandb.Settings(_tmp_dir=str(Path.cwd() / 'wandb_temp'))
            )
            logger.info(f"✅ Wandb initialized (offline): {name}")
            return True
        except Exception as e2:
            logger.error(f"Wandb offline initialization also failed: {e2}")
            return False
