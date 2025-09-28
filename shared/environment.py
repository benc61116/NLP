#!/usr/bin/env python3
"""Environment standardization utilities for consistent setup across all scripts."""

import os
import sys
import logging
import platform
import subprocess
import torch
import yaml
from pathlib import Path
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

def setup_python_path(workspace_dir: str = None):
    """Ensure the workspace directory is in Python path."""
    if workspace_dir is None:
        # Auto-detect workspace directory
        workspace_dir = Path(__file__).parent.parent.absolute()
    
    workspace_str = str(workspace_dir)
    if workspace_str not in sys.path:
        sys.path.insert(0, workspace_str)
        logger.info(f"Added {workspace_str} to Python path")
    
    return workspace_str

def setup_environment_variables(config: Dict[str, Any] = None):
    """Set up environment variables for consistent behavior."""
    # Load config if not provided
    if config is None:
        from shared.model_validation import load_config
        config = load_config()
    
    # Set up environment variables
    env_vars = {
        'TOKENIZERS_PARALLELISM': 'false',  # Suppress tokenizer warnings
        'WANDB_PROJECT': config.get('wandb', {}).get('project', 'NLP'),
        'OMP_NUM_THREADS': '1',  # Prevent OpenMP conflicts
        'MKL_NUM_THREADS': '1',  # Prevent MKL conflicts
    }
    
    # Set environment variables
    for key, value in env_vars.items():
        if key not in os.environ:  # Don't override existing values
            os.environ[key] = value
            logger.info(f"Set {key}={value}")
    
    return env_vars

def detect_compute_environment() -> Dict[str, Any]:
    """Detect the compute environment and available resources."""
    env_info = {
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'python_executable': sys.executable,
        'cpu_count': os.cpu_count(),
        'cuda_available': torch.cuda.is_available(),
        'cuda_devices': 0,
        'cuda_version': None,
        'gpu_memory': [],
        'total_memory_gb': 0,
    }
    
    # CUDA info
    if torch.cuda.is_available():
        env_info['cuda_devices'] = torch.cuda.device_count()
        env_info['cuda_version'] = torch.version.cuda
        
        # GPU memory info
        for i in range(torch.cuda.device_count()):
            try:
                props = torch.cuda.get_device_properties(i)
                memory_gb = props.total_memory / (1024**3)
                env_info['gpu_memory'].append({
                    'device': i,
                    'name': props.name,
                    'memory_gb': round(memory_gb, 1)
                })
                env_info['total_memory_gb'] += memory_gb
            except Exception as e:
                logger.warning(f"Could not get GPU {i} properties: {e}")
    
    # System memory
    try:
        import psutil
        env_info['system_memory_gb'] = round(psutil.virtual_memory().total / (1024**3), 1)
    except ImportError:
        env_info['system_memory_gb'] = 'Unknown (psutil not available)'
    
    return env_info

def validate_environment_requirements(config: Dict[str, Any] = None) -> Dict[str, Any]:
    """Validate that the environment meets the requirements."""
    if config is None:
        from shared.model_validation import load_config
        config = load_config()
    
    env_info = detect_compute_environment()
    
    results = {
        'valid': True,
        'issues': [],
        'warnings': [],
        'environment': env_info
    }
    
    # Check Python version
    python_version = tuple(map(int, platform.python_version().split('.')[:2]))
    if python_version < (3, 8):
        results['valid'] = False
        results['issues'].append(f"Python {platform.python_version()} < 3.8 (required)")
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        results['warnings'].append("CUDA not available - will use CPU (very slow)")
    else:
        # Check GPU memory
        min_memory_required = 8  # GB
        if env_info['total_memory_gb'] < min_memory_required:
            results['issues'].append(f"Total GPU memory {env_info['total_memory_gb']:.1f}GB < {min_memory_required}GB required")
            results['valid'] = False
    
    # Check required packages
    required_packages = ['transformers', 'torch', 'datasets', 'wandb', 'numpy', 'pandas']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        results['valid'] = False
        results['issues'].append(f"Missing required packages: {', '.join(missing_packages)}")
    
    # Check workspace structure
    workspace_dir = Path(__file__).parent.parent
    required_dirs = ['experiments', 'shared', 'scripts', 'data']
    missing_dirs = []
    
    for dir_name in required_dirs:
        if not (workspace_dir / dir_name).exists():
            missing_dirs.append(dir_name)
    
    if missing_dirs:
        results['valid'] = False
        results['issues'].append(f"Missing required directories: {', '.join(missing_dirs)}")
    
    return results

def log_environment_info():
    """Log comprehensive environment information."""
    env_info = detect_compute_environment()
    
    logger.info("=" * 60)
    logger.info("ENVIRONMENT INFORMATION")
    logger.info("=" * 60)
    logger.info(f"Platform: {env_info['platform']}")
    logger.info(f"Python: {env_info['python_version']} ({env_info['python_executable']})")
    logger.info(f"CPU cores: {env_info['cpu_count']}")
    logger.info(f"System memory: {env_info['system_memory_gb']} GB")
    
    if env_info['cuda_available']:
        logger.info(f"CUDA: Available (version {env_info['cuda_version']})")
        logger.info(f"GPU devices: {env_info['cuda_devices']}")
        for gpu in env_info['gpu_memory']:
            logger.info(f"  GPU {gpu['device']}: {gpu['name']} ({gpu['memory_gb']} GB)")
        logger.info(f"Total GPU memory: {env_info['total_memory_gb']:.1f} GB")
    else:
        logger.info("CUDA: Not available")
    
    logger.info("=" * 60)

def setup_logging(level: str = "INFO", format_str: str = None):
    """Set up standardized logging configuration."""
    if format_str is None:
        format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Clear any existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # Set up new logging configuration
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_str,
        handlers=[
            logging.StreamHandler(sys.stdout),
        ]
    )
    
    # Suppress some noisy loggers
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('transformers.tokenization_utils_base').setLevel(logging.WARNING)

def initialize_environment(config_path: str = None, log_level: str = "INFO") -> Dict[str, Any]:
    """Initialize the complete environment for consistent script execution."""
    # Set up logging first
    setup_logging(log_level)
    
    logger.info("Initializing standardized environment...")
    
    # Load configuration
    if config_path is None:
        config_path = Path(__file__).parent / 'config.yaml'
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Could not load config from {config_path}: {e}")
        config = {}
    
    # Set up Python path
    workspace_dir = setup_python_path()
    
    # Set up environment variables
    env_vars = setup_environment_variables(config)
    
    # Validate environment
    validation_results = validate_environment_requirements(config)
    
    # Log environment info
    log_environment_info()
    
    # Check validation results
    if not validation_results['valid']:
        logger.error("❌ ENVIRONMENT VALIDATION FAILED")
        for issue in validation_results['issues']:
            logger.error(f"   - {issue}")
        raise RuntimeError("Environment validation failed")
    
    if validation_results['warnings']:
        for warning in validation_results['warnings']:
            logger.warning(f"⚠️ {warning}")
    
    logger.info("✅ Environment initialization completed successfully")
    
    return {
        'config': config,
        'workspace_dir': workspace_dir,
        'env_vars': env_vars,
        'validation': validation_results
    }

def main():
    """Run environment validation and setup."""
    try:
        env_setup = initialize_environment()
        
        print("\n" + "=" * 60)
        print("ENVIRONMENT SETUP SUMMARY")
        print("=" * 60)
        print("✅ Environment validation passed")
        print(f"✅ Workspace: {env_setup['workspace_dir']}")
        print(f"✅ Config loaded: {len(env_setup['config'])} sections")
        print(f"✅ Environment variables: {len(env_setup['env_vars'])} set")
        
        if env_setup['validation']['warnings']:
            print(f"⚠️ Warnings: {len(env_setup['validation']['warnings'])}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Environment setup failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
