#!/usr/bin/env python3
"""
Download Phase 3 representations from WandB artifacts.
Use this to restore representations if deleted locally.
"""

import wandb
import argparse
import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def download_representation(entity: str, project: str, task: str, method: str, 
                           seed: int, output_dir: Path):
    """Download a specific representation artifact from WandB."""
    
    artifact_name = f"representations_{method}_{task}_seed{seed}"
    
    api = wandb.Api()
    try:
        artifact = api.artifact(f"{entity}/{project}/{artifact_name}:latest")
        logger.info(f"Downloading artifact '{artifact_name}' to {output_dir}...")
        
        download_path = output_dir / f"{method}_{task}_seed{seed}"
        artifact.download(root=str(download_path))
        
        logger.info(f"✅ Successfully downloaded '{artifact_name}'")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to download artifact '{artifact_name}': {e}")
        return False


def list_available_representations(entity: str, project: str):
    """List all representation artifacts in a WandB project."""
    api = wandb.Api()
    try:
        # Get all artifacts of type "representations"
        artifacts = api.artifacts(
            type_name="representations",
            per_page=100
        )
        
        representation_artifacts = []
        for artifact in artifacts:
            # Filter by project
            if f"{entity}/{project}" in artifact.source_name:
                representation_artifacts.append(artifact)
        
        if representation_artifacts:
            logger.info(f"Found {len(representation_artifacts)} representation artifacts in {entity}/{project}:")
            for art in representation_artifacts:
                artifact_name = art.name.split(':')[0]  # Remove version tag
                size_mb = art.size / (1024**2)
                logger.info(f"  - {artifact_name} ({size_mb:.2f} MB)")
        else:
            logger.info(f"No representation artifacts found in {entity}/{project}.")
        
        return [art.name.split(':')[0] for art in representation_artifacts]
    except Exception as e:
        logger.error(f"❌ Failed to list artifacts: {e}")
        return []


def download_all_representations(entity: str, project: str, output_dir: Path,
                                 tasks=None, methods=None, seeds=None):
    """Download all or filtered representation artifacts."""
    
    if tasks is None:
        tasks = ["mrpc", "sst2", "rte", "squad_v2"]
    if methods is None:
        methods = ["full_finetune", "lora"]
    if seeds is None:
        seeds = [42, 1337, 2024]
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*80)
    logger.info("DOWNLOADING PHASE 3 REPRESENTATIONS FROM WANDB")
    logger.info("="*80)
    
    downloaded = []
    failed = []
    
    for task in tasks:
        for method in methods:
            for seed in seeds:
                logger.info(f"\n📥 Downloading: {task}/{method}/seed{seed}")
                success = download_representation(entity, project, task, method, seed, output_dir)
                
                if success:
                    downloaded.append(f"{task}/{method}/seed{seed}")
                else:
                    failed.append(f"{task}/{method}/seed{seed}")
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("DOWNLOAD SUMMARY")
    logger.info("="*80)
    logger.info(f"✅ Successfully downloaded: {len(downloaded)}")
    for item in downloaded:
        logger.info(f"  - {item}")
    
    if failed:
        logger.warning(f"\n⚠️  Failed downloads: {len(failed)}")
        for item in failed:
            logger.warning(f"  - {item}")
        logger.warning("These may not exist yet in WandB.")
    
    logger.info(f"\n📁 Representations saved to: {output_dir}")
    
    return len(downloaded) > 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Phase 3 representations from WandB")
    parser.add_argument("--entity", type=str, default="galavny-tel-aviv-university", 
                       help="WandB entity name")
    parser.add_argument("--project", type=str, default="NLP-Phase3-Representations", 
                       help="WandB project name")
    parser.add_argument("--output-dir", type=str, default="results/phase3_representations", 
                       help="Directory to save downloaded representations")
    parser.add_argument("--list", action="store_true", 
                       help="List all available representation artifacts")
    parser.add_argument("--task", type=str, choices=["mrpc", "sst2", "rte", "squad_v2"],
                       help="Download specific task only")
    parser.add_argument("--method", type=str, choices=["full_finetune", "lora"],
                       help="Download specific method only")
    parser.add_argument("--seed", type=int, choices=[42, 1337, 2024],
                       help="Download specific seed only")
    parser.add_argument("--all", action="store_true",
                       help="Download all available representations")
    
    args = parser.parse_args()
    
    output_path = Path(args.output_dir)
    
    if args.list:
        list_available_representations(args.entity, args.project)
    elif args.task and args.method and args.seed:
        # Download specific representation
        output_path.mkdir(parents=True, exist_ok=True)
        download_representation(args.entity, args.project, args.task, args.method, 
                              args.seed, output_path)
    elif args.all:
        # Download all representations
        download_all_representations(args.entity, args.project, output_path)
    else:
        # Download with filters
        tasks = [args.task] if args.task else None
        methods = [args.method] if args.method else None
        seeds = [args.seed] if args.seed else None
        download_all_representations(args.entity, args.project, output_path, 
                                    tasks, methods, seeds)

