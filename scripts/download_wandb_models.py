#!/usr/bin/env python3
"""
Download trained models from WandB artifacts for Phase 3 or recovery.

This script downloads model artifacts uploaded during Phase 2 training,
enabling you to:
1. Recover models after local cleanup
2. Download models for Phase 3 representation extraction
3. Share models across machines/VMs
"""

import os
import sys
import wandb
import argparse
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def download_model_artifact(
    entity: str,
    project: str,
    artifact_name: str,
    output_dir: str = "results/downloaded_models",
    alias: str = "latest"
):
    """Download a model artifact from WandB."""
    
    logger.info(f"Downloading artifact: {artifact_name}:{alias}")
    logger.info(f"  Entity: {entity}")
    logger.info(f"  Project: {project}")
    
    try:
        # Initialize WandB API
        api = wandb.Api()
        
        # Get the artifact
        artifact_path = f"{entity}/{project}/{artifact_name}:{alias}"
        artifact = api.artifact(artifact_path, type="model")
        
        # Download to output directory
        output_path = Path(output_dir) / artifact_name
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"  Downloading to: {output_path}")
        download_path = artifact.download(root=str(output_path))
        
        logger.info(f"✅ Successfully downloaded: {artifact_name}")
        logger.info(f"  Location: {download_path}")
        
        return download_path
        
    except Exception as e:
        logger.error(f"❌ Failed to download {artifact_name}: {e}")
        return None


def download_all_phase2_models(
    entity: str,
    project: str = "NLP-Phase2",
    output_dir: str = "results/downloaded_models"
):
    """Download all Phase 2 model artifacts."""
    
    logger.info("="*80)
    logger.info("DOWNLOADING ALL PHASE 2 MODELS FROM WANDB")
    logger.info("="*80)
    
    # Define all expected models
    tasks = ["mrpc", "sst2", "rte"]
    methods = ["full_finetune", "lora"]
    seeds = [42, 1337, 2024]
    
    total = len(tasks) * len(methods) * len(seeds)
    downloaded = 0
    failed = []
    
    for task in tasks:
        for method in methods:
            for seed in seeds:
                if method == "full_finetune":
                    artifact_name = f"full_finetune_model_{task}_seed{seed}"
                else:
                    artifact_name = f"lora_adapter_{task}_seed{seed}"
                
                logger.info(f"\n[{downloaded+1}/{total}] Downloading {task}/{method}/seed{seed}...")
                
                result = download_model_artifact(
                    entity=entity,
                    project=project,
                    artifact_name=artifact_name,
                    output_dir=output_dir
                )
                
                if result:
                    downloaded += 1
                else:
                    failed.append(f"{task}/{method}/seed{seed}")
    
    logger.info("\n" + "="*80)
    logger.info("DOWNLOAD SUMMARY")
    logger.info("="*80)
    logger.info(f"Total artifacts: {total}")
    logger.info(f"Successfully downloaded: {downloaded}")
    logger.info(f"Failed: {len(failed)}")
    
    if failed:
        logger.warning("\n❌ Failed downloads:")
        for item in failed:
            logger.warning(f"  - {item}")
    else:
        logger.info("\n✅ All models downloaded successfully!")
    
    return downloaded, failed


def list_available_artifacts(entity: str, project: str = "NLP-Phase2"):
    """List all model artifacts available in WandB project."""
    
    logger.info("Listing available model artifacts...")
    
    try:
        api = wandb.Api()
        
        # Use the correct API method for wandb 0.22.0+
        # Get all runs from the project and collect their artifacts
        runs = api.runs(f"{entity}/{project}")
        
        logger.info("\n" + "="*80)
        logger.info("AVAILABLE MODEL ARTIFACTS")
        logger.info("="*80)
        
        artifacts_found = set()
        count = 0
        
        for run in runs:
            # Get artifacts logged by this run
            for artifact in run.logged_artifacts():
                if artifact.type == "model":
                    # Avoid duplicates
                    artifact_id = f"{artifact.name}:{artifact.version}"
                    if artifact_id not in artifacts_found:
                        artifacts_found.add(artifact_id)
                        count += 1
                        logger.info(f"\n{count}. {artifact.name}:{artifact.version}")
                        logger.info(f"   Type: {artifact.type}")
                        logger.info(f"   Size: {artifact.size / (1024**3):.2f} GB" if artifact.size else "   Size: N/A")
                        logger.info(f"   Created: {artifact.created_at}")
                        if artifact.metadata:
                            task = artifact.metadata.get('task', 'N/A')
                            method = artifact.metadata.get('method', 'N/A')
                            seed = artifact.metadata.get('seed', 'N/A')
                            logger.info(f"   Task: {task}, Method: {method}, Seed: {seed}")
        
        if count == 0:
            logger.warning("No model artifacts found in WandB project")
        else:
            logger.info(f"\n✅ Found {count} model artifacts")
        
        return count
        
    except Exception as e:
        logger.error(f"❌ Failed to list artifacts: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 0


def main():
    parser = argparse.ArgumentParser(
        description="Download trained models from WandB artifacts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all available models
  python scripts/download_wandb_models.py --list --entity galavny-tel-aviv-university
  
  # Download all Phase 2 models
  python scripts/download_wandb_models.py --all --entity galavny-tel-aviv-university
  
  # Download specific model
  python scripts/download_wandb_models.py \\
      --entity galavny-tel-aviv-university \\
      --artifact lora_adapter_mrpc_seed42
        """
    )
    
    parser.add_argument("--entity", type=str, required=True,
                       help="WandB entity (username or team)")
    parser.add_argument("--project", type=str, default="NLP-Phase2",
                       help="WandB project name (default: NLP-Phase2)")
    parser.add_argument("--output-dir", type=str, default="results/downloaded_models",
                       help="Output directory for downloaded models")
    
    # Action options
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--list", action="store_true",
                      help="List all available model artifacts")
    group.add_argument("--all", action="store_true",
                      help="Download all Phase 2 models")
    group.add_argument("--artifact", type=str,
                      help="Download specific artifact by name")
    
    parser.add_argument("--alias", type=str, default="latest",
                       help="Artifact alias/version (default: latest)")
    
    args = parser.parse_args()
    
    # Execute requested action
    if args.list:
        list_available_artifacts(args.entity, args.project)
    
    elif args.all:
        downloaded, failed = download_all_phase2_models(
            args.entity, args.project, args.output_dir
        )
        
        if failed:
            sys.exit(1)  # Exit with error if any downloads failed
    
    elif args.artifact:
        result = download_model_artifact(
            args.entity, args.project, args.artifact, args.output_dir, args.alias
        )
        
        if not result:
            sys.exit(1)
    
    logger.info("\n✅ Done!")


if __name__ == "__main__":
    main()

