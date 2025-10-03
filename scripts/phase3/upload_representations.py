#!/usr/bin/env python3
"""
Upload Phase 3 representations to WandB as artifacts.
Use this if extraction script didn't auto-upload.
"""

import os
import sys
import wandb
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def upload_representations_to_wandb(task: str, method: str, seed: int, 
                                   representations_dir: Path,
                                   project: str = "NLP-Phase3-Representations"):
    """Upload representations for a specific task/method/seed to WandB."""
    
    # Initialize WandB run for upload
    run = wandb.init(
        project=project,
        entity="galavny-tel-aviv-university",
        job_type="representations_upload",
        name=f"upload_{method}_{task}_seed{seed}",
        config={
            "task": task,
            "method": method,
            "seed": seed,
            "phase": "phase3"
        }
    )
    
    try:
        artifact_name = f"representations_{method}_{task}_seed{seed}"
        
        logger.info(f"Creating artifact: {artifact_name}")
        
        artifact = wandb.Artifact(
            name=artifact_name,
            type="representations",
            description=f"Extracted representations for {task}/{method}/seed{seed}",
            metadata={
                "task": task,
                "method": method,
                "seed": seed,
                "phase": "phase3"
            }
        )
        
        # Add representation directory
        logger.info(f"  Adding directory: {representations_dir}")
        artifact.add_dir(str(representations_dir))
        
        # Upload
        logger.info(f"  Uploading to WandB...")
        wandb.log_artifact(artifact)
        
        logger.info(f"✅ Successfully uploaded: {artifact_name}")
        
        # Wait for upload to complete
        artifact.wait()
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to upload {representations_dir}: {e}")
        return False
    
    finally:
        wandb.finish()


def find_and_upload_all_representations(base_dir: Path = Path("results/phase3_representations")):
    """Find all Phase 3 representations and upload to WandB."""
    
    if not base_dir.exists():
        logger.error(f"❌ Representations directory not found: {base_dir}")
        return False
    
    logger.info("="*80)
    logger.info("UPLOADING PHASE 3 REPRESENTATIONS TO WANDB")
    logger.info("="*80)
    
    uploaded = []
    failed = []
    
    # Expected pattern: results/phase3_representations/{method}_{task}_seed{seed}/
    # Example: results/phase3_representations/full_finetune_mrpc_seed42/
    
    for repr_dir in base_dir.glob("*"):
        if not repr_dir.is_dir():
            continue
        
        # Parse directory name
        dir_name = repr_dir.name
        
        # Extract method
        if dir_name.startswith("full_finetune_"):
            method = "full_finetune"
            rest = dir_name.replace("full_finetune_", "")
        elif dir_name.startswith("lora_"):
            method = "lora"
            rest = dir_name.replace("lora_", "")
        else:
            logger.warning(f"  Skipping unknown directory: {dir_name}")
            continue
        
        # Extract task and seed
        # Pattern: {task}_seed{seed}
        parts = rest.split("_seed")
        if len(parts) != 2:
            logger.warning(f"  Skipping invalid directory: {dir_name}")
            continue
        
        task = parts[0]
        try:
            seed = int(parts[1])
        except ValueError:
            logger.warning(f"  Skipping invalid seed in: {dir_name}")
            continue
        
        # Check if directory has .pt files (actual representations)
        pt_files = list(repr_dir.glob("*.pt"))
        if not pt_files:
            logger.warning(f"  Skipping empty directory: {dir_name}")
            continue
        
        repr_size = sum(f.stat().st_size for f in repr_dir.rglob('*') if f.is_file()) / (1024**2)
        logger.info(f"\n📤 Uploading: {task}/{method}/seed{seed} ({repr_size:.2f}MB)")
        
        success = upload_representations_to_wandb(task, method, seed, repr_dir)
        if success:
            uploaded.append(f"{task}/{method}/seed{seed}")
        else:
            failed.append(f"{task}/{method}/seed{seed}")
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("UPLOAD SUMMARY")
    logger.info("="*80)
    logger.info(f"✅ Successfully uploaded: {len(uploaded)}")
    for item in uploaded:
        logger.info(f"  - {item}")
    
    if failed:
        logger.warning(f"\n❌ Failed uploads: {len(failed)}")
        for item in failed:
            logger.warning(f"  - {item}")
        return False
    else:
        logger.info("\n🎉 All representations uploaded successfully to WandB!")
        logger.info("You can now safely clean up local representations if needed.")
        return True


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Upload Phase 3 representations to WandB")
    parser.add_argument("--base-dir", type=str, default="results/phase3_representations", 
                       help="Base directory containing representations")
    parser.add_argument("--task", type=str, help="Specific task to upload (optional)")
    parser.add_argument("--method", type=str, choices=["full_finetune", "lora"], 
                       help="Specific method to upload (optional)")
    parser.add_argument("--seed", type=int, help="Specific seed to upload (optional)")
    parser.add_argument("--dry-run", action="store_true", help="List representations without uploading")
    
    args = parser.parse_args()
    
    base_dir = Path(args.base_dir)
    
    if args.dry_run:
        print("\nRepresentations that will be uploaded:")
        for repr_dir in base_dir.glob("*"):
            if repr_dir.is_dir() and list(repr_dir.glob("*.pt")):
                print(f"  - {repr_dir.name}")
    elif args.task and args.method and args.seed:
        # Upload specific representation
        repr_dir = base_dir / f"{args.method}_{args.task}_seed{args.seed}"
        if not repr_dir.exists():
            logger.error(f"❌ Directory not found: {repr_dir}")
            sys.exit(1)
        success = upload_representations_to_wandb(args.task, args.method, args.seed, repr_dir)
        sys.exit(0 if success else 1)
    else:
        # Upload all representations
        success = find_and_upload_all_representations(base_dir)
        sys.exit(0 if success else 1)

