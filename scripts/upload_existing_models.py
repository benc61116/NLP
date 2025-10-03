#!/usr/bin/env python3
"""
Upload existing Phase 2 models to WandB as artifacts.
Run this BEFORE cleaning up local models.
"""

import os
import sys
import wandb
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def upload_model_to_wandb(model_path: Path, task: str, method: str, seed: int, project: str = "NLP-Phase2"):
    """Upload a single model to WandB."""
    
    # Initialize WandB run for upload
    run = wandb.init(
        project=project,
        entity="galavny-tel-aviv-university",
        job_type="model_upload",
        name=f"upload_{method}_{task}_seed{seed}",
        config={
            "task": task,
            "method": method,
            "seed": seed
        }
    )
    
    try:
        # Create artifact
        if method == "full_finetune":
            artifact_name = f"full_finetune_model_{task}_seed{seed}"
            description = f"Full fine-tuned model for {task} (seed {seed})"
        else:
            artifact_name = f"lora_adapter_{task}_seed{seed}"
            description = f"LoRA adapter for {task} (seed {seed})"
        
        logger.info(f"Creating artifact: {artifact_name}")
        
        artifact = wandb.Artifact(
            name=artifact_name,
            type="model",
            description=description,
            metadata={
                "task": task,
                "method": method,
                "seed": seed,
                "upload_script": "upload_existing_models.py"
            }
        )
        
        # Add model directory
        logger.info(f"  Adding directory: {model_path}")
        artifact.add_dir(str(model_path))
        
        # Upload
        logger.info(f"  Uploading to WandB...")
        wandb.log_artifact(artifact)
        
        logger.info(f"✅ Successfully uploaded: {artifact_name}")
        
        # Wait for upload to complete
        artifact.wait()
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to upload {model_path}: {e}")
        return False
    
    finally:
        wandb.finish()


def find_and_upload_all_models():
    """Find all Phase 2 models and upload to WandB."""
    
    results_dir = Path("results")
    
    logger.info("="*80)
    logger.info("UPLOADING EXISTING MODELS TO WANDB")
    logger.info("="*80)
    
    uploaded = []
    failed = []
    
    # Find full fine-tune models
    logger.info("\n📦 Finding Full Fine-tune Models...")
    for model_dir in results_dir.glob("full_finetune_*/*/final_model"):
        if not model_dir.exists():
            continue
        
        # Verify model file exists
        model_file = model_dir / "model.safetensors"
        if not model_file.exists():
            logger.warning(f"  Skipping {model_dir} - no model.safetensors")
            continue
        
        # Extract task and seed from path
        # Path format: results/full_finetune_20251003_034934/full_ft_rte_seed42/final_model
        parts = model_dir.parts
        task_seed_dir = parts[-2]  # e.g., "full_ft_rte_seed42"
        
        # Parse task and seed
        if "mrpc" in task_seed_dir:
            task = "mrpc"
        elif "sst2" in task_seed_dir:
            task = "sst2"
        elif "rte" in task_seed_dir:
            task = "rte"
        elif "squad" in task_seed_dir:
            task = "squad_v2"
        else:
            logger.warning(f"  Unknown task in {task_seed_dir}")
            continue
        
        # Extract seed
        if "seed42" in task_seed_dir or "seed_42" in task_seed_dir:
            seed = 42
        elif "seed1337" in task_seed_dir or "seed_1337" in task_seed_dir:
            seed = 1337
        elif "seed2024" in task_seed_dir or "seed_2024" in task_seed_dir:
            seed = 2024
        else:
            logger.warning(f"  Unknown seed in {task_seed_dir}")
            continue
        
        model_size = sum(f.stat().st_size for f in model_dir.rglob('*') if f.is_file()) / (1024**3)
        logger.info(f"\n📤 Uploading: {task}/full_finetune/seed{seed} ({model_size:.2f}GB)")
        
        success = upload_model_to_wandb(model_dir, task, "full_finetune", seed)
        if success:
            uploaded.append(f"{task}/full_finetune/seed{seed}")
        else:
            failed.append(f"{task}/full_finetune/seed{seed}")
    
    # Find LoRA adapters
    logger.info("\n📦 Finding LoRA Adapters...")
    for adapter_dir in results_dir.glob("lora_finetune_*/*/final_adapter"):
        if not adapter_dir.exists():
            continue
        
        # Verify adapter file exists
        adapter_file = adapter_dir / "adapter_model.safetensors"
        if not adapter_file.exists():
            logger.warning(f"  Skipping {adapter_dir} - no adapter_model.safetensors")
            continue
        
        # Extract task and seed from path
        # Path format: results/lora_finetune_20251003_050941/lora_mrpc_manual_seed42/final_adapter
        parts = adapter_dir.parts
        task_seed_dir = parts[-2]  # e.g., "lora_mrpc_manual_seed42"
        
        # Parse task
        if "mrpc" in task_seed_dir:
            task = "mrpc"
        elif "sst2" in task_seed_dir:
            task = "sst2"
        elif "rte" in task_seed_dir:
            task = "rte"
        elif "squad" in task_seed_dir:
            task = "squad_v2"
        else:
            logger.warning(f"  Unknown task in {task_seed_dir}")
            continue
        
        # Extract seed
        if "seed42" in task_seed_dir or "seed_42" in task_seed_dir:
            seed = 42
        elif "seed1337" in task_seed_dir or "seed_1337" in task_seed_dir:
            seed = 1337
        elif "seed2024" in task_seed_dir or "seed_2024" in task_seed_dir:
            seed = 2024
        else:
            logger.warning(f"  Unknown seed in {task_seed_dir}")
            continue
        
        adapter_size = sum(f.stat().st_size for f in adapter_dir.rglob('*') if f.is_file()) / (1024**2)
        logger.info(f"\n📤 Uploading: {task}/lora/seed{seed} ({adapter_size:.2f}MB)")
        
        success = upload_model_to_wandb(adapter_dir, task, "lora", seed)
        if success:
            uploaded.append(f"{task}/lora/seed{seed}")
        else:
            failed.append(f"{task}/lora/seed{seed}")
    
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
        logger.info("\n🎉 All models uploaded successfully to WandB!")
        logger.info("You can now safely clean up local models.")
        return True


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Upload existing models to WandB")
    parser.add_argument("--dry-run", action="store_true", help="List models without uploading")
    args = parser.parse_args()
    
    if args.dry_run:
        results_dir = Path("results")
        print("\nModels that will be uploaded:")
        print("\nFull Fine-tune:")
        for p in results_dir.glob("full_finetune_*/*/final_model/model.safetensors"):
            print(f"  - {p.parent}")
        print("\nLoRA Adapters:")
        for p in results_dir.glob("lora_finetune_*/*/final_adapter/adapter_model.safetensors"):
            print(f"  - {p.parent}")
    else:
        success = find_and_upload_all_models()
        sys.exit(0 if success else 1)

