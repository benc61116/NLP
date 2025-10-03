#!/usr/bin/env python3
"""
Download LoRA representations from WandB to local disk for Phase 4 analysis.
"""

import wandb
import logging
from pathlib import Path
import shutil

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_lora_representations():
    """Download all LoRA representation artifacts from WandB."""
    
    api = wandb.Api()
    project = 'galavny-tel-aviv-university/NLP-Phase3-Representations'
    output_base = Path("results/phase3_representations/representations")
    output_base.mkdir(parents=True, exist_ok=True)
    
    # Define what we need to download
    tasks = ["mrpc", "sst2", "rte"]
    seeds = [42, 1337, 2024]
    
    logger.info("=" * 80)
    logger.info("Downloading LoRA representations from WandB")
    logger.info("=" * 80)
    
    downloaded = 0
    skipped = 0
    failed = 0
    
    for task in tasks:
        for seed in seeds:
            artifact_name = f"representations_lora_{task}_seed{seed}"
            output_dir = output_base / f"lora_seed{seed}_{task}"
            
            # Check if already downloaded
            pt_files = list(output_dir.glob("**/*.pt")) if output_dir.exists() else []
            if len(pt_files) >= 22:
                logger.info(f"⏭️  Skipping {artifact_name} - already downloaded ({len(pt_files)} files)")
                skipped += 1
                continue
            
            try:
                logger.info(f"📥 Downloading {artifact_name}...")
                
                # Download artifact
                artifact = api.artifact(f"{project}/{artifact_name}:latest", type="representations")
                artifact_dir = artifact.download()
                
                logger.info(f"   Downloaded to: {artifact_dir}")
                
                # The artifact contains a nested directory structure
                # We need to find where the actual .pt files are
                artifact_path = Path(artifact_dir)
                
                # Find the step_000000 directory (or wherever .pt files are)
                pt_files = list(artifact_path.rglob("*.pt"))
                if not pt_files:
                    logger.warning(f"   ⚠️  No .pt files found in artifact!")
                    failed += 1
                    continue
                
                # Find the directory containing the .pt files
                source_dir = pt_files[0].parent
                logger.info(f"   Found {len(pt_files)} .pt files in {source_dir}")
                
                # Copy to correct location
                if output_dir.exists():
                    shutil.rmtree(output_dir)
                
                # Copy the entire step_000000 directory structure
                if source_dir.name == "step_000000":
                    output_dir.mkdir(parents=True)
                    shutil.copytree(source_dir, output_dir / "step_000000")
                else:
                    # Copy parent directory if it contains step_000000
                    parent_source = source_dir.parent
                    if (parent_source / "step_000000").exists():
                        shutil.copytree(parent_source, output_dir)
                    else:
                        # Fallback: create step_000000 and copy files
                        output_step_dir = output_dir / "step_000000"
                        output_step_dir.mkdir(parents=True)
                        for pt_file in pt_files:
                            shutil.copy2(pt_file, output_step_dir / pt_file.name)
                
                # Verify
                final_files = list(output_dir.glob("**/*.pt"))
                logger.info(f"   ✅ Copied {len(final_files)} files to {output_dir}")
                downloaded += 1
                
            except Exception as e:
                logger.error(f"   ❌ Failed to download {artifact_name}: {e}")
                failed += 1
    
    logger.info("=" * 80)
    logger.info("Download Summary:")
    logger.info(f"  ✅ Downloaded: {downloaded}")
    logger.info(f"  ⏭️  Skipped: {skipped}")
    logger.info(f"  ❌ Failed: {failed}")
    logger.info(f"  📊 Total: {downloaded + skipped + failed}/9 LoRA representations")
    logger.info("=" * 80)
    
    if downloaded + skipped == 9:
        logger.info("🎉 All LoRA representations are now available locally!")
        return True
    else:
        logger.warning("⚠️  Some LoRA representations are missing!")
        return False


if __name__ == "__main__":
    success = download_lora_representations()
    exit(0 if success else 1)

