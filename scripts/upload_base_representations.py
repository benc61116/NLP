#!/usr/bin/env python3
"""
Upload base model representations to WandB as artifacts for cloud storage.
This allows safe deletion of local 48GB and re-download for Phase 2/3.
"""

import os
import sys
import yaml
import wandb
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

def load_config():
    """Load the shared configuration."""
    config_path = Path(__file__).parent.parent / 'shared' / 'config.yaml'
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def upload_base_representations():
    """Upload base representations as WandB artifacts."""
    
    # Load config
    config = load_config()
    
    # Initialize WandB
    wandb.init(
        project="NLP-Phase0",
        name=f"base_representations_upload_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        tags=["base_representations", "artifacts", "phase0"],
        notes="Uploading base model representations as artifacts for cloud storage"
    )
    
    # Base directory
    base_dir = Path('results/base_model_representations/representations')
    
    if not base_dir.exists():
        print(f"❌ Base representations directory not found: {base_dir}")
        return False
    
    print(f"🔍 Found base representations directory: {base_dir}")
    
    # Get all task directories
    task_dirs = [d for d in base_dir.iterdir() if d.is_dir()]
    print(f"📁 Found {len(task_dirs)} task directories:")
    
    total_size = 0
    successful_uploads = 0
    
    for task_dir in task_dirs:
        print(f"\n📦 Processing {task_dir.name}...")
        
        # Calculate directory size
        dir_size = sum(f.stat().st_size for f in task_dir.rglob('*') if f.is_file())
        total_size += dir_size
        size_mb = dir_size / (1024 * 1024)
        size_gb = size_mb / 1024
        
        print(f"   Size: {size_gb:.2f} GB ({size_mb:.1f} MB)")
        
        try:
            # Create artifact for this task
            artifact = wandb.Artifact(
                name=f"base_representations_{task_dir.name}",
                type="base_representations",
                description=f"Base model representations for {task_dir.name} task",
                metadata={
                    "task": task_dir.name.replace("base_pretrained_", ""),
                    "model": config['model']['name'],
                    "extraction_date": datetime.now().isoformat(),
                    "size_mb": size_mb,
                    "phase": "phase0"
                }
            )
            
            # Add the entire task directory
            print(f"   📤 Uploading {task_dir.name} to WandB...")
            artifact.add_dir(str(task_dir), name=task_dir.name)
            
            # Log the artifact
            wandb.log_artifact(artifact)
            
            print(f"   ✅ Successfully uploaded {task_dir.name}")
            successful_uploads += 1
            
        except Exception as e:
            print(f"   ❌ Failed to upload {task_dir.name}: {e}")
    
    # Summary
    total_gb = total_size / (1024 * 1024 * 1024)
    print(f"\n🎯 UPLOAD SUMMARY:")
    print(f"   Successfully uploaded: {successful_uploads}/{len(task_dirs)} tasks")
    print(f"   Total size uploaded: {total_gb:.2f} GB")
    print(f"   Artifacts created: {successful_uploads}")
    
    # Log summary metrics
    wandb.log({
        "total_size_gb": total_gb,
        "tasks_uploaded": successful_uploads,
        "upload_success_rate": successful_uploads / len(task_dirs) if task_dirs else 0
    })
    
    wandb.finish()
    
    if successful_uploads == len(task_dirs):
        print(f"\n🎉 ALL BASE REPRESENTATIONS UPLOADED TO WANDB!")
        print(f"✅ You can now safely delete the local 48GB:")
        print(f"   rm -rf results/base_model_representations/")
        print(f"📥 To download for Phase 2/3, use: scripts/download_base_representations.py")
        return True
    else:
        print(f"\n⚠️  Some uploads failed. Check WandB for partial results.")
        return False

def list_available_artifacts():
    """List available base representation artifacts in WandB."""
    
    # Initialize WandB in read-only mode
    api = wandb.Api()
    
    try:
        # Get all artifacts of type base_representations
        artifacts = api.artifacts(type_name="base_representations", project="galavny-tel-aviv-university/NLP-Phase0")
        
        print("📦 Available Base Representation Artifacts:")
        print("=" * 50)
        
        for artifact in artifacts:
            print(f"🔹 {artifact.name} (v{artifact.version})")
            print(f"   Size: {artifact.size / (1024**3):.2f} GB")
            print(f"   Created: {artifact.created_at}")
            if artifact.metadata:
                task = artifact.metadata.get('task', 'unknown')
                print(f"   Task: {task}")
            print()
            
    except Exception as e:
        print(f"❌ Failed to list artifacts: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload base representations to WandB")
    parser.add_argument("--list", action="store_true", help="List available artifacts instead of uploading")
    
    args = parser.parse_args()
    
    if args.list:
        list_available_artifacts()
    else:
        success = upload_base_representations()
        sys.exit(0 if success else 1)
