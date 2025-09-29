#!/usr/bin/env python3
"""
Download base model representations from WandB artifacts for Phase 2/3.
This complements upload_base_representations.py for cloud storage workflow.
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

def download_base_representations(tasks=None, version="latest"):
    """Download base representations from WandB artifacts."""
    
    # Load config
    config = load_config()
    
    # Initialize WandB
    wandb.init(
        project="NLP-Phase0",
        name=f"base_representations_download_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        tags=["base_representations", "artifacts", "download"],
        notes="Downloading base model representations from WandB artifacts"
    )
    
    # Target directory
    target_dir = Path('results/base_model_representations/representations')
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Get available tasks if not specified
    if tasks is None:
        # Default to all 4 tasks
        tasks = ['mrpc', 'sst2', 'rte', 'squad_v2']
    
    print(f"📥 Downloading base representations for tasks: {tasks}")
    print(f"📁 Target directory: {target_dir}")
    
    successful_downloads = 0
    total_size = 0
    
    for task in tasks:
        artifact_name = f"base_representations_base_pretrained_{task}"
        
        try:
            print(f"\n📦 Downloading {task}...")
            
            # Download artifact
            artifact = wandb.use_artifact(f"{artifact_name}:{version}")
            artifact_dir = artifact.download(root=str(target_dir))
            
            # Get size info
            if artifact.size:
                size_gb = artifact.size / (1024**3)
                total_size += artifact.size
                print(f"   ✅ Downloaded {task} ({size_gb:.2f} GB)")
            else:
                print(f"   ✅ Downloaded {task}")
                
            successful_downloads += 1
            
        except Exception as e:
            print(f"   ❌ Failed to download {task}: {e}")
            print(f"      Available artifacts: run with --list to see all available")
    
    # Summary
    total_gb = total_size / (1024**3) if total_size > 0 else 0
    print(f"\n🎯 DOWNLOAD SUMMARY:")
    print(f"   Successfully downloaded: {successful_downloads}/{len(tasks)} tasks")
    if total_size > 0:
        print(f"   Total size downloaded: {total_gb:.2f} GB")
    print(f"   Downloaded to: {target_dir}")
    
    # Verify directory structure
    print(f"\n🔍 Verifying downloaded structure:")
    for task in tasks:
        task_dir = target_dir / f"base_pretrained_{task}"
        if task_dir.exists():
            files = list(task_dir.rglob('*.pt'))
            print(f"   ✅ {task}: {len(files)} .pt files")
        else:
            print(f"   ❌ {task}: directory not found")
    
    # Log summary metrics
    wandb.log({
        "total_size_gb": total_gb,
        "tasks_downloaded": successful_downloads,
        "download_success_rate": successful_downloads / len(tasks) if tasks else 0
    })
    
    wandb.finish()
    
    if successful_downloads == len(tasks):
        print(f"\n🎉 ALL BASE REPRESENTATIONS DOWNLOADED!")
        print(f"✅ Ready for Phase 2/3 analysis")
        return True
    else:
        print(f"\n⚠️  Some downloads failed. Partial results available.")
        return False

def list_available_artifacts():
    """List available base representation artifacts in WandB."""
    
    # Initialize WandB API
    api = wandb.Api()
    
    try:
        # Get all artifacts of type base_representations
        artifacts = api.artifacts(type_name="base_representations", project="galavny-tel-aviv-university/NLP-Phase0")
        
        print("📦 Available Base Representation Artifacts:")
        print("=" * 60)
        
        artifacts_list = list(artifacts)
        if not artifacts_list:
            print("❌ No base representation artifacts found.")
            print("   Run scripts/upload_base_representations.py first")
            return
        
        for artifact in artifacts_list:
            size_gb = artifact.size / (1024**3) if artifact.size else 0
            print(f"🔹 {artifact.name} (v{artifact.version})")
            print(f"   Size: {size_gb:.2f} GB")
            print(f"   Created: {artifact.created_at}")
            if artifact.metadata:
                task = artifact.metadata.get('task', 'unknown')
                model = artifact.metadata.get('model', 'unknown')
                print(f"   Task: {task}")
                print(f"   Model: {model}")
            print()
            
        # Show download command examples
        print("💡 Download Examples:")
        print("   # Download all tasks:")
        print("   python scripts/download_base_representations.py")
        print("   # Download specific tasks:")
        print("   python scripts/download_base_representations.py --tasks mrpc sst2")
        print("   # Download specific version:")
        print("   python scripts/download_base_representations.py --version v1")
            
    except Exception as e:
        print(f"❌ Failed to list artifacts: {e}")
        print("   Make sure you're logged into WandB: wandb login")

def check_local_representations():
    """Check if base representations exist locally."""
    
    base_dir = Path('results/base_model_representations/representations')
    
    if not base_dir.exists():
        print("❌ No local base representations found")
        return False
    
    tasks = ['mrpc', 'sst2', 'rte', 'squad_v2']
    found_tasks = []
    
    for task in tasks:
        task_dir = base_dir / f"base_pretrained_{task}"
        if task_dir.exists():
            files = list(task_dir.rglob('*.pt'))
            found_tasks.append((task, len(files)))
    
    if found_tasks:
        print("✅ Local base representations found:")
        for task, count in found_tasks:
            print(f"   {task}: {count} .pt files")
        return True
    else:
        print("❌ No valid base representations found locally")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download base representations from WandB")
    parser.add_argument("--tasks", nargs="+", choices=['mrpc', 'sst2', 'rte', 'squad_v2'], 
                        help="Specific tasks to download (default: all)")
    parser.add_argument("--version", default="latest", help="Artifact version to download (default: latest)")
    parser.add_argument("--list", action="store_true", help="List available artifacts")
    parser.add_argument("--check", action="store_true", help="Check local representations")
    
    args = parser.parse_args()
    
    if args.list:
        list_available_artifacts()
    elif args.check:
        check_local_representations()
    else:
        success = download_base_representations(args.tasks, args.version)
        sys.exit(0 if success else 1)
