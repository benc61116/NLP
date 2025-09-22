#!/usr/bin/env python3
"""Run baseline experiments with W&B logging enabled."""

import os
import sys
sys.path.append('.')

from experiments.baselines import BaselineExperiments

def run_baselines_with_wandb():
    """Run all baseline experiments with W&B logging enabled."""
    
    # Ensure W&B is enabled
    if 'WANDB_MODE' in os.environ:
        del os.environ['WANDB_MODE']
    
    print("🚀 Running ALL baseline experiments with W&B logging...")
    print("This will populate your W&B dashboard with all 16 experiments")
    print("Dashboard: https://wandb.ai/galavny-tel-aviv-university/NLP-Baselines")
    print("=" * 70)
    
    experiments = BaselineExperiments()
    
    # All tasks and baseline types
    tasks = ['mrpc', 'sst2', 'rte', 'squad_v2']
    
    for task in tasks:
        print(f"\n📊 Running baselines for {task.upper()}")
        print("-" * 40)
        
        try:
            # 1. Majority class baseline
            print(f"1/4: Majority class baseline for {task}")
            experiments.majority_class_baseline(task)
            print(f"✅ Completed majority class for {task}")
            
            # 2. Random baseline (reduced seeds for speed)
            print(f"2/4: Random baseline for {task}")
            experiments.random_baseline(task, num_seeds=2)
            print(f"✅ Completed random for {task}")
            
            # 3. Zero-shot baseline (may be slow)
            print(f"3/4: Zero-shot baseline for {task}")
            experiments.zero_shot_llama_baseline(task, num_prompt_templates=1)
            print(f"✅ Completed zero-shot for {task}")
            
            # 4. SOTA baseline
            print(f"4/4: SOTA baseline for {task}")
            experiments.sota_baseline(task)
            print(f"✅ Completed SOTA for {task}")
            
            print(f"🎯 All baselines completed for {task}")
            
        except Exception as e:
            print(f"❌ Error with {task}: {e}")
            continue
    
    print("\n" + "=" * 70)
    print("🏁 ALL BASELINE EXPERIMENTS COMPLETED")
    print("🔗 View results: https://wandb.ai/galavny-tel-aviv-university/NLP-Baselines")
    print("=" * 70)

if __name__ == "__main__":
    run_baselines_with_wandb()
