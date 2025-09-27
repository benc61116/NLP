#!/usr/bin/env python3
"""
Sweep-First Methodology Implementation
Implements proper academic-grade hyperparameter optimization workflow:
1. Run hyperparameter sweeps for all task/method combinations
2. Analyze sweep results to identify optimal hyperparameters  
3. Run production experiments using optimal hyperparameters with multiple seeds

This addresses the critical methodology issue identified in the plan.
"""

import os
import sys
import argparse
import yaml
import json
import subprocess
import time
from typing import Dict, List, Any, Optional
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SweepFirstWorkflow:
    """Implements the sweep-first methodology workflow"""
    
    def __init__(self, config_path: str = "shared/config.yaml", output_dir: str = "./results"):
        self.config_path = Path(config_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Load configuration
        with open(self.config_path) as f:
            self.config = yaml.safe_load(f)
        
        # Workflow state
        self.workflow_state = {
            'phase': 'not_started',
            'completed_sweeps': [],
            'optimal_configs': None,
            'completed_experiments': [],
            'start_time': None,
            'phase_times': {}
        }
        
        # Load existing state if available
        self.state_file = self.output_dir / 'sweep_first_workflow_state.json'
        self.load_state()
    
    def load_state(self):
        """Load workflow state from file"""
        if self.state_file.exists():
            try:
                with open(self.state_file) as f:
                    self.workflow_state = json.load(f)
                logger.info(f"Loaded workflow state: {self.workflow_state['phase']}")
            except Exception as e:
                logger.warning(f"Could not load workflow state: {e}")
    
    def save_state(self):
        """Save workflow state to file"""
        try:
            with open(self.state_file, 'w') as f:
                json.dump(self.workflow_state, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Could not save workflow state: {e}")
    
    def run_hyperparameter_sweeps(self, tasks: List[str], methods: List[str]) -> bool:
        """Phase 1: Run hyperparameter sweeps for all task/method combinations"""
        logger.info("🔬 PHASE 1: Running Hyperparameter Sweeps")
        logger.info("=" * 60)
        
        self.workflow_state['phase'] = 'running_sweeps'
        self.workflow_state['start_time'] = time.time()
        phase_start = time.time()
        
        total_sweeps = len(tasks) * len(methods)
        completed_sweeps = 0
        
        for task in tasks:
            for method in methods:
                sweep_id = f"{task}_{method}"
                
                if sweep_id in self.workflow_state['completed_sweeps']:
                    logger.info(f"✅ {sweep_id} sweep already completed, skipping")
                    completed_sweeps += 1
                    continue
                
                logger.info(f"🚀 [{completed_sweeps+1}/{total_sweeps}] Running {sweep_id} sweep...")
                
                # Run sweep
                success = self._run_single_sweep(task, method)
                
                if success:
                    self.workflow_state['completed_sweeps'].append(sweep_id)
                    completed_sweeps += 1
                    logger.info(f"✅ {sweep_id} sweep completed ({completed_sweeps}/{total_sweeps})")
                else:
                    logger.error(f"❌ {sweep_id} sweep failed!")
                    return False
                
                self.save_state()
        
        self.workflow_state['phase_times']['sweeps'] = time.time() - phase_start
        logger.info(f"🎉 All {total_sweeps} hyperparameter sweeps completed!")
        return True
    
    def _run_single_sweep(self, task: str, method: str) -> bool:
        """Run a single hyperparameter sweep"""
        script_name = "experiments/full_finetune.py" if method == "full_finetune" else "experiments/lora_finetune.py"
        log_file = self.output_dir / f"sweep_{task}_{method}.log"
        
        cmd = [
            "python", script_name,
            "--task", task,
            "--mode", "sweep",
            "--no-base-representations"  # Sweeps don't need representations
        ]
        
        try:
            with open(log_file, 'w') as f:
                result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, timeout=7200)  # 2 hour timeout
            
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            logger.error(f"Sweep {task}_{method} timed out after 2 hours")
            return False
        except Exception as e:
            logger.error(f"Error running sweep {task}_{method}: {e}")
            return False
    
    def analyze_sweep_results(self) -> bool:
        """Phase 2: Analyze sweep results and identify optimal hyperparameters"""
        logger.info("📊 PHASE 2: Analyzing Sweep Results")
        logger.info("=" * 60)
        
        self.workflow_state['phase'] = 'analyzing_sweeps'
        phase_start = time.time()
        
        # Run sweep analysis script
        cmd = [
            "python", "scripts/analyze_sweeps.py",
            "--export-optimal-configs",
            "--output-dir", str(self.output_dir)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Sweep analysis failed: {result.stderr}")
                return False
            
            # Load optimal configurations
            optimal_config_path = self.output_dir / 'optimal_hyperparameters.yaml'
            if not optimal_config_path.exists():
                logger.error("Optimal hyperparameters file not generated!")
                return False
            
            with open(optimal_config_path) as f:
                self.workflow_state['optimal_configs'] = yaml.safe_load(f)
            
            self.workflow_state['phase_times']['analysis'] = time.time() - phase_start
            logger.info("✅ Sweep analysis completed and optimal hyperparameters identified!")
            
            # Print summary
            self._print_optimal_summary()
            return True
            
        except Exception as e:
            logger.error(f"Error analyzing sweeps: {e}")
            return False
    
    def _print_optimal_summary(self):
        """Print summary of optimal hyperparameters"""
        logger.info("\n📋 OPTIMAL HYPERPARAMETERS IDENTIFIED:")
        logger.info("-" * 60)
        
        optimal_hp = self.workflow_state['optimal_configs']['optimal_hyperparameters']
        
        for task, methods in optimal_hp.items():
            logger.info(f"\n{task.upper()}:")
            for method, config in methods.items():
                hp = config['hyperparameters']
                perf = config['expected_performance']
                lr = hp['learning_rate']
                bs = hp['per_device_train_batch_size']
                wu = hp['warmup_ratio']
                acc = perf.get('eval_accuracy', perf.get('eval_f1', 0))
                
                logger.info(f"  {method:12}: LR={lr:8} BS={bs:2} WU={wu:.2f} → Perf={acc:.3f}")
    
    def run_production_experiments(self, tasks: List[str], methods: List[str], seeds: List[int]) -> bool:
        """Phase 3: Run production experiments using optimal hyperparameters"""
        logger.info("🎯 PHASE 3: Running Production Experiments with Optimal Hyperparameters")
        logger.info("=" * 60)
        
        self.workflow_state['phase'] = 'running_production'
        phase_start = time.time()
        
        if not self.workflow_state['optimal_configs']:
            logger.error("No optimal configurations available! Run analysis first.")
            return False
        
        optimal_hp = self.workflow_state['optimal_configs']['optimal_hyperparameters']
        
        total_experiments = len(tasks) * len(methods) * len(seeds)
        completed_experiments = 0
        
        for task in tasks:
            for method in methods:
                for seed in seeds:
                    exp_id = f"{task}_{method}_seed{seed}"
                    
                    if exp_id in self.workflow_state['completed_experiments']:
                        logger.info(f"✅ {exp_id} already completed, skipping")
                        completed_experiments += 1
                        continue
                    
                    # Get optimal hyperparameters for this task/method
                    if task not in optimal_hp or method not in optimal_hp[task]:
                        logger.warning(f"⚠️  No optimal config for {task}/{method}, skipping")
                        continue
                    
                    optimal_config = optimal_hp[task][method]['hyperparameters']
                    
                    logger.info(f"🚀 [{completed_experiments+1}/{total_experiments}] Running {exp_id} with optimal hyperparameters...")
                    
                    # Run production experiment
                    success = self._run_production_experiment(task, method, seed, optimal_config)
                    
                    if success:
                        self.workflow_state['completed_experiments'].append(exp_id)
                        completed_experiments += 1
                        logger.info(f"✅ {exp_id} completed ({completed_experiments}/{total_experiments})")
                    else:
                        logger.error(f"❌ {exp_id} failed!")
                        return False
                    
                    self.save_state()
        
        self.workflow_state['phase_times']['production'] = time.time() - phase_start
        logger.info(f"🎉 All {completed_experiments} production experiments completed!")
        return True
    
    def _run_production_experiment(self, task: str, method: str, seed: int, optimal_config: Dict) -> bool:
        """Run a single production experiment with optimal hyperparameters"""
        script_name = "experiments/full_finetune.py" if method == "full_finetune" else "experiments/lora_finetune.py"
        log_file = self.output_dir / f"production_{task}_{method}_seed{seed}.log"
        
        cmd = [
            "python", script_name,
            "--task", task,
            "--mode", "single",
            "--seed", str(seed),
            "--learning-rate", str(optimal_config['learning_rate']),
            "--batch-size", str(optimal_config['per_device_train_batch_size']),
            "--warmup-ratio", str(optimal_config['warmup_ratio']),
            "--epochs", str(optimal_config['num_train_epochs'])
        ]
        
        # Add LoRA-specific parameters
        if method == "lora" and 'lora_r' in optimal_config:
            cmd.extend([
                "--lora-r", str(optimal_config['lora_r']),
                "--lora-alpha", str(optimal_config['lora_alpha']),
                "--lora-dropout", str(optimal_config['lora_dropout'])
            ])
        
        try:
            with open(log_file, 'w') as f:
                result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, timeout=10800)  # 3 hour timeout
            
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            logger.error(f"Production experiment {task}_{method}_seed{seed} timed out after 3 hours")
            return False
        except Exception as e:
            logger.error(f"Error running production experiment {task}_{method}_seed{seed}: {e}")
            return False
    
    def run_complete_workflow(self, tasks: List[str], methods: List[str], seeds: List[int]) -> bool:
        """Run the complete sweep-first workflow"""
        logger.info("🚀 SWEEP-FIRST METHODOLOGY WORKFLOW")
        logger.info("=" * 60)
        logger.info("This implements proper academic-grade hyperparameter optimization:")
        logger.info("1. Hyperparameter sweeps for all task/method combinations")
        logger.info("2. Analysis to identify optimal hyperparameters") 
        logger.info("3. Production experiments using optimal hyperparameters")
        logger.info("=" * 60)
        
        # Phase 1: Hyperparameter Sweeps
        if self.workflow_state['phase'] in ['not_started', 'running_sweeps']:
            if not self.run_hyperparameter_sweeps(tasks, methods):
                return False
        
        # Phase 2: Sweep Analysis
        if self.workflow_state['phase'] in ['running_sweeps', 'analyzing_sweeps']:
            if not self.analyze_sweep_results():
                return False
        
        # Phase 3: Production Experiments
        if self.workflow_state['phase'] in ['analyzing_sweeps', 'running_production']:
            if not self.run_production_experiments(tasks, methods, seeds):
                return False
        
        # Workflow complete
        self.workflow_state['phase'] = 'completed'
        total_time = time.time() - self.workflow_state['start_time']
        self.workflow_state['total_time'] = total_time
        self.save_state()
        
        logger.info(f"\n🎉 SWEEP-FIRST WORKFLOW COMPLETED!")
        logger.info("=" * 60)
        logger.info(f"Total time: {total_time/3600:.1f} hours")
        logger.info(f"Sweeps: {self.workflow_state['phase_times'].get('sweeps', 0)/3600:.1f}h")
        logger.info(f"Analysis: {self.workflow_state['phase_times'].get('analysis', 0)/60:.1f}m") 
        logger.info(f"Production: {self.workflow_state['phase_times'].get('production', 0)/3600:.1f}h")
        logger.info("Results ready for drift analysis and statistical validation!")
        
        return True

def main():
    parser = argparse.ArgumentParser(description='Run sweep-first methodology workflow')
    parser.add_argument('--tasks', nargs='+', default=['mrpc', 'sst2', 'rte', 'squad_v2'],
                       help='Tasks to optimize')
    parser.add_argument('--methods', nargs='+', default=['full_finetune', 'lora'],
                       help='Methods to optimize')
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 1337, 2024],
                       help='Seeds for production experiments')
    parser.add_argument('--output-dir', default='./results/sweep_first_workflow',
                       help='Output directory for workflow results')
    parser.add_argument('--config', default='shared/config.yaml',
                       help='Configuration file path')
    parser.add_argument('--phase', choices=['sweeps', 'analysis', 'production', 'all'], default='all',
                       help='Run specific phase only')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from previous workflow state')
    
    args = parser.parse_args()
    
    # Initialize workflow
    workflow = SweepFirstWorkflow(args.config, args.output_dir)
    
    if not args.resume:
        # Reset workflow state for fresh start
        workflow.workflow_state = {
            'phase': 'not_started',
            'completed_sweeps': [],
            'optimal_configs': None,
            'completed_experiments': [],
            'start_time': None,
            'phase_times': {}
        }
    
    # Run specified phase(s)
    success = False
    
    if args.phase == 'sweeps':
        success = workflow.run_hyperparameter_sweeps(args.tasks, args.methods)
    elif args.phase == 'analysis':
        success = workflow.analyze_sweep_results()
    elif args.phase == 'production':
        success = workflow.run_production_experiments(args.tasks, args.methods, args.seeds)
    else:  # all
        success = workflow.run_complete_workflow(args.tasks, args.methods, args.seeds)
    
    if success:
        logger.info("✅ Workflow phase(s) completed successfully!")
        sys.exit(0)
    else:
        logger.error("❌ Workflow failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
