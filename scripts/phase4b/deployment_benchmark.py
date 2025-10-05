#!/usr/bin/env python3
"""
Phase 4B: Deployment Efficiency Benchmark
Research Question 2: What is the deployment latency penalty for multi-adapter setups vs merged models?

This script benchmarks:
1. Single LoRA adapter + base model
2. Multi-adapter (2 & 3 adapters) setups
3. Full fine-tuned (merged) models

Metrics measured:
- Inference latency (mean, p50, p95, p99)
- Throughput (requests/second)
- Memory usage (GPU & CPU)
- Model loading time
"""

import os
import sys
import json
import time
import psutil
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

# vLLM imports (will check if available for generation, fallback for classification)
try:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    print("⚠️  vLLM not available - using transformers for benchmarking")

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline
)
from peft import PeftModel
from datasets import load_dataset

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    config_name: str
    config_type: str  # "single_lora", "multi_lora_2", "multi_lora_3", "full_ft"
    num_adapters: int
    task: str
    num_samples: int
    
    # Latency metrics (milliseconds)
    mean_latency_ms: float
    median_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    std_latency_ms: float
    
    # Throughput
    throughput_req_per_sec: float
    
    # Memory metrics (MB)
    peak_gpu_memory_mb: float
    peak_cpu_memory_mb: float
    
    # Model loading time
    model_load_time_sec: float
    
    # Metadata
    timestamp: str
    gpu_name: str
    

class DeploymentBenchmark:
    """Benchmark deployment configurations for LoRA vs Full Fine-Tuning."""
    
    def __init__(
        self,
        base_model_name: str = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
        models_dir: str = "results/downloaded_models",
        output_dir: str = "results/phase4b_deployment",
        num_samples: int = 500,
        warmup_samples: int = 10,
        batch_size: int = 1,  # For classification tasks
    ):
        self.base_model_name = base_model_name
        self.models_dir = Path(models_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.num_samples = num_samples
        self.warmup_samples = warmup_samples
        self.batch_size = batch_size
        
        # Get GPU info
        if torch.cuda.is_available():
            self.gpu_name = torch.cuda.get_device_name(0)
            self.device = "cuda"
        else:
            self.gpu_name = "CPU"
            self.device = "cpu"
            
        logger.info(f"Initialized benchmark on {self.gpu_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
    def get_test_samples(self, task: str, num_samples: int) -> List[Tuple[str, int]]:
        """Load test samples for a given task."""
        logger.info(f"Loading {num_samples} test samples for {task}")
        
        dataset_map = {
            "mrpc": ("glue", "mrpc", "sentence1", "sentence2"),
            "sst2": ("glue", "sst2", "sentence", None),
            "rte": ("glue", "rte", "sentence1", "sentence2"),
        }
        
        if task not in dataset_map:
            raise ValueError(f"Unknown task: {task}")
            
        dataset_name, config, text1_key, text2_key = dataset_map[task]
        dataset = load_dataset(dataset_name, config, split="validation")
        
        # Sample and prepare inputs
        samples = []
        for i, example in enumerate(dataset):
            if i >= num_samples:
                break
                
            if text2_key:
                text = f"{example[text1_key]} [SEP] {example[text2_key]}"
            else:
                text = example[text1_key]
                
            label = example["label"]
            samples.append((text, label))
            
        return samples
        
    def measure_memory(self) -> Tuple[float, float]:
        """Measure current GPU and CPU memory usage in MB."""
        gpu_mem_mb = 0.0
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            gpu_mem_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
            
        cpu_mem_mb = psutil.Process().memory_info().rss / 1024 / 1024
        return gpu_mem_mb, cpu_mem_mb
        
    def benchmark_lora_single(
        self,
        task: str,
        seed: int,
        test_samples: List[Tuple[str, int]]
    ) -> BenchmarkResult:
        """Benchmark single LoRA adapter inference."""
        config_name = f"lora_{task}_seed{seed}"
        logger.info(f"Benchmarking single LoRA: {config_name}")
        
        # Reset memory tracking
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            
        # Load model
        start_load = time.time()
        
        # Determine num_labels for the task
        num_labels = 2  # All our tasks are binary classification
        
        base_model = AutoModelForSequenceClassification.from_pretrained(
            self.base_model_name,
            num_labels=num_labels,
            torch_dtype=torch.bfloat16,
            device_map=self.device
        )
        
        adapter_path = self.models_dir / f"lora_adapter_{task}_seed{seed}"
        model = PeftModel.from_pretrained(base_model, str(adapter_path))
        model.eval()
        
        load_time = time.time() - start_load
        
        # Warmup
        logger.info(f"Warmup with {self.warmup_samples} samples...")
        for text, _ in test_samples[:self.warmup_samples]:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=384)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                _ = model(**inputs)
                
        # Benchmark
        logger.info(f"Running benchmark on {len(test_samples)} samples...")
        latencies = []
        
        for text, _ in test_samples:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=384)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            start = time.time()
            with torch.no_grad():
                _ = model(**inputs)
            latency_ms = (time.time() - start) * 1000
            latencies.append(latency_ms)
            
        # Measure memory
        gpu_mem, cpu_mem = self.measure_memory()
        
        # Clean up
        del model, base_model
        torch.cuda.empty_cache()
        
        return self._create_result(
            config_name=config_name,
            config_type="single_lora",
            num_adapters=1,
            task=task,
            latencies=latencies,
            load_time=load_time,
            gpu_mem=gpu_mem,
            cpu_mem=cpu_mem
        )
        
    def benchmark_full_finetune(
        self,
        task: str,
        seed: int,
        test_samples: List[Tuple[str, int]]
    ) -> BenchmarkResult:
        """Benchmark full fine-tuned model inference."""
        config_name = f"fullft_{task}_seed{seed}"
        logger.info(f"Benchmarking full fine-tuned model: {config_name}")
        
        # Reset memory tracking
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            
        # Load model
        start_load = time.time()
        
        model_path = self.models_dir / f"full_finetune_model_{task}_seed{seed}"
        model = AutoModelForSequenceClassification.from_pretrained(
            str(model_path),
            torch_dtype=torch.bfloat16,
            device_map=self.device
        )
        model.eval()
        
        load_time = time.time() - start_load
        
        # Warmup
        logger.info(f"Warmup with {self.warmup_samples} samples...")
        for text, _ in test_samples[:self.warmup_samples]:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=384)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                _ = model(**inputs)
                
        # Benchmark
        logger.info(f"Running benchmark on {len(test_samples)} samples...")
        latencies = []
        
        for text, _ in test_samples:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=384)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            start = time.time()
            with torch.no_grad():
                _ = model(**inputs)
            latency_ms = (time.time() - start) * 1000
            latencies.append(latency_ms)
            
        # Measure memory
        gpu_mem, cpu_mem = self.measure_memory()
        
        # Clean up
        del model
        torch.cuda.empty_cache()
        
        return self._create_result(
            config_name=config_name,
            config_type="full_ft",
            num_adapters=0,
            task=task,
            latencies=latencies,
            load_time=load_time,
            gpu_mem=gpu_mem,
            cpu_mem=cpu_mem
        )
        
    def benchmark_lora_merged(
        self,
        task: str,
        seed: int,
        test_samples: List[Tuple[str, int]]
    ) -> BenchmarkResult:
        """Benchmark merged LoRA adapter inference (adapter weights merged into base model)."""
        config_name = f"lora_merged_{task}_seed{seed}"
        logger.info(f"Benchmarking merged LoRA: {config_name}")
        
        # Reset memory tracking
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            
        # Load model
        start_load = time.time()
        
        # Determine num_labels for the task
        num_labels = 2  # All our tasks are binary classification
        
        base_model = AutoModelForSequenceClassification.from_pretrained(
            self.base_model_name,
            num_labels=num_labels,
            torch_dtype=torch.bfloat16,
            device_map=self.device
        )
        
        adapter_path = self.models_dir / f"lora_adapter_{task}_seed{seed}"
        peft_model = PeftModel.from_pretrained(base_model, str(adapter_path))
        
        # CRITICAL: Merge adapter weights into base model
        # This eliminates the runtime adapter computation overhead
        model = peft_model.merge_and_unload()
        model.eval()
        
        load_time = time.time() - start_load
        
        # Warmup
        logger.info(f"Warmup with {self.warmup_samples} samples...")
        for text, _ in test_samples[:self.warmup_samples]:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=384)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                _ = model(**inputs)
                
        # Benchmark
        logger.info(f"Running benchmark on {len(test_samples)} samples...")
        latencies = []
        
        for text, _ in test_samples:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=384)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            start = time.time()
            with torch.no_grad():
                _ = model(**inputs)
            latency_ms = (time.time() - start) * 1000
            latencies.append(latency_ms)
            
        # Measure memory
        gpu_mem, cpu_mem = self.measure_memory()
        
        # Clean up
        del model
        torch.cuda.empty_cache()
        
        return self._create_result(
            config_name=config_name,
            config_type="lora_merged",
            num_adapters=0,  # Merged, so no separate adapter
            task=task,
            latencies=latencies,
            load_time=load_time,
            gpu_mem=gpu_mem,
            cpu_mem=cpu_mem
        )
        
    def benchmark_multi_adapter(
        self,
        tasks: List[str],
        seeds: List[int],
        test_samples_dict: Dict[str, List[Tuple[str, int]]]
    ) -> BenchmarkResult:
        """
        Benchmark multi-adapter setup.
        
        In a real multi-adapter deployment, one base model serves multiple adapters.
        We simulate this by:
        1. Loading base model once
        2. Swapping adapters for each task
        3. Measuring the overhead of adapter switching
        """
        num_adapters = len(tasks)
        config_name = f"multi_lora_{num_adapters}_adapters"
        logger.info(f"Benchmarking multi-adapter setup: {config_name} ({tasks})")
        
        # Reset memory tracking
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            
        # Load base model once
        start_load = time.time()
        
        base_model = AutoModelForSequenceClassification.from_pretrained(
            self.base_model_name,
            num_labels=2,
            torch_dtype=torch.bfloat16,
            device_map=self.device
        )
        
        # Load all adapters
        adapters = {}
        for task, seed in zip(tasks, seeds):
            adapter_path = self.models_dir / f"lora_adapter_{task}_seed{seed}"
            adapters[task] = adapter_path
            
        load_time = time.time() - start_load
        
        # Benchmark with adapter switching
        latencies = []
        all_samples = []
        
        # Interleave samples from different tasks to simulate multi-tenant serving
        for task in tasks:
            samples = test_samples_dict[task][:self.num_samples // num_adapters]
            for text, label in samples:
                all_samples.append((task, text, label))
                
        logger.info(f"Running benchmark on {len(all_samples)} samples with adapter switching...")
        
        current_task = None
        current_model = None
        
        for task, text, _ in all_samples:
            # Switch adapter if needed (this is the key overhead we're measuring)
            if task != current_task:
                if current_model is not None:
                    del current_model
                    torch.cuda.empty_cache()
                    
                # Load adapter for this task
                switch_start = time.time()
                current_model = PeftModel.from_pretrained(
                    base_model,
                    str(adapters[task])
                )
                current_model.eval()
                switch_time = (time.time() - switch_start) * 1000
                current_task = task
                logger.debug(f"Adapter switch to {task}: {switch_time:.2f}ms")
                
            # Run inference
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=384)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            start = time.time()
            with torch.no_grad():
                _ = current_model(**inputs)
            latency_ms = (time.time() - start) * 1000
            latencies.append(latency_ms)
            
        # Measure memory
        gpu_mem, cpu_mem = self.measure_memory()
        
        # Clean up
        del current_model, base_model
        torch.cuda.empty_cache()
        
        return self._create_result(
            config_name=config_name,
            config_type=f"multi_lora_{num_adapters}",
            num_adapters=num_adapters,
            task="+".join(tasks),
            latencies=latencies,
            load_time=load_time,
            gpu_mem=gpu_mem,
            cpu_mem=cpu_mem
        )
        
    def _create_result(
        self,
        config_name: str,
        config_type: str,
        num_adapters: int,
        task: str,
        latencies: List[float],
        load_time: float,
        gpu_mem: float,
        cpu_mem: float
    ) -> BenchmarkResult:
        """Create a BenchmarkResult from measurements."""
        latencies_arr = np.array(latencies)
        
        return BenchmarkResult(
            config_name=config_name,
            config_type=config_type,
            num_adapters=num_adapters,
            task=task,
            num_samples=len(latencies),
            mean_latency_ms=float(np.mean(latencies_arr)),
            median_latency_ms=float(np.median(latencies_arr)),
            p95_latency_ms=float(np.percentile(latencies_arr, 95)),
            p99_latency_ms=float(np.percentile(latencies_arr, 99)),
            std_latency_ms=float(np.std(latencies_arr)),
            throughput_req_per_sec=1000.0 / np.mean(latencies_arr),
            peak_gpu_memory_mb=gpu_mem,
            peak_cpu_memory_mb=cpu_mem,
            model_load_time_sec=load_time,
            timestamp=datetime.now().isoformat(),
            gpu_name=self.gpu_name
        )
        
    def run_full_benchmark(self) -> List[BenchmarkResult]:
        """Run complete benchmark suite."""
        logger.info("=" * 80)
        logger.info("PHASE 4B: DEPLOYMENT EFFICIENCY BENCHMARK")
        logger.info("=" * 80)
        
        results = []
        tasks = ["mrpc", "sst2", "rte"]
        seeds = [42, 1337, 2024]
        
        # Load all test samples once
        test_samples_dict = {}
        for task in tasks:
            test_samples_dict[task] = self.get_test_samples(task, self.num_samples)
            
        # 1. Benchmark single LoRA adapters (all 9)
        logger.info("\n" + "=" * 80)
        logger.info("1. SINGLE LORA ADAPTER BENCHMARKS (9 models)")
        logger.info("=" * 80)
        
        for task in tasks:
            for seed in seeds:
                result = self.benchmark_lora_single(
                    task=task,
                    seed=seed,
                    test_samples=test_samples_dict[task]
                )
                results.append(result)
                logger.info(f"✓ {result.config_name}: {result.mean_latency_ms:.2f}ms mean latency")
                
        # 2. Benchmark full fine-tuned models (all 9)
        logger.info("\n" + "=" * 80)
        logger.info("2. FULL FINE-TUNED MODEL BENCHMARKS (9 models)")
        logger.info("=" * 80)
        
        for task in tasks:
            for seed in seeds:
                result = self.benchmark_full_finetune(
                    task=task,
                    seed=seed,
                    test_samples=test_samples_dict[task]
                )
                results.append(result)
                logger.info(f"✓ {result.config_name}: {result.mean_latency_ms:.2f}ms mean latency")
                
        # 3. Benchmark merged LoRA adapters (all 9)
        logger.info("\n" + "=" * 80)
        logger.info("3. MERGED LORA ADAPTER BENCHMARKS (9 models)")
        logger.info("=" * 80)
        logger.info("Testing: LoRA adapters merged into base model (W = W_base + B×A)")
        
        for task in tasks:
            for seed in seeds:
                result = self.benchmark_lora_merged(
                    task=task,
                    seed=seed,
                    test_samples=test_samples_dict[task]
                )
                results.append(result)
                logger.info(f"✓ {result.config_name}: {result.mean_latency_ms:.2f}ms mean latency")
                
        # 4. Benchmark multi-adapter (2 adapters)
        logger.info("\n" + "=" * 80)
        logger.info("4. MULTI-ADAPTER BENCHMARK (2 adapters)")
        logger.info("=" * 80)
        
        # Use first 2 tasks, first seed
        multi_2_tasks = tasks[:2]
        multi_2_seeds = [seeds[0]] * 2
        result = self.benchmark_multi_adapter(
            tasks=multi_2_tasks,
            seeds=multi_2_seeds,
            test_samples_dict=test_samples_dict
        )
        results.append(result)
        logger.info(f"✓ {result.config_name}: {result.mean_latency_ms:.2f}ms mean latency")
        
        # 5. Benchmark multi-adapter (3 adapters)
        logger.info("\n" + "=" * 80)
        logger.info("5. MULTI-ADAPTER BENCHMARK (3 adapters)")
        logger.info("=" * 80)
        
        # Use all 3 tasks, first seed
        multi_3_tasks = tasks
        multi_3_seeds = [seeds[0]] * 3
        result = self.benchmark_multi_adapter(
            tasks=multi_3_tasks,
            seeds=multi_3_seeds,
            test_samples_dict=test_samples_dict
        )
        results.append(result)
        logger.info(f"✓ {result.config_name}: {result.mean_latency_ms:.2f}ms mean latency")
        
        return results
        
    def save_results(self, results: List[BenchmarkResult]):
        """Save benchmark results to JSON."""
        output_file = self.output_dir / "deployment_benchmark_results.json"
        
        results_dict = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "gpu_name": self.gpu_name,
                "base_model": self.base_model_name,
                "num_samples": self.num_samples,
                "warmup_samples": self.warmup_samples,
            },
            "results": [asdict(r) for r in results]
        }
        
        with open(output_file, 'w') as f:
            json.dump(results_dict, f, indent=2)
            
        logger.info(f"\n✓ Results saved to {output_file}")
        
        # Also save summary CSV
        self.save_summary_csv(results)
        
    def save_summary_csv(self, results: List[BenchmarkResult]):
        """Save summary results as CSV."""
        import pandas as pd
        
        df = pd.DataFrame([asdict(r) for r in results])
        output_file = self.output_dir / "deployment_benchmark_summary.csv"
        df.to_csv(output_file, index=False)
        
        logger.info(f"✓ Summary saved to {output_file}")
        
    def print_summary(self, results: List[BenchmarkResult]):
        """Print benchmark summary."""
        logger.info("\n" + "=" * 80)
        logger.info("BENCHMARK SUMMARY")
        logger.info("=" * 80)
        
        # Group by config type
        single_lora = [r for r in results if r.config_type == "single_lora"]
        full_ft = [r for r in results if r.config_type == "full_ft"]
        lora_merged = [r for r in results if r.config_type == "lora_merged"]
        multi_2 = [r for r in results if r.config_type == "multi_lora_2"]
        multi_3 = [r for r in results if r.config_type == "multi_lora_3"]
        
        def print_group_stats(name: str, group: List[BenchmarkResult]):
            if not group:
                return
            mean_latencies = [r.mean_latency_ms for r in group]
            mean_throughputs = [r.throughput_req_per_sec for r in group]
            mean_gpu_mem = [r.peak_gpu_memory_mb for r in group]
            
            logger.info(f"\n{name} ({len(group)} configs):")
            logger.info(f"  Latency: {np.mean(mean_latencies):.2f} ± {np.std(mean_latencies):.2f} ms")
            logger.info(f"  Throughput: {np.mean(mean_throughputs):.2f} ± {np.std(mean_throughputs):.2f} req/s")
            logger.info(f"  GPU Memory: {np.mean(mean_gpu_mem):.0f} ± {np.std(mean_gpu_mem):.0f} MB")
            
        print_group_stats("Single LoRA Adapter (Separate)", single_lora)
        print_group_stats("Full Fine-Tuned Model", full_ft)
        print_group_stats("Merged LoRA Adapter", lora_merged)
        print_group_stats("Multi-Adapter (2)", multi_2)
        print_group_stats("Multi-Adapter (3)", multi_3)
        
        logger.info("\n" + "=" * 80)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Phase 4B Deployment Benchmark")
    parser.add_argument("--models-dir", type=str, default="results/downloaded_models",
                      help="Directory containing models")
    parser.add_argument("--output-dir", type=str, default="results/phase4b_deployment",
                      help="Output directory for results")
    parser.add_argument("--num-samples", type=int, default=500,
                      help="Number of samples to benchmark")
    parser.add_argument("--warmup-samples", type=int, default=10,
                      help="Number of warmup samples")
    
    args = parser.parse_args()
    
    # Run benchmark
    benchmark = DeploymentBenchmark(
        models_dir=args.models_dir,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        warmup_samples=args.warmup_samples
    )
    
    results = benchmark.run_full_benchmark()
    benchmark.save_results(results)
    benchmark.print_summary(results)
    
    logger.info("\n✅ Phase 4B deployment benchmark complete!")
    

if __name__ == "__main__":
    main()
