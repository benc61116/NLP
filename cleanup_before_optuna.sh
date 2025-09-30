#!/bin/bash
# Safe cleanup script before running Optuna optimization
# This will free up ~50-60GB of disk space

echo "🧹 NLP Project Cleanup Before Optuna Run"
echo "========================================"

# Show current disk usage
echo "📊 Current disk usage:"
df -h / | tail -1
echo ""

# Show current NLP directory size
echo "📂 Current NLP directory size:"
du -sh /home/galavny13/workspace/NLP
echo ""

echo "🗑️  Starting cleanup..."

# 1. DELETE: Base model representations (48GB) - Not needed for Optuna
echo "1. Deleting base model representations (48GB)..."
if [ -d "results/base_model_representations" ]; then
    rm -rf results/base_model_representations
    echo "   ✅ Deleted base_model_representations"
else
    echo "   ⏭️  base_model_representations not found"
fi

# 2. DELETE: Failed experiment directories (keep only recent ones)
echo "2. Cleaning up old experiment directories..."
cd results/

# Delete full finetune experiments older than today (keep today's for reference)
for dir in full_finetune_20250929_*; do
    if [ -d "$dir" ]; then
        # Keep the most recent one, delete the rest
        if [[ "$dir" != *"224208"* ]]; then  # Keep the latest validation run
            echo "   🗑️  Deleting $dir"
            rm -rf "$dir"
        else
            echo "   📁 Keeping recent experiment: $dir"
        fi
    fi
done

# Delete LoRA experiments older than today (they're small but still cleanup)
for dir in lora_finetune_20250929_*; do
    if [ -d "$dir" ]; then
        if [[ "$dir" < "lora_finetune_20250929_220000" ]]; then  # Keep recent ones
            echo "   🗑️  Deleting $dir"
            rm -rf "$dir"
        else
            echo "   📁 Keeping recent LoRA experiment: $dir"
        fi
    fi
done

cd ..

# 3. CLEAN: Wandb cache and old runs
echo "3. Cleaning wandb cache..."
if [ -d "wandb" ]; then
    # Keep the latest run, delete old ones
    cd wandb/
    for dir in run-*; do
        if [ -d "$dir" ] && [[ "$dir" != "run-20250929_224208-yle3mroj" ]]; then
            echo "   🗑️  Deleting old wandb run: $dir"
            rm -rf "$dir"
        fi
    done
    cd ..
fi

# Clean wandb cache directory
if [ -d "wandb_cache" ]; then
    echo "   🗑️  Cleaning wandb_cache"
    rm -rf wandb_cache/*
fi

# 4. CLEAN: Temporary files and caches
echo "4. Cleaning temporary files..."
find . -name "*.pyc" -delete 2>/dev/null || true
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find . -name "*.tmp" -delete 2>/dev/null || true

# 5. CLEAN: Large log files (keep recent ones)
echo "5. Cleaning large log files..."
find logs/ -name "*.log" -size +100M -mtime +1 -delete 2>/dev/null || true

echo ""
echo "✅ Cleanup completed!"
echo ""

# Show final disk usage
echo "📊 Final disk usage:"
df -h / | tail -1
echo ""

echo "📂 Final NLP directory size:"
du -sh /home/galavny13/workspace/NLP
echo ""

echo "🚀 Ready for Optuna optimization!"
echo "💡 Estimated space freed: ~50-60GB"
