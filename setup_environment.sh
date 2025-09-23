#!/bin/bash
# Environment Setup Script for NLP Research Project
# Ensures complete compatibility between all VMs

set -e  # Exit on error

echo "🔧 Setting up PyTorch-only environment for NLP research..."
echo "=================================================="

# Remove any existing TensorFlow installations that cause conflicts
echo "🧹 Cleaning up potential TensorFlow conflicts..."
pip uninstall tensorflow tensorflow-gpu tf-keras keras -y 2>/dev/null || true

# Remove torch_xla that causes compatibility issues
echo "🧹 Removing torch_xla if present..."
pip uninstall torch_xla -y 2>/dev/null || true

# Install exact requirements
echo "📦 Installing exact package versions..."
pip install -r requirements.txt --force-reinstall

echo ""
echo "✅ Environment setup complete!"
echo "🔍 Verifying installation..."

# Verify critical packages
python -c "
import torch
print(f'✅ PyTorch: {torch.__version__}')

import transformers
print(f'✅ Transformers: {transformers.__version__}')

import peft
print(f'✅ PEFT: {peft.__version__}')

# Verify no TensorFlow
try:
    import tensorflow
    print('❌ WARNING: TensorFlow is still installed - this may cause conflicts!')
except ImportError:
    print('✅ TensorFlow correctly not installed')

print('\\n🎉 Environment verified and ready for training!')
"

echo ""
echo "🚀 Ready to run Phase 1 experiments!"
echo "Run: chmod +x scripts/phase1/vm*.sh && ./scripts/phase1/vm1.sh"
