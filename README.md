# FlashTensor H2O Diagonal Caching Evaluation

This repository contains the implementation and evaluation of **H2O Diagonal Caching optimization** for FlashTensor. We compare our optimized kernels against standard baselines including PyTorch, TorchInductor, TensorRT, and the original FlashTensor implementation.

## Environment Setup

### Prerequisites
- NVIDIA GPU with CUDA support
- Python 3.10+
- CUDA 12.3+

### Quick Setup with Conda

```bash
# Create and activate conda environment
conda create -n ft python=3.10
conda activate ft

# Install dependencies from exported requirements
pip install -r requirements_pip.txt

# Or install key packages manually:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install triton
pip install click
```

### Manual Environment Setup

If you need to recreate the environment from scratch:

```bash
conda create -n ft python=3.10
conda activate ft

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install Triton for GPU kernels
pip install triton

# Install additional dependencies
pip install click numpy scipy pandas matplotlib seaborn
pip install tensorrt
pip install onnx onnxruntime

# Install FlashTensor dependencies
cd 3rd/asuka
pip install -e .
```

## Repository Structure

```
evaluation/
├── test.py                           # Main evaluation script
├── compile.py                        # System compilation logic
├── kernel/                           # Kernel implementations
│   ├── h2o.py                        # Original FlashTensor H2O kernels
│   ├── h2o_p1_simple_diagonal.py    # Our P1 kernel with diagonal caching
│   └── h2o_p4_simple_cache.py       # Our P4 kernel with cache usage
├── requirements_pip.txt              # Exported pip packages
└── README.md                         # This file
```

## Usage

The evaluation script `test.py` supports multiple systems and provides detailed performance analysis:

### Command Line Interface

```bash
python test.py --system=<SYSTEM> [--diagonal_k=<K>]
```

**Parameters:**
- `--system`: System to test (required)
  - `torch`: PyTorch baseline
  - `torchinductor`: TorchInductor optimization
  - `tensorrt`: TensorRT optimization
  - `flashtensor`: Original FlashTensor implementation
  - `ours`: Our diagonal caching optimization
- `--diagonal_k`: Diagonal cache parameter for `ours` system (1-31, default=1)

### Baseline Systems

Test standard optimization systems:

```bash
# PyTorch baseline (slowest, reference implementation)
python test.py --system=torch

# TorchInductor (fastest standard optimization)
python test.py --system=torchinductor

# TensorRT optimization
python test.py --system=tensorrt
```

### FlashTensor System

Test the original FlashTensor implementation with detailed kernel breakdown:

```bash
# FlashTensor with P1/P4 kernel latency reporting
python test.py --system=flashtensor
```


### Our Optimized System

Test our diagonal caching optimization with different cache sizes:

```bash
# Minimal caching (k=1, baseline)
python test.py --system=ours --diagonal_k=1

# Small cache (k=2)
python test.py --system=ours --diagonal_k=2

# Medium cache (k=4)
python test.py --system=ours --diagonal_k=4

# Large cache (k=8)
python test.py --system=ours --diagonal_k=8

# Maximum cache (k=16)
python test.py --system=ours --diagonal_k=16
```
