# Setup Guide

Complete setup guide for getting started with SHViT.

## Table of Contents
1. [System Requirements](#system-requirements)
2. [Environment Setup](#environment-setup)
3. [Dataset Preparation](#dataset-preparation)
4. [Downloading Pre-trained Models](#downloading-pre-trained-models)
5. [Verification](#verification)
6. [Common Issues](#common-issues)

---

## System Requirements

### Minimum Requirements
- **OS**: Linux, macOS, or Windows 10+
- **Python**: 3.9 or higher
- **RAM**: 8 GB (16 GB recommended)
- **Storage**: 200 GB (for ImageNet-1K)
- **GPU**: Optional for inference, required for training
  - Minimum: NVIDIA GPU with 6 GB VRAM
  - Recommended: NVIDIA GPU with 16+ GB VRAM

### Recommended Setup
- **GPU**: NVIDIA RTX 3090/4090 or A100
- **RAM**: 32 GB+
- **CPU**: 16+ cores for efficient data loading
- **Storage**: SSD for faster data loading

---

## Environment Setup

### Option 1: Conda (Recommended)

```bash
# Create a new conda environment
conda create -n shvit python=3.9 -y
conda activate shvit

# Install PyTorch (choose appropriate CUDA version)
# For CUDA 11.8
conda install pytorch==2.0.0 torchvision==0.15.0 pytorch-cuda=11.8 -c pytorch -c nvidia

# For CUDA 12.1
conda install pytorch==2.0.0 torchvision==0.15.0 pytorch-cuda=12.1 -c pytorch -c nvidia

# For CPU only
conda install pytorch==2.0.0 torchvision==0.15.0 cpuonly -c pytorch

# Install other dependencies
pip install -r requirements.txt
```

### Option 2: Python Virtual Environment

```bash
# Create virtual environment
python -m venv shvit_env

# Activate (Linux/macOS)
source shvit_env/bin/activate

# Activate (Windows)
shvit_env\Scripts\activate

# Install dependencies
pip install torch==2.0.0 torchvision==0.15.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### Option 3: Docker (Coming Soon)

```bash
# Pull Docker image
docker pull shvit/shvit:latest

# Run container
docker run --gpus all -it -v $(pwd):/workspace shvit/shvit:latest
```

---

## Dataset Preparation

### ImageNet-1K

#### Method 1: Official Download

1. **Register** at [ImageNet website](http://image-net.org/)

2. **Download** ImageNet-1K (ILSVRC2012)
   - Training images: `ILSVRC2012_img_train.tar` (~138 GB)
   - Validation images: `ILSVRC2012_img_val.tar` (~6.3 GB)

3. **Extract and organize**:

```bash
# Create directory structure
mkdir -p datasets/imagenet-1k/{train,val}

# Extract training data
tar -xf ILSVRC2012_img_train.tar -C datasets/imagenet-1k/train/
cd datasets/imagenet-1k/train/

# Each tar file is a class, extract all
for f in *.tar; do
  class_name="${f%.tar}"
  mkdir -p "$class_name"
  tar -xf "$f" -C "$class_name"
  rm "$f"
done

cd ../../..

# Extract validation data
tar -xf ILSVRC2012_img_val.tar -C datasets/imagenet-1k/val/

# Organize validation by class (requires helper script)
# Download validation ground truth and organize accordingly
```

#### Method 2: Kaggle

```bash
# Install Kaggle API
pip install kaggle

# Configure Kaggle credentials (get from kaggle.com/account)
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Download dataset
kaggle competitions download -c imagenet-object-localization-challenge
unzip imagenet-object-localization-challenge.zip -d datasets/
```

#### Validation-Only Setup (Quick Start)

For quick testing, download only the validation set (~7 GB):

```bash
# Download validation set
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar

# Extract
mkdir -p datasets/imagenet-1k/val
tar -xf ILSVRC2012_img_val.tar -C datasets/imagenet-1k/val/
```

### MS COCO (For Object Detection)

```bash
# Create directory
mkdir -p datasets/coco

# Download (choose one)
# Option 1: wget
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip

# Option 2: Direct from website
# Visit https://cocodataset.org/#download

# Extract
unzip train2017.zip -d datasets/coco/
unzip val2017.zip -d datasets/coco/
unzip annotations_trainval2017.zip -d datasets/coco/
```

---

## Downloading Pre-trained Models

### Automatic Download (Recommended)

Models will be downloaded automatically when running evaluation:

```bash
python main.py --eval --model shvit_s4 \
  --resume https://github.com/ysj9909/SHViT/releases/download/v1.0/shvit_s4.pth \
  --data-path datasets/imagenet-1k --input-size 256
```

### Manual Download

```bash
# Create checkpoints directory
mkdir -p checkpoints

# Download model weights
cd checkpoints

# SHViT-S1
wget https://github.com/ysj9909/SHViT/releases/download/v1.0/shvit_s1.pth

# SHViT-S2
wget https://github.com/ysj9909/SHViT/releases/download/v1.0/shvit_s2.pth

# SHViT-S3
wget https://github.com/ysj9909/SHViT/releases/download/v1.0/shvit_s3.pth

# SHViT-S4
wget https://github.com/ysj9909/SHViT/releases/download/v1.0/shvit_s4.pth

cd ..
```

---

## Verification

### 1. Check PyTorch Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"
```

Expected output:
```
PyTorch: 2.0.0+cu118
CUDA available: True
CUDA version: 11.8
```

### 2. Check Dependencies

```bash
python -c "import timm, einops, matplotlib; print('All dependencies OK')"
```

### 3. Test Model Loading

```bash
python -c "from model.shvit import shvit_s4; model = shvit_s4(); print(f'Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M')"
```

Expected output:
```
Model parameters: 16.59M
```

### 4. Quick Inference Test

```bash
python speed_test.py
```

---

## Common Issues

### Issue 1: CUDA Out of Memory

**Solution**: Reduce batch size or use gradient accumulation

```bash
# Reduce batch size
python main.py --batch-size 128  # instead of 256

# Enable gradient accumulation
python main.py --batch-size 64 --gradient-accumulation-steps 4
```

### Issue 2: Dataset Not Found

**Error**: `FileNotFoundError: [Errno 2] No such file or directory: 'datasets/imagenet-1k/val'`

**Solution**: Verify dataset path
```bash
ls -l datasets/imagenet-1k/val/
# Should show 1000 class directories
```

### Issue 3: ImportError for timm

**Error**: `ImportError: cannot import name 'create_model' from 'timm'`

**Solution**: Update timm
```bash
pip install --upgrade timm>=0.9.0
```

### Issue 4: Slow Data Loading

**Solution**: Increase num_workers
```bash
python main.py --num_workers 16  # Adjust based on CPU cores
```

### Issue 5: Multi-GPU Training Issues

**Solution**: Use torch.distributed.launch
```bash
python -m torch.distributed.launch --nproc_per_node=4 \
  main.py --model shvit_s4 --data-path datasets/imagenet-1k
```

### Issue 6: Windows Path Issues

**Solution**: Use raw strings or forward slashes
```python
# Instead of: "datasets\imagenet-1k\val"
# Use: "datasets/imagenet-1k/val"
```

---

## Next Steps

After completing setup:

1. **Quick Evaluation**: Test pre-trained model
   ```bash
   python main.py --eval --model shvit_s4 \
     --resume checkpoints/shvit_s4.pth \
     --data-path datasets/imagenet-1k --input-size 256
   ```

2. **Speed Benchmark**: Measure throughput
   ```bash
   python speed_test.py
   ```

3. **Training**: Start training from scratch
   ```bash
   # See README.md for training commands
   ```

4. **Robustness Evaluation**: Test adversarial robustness
   ```bash
   cd robustness
   python eval_attacks.py --model shvit_s4 --attack fgsm
   ```

---

## Getting Help

- **Documentation**: See [README.md](README.md) for detailed usage
- **Issues**: Report bugs at [GitHub Issues](https://github.com/ysj9909/SHViT/issues)
- **Discussions**: Ask questions in [GitHub Discussions](https://github.com/ysj9909/SHViT/discussions)

---

**Setup complete! You're ready to use SHViT.** 🚀

