# SHViT: Single-Head Vision Transformer with Memory Efficient Macro Design

[![CVPR 2024](https://img.shields.io/badge/CVPR-2024-blue.svg)](https://arxiv.org/abs/2401.16456)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

Official PyTorch implementation of **"SHViT: Single-Head Vision Transformer with Memory Efficient Macro Design"** (CVPR 2024).

📄 [**Paper**](https://arxiv.org/abs/2401.16456) | 🚀 [**Pre-trained Models**](#pre-trained-models)

*Seokju Yun, Youngmin Ro*

![SHViT Performance](acc_vs_thro.png)

---

## 🌟 Highlights

- **State-of-the-art Speed-Accuracy Trade-off**: SHViT achieves superior performance compared to existing efficient vision transformers
- **Memory Efficient Design**: Single-head attention mechanism reduces computational redundancy while maintaining accuracy
- **Fast Inference**: 3.3×, 8.1×, and 2.4× faster than MobileViTv2 on GPU, CPU, and iPhone12, respectively
- **Multiple Variants**: Four model sizes (S1-S4) for different resource constraints
- **Versatile Applications**: Image classification, object detection, and instance segmentation

---

## 📋 Abstract

Recently, efficient Vision Transformers have shown great performance with low latency on resource-constrained devices. Conventionally, they use 4×4 patch embeddings and a 4-stage structure at the macro level, while utilizing sophisticated attention with multi-head configuration at the micro level. 

This paper aims to address computational redundancy at all design levels in a memory-efficient manner. We discover that using larger-stride patchify stem not only reduces memory access costs but also achieves competitive performance by leveraging token representations with reduced spatial redundancy from the early stages.

Furthermore, our preliminary analyses suggest that attention layers in the early stages can be substituted with convolutions, and several attention heads in the latter stages are computationally redundant. To handle this, we introduce a **single-head attention module** that inherently prevents head redundancy and simultaneously boosts accuracy by parallelly combining global and local information.

---

## 🏆 Pre-trained Models

| Model | Resolution | Top-1 Acc | #Params | FLOPs | Throughput | Download |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **SHViT-S1** | 224×224 | 72.8% | 6.3M | 241M | 33,489 | [model](https://github.com/ysj9909/SHViT/releases/download/v1.0/shvit_s1.pth) |
| **SHViT-S2** | 224×224 | 75.2% | 11.4M | 366M | 26,878 | [model](https://github.com/ysj9909/SHViT/releases/download/v1.0/shvit_s2.pth) |
| **SHViT-S3** | 224×224 | 77.4% | 14.2M | 601M | 20,522 | [model](https://github.com/ysj9909/SHViT/releases/download/v1.0/shvit_s3.pth) |
| **SHViT-S4** | 256×256 | 79.4% | 16.5M | 986M | 14,283 | [model](https://github.com/ysj9909/SHViT/releases/download/v1.0/shvit_s4.pth) |

*Throughput measured on NVIDIA V100 GPU (images/sec)*

---

## 🚀 Quick Start

### Installation

```bash
# Create conda environment
conda create -n shvit python=3.9
conda activate shvit

# Install PyTorch (adjust CUDA version as needed)
conda install pytorch==2.0.0 torchvision==0.15.0 cudatoolkit=11.8 -c pytorch

# Install dependencies
pip install -r requirements.txt
```

### Dataset Preparation

Download the [ImageNet-1K](http://image-net.org/) dataset and organize it as follows:

```
datasets/imagenet-1k/
├── train/
│   ├── n01440764/
│   │   ├── n01440764_10026.JPEG
│   │   └── ...
│   └── ...
└── val/
    ├── n01440764/
    │   ├── ILSVRC2012_val_00000293.JPEG
    │   └── ...
    └── ...
```

### Evaluation

Run the following command to evaluate a pre-trained model:

```bash
python main.py --eval --model shvit_s4 --resume ./shvit_s4.pth \
  --data-path datasets/imagenet-1k --input-size 256
```

### Training

Train SHViT models from scratch using 8 GPUs:

<details>
<summary>SHViT-S1</summary>

```bash
python -m torch.distributed.launch --nproc_per_node=8 --master_port 12345 --use_env \
  main.py --model shvit_s1 --data-path datasets/imagenet-1k --dist-eval --weight-decay 0.025
```
</details>

<details>
<summary>SHViT-S2</summary>

```bash
python -m torch.distributed.launch --nproc_per_node=8 --master_port 12345 --use_env \
  main.py --model shvit_s2 --data-path datasets/imagenet-1k --dist-eval --weight-decay 0.032
```
</details>

<details>
<summary>SHViT-S3</summary>

```bash
python -m torch.distributed.launch --nproc_per_node=8 --master_port 12345 --use_env \
  main.py --model shvit_s3 --data-path datasets/imagenet-1k --dist-eval --weight-decay 0.035
```
</details>

<details>
<summary>SHViT-S4</summary>

```bash
python -m torch.distributed.launch --nproc_per_node=8 --master_port 12345 --use_env \
  main.py --model shvit_s4 --data-path datasets/imagenet-1k --dist-eval --weight-decay 0.03 --input-size 256
```
</details>

---

## 🔧 Advanced Usage

### Latency Measurement

Compare throughput performance on GPU/CPU:

```bash
python speed_test.py
```

### Model Export

Export the model to Core ML format for mobile deployment:

```bash
python export_model.py --variant shvit_s4 --output-dir ./exported \
  --checkpoint ./shvit_s4.pth
```

The mobile latency reported in the paper for iPhone 12 uses the deployment tool from [XCode 14](https://developer.apple.com/videos/play/wwdc2022/10027/).

### Object Detection & Instance Segmentation

See the `downstream/` directory for:
- Mask R-CNN with FPN
- RetinaNet
- Training and evaluation scripts for MS COCO

```bash
cd downstream
# See downstream/README.md for detailed instructions
```

---

## 🧪 Adversarial Robustness Evaluation

This repository includes tools for evaluating adversarial robustness:

### GAP-1: Robustness Evaluation
```bash
cd robustness
python eval_attacks.py --model shvit_s4 --data-path ../datasets/imagenet-1k/val \
  --attack fgsm --epsilon 0.002
```

### GAP-4: Attention Disruption Visualization
```bash
cd robustness
python eval_gap4_complete.py --model shvit_s4 --samples 10 \
  --attack pgd --epsilon 0.004
```

### Visualization Tools
```bash
cd scripts
python visualize_samples.py --output-dir ../outputs/visualizations
python generate_publication_figures.py
```

---

## 📁 Project Structure

```
SHViT-Clean/
├── model/              # SHViT model architecture
│   ├── shvit.py       # Core model implementation
│   └── build.py       # Model factory
├── data/              # Data loading and augmentation
│   ├── datasets.py
│   ├── samplers.py
│   └── threeaugment.py
├── robustness/        # Adversarial robustness evaluation
│   ├── attacks.py
│   ├── eval_attacks.py
│   └── eval_gap4_complete.py
├── viz/               # Attention visualization tools
│   ├── attention.py
│   └── gradcam.py
├── downstream/        # Object detection & segmentation
│   ├── configs/
│   ├── train.py
│   └── test.py
├── scripts/           # Utility scripts
├── main.py           # Training & evaluation entry point
├── engine.py         # Training & validation loops
├── losses.py         # Loss functions
├── utils.py          # Utility functions
├── speed_test.py     # Throughput benchmarking
├── export_model.py   # Model export utilities
├── requirements.txt  # Python dependencies
└── LICENSE           # MIT License
```

---

## 📊 Results

### ImageNet-1K Classification

SHViT achieves state-of-the-art speed-accuracy trade-off on ImageNet-1K:

| Model | Top-1 Acc | GPU Latency | CPU Latency | Mobile Latency |
|-------|-----------|-------------|-------------|----------------|
| MobileViTv2 x1.0 | 78.1% | 3.3× slower | 8.1× slower | 2.4× slower |
| **SHViT-S4** | **79.4%** | **Baseline** | **Baseline** | **Baseline** |

### MS COCO Object Detection

Using Mask R-CNN head:

| Backbone | Box AP | Mask AP | GPU Latency | Mobile Latency |
|----------|--------|---------|-------------|----------------|
| FastViT-SA12 | 39.8 | 36.8 | 3.8× slower | 2.0× slower |
| **SHViT-S4** | **39.9** | **36.7** | **Baseline** | **Baseline** |

---

## 🤝 Citation

If our work or code helps your research, please cite our paper:

```bibtex
@inproceedings{yun2024shvit,
  title={SHViT: Single-Head Vision Transformer with Memory Efficient Macro Design},
  author={Yun, Seokju and Ro, Youngmin},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={5756--5767},
  year={2024}
}
```

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

The codebase builds upon the following excellent projects:
- [Swin Transformer](https://github.com/microsoft/swin-transformer) (MIT)
- [LeViT](https://github.com/facebookresearch/LeViT) (Apache 2.0)
- [pytorch-image-models](https://github.com/rwightman/pytorch-image-models) (Apache 2.0)
- [EfficientViT](https://github.com/microsoft/Cream/tree/main/EfficientViT) (MIT)
- [PyTorch](https://github.com/pytorch/pytorch) (BSD)

---

## 🙏 Acknowledgements

We sincerely appreciate the open-source contributions from the computer vision and deep learning community that made this work possible.

---

## 📧 Contact

For questions and discussions, please open an issue or contact:
- Seokju Yun: [ysj9909](https://github.com/ysj9909)

---

**Keywords**: Vision Transformer, Efficient Architecture, Single-Head Attention, Image Classification, Object Detection, Mobile Deployment, Adversarial Robustness

