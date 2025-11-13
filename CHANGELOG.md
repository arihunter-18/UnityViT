# Changelog

All notable changes to the SHViT project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.1.0] - 2025-11-13

### Added
- Adversarial robustness evaluation framework (GAP-1)
  - FGSM and PGD attack implementations
  - Comprehensive metrics logging and analysis
  - Support for multiple epsilon values
- Attention disruption visualization (GAP-4)
  - Token Activation Heatmaps (TAH)
  - Attention Stability Maps (ASM)
  - Grad-CAM visualization
  - Statistical analysis and plotting
- Visualization scripts
  - Sample visualization utilities
  - Publication-ready figure generation
  - Attention pattern analysis
- Infrastructure improvements
  - Enhanced GPU support and detection
  - Better error handling and logging
  - Progress tracking with tqdm
- Documentation enhancements
  - Comprehensive README with badges and examples
  - Contributing guidelines
  - Detailed code structure documentation

### Changed
- Updated requirements.txt for Python 3.11+ compatibility
- Improved PyTorch compatibility (2.0+)
- Enhanced model loading with better error messages
- Optimized data loading pipeline

### Fixed
- GPU training compatibility issues
- CUDA device selection and management
- Data path handling for Windows and Unix systems
- Model checkpoint loading edge cases

---

## [1.0.0] - 2024-06-01

### Added
- Initial release
- Four SHViT model variants (S1, S2, S3, S4)
- ImageNet-1K training and evaluation
- Pre-trained model weights
- Object detection with Mask R-CNN
- Instance segmentation support
- RetinaNet integration
- Model export utilities (ONNX, Core ML)
- Speed benchmarking tools
- Comprehensive training scripts
- Data augmentation strategies (ThreeAugment)
- Distributed training support
- Model EMA (Exponential Moving Average)
- Knowledge distillation framework
- Mixed precision training support

### Model Architectures
- SHViT-S1: 6.3M params, 72.8% top-1 accuracy
- SHViT-S2: 11.4M params, 75.2% top-1 accuracy
- SHViT-S3: 14.2M params, 77.4% top-1 accuracy
- SHViT-S4: 16.5M params, 79.4% top-1 accuracy

### Features
- Single-head attention mechanism
- Memory-efficient macro design
- Larger-stride patchify stem
- Hybrid convolutional-attention architecture
- FPN integration for dense prediction
- Mobile deployment support

---

## Future Plans

### [1.2.0] - Planned
- [ ] Quantization support (INT8, FP16)
- [ ] TensorRT optimization
- [ ] Additional robustness benchmarks
- [ ] AutoAttack integration
- [ ] Semantic segmentation support
- [ ] Video understanding tasks
- [ ] Multi-modal extensions

### Research Extensions
- [ ] Self-supervised pre-training
- [ ] Few-shot learning
- [ ] Domain adaptation
- [ ] Neural architecture search integration
- [ ] Continual learning support

---

## Versioning

- **Major version** (X.0.0): Breaking API changes
- **Minor version** (0.X.0): New features, backward compatible
- **Patch version** (0.0.X): Bug fixes, minor improvements

---

## Links
- [GitHub Repository](https://github.com/ysj9909/SHViT)
- [Paper](https://arxiv.org/abs/2401.16456)
- [Pre-trained Models](https://github.com/ysj9909/SHViT/releases)

---

**Note**: This changelog is maintained manually. For detailed commit history, please refer to the Git log.

