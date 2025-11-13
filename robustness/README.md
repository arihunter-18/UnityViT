## Adversarial Robustness Evaluation for SHViT

This module provides GPU-accelerated tools for evaluating adversarial robustness of Single-Head Vision Transformers, specifically addressing **Gap 1** (single-head vulnerability) and **Gap 4** (visualization).

---

## 🚀 Features

### Attack Implementations
- **FGSM** (Fast Gradient Sign Method) with attention guidance
- **PGD** (Projected Gradient Descent) with time budgeting
- **GPU-accelerated** for high-speed evaluation
- **Attention-guided attacks** that exploit single-head architecture

### Visualization Toolkit
- **Token Activation Heatmaps (TAH)**: Shows which patches the single head attends to
- **Attention Stability Maps (ASM)**: Compares attention between clean and adversarial inputs
- **Grad-CAM Integration**: Gradient-weighted Class Activation Mapping for SHViT

---

## 📁 File Structure

```
robustness/
├── __init__.py           # Module exports
├── attacks.py            # FGSM, PGD, evaluation framework
├── eval_attacks.py       # Main evaluation script for Gap 1
└── README.md             # This file

viz/
├── __init__.py
├── attention.py          # TAH and ASM visualization
└── gradcam.py            # Grad-CAM for SHViT

scripts/
└── visualize_samples.py  # Generate visualization panels

model/
└── shvit.py             # Modified with attention caching
```

---

## 🔧 Installation

All dependencies are already in `requirements.txt`:

```bash
pip install -r requirements.txt
```

**Required packages:**
- PyTorch >= 2.0.0
- torchvision >= 0.15.0
- timm >= 0.9.0
- matplotlib >= 3.6.0
- numpy >= 1.24.0
- pillow >= 9.0.0

---

## 📊 Usage

### 1. Gap 1: Adversarial Robustness Evaluation

Compare SHViT vs ViT-B under identical attacks:

```bash
cd robustness

# Quick test on subset
python eval_attacks.py \
    --shvit-model shvit_s4 \
    --vit-model vit_base_patch16_224 \
    --data-path ../datasets/imagenet-1k-subset \
    --batch-size 32 \
    --subset-size 1000 \
    --attack both \
    --epsilon 0.004 0.008 0.016 \
    --pgd-steps 10 \
    --device cuda \
    --output-dir ../outputs/gap1_results

# Full evaluation
python eval_attacks.py \
    --shvit-model shvit_s4 \
    --shvit-weights ../checkpoints/shvit_s4.pth \
    --vit-model vit_base_patch16_224 \
    --data-path ../datasets/imagenet-1k \
    --batch-size 64 \
    --attack both \
    --epsilon 0.001 0.002 0.004 0.008 0.016 \
    --pgd-steps 10 \
    --device cuda \
    --output-dir ../outputs/gap1_results
```

**Output:**
- `results.json`: Detailed metrics for all configurations
- `summary.txt`: Human-readable summary with vulnerability gap analysis

**Key Metrics:**
- **Clean Accuracy**: Accuracy on unperturbed images
- **Robust Accuracy**: Accuracy under adversarial attack
- **Attack Success Rate (ASR)**: Percentage of successful attacks
- **Vulnerability Gap**: Difference in ASR between SHViT and ViT

---

### 2. Gap 4: Visualization Toolkit

Generate comprehensive visualizations:

```bash
cd scripts

# Basic visualization (20 samples)
python visualize_samples.py \
    --model shvit_s4 \
    --data-path ../datasets/imagenet-1k-subset \
    --num-samples 20 \
    --epsilon 0.016 \
    --attack pgd \
    --device cuda \
    --output-dir ../outputs/viz

# Comprehensive visualization (all blocks + Grad-CAM)
python visualize_samples.py \
    --model shvit_s4 \
    --weights ../checkpoints/shvit_s4.pth \
    --data-path ../datasets/imagenet-1k-subset \
    --num-samples 30 \
    --epsilon 0.016 \
    --attack pgd \
    --visualize-all-blocks \
    --include-gradcam \
    --device cuda \
    --output-dir ../outputs/viz
```

**Output:**
Each sample generates:
- `sample_XXX_comparison.png`: 2x3 grid with clean/adv images, TAH, ASM
- `sample_XXX_all_blocks.png`: Attention maps from all SHSA blocks (optional)
- `sample_XXX_gradcam.png`: Grad-CAM comparison clean vs adversarial (optional)

---

## 🔬 Programmatic API

### Attack Examples

```python
import torch
from robustness.attacks import fgsm, pgd, AttackHooks

# Load your model
model = load_shvit_model()
model.eval()

# Prepare data
images = torch.randn(8, 3, 224, 224).cuda()
labels = torch.randint(0, 1000, (8,)).cuda()

# FGSM attack
x_adv, info = fgsm(
    model=model,
    x=images,
    y=labels,
    epsilon=4/255,
    device='cuda'
)

print(f"Attack Success Rate: {info['attack_success_rate']:.2%}")
print(f"Time per image: {info['time_per_image']*1000:.2f}ms")

# PGD attack with attention guidance
attn_hooks = AttackHooks(device='cuda')
attn_hooks.register(model)

x_adv, info = pgd(
    model=model,
    x=images,
    y=labels,
    epsilon=4/255,
    steps=10,
    attn_guided=True,
    attn_hooks=attn_hooks,
    device='cuda'
)

attn_hooks.remove()
```

### Visualization Examples

```python
from viz.attention import AttentionVisualizer
from viz.gradcam import GradCAM

# Initialize visualizers
attn_viz = AttentionVisualizer(model, device='cuda')
attn_viz.register_hooks()

# Generate Token Activation Heatmap
attn_viz.clear_cache()
with torch.no_grad():
    _ = model(image)
heatmap = attn_viz.token_activation_heatmap(image, block_idx=-1)

# Generate Attention Stability Map
diff_map, stability_score = attn_viz.attention_stability_map(
    image_clean=clean_image,
    image_adv=adv_image,
    metric='l1'
)

print(f"Attention stability: {stability_score:.4f}")

# Comprehensive visualization
fig = attn_viz.plot_comparison_grid(
    image_clean=clean_image,
    image_adv=adv_image,
    label_clean=label,
    pred_clean=pred_clean,
    pred_adv=pred_adv,
    epsilon=0.016,
    save_path='output.png'
)

# Grad-CAM
gradcam = GradCAM(model, device='cuda')
gradcam.register_hooks()

cam = gradcam.generate_cam(image, target_class=label)
fig = gradcam.compare_clean_vs_adversarial(
    image_clean=clean_image,
    image_adv=adv_image,
    true_label=label,
    save_path='gradcam.png'
)

# Cleanup
attn_viz.remove_hooks()
gradcam.remove_hooks()
```

---

## 📈 Expected Results (Gap 1)

Based on research hypothesis:

| Model | Clean Acc | FGSM (ε=4/255) | PGD-10 (ε=4/255) |
|-------|-----------|----------------|------------------|
| ViT-B | ~80% | ~45% | ~35% |
| SHViT-S4 | ~80% | **~30-35%** | **~20-25%** |
| **Vulnerability Gap** | - | **+10-15 pp** | **+10-15 pp** |

**Hypothesis**: SHViT's single-head architecture increases vulnerability by 15-25% due to lack of attention redundancy.

---

## 🎨 Visualization Examples (Gap 4)

### Token Activation Heatmap (TAH)
- Shows spatial attention distribution
- Highlights which image patches the single head focuses on
- Clean vs adversarial comparison reveals attention drift

### Attention Stability Map (ASM)
- Quantifies attention pattern changes under attack
- Lower stability score = more disrupted attention
- Identifies vulnerable spatial regions

### Grad-CAM
- Shows class-discriminative regions
- Reveals how attacks shift model focus
- Complements TAH with gradient information

---

## ⚡ Performance Notes

### GPU Acceleration
- All attacks and visualizations run on GPU
- Typical speeds (RTX 3090):
  - FGSM: ~2-3ms per image
  - PGD-10: ~20-30ms per image
  - Visualization: ~50ms per sample

### Memory Requirements
- Batch size 32: ~4GB VRAM
- Batch size 64: ~8GB VRAM
- Visualization: ~2GB VRAM per sample

### Optimization Tips
```python
# Use mixed precision for faster evaluation
with torch.cuda.amp.autocast():
    output = model(x)

# Increase batch size on high-end GPUs
--batch-size 128  # For A100/V100

# Use subset for quick tests
--subset-size 1000  # ~5 minutes
```

---

## 🐛 Troubleshooting

### "No SHSA modules found"
- Ensure model is properly loaded
- Check that `model/shvit.py` has attention caching (`self._attn_cache = attn`)

### CUDA out of memory
- Reduce `--batch-size`
- Use `--subset-size` for testing
- Close other GPU processes

### Visualization errors
- Ensure matplotlib backend is set: `matplotlib.use('Agg')`
- Check output directory has write permissions

### Slow PGD attacks
- Reduce `--pgd-steps` (4-8 is often sufficient)
- Use `time_budget` parameter to cap wall-clock time

---

## 📚 Citation

If you use this code for research, please cite:

```bibtex
@inproceedings{shvit2024,
  title={SHViT: Single-Head Vision Transformer with Memory Efficient Macro Design},
  author={...},
  booktitle={CVPR},
  year={2024}
}
```

---

## 🔗 Related Work

- **Adversarial Robustness**: [Madry et al., ICLR 2018](https://arxiv.org/abs/1706.06083)
- **ViT Robustness**: [Bhojanapalli et al., NeurIPS 2021](https://arxiv.org/abs/2105.07926)
- **Grad-CAM**: [Selvaraju et al., ICCV 2017](https://arxiv.org/abs/1610.02391)

---

## 📝 TODO

- [ ] Add AutoAttack support
- [ ] Implement certified defenses (randomized smoothing)
- [ ] Add multi-GPU distributed evaluation
- [ ] Export adversarial examples for analysis
- [ ] Add interpretability metrics (IoU, pointing game)

---

**Questions?** Open an issue or contact the authors.

