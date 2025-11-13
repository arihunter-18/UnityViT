"""
Comprehensive visualization script for Gap 4:
Token Activation Heatmaps, Attention Stability Maps, and Grad-CAM
Generates 20-30 high-quality visualization panels
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import argparse
import os
from pathlib import Path
import numpy as np
from datetime import datetime
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from model.build import shvit_s1, shvit_s2, shvit_s3, shvit_s4
from robustness.attacks import fgsm, pgd
from viz.attention import AttentionVisualizer
from viz.gradcam import GradCAM


def get_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Gap 4: Visualization Toolkit')
    
    # Model configuration
    parser.add_argument('--model', type=str, default='shvit_s4',
                       choices=['shvit_s1', 'shvit_s2', 'shvit_s3', 'shvit_s4'],
                       help='SHViT model variant')
    parser.add_argument('--weights', type=str, default=None,
                       help='Path to pretrained weights')
    
    # Data configuration
    parser.add_argument('--data-path', type=str, default='../datasets/imagenet-1k-subset',
                       help='Path to ImageNet dataset')
    parser.add_argument('--num-samples', type=int, default=20,
                       help='Number of samples to visualize')
    
    # Attack configuration for ASM
    parser.add_argument('--epsilon', type=float, default=4/255,
                       help='Epsilon for adversarial examples')
    parser.add_argument('--attack', type=str, default='pgd',
                       choices=['fgsm', 'pgd'],
                       help='Attack type for generating adversarial examples')
    
    # Visualization configuration
    parser.add_argument('--visualize-all-blocks', action='store_true',
                       help='Visualize attention from all SHSA blocks')
    parser.add_argument('--include-gradcam', action='store_true',
                       help='Include Grad-CAM visualizations')
    
    # System configuration
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use')
    parser.add_argument('--output-dir', type=str, default='../outputs/viz',
                       help='Output directory for visualizations')
    
    return parser.parse_args()


def load_model(model_name, weights_path=None, device='cuda'):
    """Load SHViT model"""
    print(f"\nLoading model: {model_name}")
    
    model_fn = eval(model_name)
    model = model_fn(pretrained=False, num_classes=1000)
    
    if weights_path and os.path.exists(weights_path):
        print(f"Loading weights from: {weights_path}")
        checkpoint = torch.load(weights_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'] if 'model' in checkpoint else checkpoint)
    else:
        print("Warning: No pretrained weights loaded")
    
    model = model.to(device)
    model.eval()
    
    return model


def get_sample_images(data_path, num_samples=20):
    """Load sample images from dataset"""
    print(f"\nLoading {num_samples} sample images...")
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    
    dataset = torchvision.datasets.ImageFolder(
        os.path.join(data_path, 'val'),
        transform=transform
    )
    
    # Select diverse samples
    indices = torch.linspace(0, len(dataset)-1, num_samples).long()
    samples = [dataset[i] for i in indices]
    
    return samples


def generate_adversarial(model, image, label, attack_type='pgd', epsilon=4/255, device='cuda'):
    """Generate adversarial example"""
    image = image.unsqueeze(0).to(device)
    label = torch.tensor([label]).to(device)
    
    if attack_type == 'fgsm':
        x_adv, info = fgsm(model, image, label, epsilon=epsilon, device=device)
    else:  # pgd
        x_adv, info = pgd(model, image, label, epsilon=epsilon, steps=10, device=device)
    
    return x_adv, info


def main():
    args = get_args()
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"gap4_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"Gap 4: Visualization Toolkit for Single-Head Attention")
    print(f"Output directory: {run_dir}")
    print(f"{'='*70}")
    
    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU")
        args.device = 'cpu'
    
    # Load model
    model = load_model(args.model, args.weights, args.device)
    
    # Initialize visualizers
    attn_viz = AttentionVisualizer(model, patch_size=16, device=args.device)
    attn_viz.register_hooks()
    
    if args.include_gradcam:
        gradcam = GradCAM(model, device=args.device)
        gradcam.register_hooks()
    
    # Load sample images
    samples = get_sample_images(args.data_path, args.num_samples)
    
    print(f"\nGenerating visualizations for {len(samples)} samples...")
    print(f"Attack: {args.attack.upper()}, Epsilon: {args.epsilon:.5f}")
    
    # Process each sample
    for idx, (image, label) in enumerate(samples):
        print(f"\n--- Sample {idx+1}/{len(samples)} ---")
        
        # Get clean prediction
        with torch.no_grad():
            output = model(image.unsqueeze(0).to(args.device))
            pred_clean = output.argmax(dim=1).item()
        
        # Generate adversarial example
        image_adv, attack_info = generate_adversarial(
            model, image, label, args.attack, args.epsilon, args.device
        )
        
        # Get adversarial prediction
        with torch.no_grad():
            output_adv = model(image_adv)
            pred_adv = output_adv.argmax(dim=1).item()
        
        success = "SUCCESS" if pred_adv != label else "FAILED"
        print(f"  True: {label}, Pred (clean): {pred_clean}, Pred (adv): {pred_adv} [{success}]")
        print(f"  ASR: {attack_info['attack_success_rate']:.2%}, "
              f"L∞: {attack_info['linf_norm']:.4f}")
        
        # 1. Comprehensive comparison grid (TAH + ASM)
        fig = attn_viz.plot_comparison_grid(
            image_clean=image.unsqueeze(0),
            image_adv=image_adv,
            label_clean=label,
            pred_clean=pred_clean,
            pred_adv=pred_adv,
            epsilon=args.epsilon,
            save_path=run_dir / f"sample_{idx:03d}_comparison.png",
            block_idx=-1  # Last block
        )
        plt.close(fig)
        
        # 2. All blocks visualization (optional)
        if args.visualize_all_blocks and idx < 5:  # Only first 5 samples
            fig = attn_viz.visualize_all_blocks(
                image=image.unsqueeze(0),
                save_path=run_dir / f"sample_{idx:03d}_all_blocks.png"
            )
            if fig:
                plt.close(fig)
        
        # 3. Grad-CAM comparison (optional)
        if args.include_gradcam:
            fig = gradcam.compare_clean_vs_adversarial(
                image_clean=image.unsqueeze(0),
                image_adv=image_adv,
                true_label=label,
                save_path=run_dir / f"sample_{idx:03d}_gradcam.png"
            )
            plt.close(fig)
    
    # Cleanup
    attn_viz.remove_hooks()
    if args.include_gradcam:
        gradcam.remove_hooks()
    
    print(f"\n{'='*70}")
    print(f"VISUALIZATION COMPLETE")
    print(f"Generated {len(samples)} visualization panels in: {run_dir}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    
    main()

