"""
GAP-4: Complete Attention Disruption Visualization and Analysis
Comprehensive system for analyzing how adversarial attacks affect single-head attention
Features:
- Attention visualization (clean vs adversarial)
- Attention Shift Metric (ASM) quantification
- Grad-CAM comparison
- Statistical analysis
- Publication-ready figures
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset
import argparse
import os
import json
import csv
from pathlib import Path
import numpy as np
from datetime import datetime
import sys
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from model.build import shvit_s1, shvit_s2, shvit_s3, shvit_s4
from robustness.attacks import fgsm, pgd
from viz.attention import AttentionVisualizer
from viz.gradcam import GradCAM


def get_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='GAP-4: Complete Attention Disruption Analysis')
    
    # Model configuration
    parser.add_argument('--model', type=str, default='shvit_s4',
                       choices=['shvit_s1', 'shvit_s2', 'shvit_s3', 'shvit_s4'],
                       help='SHViT model variant')
    parser.add_argument('--weights', type=str, default=None,
                       help='Path to pretrained weights')
    
    # Data configuration
    parser.add_argument('--data-path', type=str, required=True,
                       help='Path to ImageNet dataset')
    parser.add_argument('--num-samples', type=int, default=30,
                       help='Number of samples to analyze')
    parser.add_argument('--subset-size', type=int, default=None,
                       help='Use subset of dataset (for selection)')
    
    # Attack configuration
    parser.add_argument('--epsilon', type=float, nargs='+',
                       default=[0.004, 0.008, 0.016],
                       help='Epsilon values for adversarial examples')
    parser.add_argument('--attack', type=str, default='pgd',
                       choices=['fgsm', 'pgd', 'both'],
                       help='Attack type')
    parser.add_argument('--pgd-steps', type=int, default=10,
                       help='Number of PGD iterations')
    
    # Visualization configuration
    parser.add_argument('--visualize-all-blocks', action='store_true',
                       help='Visualize attention from all SHSA blocks')
    parser.add_argument('--include-gradcam', action='store_true',
                       help='Include Grad-CAM visualizations')
    parser.add_argument('--save-top-k', type=int, default=10,
                       help='Save top-k best visualizations')
    
    # System configuration
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use')
    parser.add_argument('--output-dir', type=str, default='outputs/gap4_complete',
                       help='Output directory')
    parser.add_argument('--exp-name', type=str, default=None,
                       help='Experiment name')
    
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
        print("[SUCCESS] Pretrained weights loaded")
    else:
        print("[WARNING] No pretrained weights loaded - using random initialization")
    
    model = model.to(device)
    model.eval()
    
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Parameters: {num_params:.2f}M")
    
    return model


def get_dataset(data_path, subset_size=None):
    """Load dataset"""
    print(f"\nLoading dataset from: {data_path}")
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    
    dataset = torchvision.datasets.ImageFolder(
        os.path.join(data_path, 'val'),
        transform=transform
    )
    
    if subset_size and subset_size < len(dataset):
        indices = torch.randperm(len(dataset))[:subset_size]
        dataset = Subset(dataset, indices)
        print(f"Using subset of {subset_size} images")
    
    print(f"Dataset size: {len(dataset)}")
    return dataset


def generate_adversarial(model, image, label, attack_type='pgd', epsilon=0.016, 
                         pgd_steps=10, device='cuda'):
    """Generate adversarial example"""
    image = image.unsqueeze(0).to(device)
    label = torch.tensor([label]).to(device)
    
    if attack_type == 'fgsm':
        x_adv, info = fgsm(model, image, label, epsilon=epsilon, device=device)
    else:  # pgd
        x_adv, info = pgd(model, image, label, epsilon=epsilon, 
                         steps=pgd_steps, device=device)
    
    return x_adv, info


class ASMAnalyzer:
    """Analyze Attention Shift Metrics across samples"""
    
    def __init__(self):
        self.asm_scores = []
        self.attack_success = []
        self.epsilon_values = []
        self.sample_ids = []
    
    def add_result(self, sample_id, epsilon, asm_score, success):
        """Add a result"""
        self.sample_ids.append(sample_id)
        self.epsilon_values.append(epsilon)
        self.asm_scores.append(asm_score)
        self.attack_success.append(1 if success else 0)
    
    def compute_statistics(self):
        """Compute statistical analysis"""
        asm_array = np.array(self.asm_scores)
        success_array = np.array(self.attack_success)
        
        stats_dict = {
            'mean_asm': float(np.mean(asm_array)),
            'std_asm': float(np.std(asm_array)),
            'median_asm': float(np.median(asm_array)),
            'min_asm': float(np.min(asm_array)),
            'max_asm': float(np.max(asm_array)),
            'asr': float(np.mean(success_array)),
            'num_samples': len(asm_array)
        }
        
        # Correlation between ASM and attack success
        if len(set(success_array)) > 1:  # Need variability
            correlation, p_value = stats.pointbiserialr(success_array, asm_array)
            stats_dict['correlation_asm_success'] = float(correlation)
            stats_dict['p_value'] = float(p_value)
        
        return stats_dict
    
    def plot_asm_distribution(self, save_path):
        """Plot ASM distribution"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram
        axes[0].hist(self.asm_scores, bins=20, edgecolor='black', alpha=0.7)
        axes[0].set_xlabel('Attention Shift Metric (ASM)', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].set_title('Distribution of ASM Scores', fontsize=14, fontweight='bold')
        axes[0].axvline(np.mean(self.asm_scores), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(self.asm_scores):.4f}')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Box plot by attack success
        success_labels = ['Failed', 'Success']
        data_by_success = [
            [self.asm_scores[i] for i in range(len(self.asm_scores)) 
             if self.attack_success[i] == 0],
            [self.asm_scores[i] for i in range(len(self.asm_scores)) 
             if self.attack_success[i] == 1]
        ]
        
        axes[1].boxplot(data_by_success, labels=success_labels)
        axes[1].set_ylabel('Attention Shift Metric (ASM)', fontsize=12)
        axes[1].set_title('ASM by Attack Success', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved ASM distribution plot: {save_path}")
    
    def plot_asm_vs_epsilon(self, save_path):
        """Plot ASM vs epsilon"""
        # Group by epsilon
        epsilon_unique = sorted(set(self.epsilon_values))
        asm_by_eps = {eps: [] for eps in epsilon_unique}
        
        for eps, asm in zip(self.epsilon_values, self.asm_scores):
            asm_by_eps[eps].append(asm)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Box plot
        data = [asm_by_eps[eps] for eps in epsilon_unique]
        labels = [f"{eps:.4f}\n({eps*255:.1f}/255)" for eps in epsilon_unique]
        
        bp = ax.boxplot(data, labels=labels, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        
        ax.set_xlabel('Epsilon (eps)', fontsize=12)
        ax.set_ylabel('Attention Shift Metric (ASM)', fontsize=12)
        ax.set_title('ASM vs Attack Strength', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add trend line
        means = [np.mean(asm_by_eps[eps]) for eps in epsilon_unique]
        ax.plot(range(1, len(epsilon_unique)+1), means, 'r-', 
               marker='o', linewidth=2, markersize=8, label='Mean ASM')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved ASM vs epsilon plot: {save_path}")


def main():
    args = get_args()
    
    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = args.exp_name or f"gap4_{timestamp}"
    output_dir = Path(args.output_dir) / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    viz_dir = output_dir / 'visualizations'
    viz_dir.mkdir(exist_ok=True)
    
    print("="*70)
    print("GAP-4: Attention Disruption Visualization and Analysis")
    print("="*70)
    print(f"Experiment: {exp_name}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("[WARNING] CUDA not available, using CPU")
        args.device = 'cpu'
    elif args.device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load model
    model = load_model(args.model, args.weights, args.device)
    
    # Initialize visualizers
    print("\nInitializing visualization tools...")
    attn_viz = AttentionVisualizer(model, patch_size=16, device=args.device)
    attn_viz.register_hooks()
    
    if args.include_gradcam:
        gradcam = GradCAM(model, device=args.device)
        gradcam.register_hooks()
    
    # Load dataset
    dataset = get_dataset(args.data_path, args.subset_size)
    
    # Select samples for visualization
    num_viz_samples = min(args.num_samples, len(dataset))
    indices = np.linspace(0, len(dataset)-1, num_viz_samples, dtype=int)
    
    # Initialize analyzer
    asm_analyzer = ASMAnalyzer()
    
    # Results storage
    all_results = []
    csv_path = output_dir / 'asm_metrics.csv'
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['sample_id', 'true_label', 'pred_clean', 'pred_adv', 
                     'attack', 'epsilon', 'asm_score', 'attack_success', 
                     'linf_norm', 'l2_norm']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # Process each epsilon value
        for epsilon in args.epsilon:
            print(f"\n{'='*70}")
            print(f"Processing Epsilon: {epsilon:.5f} ({epsilon*255:.2f}/255)")
            print(f"{'='*70}")
            
            # Determine attack types
            attack_types = ['fgsm', 'pgd'] if args.attack == 'both' else [args.attack]
            
            for attack_type in attack_types:
                print(f"\nAttack: {attack_type.upper()}")
                
                # Process samples
                pbar = tqdm(indices, desc=f"{attack_type.upper()} eps={epsilon:.4f}")
                
                for sample_idx in pbar:
                    image, label = dataset[sample_idx]
                    
                    # Get clean prediction
                    with torch.no_grad():
                        output_clean = model(image.unsqueeze(0).to(args.device))
                        pred_clean = output_clean.argmax(dim=1).item()
                    
                    # Generate adversarial example
                    try:
                        image_adv, attack_info = generate_adversarial(
                            model, image, label, attack_type, epsilon, 
                            args.pgd_steps, args.device
                        )
                    except Exception as e:
                        print(f"\nError generating adversarial for sample {sample_idx}: {e}")
                        continue
                    
                    # Get adversarial prediction
                    with torch.no_grad():
                        output_adv = model(image_adv)
                        pred_adv = output_adv.argmax(dim=1).item()
                    
                    attack_success = (pred_adv != label)
                    
                    # Compute ASM
                    try:
                        asm_map, asm_score = attn_viz.attention_stability_map(
                            image.unsqueeze(0), image_adv, block_idx=-1, metric='l1'
                        )
                    except Exception as e:
                        print(f"\nError computing ASM for sample {sample_idx}: {e}")
                        continue
                    
                    # Store results
                    result = {
                        'sample_id': int(sample_idx),
                        'true_label': int(label),
                        'pred_clean': int(pred_clean),
                        'pred_adv': int(pred_adv),
                        'attack': attack_type,
                        'epsilon': float(epsilon),
                        'asm_score': float(asm_score),
                        'attack_success': int(attack_success),
                        'linf_norm': float(attack_info['linf_norm']),
                        'l2_norm': float(attack_info['l2_norm'])
                    }
                    
                    all_results.append(result)
                    writer.writerow(result)
                    csvfile.flush()
                    
                    # Add to analyzer
                    asm_analyzer.add_result(sample_idx, epsilon, asm_score, attack_success)
                    
                    # Update progress bar
                    pbar.set_postfix({
                        'ASM': f'{asm_score:.4f}',
                        'Success': 'Y' if attack_success else 'N'
                    })
                    
                    # Generate visualizations for selected samples
                    # Save top-k by ASM score or first N samples
                    should_visualize = (
                        sample_idx < args.save_top_k or 
                        asm_score > 0.2  # High disruption
                    )
                    
                    if should_visualize:
                        try:
                            # Main comparison grid
                            fig = attn_viz.plot_comparison_grid(
                                image_clean=image.unsqueeze(0),
                                image_adv=image_adv,
                                label_clean=label,
                                pred_clean=pred_clean,
                                pred_adv=pred_adv,
                                epsilon=epsilon,
                                save_path=viz_dir / f"sample_{sample_idx:03d}_{attack_type}_eps{epsilon:.4f}_comparison.png",
                                block_idx=-1
                            )
                            plt.close(fig)
                            
                            # Grad-CAM (if enabled and first few samples)
                            if args.include_gradcam and sample_idx < 5:
                                fig = gradcam.compare_clean_vs_adversarial(
                                    image_clean=image.unsqueeze(0),
                                    image_adv=image_adv,
                                    true_label=label,
                                    save_path=viz_dir / f"sample_{sample_idx:03d}_{attack_type}_eps{epsilon:.4f}_gradcam.png"
                                )
                                plt.close(fig)
                            
                            # All blocks (if enabled and first few samples)
                            if args.visualize_all_blocks and sample_idx < 3:
                                fig = attn_viz.visualize_all_blocks(
                                    image=image.unsqueeze(0),
                                    save_path=viz_dir / f"sample_{sample_idx:03d}_all_blocks.png"
                                )
                                if fig:
                                    plt.close(fig)
                        
                        except Exception as e:
                            print(f"\nError generating visualization for sample {sample_idx}: {e}")
    
    # Cleanup hooks
    attn_viz.remove_hooks()
    if args.include_gradcam:
        gradcam.remove_hooks()
    
    # Statistical analysis
    print(f"\n{'='*70}")
    print("Statistical Analysis")
    print(f"{'='*70}")
    
    stats_results = asm_analyzer.compute_statistics()
    
    print(f"\nASM Statistics:")
    print(f"  Mean:   {stats_results['mean_asm']:.4f}")
    print(f"  Std:    {stats_results['std_asm']:.4f}")
    print(f"  Median: {stats_results['median_asm']:.4f}")
    print(f"  Range:  [{stats_results['min_asm']:.4f}, {stats_results['max_asm']:.4f}]")
    print(f"  ASR:    {stats_results['asr']:.2%}")
    
    if 'correlation_asm_success' in stats_results:
        print(f"\nCorrelation (ASM vs Success): {stats_results['correlation_asm_success']:.4f}")
        print(f"  P-value: {stats_results['p_value']:.4e}")
    
    # Save statistics
    stats_path = output_dir / 'statistics.json'
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump({
            'statistics': stats_results,
            'all_results': all_results,
            'config': vars(args)
        }, f, indent=2, default=str)
    
    # Generate analysis plots
    print(f"\n{'='*70}")
    print("Generating Analysis Plots")
    print(f"{'='*70}")
    
    asm_analyzer.plot_asm_distribution(output_dir / 'asm_distribution.png')
    asm_analyzer.plot_asm_vs_epsilon(output_dir / 'asm_vs_epsilon.png')
    
    # Summary
    print(f"\n{'='*70}")
    print("GAP-4 Analysis Complete!")
    print(f"{'='*70}")
    print(f"\nResults saved to: {output_dir}")
    print(f"\nGenerated files:")
    print(f"  - asm_metrics.csv: Per-sample metrics")
    print(f"  - statistics.json: Statistical analysis")
    print(f"  - asm_distribution.png: ASM distribution plot")
    print(f"  - asm_vs_epsilon.png: ASM vs attack strength")
    print(f"  - visualizations/: {len(list(viz_dir.glob('*.png')))} visualization images")
    print()


if __name__ == '__main__':
    main()

