"""
Enhanced Gap 1 Evaluation with Comprehensive Logging and Visualization
- Real-time progress tracking with ETA
- Detailed per-batch logging
- CSV export for Excel analysis
- Automatic visualization generation
- HTML report generation
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import timm
import argparse
import json
import os
import csv
from pathlib import Path
import time
from datetime import datetime, timedelta
import logging
import sys
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from robustness.attacks import fgsm, pgd, AttackHooks, evaluate_robustness
from model.build import shvit_s1, shvit_s2, shvit_s3, shvit_s4


class ExperimentLogger:
    """Comprehensive logging system for experiments"""
    
    def __init__(self, log_dir, console_level=logging.INFO, file_level=logging.DEBUG):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create logger
        self.logger = logging.getLogger('GAP1_Experiment')
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers = []  # Clear existing handlers
        
        # Console handler (with encoding fallback for Windows)
        import codecs
        if hasattr(sys.stdout, 'reconfigure'):
            try:
                sys.stdout.reconfigure(encoding='utf-8')
            except:
                pass
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(console_level)
        console_format = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_format)
        self.logger.addHandler(console_handler)
        
        # File handler (detailed)
        log_file = self.log_dir / 'experiment.log'
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(file_level)
        file_format = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        self.logger.addHandler(file_handler)
        
        self.logger.info(f"Logging initialized. Log file: {log_file}")
    
    def info(self, msg):
        self.logger.info(msg)
    
    def debug(self, msg):
        self.logger.debug(msg)
    
    def warning(self, msg):
        self.logger.warning(msg)
    
    def error(self, msg):
        self.logger.error(msg)
    
    def section(self, title, char='=', width=70):
        """Log a section header"""
        self.logger.info(char * width)
        self.logger.info(title.center(width))
        self.logger.info(char * width)
    
    def subsection(self, title, char='-', width=70):
        """Log a subsection header"""
        self.logger.info(char * width)
        self.logger.info(title)
        self.logger.info(char * width)


class ResultsManager:
    """Manage and export results in multiple formats"""
    
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results = {
            'metadata': {},
            'shvit_results': {},
            'vit_results': {},
            'comparison': {},
            'per_batch_metrics': []
        }
        
        # CSV file for batch metrics
        self.csv_path = self.output_dir / 'batch_metrics.csv'
        self.csv_file = None
        self.csv_writer = None
    
    def init_csv(self):
        """Initialize CSV file for batch-level metrics"""
        self.csv_file = open(self.csv_path, 'w', newline='')
        fieldnames = [
            'timestamp', 'model', 'attack', 'epsilon', 'batch_idx',
            'batch_size', 'clean_acc', 'robust_acc', 'asr',
            'avg_time_ms', 'l2_norm', 'linf_norm'
        ]
        self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=fieldnames)
        self.csv_writer.writeheader()
        self.csv_file.flush()
    
    def log_batch(self, model_name, attack_type, epsilon, batch_idx, metrics):
        """Log batch-level metrics to CSV"""
        if self.csv_writer:
            row = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'model': model_name,
                'attack': attack_type,
                'epsilon': epsilon,
                'batch_idx': batch_idx,
                'batch_size': metrics.get('batch_size', 0),
                'clean_acc': metrics.get('clean_acc', 0),
                'robust_acc': metrics.get('robust_acc', 0),
                'asr': metrics.get('asr', 0),
                'avg_time_ms': metrics.get('avg_time_ms', 0),
                'l2_norm': metrics.get('l2_norm', 0),
                'linf_norm': metrics.get('linf_norm', 0)
            }
            self.csv_writer.writerow(row)
            self.csv_file.flush()
            
            # Also add to in-memory storage
            self.results['per_batch_metrics'].append(row)
    
    def add_result(self, model_type, epsilon, attack_type, metrics):
        """Add model evaluation results"""
        if model_type not in self.results:
            self.results[model_type] = {}
        if epsilon not in self.results[model_type]:
            self.results[model_type][epsilon] = {}
        self.results[model_type][epsilon][attack_type] = metrics
    
    def compute_comparison(self, epsilon, attack_type):
        """Compute comparison between SHViT and ViT"""
        if (epsilon in self.results.get('shvit_results', {}) and 
            epsilon in self.results.get('vit_results', {})):
            
            shvit_metrics = self.results['shvit_results'][epsilon].get(attack_type, {})
            vit_metrics = self.results['vit_results'][epsilon].get(attack_type, {})
            
            if shvit_metrics and vit_metrics:
                shvit_asr = shvit_metrics['attack_success_rate']
                vit_asr = vit_metrics['attack_success_rate']
                gap = (shvit_asr - vit_asr) * 100  # percentage points
                
                comparison = {
                    'shvit_asr': shvit_asr,
                    'vit_asr': vit_asr,
                    'vulnerability_gap_pp': gap,
                    'shvit_clean_acc': shvit_metrics['clean_accuracy'],
                    'vit_clean_acc': vit_metrics['clean_accuracy'],
                    'shvit_time_ms': shvit_metrics['avg_time_per_image'] * 1000,
                    'vit_time_ms': vit_metrics['avg_time_per_image'] * 1000
                }
                
                if epsilon not in self.results['comparison']:
                    self.results['comparison'][epsilon] = {}
                self.results['comparison'][epsilon][attack_type] = comparison
                
                return comparison
        return None
    
    def save_json(self):
        """Save all results to JSON"""
        json_path = self.output_dir / 'results.json'
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        return json_path
    
    def save_summary(self, logger):
        """Save human-readable summary"""
        summary_path = self.output_dir / 'summary.txt'
        
        with open(summary_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("GAP-1: Single-Head Attention Vulnerability Evaluation\n")
            f.write("="*70 + "\n\n")
            
            # Metadata
            if 'metadata' in self.results:
                f.write("Experiment Configuration:\n")
                f.write("-" * 70 + "\n")
                for key, value in self.results['metadata'].items():
                    f.write(f"  {key}: {value}\n")
                f.write("\n")
            
            # Results per epsilon
            for eps in sorted(self.results.get('comparison', {}).keys()):
                f.write(f"\nEpsilon: {eps:.5f} ({eps*255:.2f}/255)\n")
                f.write("=" * 70 + "\n")
                
                for attack_type in ['fgsm', 'pgd']:
                    if attack_type in self.results['comparison'][eps]:
                        comp = self.results['comparison'][eps][attack_type]
                        f.write(f"\n{attack_type.upper()} Attack:\n")
                        f.write("-" * 70 + "\n")
                        f.write(f"  SHViT ASR:         {comp['shvit_asr']:7.2%}\n")
                        f.write(f"  ViT ASR:           {comp['vit_asr']:7.2%}\n")
                        f.write(f"  Vulnerability Gap: {comp['vulnerability_gap_pp']:+7.2f} pp\n")
                        f.write(f"  SHViT Clean Acc:   {comp['shvit_clean_acc']:7.2%}\n")
                        f.write(f"  ViT Clean Acc:     {comp['vit_clean_acc']:7.2%}\n")
                        f.write(f"  SHViT Time:        {comp['shvit_time_ms']:7.2f} ms/img\n")
                        f.write(f"  ViT Time:          {comp['vit_time_ms']:7.2f} ms/img\n")
                        f.write("\n")
        
        logger.info(f"Summary saved to: {summary_path}")
        return summary_path
    
    def close(self):
        """Close CSV file"""
        if self.csv_file:
            self.csv_file.close()


class Visualizer:
    """Generate visualizations for results"""
    
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        sns.set_style("whitegrid")
        plt.rcParams['figure.dpi'] = 150
    
    def plot_asr_vs_epsilon(self, results, logger):
        """Plot ASR vs epsilon for both models"""
        logger.info("Generating ASR vs Epsilon plot...")
        
        epsilons = sorted(results.get('comparison', {}).keys())
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        for attack_idx, attack_type in enumerate(['fgsm', 'pgd']):
            ax = axes[attack_idx]
            
            shvit_asrs = []
            vit_asrs = []
            
            for eps in epsilons:
                if attack_type in results.get('comparison', {}).get(eps, {}):
                    comp = results['comparison'][eps][attack_type]
                    shvit_asrs.append(comp['shvit_asr'] * 100)
                    vit_asrs.append(comp['vit_asr'] * 100)
                else:
                    shvit_asrs.append(0)
                    vit_asrs.append(0)
            
            eps_labels = [f"{e*255:.1f}/255" for e in epsilons]
            
            ax.plot(eps_labels, shvit_asrs, 'o-', label='SHViT', linewidth=2, markersize=8)
            ax.plot(eps_labels, vit_asrs, 's-', label='ViT', linewidth=2, markersize=8)
            
            ax.set_xlabel('Epsilon (ε)', fontsize=12)
            ax.set_ylabel('Attack Success Rate (%)', fontsize=12)
            ax.set_title(f'{attack_type.upper()} Attack', fontsize=14, fontweight='bold')
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
            
            # Add vulnerability gap annotations
            for i, eps in enumerate(epsilons):
                if i < len(shvit_asrs) and i < len(vit_asrs):
                    gap = shvit_asrs[i] - vit_asrs[i]
                    mid_y = (shvit_asrs[i] + vit_asrs[i]) / 2
                    ax.annotate(f'+{gap:.1f}pp', 
                               xy=(i, mid_y), 
                               fontsize=9, 
                               ha='center',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))
        
        plt.tight_layout()
        save_path = self.output_dir / 'asr_vs_epsilon.png'
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        
        logger.info(f"ASR vs Epsilon plot saved to: {save_path}")
        return save_path
    
    def plot_vulnerability_gap(self, results, logger):
        """Plot vulnerability gap across epsilons"""
        logger.info("Generating Vulnerability Gap plot...")
        
        epsilons = sorted(results.get('comparison', {}).keys())
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        fgsm_gaps = []
        pgd_gaps = []
        
        for eps in epsilons:
            for attack_type, gaps_list in [('fgsm', fgsm_gaps), ('pgd', pgd_gaps)]:
                if attack_type in results.get('comparison', {}).get(eps, {}):
                    comp = results['comparison'][eps][attack_type]
                    gaps_list.append(comp['vulnerability_gap_pp'])
                else:
                    gaps_list.append(0)
        
        x = np.arange(len(epsilons))
        width = 0.35
        
        eps_labels = [f"{e*255:.1f}/255" for e in epsilons]
        
        bars1 = ax.bar(x - width/2, fgsm_gaps, width, label='FGSM', alpha=0.8)
        bars2 = ax.bar(x + width/2, pgd_gaps, width, label='PGD', alpha=0.8)
        
        ax.set_xlabel('Epsilon (ε)', fontsize=12)
        ax.set_ylabel('Vulnerability Gap (pp)', fontsize=12)
        ax.set_title('SHViT vs ViT Vulnerability Gap', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(eps_labels)
        ax.legend(fontsize=11)
        ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}',
                       ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        save_path = self.output_dir / 'vulnerability_gap.png'
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Vulnerability Gap plot saved to: {save_path}")
        return save_path
    
    def plot_clean_vs_robust_accuracy(self, results, logger):
        """Plot clean vs robust accuracy"""
        logger.info("Generating Clean vs Robust Accuracy plot...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle('Clean vs Robust Accuracy Analysis', fontsize=16, fontweight='bold')
        
        epsilons = sorted(results.get('comparison', {}).keys())
        eps_labels = [f"{e*255:.1f}/255" for e in epsilons]
        
        models = [('shvit_results', 'SHViT'), ('vit_results', 'ViT')]
        attacks = ['fgsm', 'pgd']
        
        for model_idx, (model_key, model_name) in enumerate(models):
            for attack_idx, attack_type in enumerate(attacks):
                ax = axes[model_idx][attack_idx]
                
                clean_accs = []
                robust_accs = []
                
                for eps in epsilons:
                    if eps in results.get(model_key, {}) and attack_type in results[model_key][eps]:
                        metrics = results[model_key][eps][attack_type]
                        clean_accs.append(metrics['clean_accuracy'] * 100)
                        robust_accs.append(metrics['robust_accuracy'] * 100)
                    else:
                        clean_accs.append(0)
                        robust_accs.append(0)
                
                x = np.arange(len(epsilons))
                width = 0.35
                
                ax.bar(x - width/2, clean_accs, width, label='Clean', alpha=0.8)
                ax.bar(x + width/2, robust_accs, width, label='Robust', alpha=0.8)
                
                ax.set_xlabel('Epsilon (ε)', fontsize=10)
                ax.set_ylabel('Accuracy (%)', fontsize=10)
                ax.set_title(f'{model_name} - {attack_type.upper()}', fontsize=12, fontweight='bold')
                ax.set_xticks(x)
                ax.set_xticklabels(eps_labels)
                ax.legend(fontsize=9)
                ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        save_path = self.output_dir / 'clean_vs_robust_accuracy.png'
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Clean vs Robust Accuracy plot saved to: {save_path}")
        return save_path
    
    def generate_all_plots(self, results, logger):
        """Generate all visualization plots"""
        logger.section("Generating Visualizations")
        
        plots = []
        try:
            plots.append(self.plot_asr_vs_epsilon(results, logger))
            plots.append(self.plot_vulnerability_gap(results, logger))
            plots.append(self.plot_clean_vs_robust_accuracy(results, logger))
        except Exception as e:
            logger.error(f"Error generating plots: {e}")
        
        return plots


def evaluate_model_with_logging(model, model_name, dataloader, attack_fn, attack_kwargs,
                                results_manager, logger, device='cuda'):
    """Evaluate model with detailed batch-level logging"""
    model.eval()
    model.to(device)
    
    total_samples = 0
    total_correct_clean = 0
    total_correct_adv = 0
    total_time = 0
    all_info = []
    
    attack_name = attack_fn.__name__.upper()
    epsilon = attack_kwargs.get('epsilon', 0)
    
    logger.subsection(f"Evaluating {model_name} - {attack_name} (eps={epsilon:.5f})")
    
    # Progress bar
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), 
                desc=f"{model_name} {attack_name}", 
                leave=True)
    
    for batch_idx, (x, y) in pbar:
        x, y = x.to(device), y.to(device)
        batch_size = len(x)
        
        # Clean accuracy
        with torch.no_grad():
            outputs_clean = model(x)
            pred_clean = outputs_clean.argmax(dim=1)
            correct_clean = (pred_clean == y).sum().item()
            total_correct_clean += correct_clean
        
        # Adversarial attack
        x_adv, info = attack_fn(model, x, y, device=device, **attack_kwargs)
        
        # Adversarial accuracy
        with torch.no_grad():
            outputs_adv = model(x_adv)
            pred_adv = outputs_adv.argmax(dim=1)
            correct_adv = (pred_adv == y).sum().item()
            total_correct_adv += correct_adv
        
        total_samples += batch_size
        total_time += info['time_per_image'] * batch_size
        all_info.append(info)
        
        # Compute running metrics
        clean_acc = total_correct_clean / total_samples
        robust_acc = total_correct_adv / total_samples
        asr = 1 - robust_acc
        avg_time_ms = (total_time / total_samples) * 1000
        
        # Log batch metrics to CSV
        batch_metrics = {
            'batch_size': batch_size,
            'clean_acc': clean_acc,
            'robust_acc': robust_acc,
            'asr': asr,
            'avg_time_ms': avg_time_ms,
            'l2_norm': info.get('l2_norm', 0),
            'linf_norm': info.get('linf_norm', 0)
        }
        results_manager.log_batch(model_name, attack_name, epsilon, batch_idx, batch_metrics)
        
        # Update progress bar
        pbar.set_postfix({
            'Clean': f'{clean_acc:.2%}',
            'Robust': f'{robust_acc:.2%}',
            'ASR': f'{asr:.2%}',
            'Time': f'{avg_time_ms:.1f}ms'
        })
        
        # Log every 10 batches
        if batch_idx % 10 == 0:
            logger.debug(f"Batch {batch_idx:3d} | Clean: {clean_acc:.2%} | "
                        f"Robust: {robust_acc:.2%} | ASR: {asr:.2%} | "
                        f"Time: {avg_time_ms:.1f}ms/img")
    
    pbar.close()
    
    # Final metrics
    clean_accuracy = total_correct_clean / total_samples
    robust_accuracy = total_correct_adv / total_samples
    attack_success_rate = 1 - robust_accuracy
    avg_time_per_image = total_time / total_samples
    avg_l2 = np.mean([info['l2_norm'] for info in all_info])
    avg_linf = np.mean([info['linf_norm'] for info in all_info])
    
    metrics = {
        'clean_accuracy': clean_accuracy,
        'robust_accuracy': robust_accuracy,
        'attack_success_rate': attack_success_rate,
        'avg_time_per_image': avg_time_per_image,
        'avg_l2_norm': avg_l2,
        'avg_linf_norm': avg_linf,
        'total_samples': total_samples,
        'device': device
    }
    
    logger.info(f"FINAL | Clean: {clean_accuracy:.2%} | Robust: {robust_accuracy:.2%} | "
               f"ASR: {attack_success_rate:.2%} | Time: {avg_time_per_image*1000:.2f}ms/img")
    
    return metrics


def get_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Enhanced Gap 1: Adversarial Robustness Evaluation')
    
    # Model configuration
    parser.add_argument('--shvit-model', type=str, default='shvit_s4',
                       choices=['shvit_s1', 'shvit_s2', 'shvit_s3', 'shvit_s4'],
                       help='SHViT model variant')
    parser.add_argument('--vit-model', type=str, default=None,
                       help='ViT baseline model from timm (optional)')
    parser.add_argument('--shvit-weights', type=str, default=None,
                       help='Path to SHViT pretrained weights')
    
    # Data configuration
    parser.add_argument('--data-path', type=str, required=True,
                       help='Path to ImageNet dataset')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for evaluation')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--subset-size', type=int, default=None,
                       help='Use subset of dataset (for quick tests)')
    
    # Attack configuration
    parser.add_argument('--attack', type=str, default='both',
                       choices=['fgsm', 'pgd', 'both'],
                       help='Attack type to evaluate')
    parser.add_argument('--epsilon', type=float, nargs='+', 
                       default=[0.001, 0.002, 0.004, 0.008, 0.016],
                       help='List of epsilon values for L∞ perturbations')
    parser.add_argument('--pgd-steps', type=int, default=10,
                       help='Number of PGD iterations')
    parser.add_argument('--pgd-alpha', type=float, default=None,
                       help='PGD step size (default: epsilon/4)')
    
    # System configuration
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use for computation')
    parser.add_argument('--output-dir', type=str, default='outputs/gap1_enhanced',
                       help='Output directory for results')
    parser.add_argument('--exp-name', type=str, default=None,
                       help='Experiment name (default: timestamp)')
    
    return parser.parse_args()


def load_model(model_name_or_fn, weights_path=None, device='cuda', is_timm=False):
    """Generic model loader"""
    if is_timm:
        print(f"\nLoading ViT model from timm: {model_name_or_fn}")
        try:
            model = timm.create_model(model_name_or_fn, pretrained=True, num_classes=1000)
        except:
            print(f"Warning: Could not load pretrained {model_name_or_fn}, using random weights")
            model = timm.create_model(model_name_or_fn, pretrained=False, num_classes=1000)
    else:
        print(f"\nLoading SHViT model: {model_name_or_fn}")
        model_fn = eval(model_name_or_fn)
        model = model_fn(pretrained=False, num_classes=1000)
        
        if weights_path and os.path.exists(weights_path):
            print(f"Loading weights from: {weights_path}")
            checkpoint = torch.load(weights_path, map_location='cpu')
            model.load_state_dict(checkpoint['model'] if 'model' in checkpoint else checkpoint)
    
    model = model.to(device)
    model.eval()
    
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Parameters: {num_params:.2f}M")
    
    return model


def get_dataloader(data_path, batch_size=32, num_workers=4, subset_size=None):
    """Create ImageNet validation dataloader"""
    print(f"\nPreparing dataset from: {data_path}")
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    
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
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader


def main():
    args = get_args()
    
    # Setup experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = args.exp_name or f"exp_{timestamp}"
    output_dir = Path(args.output_dir) / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize logging
    logger = ExperimentLogger(output_dir)
    logger.section("GAP-1: Single-Head Attention Vulnerability Evaluation")
    logger.info(f"Experiment: {exp_name}")
    logger.info(f"Output directory: {output_dir}")
    
    # Check GPU
    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        args.device = 'cpu'
    
    if args.device == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Initialize results manager
    results_manager = ResultsManager(output_dir)
    results_manager.init_csv()
    
    # Store metadata
    results_manager.results['metadata'] = {
        'experiment_name': exp_name,
        'timestamp': timestamp,
        'shvit_model': args.shvit_model,
        'vit_model': args.vit_model,
        'data_path': args.data_path,
        'batch_size': args.batch_size,
        'subset_size': args.subset_size,
        'attack_types': args.attack,
        'epsilons': args.epsilon,
        'pgd_steps': args.pgd_steps,
        'device': args.device
    }
    
    # Load models
    logger.section("Loading Models")
    start_time = time.time()
    
    shvit_model = load_model(args.shvit_model, args.shvit_weights, args.device, is_timm=False)
    
    vit_model = None
    if args.vit_model:
        vit_model = load_model(args.vit_model, None, args.device, is_timm=True)
    
    logger.info(f"Model loading time: {time.time() - start_time:.2f}s")
    
    # Load data
    logger.section("Loading Dataset")
    dataloader = get_dataloader(
        args.data_path,
        args.batch_size,
        args.num_workers,
        args.subset_size
    )
    
    # Run evaluation
    logger.section("Starting Evaluation")
    experiment_start = time.time()
    
    models_to_eval = [('shvit', shvit_model, args.shvit_model)]
    if vit_model:
        models_to_eval.append(('vit', vit_model, args.vit_model))
    
    for eps in args.epsilon:
        logger.section(f"Epsilon: {eps:.5f} ({eps*255:.2f}/255)")
        
        # FGSM
        if args.attack in ['fgsm', 'both']:
            logger.subsection(f"FGSM Attack")
            fgsm_kwargs = {'epsilon': eps}
            
            for model_type, model, model_name in models_to_eval:
                metrics = evaluate_model_with_logging(
                    model, f"{model_type.upper()}-{model_name}", dataloader,
                    fgsm, fgsm_kwargs, results_manager, logger, args.device
                )
                results_manager.add_result(f'{model_type}_results', eps, 'fgsm', metrics)
            
            if vit_model:
                results_manager.compute_comparison(eps, 'fgsm')
        
        # PGD
        if args.attack in ['pgd', 'both']:
            logger.subsection(f"PGD Attack")
            pgd_kwargs = {
                'epsilon': eps,
                'steps': args.pgd_steps,
                'alpha': args.pgd_alpha if args.pgd_alpha else eps / 4
            }
            
            for model_type, model, model_name in models_to_eval:
                metrics = evaluate_model_with_logging(
                    model, f"{model_type.upper()}-{model_name}", dataloader,
                    pgd, pgd_kwargs, results_manager, logger, args.device
                )
                results_manager.add_result(f'{model_type}_results', eps, 'pgd', metrics)
            
            if vit_model:
                results_manager.compute_comparison(eps, 'pgd')
    
    total_time = time.time() - experiment_start
    logger.info(f"Total evaluation time: {str(timedelta(seconds=int(total_time)))}")
    
    # Save results
    logger.section("Saving Results")
    results_manager.save_json()
    results_manager.save_summary(logger)
    results_manager.close()
    
    # Generate visualizations
    if vit_model:  # Only generate comparison plots if we have both models
        visualizer = Visualizer(output_dir)
        visualizer.generate_all_plots(results_manager.results, logger)
    
    logger.section("Experiment Complete!")
    logger.info(f"All results saved to: {output_dir}")


if __name__ == '__main__':
    main()

