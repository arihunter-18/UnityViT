"""
Main evaluation script for Gap 1: Single-Head Attention Vulnerability
Compares SHViT-S vs ViT-B under identical FGSM/PGD attacks
GPU-accelerated with comprehensive logging
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
from pathlib import Path
import time
from datetime import datetime

from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from robustness.attacks import fgsm, pgd, AttackHooks, evaluate_robustness
from model.build import shvit_s1, shvit_s2, shvit_s3, shvit_s4


def get_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Gap 1: Adversarial Robustness Evaluation')
    
    # Model configuration
    parser.add_argument('--shvit-model', type=str, default='shvit_s4',
                       choices=['shvit_s1', 'shvit_s2', 'shvit_s3', 'shvit_s4'],
                       help='SHViT model variant')
    parser.add_argument('--vit-model', type=str, default='vit_base_patch16_224',
                       help='ViT baseline model from timm')
    parser.add_argument('--shvit-weights', type=str, default=None,
                       help='Path to SHViT pretrained weights')
    
    # Data configuration
    parser.add_argument('--data-path', type=str, default='../datasets/imagenet-1k-subset',
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
                       default=[1/255, 2/255, 4/255, 8/255],
                       help='List of epsilon values for L∞ perturbations')
    parser.add_argument('--pgd-steps', type=int, default=10,
                       help='Number of PGD iterations')
    parser.add_argument('--pgd-alpha', type=float, default=None,
                       help='PGD step size (default: epsilon/4)')
    parser.add_argument('--attn-guided', action='store_true',
                       help='Use attention-guided attacks')
    
    # System configuration
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use for computation')
    parser.add_argument('--output-dir', type=str, default='../outputs/gap1_results',
                       help='Output directory for results')
    parser.add_argument('--save-examples', action='store_true',
                       help='Save adversarial examples')
    
    return parser.parse_args()


def load_shvit_model(model_name, weights_path=None, device='cuda'):
    """Load SHViT model"""
    print(f"\nLoading SHViT model: {model_name}")
    
    model_fn = eval(model_name)
    model = model_fn(pretrained=False, num_classes=1000)
    
    if weights_path and os.path.exists(weights_path):
        print(f"Loading weights from: {weights_path}")
        checkpoint = torch.load(weights_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'] if 'model' in checkpoint else checkpoint)
    else:
        print("Warning: No pretrained weights loaded for SHViT")
    
    model = model.to(device)
    model.eval()
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"SHViT parameters: {num_params:.2f}M")
    
    return model


def load_vit_model(model_name, device='cuda'):
    """Load ViT baseline model from timm"""
    print(f"\nLoading ViT model: {model_name}")
    
    try:
        model = timm.create_model(model_name, pretrained=True, num_classes=1000)
    except:
        print(f"Warning: Could not load pretrained {model_name}, using random weights")
        model = timm.create_model(model_name, pretrained=False, num_classes=1000)
    
    model = model.to(device)
    model.eval()
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"ViT parameters: {num_params:.2f}M")
    
    return model


def get_dataloader(data_path, batch_size=32, num_workers=4, subset_size=None):
    """Create ImageNet validation dataloader"""
    print(f"\nPreparing dataset from: {data_path}")
    
    # ImageNet normalization
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        # Note: We apply normalization AFTER attacks, so images stay in [0,1]
        # normalize,
    ])
    
    # Load dataset
    dataset = torchvision.datasets.ImageFolder(
        os.path.join(data_path, 'val'),
        transform=transform
    )
    
    # Create subset if requested
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


def evaluate_model_on_attack(model, model_name, dataloader, attack_fn, attack_kwargs, 
                             attn_hooks=None, device='cuda'):
    """Evaluate a single model on a single attack configuration"""
    print(f"\n{'='*70}")
    print(f"Evaluating {model_name}")
    print(f"Attack: {attack_fn.__name__.upper()}")
    print(f"Parameters: {attack_kwargs}")
    print(f"{'='*70}")
    
    # Register attention hooks if using guided attacks
    if attack_kwargs.get('attn_guided', False) and attn_hooks:
        attn_hooks.register(model)
    
    # Run evaluation
    metrics = evaluate_robustness(
        model=model,
        dataloader=dataloader,
        attack_fn=attack_fn,
        attack_kwargs=attack_kwargs,
        device=device
    )
    
    # Remove hooks
    if attn_hooks:
        attn_hooks.remove()
    
    return metrics


def main():
    args = get_args()
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"Gap 1: Single-Head Attention Vulnerability Evaluation")
    print(f"Output directory: {run_dir}")
    print(f"{'='*70}")
    
    # Check GPU availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        args.device = 'cpu'
    
    if args.device == 'cuda':
        print(f"\nGPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Load models
    shvit_model = load_shvit_model(args.shvit_model, args.shvit_weights, args.device)
    vit_model = load_vit_model(args.vit_model, args.device)
    
    # Load data
    dataloader = get_dataloader(
        args.data_path, 
        args.batch_size, 
        args.num_workers,
        args.subset_size
    )
    
    # Initialize attention hooks
    attn_hooks_shvit = AttackHooks(device=args.device)
    attn_hooks_vit = AttackHooks(device=args.device)
    
    # Results storage
    all_results = {
        'args': vars(args),
        'timestamp': timestamp,
        'shvit_results': {},
        'vit_results': {},
        'comparison': {}
    }
    
    # Evaluate across all epsilon values
    for eps in args.epsilon:
        print(f"\n{'#'*70}")
        print(f"# Epsilon: {eps:.5f} ({eps*255:.2f}/255)")
        print(f"{'#'*70}")
        
        all_results['shvit_results'][eps] = {}
        all_results['vit_results'][eps] = {}
        
        # FGSM Evaluation
        if args.attack in ['fgsm', 'both']:
            print(f"\n{'='*70}")
            print(f"FGSM ATTACK (ε={eps:.5f})")
            print(f"{'='*70}")
            
            fgsm_kwargs = {
                'epsilon': eps,
                'attn_guided': args.attn_guided,
                'attn_hooks': attn_hooks_shvit if args.attn_guided else None
            }
            
            # Evaluate SHViT
            shvit_fgsm = evaluate_model_on_attack(
                shvit_model, f"SHViT-{args.shvit_model}", dataloader,
                fgsm, fgsm_kwargs, attn_hooks_shvit, args.device
            )
            all_results['shvit_results'][eps]['fgsm'] = shvit_fgsm
            
            # Evaluate ViT
            fgsm_kwargs['attn_hooks'] = attn_hooks_vit if args.attn_guided else None
            vit_fgsm = evaluate_model_on_attack(
                vit_model, f"ViT-{args.vit_model}", dataloader,
                fgsm, fgsm_kwargs, attn_hooks_vit, args.device
            )
            all_results['vit_results'][eps]['fgsm'] = vit_fgsm
        
        # PGD Evaluation
        if args.attack in ['pgd', 'both']:
            print(f"\n{'='*70}")
            print(f"PGD ATTACK (ε={eps:.5f}, steps={args.pgd_steps})")
            print(f"{'='*70}")
            
            pgd_kwargs = {
                'epsilon': eps,
                'steps': args.pgd_steps,
                'alpha': args.pgd_alpha if args.pgd_alpha else eps / 4,
                'attn_guided': args.attn_guided,
                'attn_hooks': attn_hooks_shvit if args.attn_guided else None
            }
            
            # Evaluate SHViT
            shvit_pgd = evaluate_model_on_attack(
                shvit_model, f"SHViT-{args.shvit_model}", dataloader,
                pgd, pgd_kwargs, attn_hooks_shvit, args.device
            )
            all_results['shvit_results'][eps]['pgd'] = shvit_pgd
            
            # Evaluate ViT
            pgd_kwargs['attn_hooks'] = attn_hooks_vit if args.attn_guided else None
            vit_pgd = evaluate_model_on_attack(
                vit_model, f"ViT-{args.vit_model}", dataloader,
                pgd, pgd_kwargs, attn_hooks_vit, args.device
            )
            all_results['vit_results'][eps]['pgd'] = vit_pgd
    
    # Compute comparisons
    print(f"\n{'='*70}")
    print(f"COMPARATIVE ANALYSIS")
    print(f"{'='*70}")
    
    for eps in args.epsilon:
        print(f"\n--- Epsilon: {eps:.5f} ---")
        all_results['comparison'][eps] = {}
        
        for attack_type in ['fgsm', 'pgd']:
            if attack_type in all_results['shvit_results'][eps]:
                shvit_asr = all_results['shvit_results'][eps][attack_type]['attack_success_rate']
                vit_asr = all_results['vit_results'][eps][attack_type]['attack_success_rate']
                
                vulnerability_gap = (shvit_asr - vit_asr) * 100  # percentage points
                
                print(f"\n{attack_type.upper()}:")
                print(f"  SHViT ASR:        {shvit_asr:6.2%}")
                print(f"  ViT ASR:          {vit_asr:6.2%}")
                print(f"  Vulnerability Gap: {vulnerability_gap:+.2f} pp")
                
                all_results['comparison'][eps][attack_type] = {
                    'shvit_asr': shvit_asr,
                    'vit_asr': vit_asr,
                    'vulnerability_gap_pp': vulnerability_gap
                }
    
    # Save results
    results_path = run_dir / 'results.json'
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to: {results_path}")
    
    # Save summary
    summary_path = run_dir / 'summary.txt'
    with open(summary_path, 'w') as f:
        f.write(f"Gap 1: Single-Head Attention Vulnerability\n")
        f.write(f"{'='*70}\n\n")
        f.write(f"Models:\n")
        f.write(f"  SHViT: {args.shvit_model}\n")
        f.write(f"  ViT:   {args.vit_model}\n\n")
        
        for eps in args.epsilon:
            f.write(f"\nEpsilon: {eps:.5f}\n")
            f.write(f"-" * 50 + "\n")
            
            for attack_type in ['fgsm', 'pgd']:
                if attack_type in all_results['comparison'].get(eps, {}):
                    comp = all_results['comparison'][eps][attack_type]
                    f.write(f"{attack_type.upper()}:\n")
                    f.write(f"  SHViT ASR: {comp['shvit_asr']:.2%}\n")
                    f.write(f"  ViT ASR:   {comp['vit_asr']:.2%}\n")
                    f.write(f"  Gap:       {comp['vulnerability_gap_pp']:+.2f} pp\n\n")
    
    print(f"Summary saved to: {summary_path}")
    
    print(f"\n{'='*70}")
    print(f"EVALUATION COMPLETE")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()

