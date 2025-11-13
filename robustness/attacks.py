"""
Adversarial attack implementations for SHViT robustness evaluation
GPU-accelerated FGSM, PGD, and attention-guided variants
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
import time
import numpy as np


class AttackHooks:
    """Collect attention maps from SHSA modules during forward pass"""
    
    def __init__(self, device='cuda'):
        self.attention_maps = []
        self.hooks = []
        self.device = device
    
    def register(self, model):
        """Register hooks on all SHSA modules"""
        def hook_fn(module, input, output):
            """Extract attention matrix from SHSA forward pass"""
            if hasattr(module, '_attn_cache'):
                self.attention_maps.append(module._attn_cache.detach())
        
        for name, module in model.named_modules():
            # Check for SHSA module
            if 'SHSA' in type(module).__name__ or 'shsa' in name.lower():
                hook = module.register_forward_hook(hook_fn)
                self.hooks.append(hook)
                print(f"Registered hook on: {name} ({type(module).__name__})")
        
        if len(self.hooks) == 0:
            print("Warning: No SHSA modules found for hooking!")
    
    def clear(self):
        """Clear cached attention maps"""
        self.attention_maps = []
    
    def remove(self):
        """Remove all hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def get_attention_mask(self, aggregate='mean'):
        """
        Get aggregated spatial attention mask
        
        Args:
            aggregate: 'mean' or 'last' - how to combine attention maps
        
        Returns:
            mask: (B, num_patches) attention mask
        """
        if not self.attention_maps:
            return None
        
        if aggregate == 'last':
            attn = self.attention_maps[-1]
        else:  # mean
            # Attention maps may have different spatial sizes across stages
            # Use only the last stage or largest resolution
            attn = self.attention_maps[-1]  # Use last stage attention
        
        # Average across query dimension to get per-token importance
        mask = attn.mean(dim=1)  # (B, HW)
        return mask


def fgsm(model: nn.Module, 
         x: torch.Tensor, 
         y: torch.Tensor, 
         epsilon: float,
         targeted: bool = False,
         attn_guided: bool = False,
         attn_hooks: Optional[AttackHooks] = None,
         device: str = 'cuda') -> Tuple[torch.Tensor, Dict]:
    """
    Fast Gradient Sign Method (FGSM) attack - GPU accelerated
    
    Args:
        model: Target model
        x: Input images (B, C, H, W) on device
        y: True labels (B,) on device
        epsilon: Perturbation budget (L∞)
        targeted: If True, minimize loss; otherwise maximize
        attn_guided: Use attention maps to guide perturbation
        attn_hooks: AttackHooks instance for attention guidance
        device: 'cuda' or 'cpu'
    
    Returns:
        x_adv: Adversarial examples
        info: Attack statistics
    """
    model.eval()
    model.to(device)
    x = x.to(device)
    y = y.to(device)
    
    x_adv = x.clone().detach().requires_grad_(True)
    
    # Synchronize for accurate timing
    if device == 'cuda':
        torch.cuda.synchronize()
    start_time = time.time()
    
    # Clear attention cache if using guidance
    if attn_guided and attn_hooks:
        attn_hooks.clear()
    
    # Forward pass
    outputs = model(x_adv)
    loss = F.cross_entropy(outputs, y)
    
    # Backward pass
    model.zero_grad()
    loss.backward()
    grad = x_adv.grad.data
    
    # Apply attention mask if provided
    if attn_guided and attn_hooks:
        attn_mask = attn_hooks.get_attention_mask(aggregate='last')
        if attn_mask is not None:
            # Reshape attention mask to spatial dimensions
            B, HW = attn_mask.shape
            H = W = int(np.sqrt(HW))
            spatial_mask = attn_mask.view(B, 1, H, W)
            
            # Upsample mask to image size
            mask = F.interpolate(spatial_mask, 
                                size=x.shape[2:], 
                                mode='bilinear', 
                                align_corners=False)
            # Apply mask to gradients
            grad = grad * mask
    
    # FGSM update
    sign_grad = grad.sign()
    if targeted:
        x_adv = x_adv - epsilon * sign_grad
    else:
        x_adv = x_adv + epsilon * sign_grad
    
    # Clamp to valid range [0, 1]
    x_adv = torch.clamp(x_adv, 0, 1)
    
    if device == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.time() - start_time
    
    # Compute statistics
    with torch.no_grad():
        outputs_adv = model(x_adv)
        pred_clean = outputs.argmax(dim=1)
        pred_adv = outputs_adv.argmax(dim=1)
        success = (pred_adv != y).float().mean().item()
        
        # Perturbation norms
        delta = x_adv - x
        l2_norm = delta.norm(p=2, dim=(1,2,3)).mean().item()
        linf_norm = delta.abs().max().item()
    
    info = {
        'epsilon': epsilon,
        'attack_success_rate': success,
        'time_per_image': elapsed / len(x),
        'l2_norm': l2_norm,
        'linf_norm': linf_norm,
        'device': device
    }
    
    return x_adv.detach(), info


def pgd(model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        epsilon: float,
        steps: int = 10,
        alpha: Optional[float] = None,
        restarts: int = 1,
        attn_guided: bool = False,
        attn_hooks: Optional[AttackHooks] = None,
        time_budget: Optional[float] = None,
        device: str = 'cuda') -> Tuple[torch.Tensor, Dict]:
    """
    Projected Gradient Descent (PGD) attack - GPU accelerated with time budgeting
    
    Args:
        model: Target model
        x: Input images (B, C, H, W)
        y: True labels (B,)
        epsilon: Perturbation budget (L∞)
        steps: Number of PGD iterations
        alpha: Step size (default: epsilon/4)
        restarts: Number of random restarts
        attn_guided: Use attention to guide perturbation
        attn_hooks: AttackHooks instance
        time_budget: Max wall-clock seconds per batch (adaptive steps)
        device: 'cuda' or 'cpu'
    
    Returns:
        x_adv: Adversarial examples
        info: Attack statistics
    """
    model.eval()
    model.to(device)
    x = x.to(device)
    y = y.to(device)
    
    if alpha is None:
        alpha = epsilon / 4
    
    if device == 'cuda':
        torch.cuda.synchronize()
    start_time = time.time()
    
    # Track best adversarial examples across restarts
    best_loss = torch.ones(len(x), device=device) * -float('inf')
    best_adv = x.clone()
    
    actual_steps = 0
    
    for restart in range(restarts):
        # Random initialization
        delta = torch.zeros_like(x).uniform_(-epsilon, epsilon)
        delta.data = torch.clamp(x + delta.data, 0, 1) - x
        delta.requires_grad_(True)
        
        for step in range(steps):
            # Check time budget
            if time_budget:
                if device == 'cuda':
                    torch.cuda.synchronize()
                if (time.time() - start_time) > time_budget:
                    break
            
            # Clear attention cache
            if attn_guided and attn_hooks:
                attn_hooks.clear()
            
            x_adv = x + delta
            
            # Forward pass
            outputs = model(x_adv)
            loss = F.cross_entropy(outputs, y, reduction='none')
            
            # Backward pass
            model.zero_grad()
            loss.sum().backward()
            grad = delta.grad.data
            
            # Apply attention guidance
            if attn_guided and attn_hooks:
                attn_mask = attn_hooks.get_attention_mask(aggregate='mean')
                if attn_mask is not None:
                    B, HW = attn_mask.shape
                    H = W = int(np.sqrt(HW))
                    spatial_mask = attn_mask.view(B, 1, H, W)
                    mask = F.interpolate(spatial_mask, 
                                        size=x.shape[2:], 
                                        mode='bilinear', 
                                        align_corners=False)
                    grad = grad * mask
            
            # PGD update
            delta.data = delta.data + alpha * grad.sign()
            delta.data = torch.clamp(delta.data, -epsilon, epsilon)
            delta.data = torch.clamp(x + delta.data, 0, 1) - x
            
            # Track best adversarial examples (highest loss)
            with torch.no_grad():
                improved = loss > best_loss
                best_loss = torch.where(improved, loss, best_loss)
                best_adv = torch.where(improved.view(-1,1,1,1), x_adv.detach(), best_adv)
            
            delta.grad.zero_()
            actual_steps += 1
    
    if device == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.time() - start_time
    
    # Compute final statistics
    with torch.no_grad():
        outputs_clean = model(x)
        outputs_adv = model(best_adv)
        pred_clean = outputs_clean.argmax(dim=1)
        pred_adv = outputs_adv.argmax(dim=1)
        success = (pred_adv != y).float().mean().item()
        
        # Perturbation norms
        delta = best_adv - x
        l2_norm = delta.norm(p=2, dim=(1,2,3)).mean().item()
        linf_norm = delta.abs().max().item()
    
    info = {
        'epsilon': epsilon,
        'steps': steps,
        'actual_steps': actual_steps,
        'alpha': alpha,
        'restarts': restarts,
        'attack_success_rate': success,
        'time_per_image': elapsed / len(x),
        'perturbation_quality_per_second': success / (elapsed / len(x)) if elapsed > 0 else 0,
        'l2_norm': l2_norm,
        'linf_norm': linf_norm,
        'device': device
    }
    
    return best_adv, info


def evaluate_robustness(model: nn.Module,
                       dataloader,
                       attack_fn,
                       attack_kwargs: Dict,
                       device: str = 'cuda',
                       max_batches: Optional[int] = None) -> Dict:
    """
    Evaluate model robustness on a dataset - GPU accelerated
    
    Args:
        model: Model to evaluate
        dataloader: Test data loader
        attack_fn: Attack function (fgsm or pgd)
        attack_kwargs: Attack parameters
        device: Device to run on ('cuda' or 'cpu')
        max_batches: Limit evaluation to N batches (for quick tests)
    
    Returns:
        metrics: Robustness metrics dictionary
    """
    model.eval()
    model.to(device)
    
    total_samples = 0
    total_correct_clean = 0
    total_correct_adv = 0
    total_time = 0
    
    all_info = []
    
    print(f"\n{'='*70}")
    print(f"Robustness Evaluation on {device.upper()}")
    print(f"{'='*70}")
    
    for batch_idx, (x, y) in enumerate(dataloader):
        if max_batches and batch_idx >= max_batches:
            break
        
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
        
        # Progress update
        if batch_idx % 10 == 0:
            clean_acc = total_correct_clean / total_samples
            robust_acc = total_correct_adv / total_samples
            asr = 1 - robust_acc
            avg_time = total_time / total_samples
            
            print(f"Batch {batch_idx:3d} | "
                  f"Clean Acc: {clean_acc:6.2%} | "
                  f"Robust Acc: {robust_acc:6.2%} | "
                  f"ASR: {asr:6.2%} | "
                  f"Time: {avg_time*1000:.1f}ms/img")
    
    # Final metrics
    clean_accuracy = total_correct_clean / total_samples
    robust_accuracy = total_correct_adv / total_samples
    attack_success_rate = 1 - robust_accuracy
    avg_time_per_image = total_time / total_samples
    
    # Aggregate info from all batches
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
    
    print(f"\n{'='*70}")
    print(f"FINAL RESULTS:")
    print(f"{'='*70}")
    print(f"Clean Accuracy:   {clean_accuracy:6.2%}")
    print(f"Robust Accuracy:  {robust_accuracy:6.2%}")
    print(f"Attack Success:   {attack_success_rate:6.2%}")
    print(f"Avg Time/Image:   {avg_time_per_image*1000:.2f}ms")
    print(f"Avg L2 Norm:      {avg_l2:.4f}")
    print(f"Avg L∞ Norm:      {avg_linf:.4f}")
    print(f"Total Samples:    {total_samples}")
    print(f"{'='*70}\n")
    
    return metrics

