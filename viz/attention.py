"""
Attention visualization utilities for single-head transformers
Token Activation Heatmaps (TAH) and Attention Stability Maps (ASM)
GPU-accelerated visualization
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from typing import List, Tuple, Optional, Dict
import warnings
from PIL import Image


class AttentionVisualizer:
    """Visualize attention maps from SHViT's single-head attention - GPU accelerated"""
    
    def __init__(self, model, patch_size: int = 16, device: str = 'cuda'):
        """
        Initialize attention visualizer
        
        Args:
            model: SHViT model
            patch_size: Patch size for ViT (usually 16)
            device: 'cuda' or 'cpu'
        """
        self.model = model.to(device)
        self.patch_size = patch_size
        self.device = device
        self.attention_maps = []
        self.hooks = []
        self.block_names = []
    
    def register_hooks(self):
        """Register forward hooks to capture attention maps from all SHSA modules"""
        def hook_fn(name):
            def fn(module, input, output):
                # Capture attention from SHSA modules
                if hasattr(module, '_attn_cache'):
                    self.attention_maps.append({
                        'attn': module._attn_cache.detach().cpu(),
                        'module_name': name,
                        'block_idx': len(self.attention_maps)
                    })
            return fn
        
        for name, module in self.model.named_modules():
            if 'SHSA' in type(module).__name__ or 'shsa' in name.lower():
                hook = module.register_forward_hook(hook_fn(name))
                self.hooks.append(hook)
                self.block_names.append(name)
                print(f"Registered visualization hook on: {name}")
        
        if len(self.hooks) == 0:
            warnings.warn("No SHSA modules found for visualization!")
    
    def remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def clear_cache(self):
        """Clear cached attention maps"""
        self.attention_maps = []
    
    def token_activation_heatmap(self, 
                                 image: torch.Tensor,
                                 block_idx: int = -1,
                                 aggregate: str = 'mean',
                                 top_k: Optional[int] = None) -> np.ndarray:
        """
        Create Token Activation Heatmap (TAH)
        Shows which image patches the attention head focuses on
        
        Args:
            image: Input image (1, 3, H, W)
            block_idx: Which SHSA block to visualize (-1 for last)
            aggregate: 'mean' or 'max' - how to aggregate attention scores
            top_k: If set, only show top-k attended tokens
        
        Returns:
            heatmap: (H, W) attention heatmap in [0, 1]
        """
        if not self.attention_maps:
            warnings.warn("No attention maps captured. Run forward pass first.")
            return None
        
        attn_data = self.attention_maps[block_idx]
        attn = attn_data['attn'][0]  # (HW, HW)
        
        # Aggregate attention received by each token
        if aggregate == 'mean':
            attn_scores = attn.mean(dim=0)  # (HW,)
        elif aggregate == 'max':
            attn_scores = attn.max(dim=0)[0]  # (HW,)
        else:
            attn_scores = attn.mean(dim=0)
        
        # Top-k filtering
        if top_k is not None:
            threshold = attn_scores.topk(top_k)[0][-1]
            attn_scores = torch.where(attn_scores >= threshold, 
                                     attn_scores, 
                                     torch.zeros_like(attn_scores))
        
        # Reshape to spatial grid
        num_patches = int(np.sqrt(attn_scores.shape[0]))
        heatmap = attn_scores.reshape(num_patches, num_patches).numpy()
        
        # Normalize to [0, 1]
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        
        return heatmap
    
    def attention_stability_map(self,
                               image_clean: torch.Tensor,
                               image_adv: torch.Tensor,
                               block_idx: int = -1,
                               metric: str = 'l1') -> Tuple[np.ndarray, float]:
        """
        Compute Attention Stability Map (ASM)
        Compares attention patterns between clean and adversarial inputs
        
        Args:
            image_clean: Clean image (1, 3, H, W)
            image_adv: Adversarial image (1, 3, H, W)
            block_idx: Which block to analyze (-1 for last)
            metric: 'l1', 'l2', or 'jsd' (Jensen-Shannon divergence)
        
        Returns:
            diff_map: (H, W) spatial difference map
            stability_score: Overall stability metric (lower = more stable)
        """
        # Move images to device
        image_clean = image_clean.to(self.device)
        image_adv = image_adv.to(self.device)
        
        # Get attention for clean image
        self.clear_cache()
        with torch.no_grad():
            _ = self.model(image_clean)
        attn_clean = self.attention_maps[block_idx]['attn'][0]  # (HW, HW)
        
        # Get attention for adversarial image
        self.clear_cache()
        with torch.no_grad():
            _ = self.model(image_adv)
        attn_adv = self.attention_maps[block_idx]['attn'][0]  # (HW, HW)
        
        # Compute difference
        if metric == 'l1':
            diff = torch.abs(attn_clean - attn_adv).mean(dim=0)
            stability_score = diff.mean().item()
        elif metric == 'l2':
            diff = ((attn_clean - attn_adv) ** 2).mean(dim=0).sqrt()
            stability_score = diff.mean().item()
        elif metric == 'jsd':
            # Jensen-Shannon Divergence per token
            attn_clean_np = attn_clean.mean(dim=0).numpy()
            attn_adv_np = attn_adv.mean(dim=0).numpy()
            
            # Normalize to probability distributions
            attn_clean_np = attn_clean_np / (attn_clean_np.sum() + 1e-8)
            attn_adv_np = attn_adv_np / (attn_adv_np.sum() + 1e-8)
            
            # JSD
            m = (attn_clean_np + attn_adv_np) / 2
            jsd = 0.5 * (np.sum(attn_clean_np * np.log(attn_clean_np / (m + 1e-8) + 1e-8)) +
                        np.sum(attn_adv_np * np.log(attn_adv_np / (m + 1e-8) + 1e-8)))
            stability_score = jsd
            diff = torch.from_numpy(np.abs(attn_clean_np - attn_adv_np))
        else:
            diff = torch.abs(attn_clean - attn_adv).mean(dim=0)
            stability_score = diff.mean().item()
        
        # Reshape to spatial grid
        num_patches = int(np.sqrt(diff.shape[0]))
        diff_map = diff.reshape(num_patches, num_patches).numpy()
        
        # Normalize for visualization
        diff_map = (diff_map - diff_map.min()) / (diff_map.max() - diff_map.min() + 1e-8)
        
        return diff_map, stability_score
    
    def plot_comparison_grid(self,
                            image_clean: torch.Tensor,
                            image_adv: torch.Tensor,
                            label_clean: int,
                            pred_clean: int,
                            pred_adv: int,
                            epsilon: float,
                            save_path: Optional[str] = None,
                            block_idx: int = -1):
        """
        Create comprehensive visualization grid:
        Row 1: [Clean Image, Adversarial Image, Perturbation (amplified)]
        Row 2: [TAH Clean, TAH Adversarial, Attention Stability Map]
        
        Args:
            image_clean: Clean image (1, 3, H, W)
            image_adv: Adversarial image (1, 3, H, W)
            label_clean: Ground truth label
            pred_clean: Clean prediction
            pred_adv: Adversarial prediction
            epsilon: Attack epsilon
            save_path: Where to save figure
            block_idx: Which attention block to visualize
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Adversarial Robustness Analysis (ε={epsilon:.3f})', 
                    fontsize=16, fontweight='bold')
        
        # Convert images to numpy for display (denormalize if needed)
        img_clean = image_clean[0].permute(1, 2, 0).cpu().numpy()
        img_adv = image_adv[0].permute(1, 2, 0).cpu().numpy()
        
        # Ensure images are in [0, 1]
        img_clean = np.clip(img_clean, 0, 1)
        img_adv = np.clip(img_adv, 0, 1)
        
        # Row 1: Images
        axes[0, 0].imshow(img_clean)
        axes[0, 0].set_title(f'Clean Image\nTrue: {label_clean}, Pred: {pred_clean}', 
                            fontsize=11)
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(img_adv)
        axes[0, 1].set_title(f'Adversarial Image\nPred: {pred_adv}', 
                            fontsize=11)
        axes[0, 1].axis('off')
        
        # Perturbation visualization (amplified)
        diff = np.abs(img_clean - img_adv)
        perturbation = diff / (diff.max() + 1e-8)  # Normalize
        axes[0, 2].imshow(perturbation, cmap='hot')
        axes[0, 2].set_title(f'Perturbation\nL∞: {diff.max():.4f}', 
                            fontsize=11)
        axes[0, 2].axis('off')
        
        # Row 2: Attention heatmaps
        # TAH for clean image
        self.clear_cache()
        with torch.no_grad():
            _ = self.model(image_clean.to(self.device))
        heatmap_clean = self.token_activation_heatmap(image_clean, block_idx=block_idx)
        
        if heatmap_clean is not None:
            im1 = axes[1, 0].imshow(heatmap_clean, cmap='hot', interpolation='bilinear')
            axes[1, 0].set_title('Token Activation\nHeatmap (Clean)', fontsize=11)
            axes[1, 0].axis('off')
            plt.colorbar(im1, ax=axes[1, 0], fraction=0.046)
        
        # TAH for adversarial image
        self.clear_cache()
        with torch.no_grad():
            _ = self.model(image_adv.to(self.device))
        heatmap_adv = self.token_activation_heatmap(image_adv, block_idx=block_idx)
        
        if heatmap_adv is not None:
            im2 = axes[1, 1].imshow(heatmap_adv, cmap='hot', interpolation='bilinear')
            axes[1, 1].set_title('Token Activation\nHeatmap (Adversarial)', fontsize=11)
            axes[1, 1].axis('off')
            plt.colorbar(im2, ax=axes[1, 1], fraction=0.046)
        
        # Attention Stability Map
        diff_map, stability_score = self.attention_stability_map(
            image_clean, image_adv, block_idx=block_idx, metric='l1'
        )
        
        im3 = axes[1, 2].imshow(diff_map, cmap='RdYlGn_r', interpolation='bilinear')
        axes[1, 2].set_title(f'Attention Stability Map\nScore: {stability_score:.4f}', 
                            fontsize=11)
        axes[1, 2].axis('off')
        plt.colorbar(im3, ax=axes[1, 2], fraction=0.046)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved visualization to: {save_path}")
        
        return fig
    
    def visualize_all_blocks(self,
                            image: torch.Tensor,
                            save_path: Optional[str] = None):
        """
        Visualize attention heatmaps from all SHSA blocks
        
        Args:
            image: Input image (1, 3, H, W)
            save_path: Where to save figure
        """
        self.clear_cache()
        with torch.no_grad():
            _ = self.model(image.to(self.device))
        
        num_blocks = len(self.attention_maps)
        
        if num_blocks == 0:
            warnings.warn("No attention maps captured!")
            return None
        
        # Create grid
        cols = 4
        rows = (num_blocks + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
        axes = axes.flatten() if num_blocks > 1 else [axes]
        
        fig.suptitle('Attention Maps Across All SHSA Blocks', fontsize=16, fontweight='bold')
        
        for idx in range(num_blocks):
            heatmap = self.token_activation_heatmap(image, block_idx=idx)
            if heatmap is not None:
                im = axes[idx].imshow(heatmap, cmap='hot', interpolation='bilinear')
                axes[idx].set_title(f'Block {idx}\n{self.block_names[idx].split(".")[-1]}', 
                                  fontsize=10)
                axes[idx].axis('off')
                plt.colorbar(im, ax=axes[idx], fraction=0.046)
        
        # Hide unused subplots
        for idx in range(num_blocks, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved all-blocks visualization to: {save_path}")
        
        return fig

