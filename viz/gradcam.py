"""
Grad-CAM implementation for SHViT
Gradient-weighted Class Activation Mapping for single-head transformers
GPU-accelerated implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List
from PIL import Image


class GradCAM:
    """Grad-CAM for SHViT feature maps - GPU accelerated"""
    
    def __init__(self, model, target_layer_name: str = None, device: str = 'cuda'):
        """
        Initialize Grad-CAM
        
        Args:
            model: SHViT model
            target_layer_name: Name pattern to match target layer (default: last stage)
            device: 'cuda' or 'cpu'
        """
        self.model = model.to(device)
        self.device = device
        self.target_layer_name = target_layer_name
        self.gradients = None
        self.activations = None
        self.hooks = []
        self.target_layer = None
    
    def _find_target_layer(self):
        """Find the target layer for Grad-CAM"""
        if self.target_layer_name:
            # User-specified layer
            for name, module in self.model.named_modules():
                if self.target_layer_name in name:
                    return name, module
        
        # Default: find last convolutional or last SHSA block
        last_conv = None
        last_shsa = None
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                last_conv = (name, module)
            if 'SHSA' in type(module).__name__ or 'shsa' in name.lower():
                last_shsa = (name, module)
        
        # Prefer last SHSA block, fall back to last conv
        target = last_shsa if last_shsa else last_conv
        
        if target:
            print(f"Grad-CAM target layer: {target[0]}")
            return target
        else:
            raise ValueError("Could not find suitable target layer for Grad-CAM")
    
    def register_hooks(self):
        """Register forward and backward hooks on target layer"""
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        layer_name, layer_module = self._find_target_layer()
        self.target_layer = layer_name
        
        self.hooks.append(layer_module.register_forward_hook(forward_hook))
        self.hooks.append(layer_module.register_full_backward_hook(backward_hook))
        
        print(f"Registered Grad-CAM hooks on: {layer_name}")
    
    def remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def generate_cam(self, 
                     input_image: torch.Tensor, 
                     target_class: Optional[int] = None,
                     use_relu: bool = True) -> np.ndarray:
        """
        Generate Grad-CAM heatmap - GPU accelerated
        
        Args:
            input_image: Input tensor (1, 3, H, W) on device
            target_class: Target class index (None = predicted class)
            use_relu: Apply ReLU to CAM (standard practice)
        
        Returns:
            cam: Grad-CAM heatmap (H, W) in [0, 1]
        """
        self.model.eval()
        input_image = input_image.to(self.device)
        input_image.requires_grad_(True)
        
        # Forward pass
        output = self.model(input_image)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass for target class
        self.model.zero_grad()
        class_score = output[0, target_class]
        class_score.backward()
        
        # Get gradients and activations
        gradients = self.gradients[0]  # (C, H, W)
        activations = self.activations[0]  # (C, H, W)
        
        # Global average pooling of gradients (weights)
        weights = gradients.mean(dim=(1, 2))  # (C,)
        
        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[1:], device=self.device)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # Apply ReLU
        if use_relu:
            cam = F.relu(cam)
        
        # Move to CPU and convert to numpy
        cam = cam.cpu().numpy()
        
        # Normalize to [0, 1]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam
    
    def overlay_cam_on_image(self, 
                             image: np.ndarray, 
                             cam: np.ndarray, 
                             alpha: float = 0.5,
                             colormap: str = 'jet') -> np.ndarray:
        """
        Overlay CAM heatmap on image
        
        Args:
            image: Original image (H, W, 3) in [0, 1]
            cam: CAM heatmap (h, w)
            alpha: Blending factor (0=only image, 1=only CAM)
            colormap: Matplotlib colormap name
        
        Returns:
            overlayed: Blended visualization (H, W, 3)
        """
        # Resize CAM to image size
        cam_resized = np.array(Image.fromarray(
            (cam * 255).astype(np.uint8)
        ).resize((image.shape[1], image.shape[0]), Image.BILINEAR)) / 255.0
        
        # Apply colormap
        import matplotlib.cm as cm
        cmap = cm.get_cmap(colormap)
        heatmap = cmap(cam_resized)[:, :, :3]
        
        # Ensure image is in correct range
        if image.max() > 1.0:
            image = image / 255.0
        
        # Blend
        overlayed = alpha * heatmap + (1 - alpha) * image
        overlayed = np.clip(overlayed, 0, 1)
        
        return overlayed
    
    def visualize(self,
                  image: torch.Tensor,
                  target_class: Optional[int] = None,
                  save_path: Optional[str] = None,
                  show_prediction: bool = True):
        """
        Create complete Grad-CAM visualization
        
        Args:
            image: Input image (1, 3, H, W)
            target_class: Target class for CAM
            save_path: Where to save figure
            show_prediction: Display model prediction
        """
        # Generate CAM
        cam = self.generate_cam(image, target_class)
        
        # Get prediction
        with torch.no_grad():
            output = self.model(image.to(self.device))
            pred_class = output.argmax(dim=1).item()
            confidence = F.softmax(output, dim=1)[0, pred_class].item()
        
        # Prepare image for display
        img_np = image[0].permute(1, 2, 0).cpu().numpy()
        img_np = np.clip(img_np, 0, 1)
        
        # Create overlay
        overlay = self.overlay_cam_on_image(img_np, cam, alpha=0.5)
        
        # Plot
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(img_np)
        axes[0].set_title('Original Image', fontsize=12)
        axes[0].axis('off')
        
        # Grad-CAM heatmap
        im = axes[1].imshow(cam, cmap='jet')
        axes[1].set_title(f'Grad-CAM Heatmap\nTarget Layer: {self.target_layer}', 
                         fontsize=12)
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], fraction=0.046)
        
        # Overlay
        axes[2].imshow(overlay)
        if show_prediction:
            title = f'Grad-CAM Overlay\nPredicted: Class {pred_class} ({confidence:.2%})'
        else:
            title = 'Grad-CAM Overlay'
        axes[2].set_title(title, fontsize=12)
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved Grad-CAM visualization to: {save_path}")
        
        return fig, cam
    
    def compare_clean_vs_adversarial(self,
                                     image_clean: torch.Tensor,
                                     image_adv: torch.Tensor,
                                     true_label: int,
                                     save_path: Optional[str] = None):
        """
        Compare Grad-CAM for clean vs adversarial images
        
        Args:
            image_clean: Clean image (1, 3, H, W)
            image_adv: Adversarial image (1, 3, H, W)
            true_label: Ground truth label
            save_path: Where to save figure
        """
        # Generate CAMs
        cam_clean = self.generate_cam(image_clean, target_class=true_label)
        cam_adv = self.generate_cam(image_adv, target_class=true_label)
        
        # Get predictions
        with torch.no_grad():
            out_clean = self.model(image_clean.to(self.device))
            out_adv = self.model(image_adv.to(self.device))
            pred_clean = out_clean.argmax(dim=1).item()
            pred_adv = out_adv.argmax(dim=1).item()
            conf_clean = F.softmax(out_clean, dim=1)[0, pred_clean].item()
            conf_adv = F.softmax(out_adv, dim=1)[0, pred_adv].item()
        
        # Prepare images
        img_clean = image_clean[0].permute(1, 2, 0).cpu().numpy()
        img_adv = image_adv[0].permute(1, 2, 0).cpu().numpy()
        img_clean = np.clip(img_clean, 0, 1)
        img_adv = np.clip(img_adv, 0, 1)
        
        # Create overlays
        overlay_clean = self.overlay_cam_on_image(img_clean, cam_clean, alpha=0.5)
        overlay_adv = self.overlay_cam_on_image(img_adv, cam_adv, alpha=0.5)
        
        # Plot comparison
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Grad-CAM: Clean vs Adversarial (True Label: {true_label})', 
                    fontsize=16, fontweight='bold')
        
        # Row 1: Clean
        axes[0, 0].imshow(img_clean)
        axes[0, 0].set_title(f'Clean Image\nPred: {pred_clean} ({conf_clean:.2%})', 
                            fontsize=11)
        axes[0, 0].axis('off')
        
        im1 = axes[0, 1].imshow(cam_clean, cmap='jet')
        axes[0, 1].set_title('Grad-CAM (Clean)', fontsize=11)
        axes[0, 1].axis('off')
        plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)
        
        axes[0, 2].imshow(overlay_clean)
        axes[0, 2].set_title('Overlay (Clean)', fontsize=11)
        axes[0, 2].axis('off')
        
        # Row 2: Adversarial
        axes[1, 0].imshow(img_adv)
        axes[1, 0].set_title(f'Adversarial Image\nPred: {pred_adv} ({conf_adv:.2%})', 
                            fontsize=11)
        axes[1, 0].axis('off')
        
        im2 = axes[1, 1].imshow(cam_adv, cmap='jet')
        axes[1, 1].set_title('Grad-CAM (Adversarial)', fontsize=11)
        axes[1, 1].axis('off')
        plt.colorbar(im2, ax=axes[1, 1], fraction=0.046)
        
        axes[1, 2].imshow(overlay_adv)
        axes[1, 2].set_title('Overlay (Adversarial)', fontsize=11)
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved comparison to: {save_path}")
        
        return fig

