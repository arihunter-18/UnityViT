"""
Quick test script to verify all infrastructure components work
Tests attacks, visualization, and Grad-CAM on synthetic data
"""

import torch
import torch.nn as nn
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from model.build import shvit_s4
from robustness.attacks import fgsm, pgd, AttackHooks
from viz.attention import AttentionVisualizer
from viz.gradcam import GradCAM


def test_model_loading(device='cuda'):
    """Test 1: Model loading and forward pass"""
    print("\n" + "="*70)
    print("TEST 1: Model Loading & Forward Pass")
    print("="*70)
    
    try:
        model = shvit_s4(pretrained=False, num_classes=1000)
        model = model.to(device)
        model.eval()
        
        # Test forward pass
        x = torch.randn(2, 3, 224, 224).to(device)
        with torch.no_grad():
            output = model(x)
        
        assert output.shape == (2, 1000), f"Expected (2, 1000), got {output.shape}"
        print("[OK] Model loaded successfully")
        print(f"[OK] Output shape: {output.shape}")
        print(f"[OK] Device: {device}")
        
        # Check attention caching
        has_cache = False
        for name, module in model.named_modules():
            if 'SHSA' in type(module).__name__:
                if hasattr(module, '_attn_cache'):
                    has_cache = True
                    cache_shape = module._attn_cache.shape
                    print(f"[OK] Attention cache found: {cache_shape}")
                    break
        
        if not has_cache:
            print("[WARNING] No attention cache found. Visualizations may not work.")
        
        return model
        
    except Exception as e:
        print(f"[FAILED] {e}")
        raise


def test_fgsm_attack(model, device='cuda'):
    """Test 2: FGSM attack"""
    print("\n" + "="*70)
    print("TEST 2: FGSM Attack")
    print("="*70)
    
    try:
        # Create synthetic data
        x = torch.randn(4, 3, 224, 224).to(device)
        y = torch.randint(0, 1000, (4,)).to(device)
        
        # Run FGSM
        x_adv, info = fgsm(model, x, y, epsilon=4/255, device=device)
        
        assert x_adv.shape == x.shape, f"Shape mismatch: {x_adv.shape} vs {x.shape}"
        assert 'attack_success_rate' in info, "Missing ASR in info"
        
        print(f"[OK] FGSM completed successfully")
        print(f"[OK] Attack Success Rate: {info['attack_success_rate']:.2%}")
        print(f"[OK] Time per image: {info['time_per_image']*1000:.2f}ms")
        print(f"[OK] Linf norm: {info['linf_norm']:.4f}")
        print(f"[OK] L2 norm: {info['l2_norm']:.4f}")
        
        return x_adv
        
    except Exception as e:
        print(f"[FAILED] {e}")
        raise


def test_pgd_attack(model, device='cuda'):
    """Test 3: PGD attack"""
    print("\n" + "="*70)
    print("TEST 3: PGD Attack")
    print("="*70)
    
    try:
        # Create synthetic data
        x = torch.randn(4, 3, 224, 224).to(device)
        y = torch.randint(0, 1000, (4,)).to(device)
        
        # Run PGD
        x_adv, info = pgd(model, x, y, epsilon=4/255, steps=10, device=device)
        
        assert x_adv.shape == x.shape, f"Shape mismatch: {x_adv.shape} vs {x.shape}"
        assert 'attack_success_rate' in info, "Missing ASR in info"
        assert 'perturbation_quality_per_second' in info, "Missing PQpS in info"
        
        print(f"[OK] PGD completed successfully")
        print(f"[OK] Attack Success Rate: {info['attack_success_rate']:.2%}")
        print(f"[OK] Time per image: {info['time_per_image']*1000:.2f}ms")
        print(f"[OK] PQpS: {info['perturbation_quality_per_second']:.2f}")
        print(f"[OK] Actual steps: {info['actual_steps']}")
        
        return x_adv
        
    except Exception as e:
        print(f"[FAILED] {e}")
        raise


def test_attention_hooks(model, device='cuda'):
    """Test 4: Attention hooks"""
    print("\n" + "="*70)
    print("TEST 4: Attention Hooks")
    print("="*70)
    
    try:
        hooks = AttackHooks(device=device)
        hooks.register(model)
        
        # Forward pass
        x = torch.randn(2, 3, 224, 224).to(device)
        with torch.no_grad():
            _ = model(x)
        
        # Check captured attention
        num_attn_maps = len(hooks.attention_maps)
        assert num_attn_maps > 0, "No attention maps captured"
        
        print(f"[OK] Captured {num_attn_maps} attention maps")
        
        # Test attention mask generation
        mask = hooks.get_attention_mask(aggregate='mean')
        if mask is not None:
            print(f"[OK] Attention mask shape: {mask.shape}")
        
        hooks.remove()
        print(f"[OK] Hooks removed successfully")
        
        return True
        
    except Exception as e:
        print(f"[FAILED] {e}")
        raise


def test_attention_guided_attack(model, device='cuda'):
    """Test 5: Attention-guided attack"""
    print("\n" + "="*70)
    print("TEST 5: Attention-Guided Attack")
    print("="*70)
    
    try:
        # Setup hooks
        hooks = AttackHooks(device=device)
        hooks.register(model)
        
        # Create synthetic data
        x = torch.randn(4, 3, 224, 224).to(device)
        y = torch.randint(0, 1000, (4,)).to(device)
        
        # Run attention-guided FGSM
        x_adv, info = fgsm(model, x, y, epsilon=4/255, 
                          attn_guided=True, attn_hooks=hooks, device=device)
        
        print(f"[OK] Attention-guided FGSM completed")
        print(f"[OK] Attack Success Rate: {info['attack_success_rate']:.2%}")
        
        hooks.remove()
        return True
        
    except Exception as e:
        print(f"[FAILED] {e}")
        raise


def test_attention_visualizer(model, device='cuda'):
    """Test 6: Attention visualization"""
    print("\n" + "="*70)
    print("TEST 6: Attention Visualization")
    print("="*70)
    
    try:
        viz = AttentionVisualizer(model, device=device)
        viz.register_hooks()
        
        # Create synthetic data
        x = torch.randn(1, 3, 224, 224).to(device)
        
        # Forward pass
        viz.clear_cache()
        with torch.no_grad():
            _ = model(x)
        
        # Generate TAH
        heatmap = viz.token_activation_heatmap(x, block_idx=-1)
        
        if heatmap is not None:
            print(f"[OK] Token Activation Heatmap generated: {heatmap.shape}")
        else:
            print("[WARNING] Could not generate TAH")
        
        # Test ASM
        x_clean = torch.randn(1, 3, 224, 224).to(device)
        x_adv = x_clean + 0.01 * torch.randn_like(x_clean)
        
        diff_map, stability = viz.attention_stability_map(x_clean, x_adv, metric='l1')
        print(f"[OK] Attention Stability Map generated: {diff_map.shape}")
        print(f"[OK] Stability score: {stability:.4f}")
        
        viz.remove_hooks()
        print(f"[OK] Visualization hooks removed")
        
        return True
        
    except Exception as e:
        print(f"[FAILED] {e}")
        raise


def test_gradcam(model, device='cuda'):
    """Test 7: Grad-CAM"""
    print("\n" + "="*70)
    print("TEST 7: Grad-CAM")
    print("="*70)
    
    try:
        gradcam = GradCAM(model, device=device)
        gradcam.register_hooks()
        
        # Create synthetic data
        x = torch.randn(1, 3, 224, 224).to(device)
        
        # Generate CAM
        cam = gradcam.generate_cam(x, target_class=0)
        
        print(f"[OK] Grad-CAM generated: {cam.shape}")
        print(f"[OK] CAM range: [{cam.min():.4f}, {cam.max():.4f}]")
        
        gradcam.remove_hooks()
        print(f"[OK] Grad-CAM hooks removed")
        
        return True
        
    except Exception as e:
        print(f"[FAILED] {e}")
        raise


def main():
    """Run all tests"""
    print("\n" + "#"*70)
    print("# Infrastructure Test Suite")
    print("# Testing all Gap 1 & Gap 4 components")
    print("#"*70)
    
    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    try:
        # Run all tests
        model = test_model_loading(device)
        test_fgsm_attack(model, device)
        test_pgd_attack(model, device)
        test_attention_hooks(model, device)
        test_attention_guided_attack(model, device)
        test_attention_visualizer(model, device)
        test_gradcam(model, device)
        
        # Summary
        print("\n" + "="*70)
        print("[SUCCESS] ALL TESTS PASSED")
        print("="*70)
        print("\nInfrastructure is ready for Gap 1 & Gap 4 experiments!")
        print("\nNext steps:")
        print("  1. Download ImageNet dataset")
        print("  2. Run: python robustness/eval_attacks.py --help")
        print("  3. Run: python scripts/visualize_samples.py --help")
        print("="*70 + "\n")
        
        return True
        
    except Exception as e:
        print("\n" + "="*70)
        print("[FAILED] TEST SUITE FAILED")
        print("="*70)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

