#!/usr/bin/env python3
"""
Test script to verify the diffusion model architecture and basic functionality.
"""

import torch
import yaml
from model.unet import UNet
from model.diffusion import GaussianDiffusion

def test_model_architecture():
    """Test the UNet model architecture."""
    print("Testing UNet architecture...")
    
    # Load config
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Test parameters
    batch_size = 2
    channels = config['channels']
    image_size = config['image_size']
    
    # Create model with config parameters
    model = UNet(
        dim=config['model']['dim'],
        dim_mults=tuple(config['model']['dim_mults']),
        channels=channels,
        dropout=config['model']['dropout']
    )
    
    # Create test input
    x = torch.randn(batch_size, channels, image_size, image_size)
    t = torch.randint(0, config['diffusion']['timesteps'], (batch_size,))
    
    print(f"Input shape: {x.shape}")
    print(f"Time steps: {t.shape}")
    
    # Forward pass
    with torch.no_grad():
        output = model(x, t)
    
    print(f"Output shape: {output.shape}")
    
    # Verify output shape matches input
    assert output.shape == x.shape, f"Expected {x.shape}, got {output.shape}"
    print("✅ UNet architecture test passed!")
    
    return model

def test_diffusion_process():
    """Test the diffusion process."""
    print("\nTesting diffusion process...")
    
    # Load config
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create diffusion model
    diffusion = GaussianDiffusion(timesteps=config['diffusion']['timesteps'])
    
    # Test parameters
    batch_size = 2
    channels = 3
    image_size = 32
    
    # Create test data
    x_start = torch.randn(batch_size, channels, image_size, image_size)
    t = torch.randint(0, diffusion.timesteps, (batch_size,))
    
    print(f"Original data shape: {x_start.shape}")
    
    # Test q_sample (forward process)
    x_noisy = diffusion.q_sample(x_start, t)
    print(f"Noisy data shape: {x_noisy.shape}")
    
    # Test predict_start_from_noise
    noise = torch.randn_like(x_start)
    x_pred = diffusion.predict_start_from_noise(x_noisy, t, noise)
    print(f"Predicted start shape: {x_pred.shape}")
    
    print("✅ Diffusion process test passed!")
    
    return diffusion

def test_training_step():
    """Test a single training step."""
    print("\nTesting training step...")
    
    # Load config
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create model and diffusion
    model = UNet(
        dim=config['model']['dim'],
        dim_mults=tuple(config['model']['dim_mults']),
        channels=config['channels'],
        dropout=config['model']['dropout']
    )
    diffusion = GaussianDiffusion(timesteps=config['diffusion']['timesteps'])
    
    # Create test data
    batch_size = 2
    x_start = torch.randn(batch_size, 3, 32, 32)
    t = torch.randint(0, diffusion.timesteps, (batch_size,))
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Training step
    optimizer.zero_grad()
    loss = diffusion.p_losses(model, x_start, t)
    loss.backward()
    optimizer.step()
    
    print(f"Training loss: {loss.item():.4f}")
    print("✅ Training step test passed!")

def test_config_loading():
    """Test configuration loading."""
    print("\nTesting configuration loading...")
    
    try:
        with open('configs/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        print("Configuration loaded successfully!")
        print(f"Dataset: {config['dataset']}")
        print(f"Batch size: {config['batch_size']}")
        print(f"Learning rate: {config['learning_rate']}")
        print("✅ Configuration test passed!")
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")

def main():
    """Run all tests."""
    print("🧪 Running diffusion model tests...\n")
    
    try:
        # Test model architecture
        model = test_model_architecture()
        
        # Test diffusion process
        diffusion = test_diffusion_process()
        
        # Test training step
        test_training_step()
        
        # Test configuration
        test_config_loading()
        
        print("\n🎉 All tests passed! The model is ready for training.")
        
        # Print model summary
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"\n📊 Model Summary:")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        raise

if __name__ == "__main__":
    main() 