#!/usr/bin/env python3
"""
Debug script to test training step by step.
"""

import torch
import yaml
from model.unet import UNet
from model.diffusion import GaussianDiffusion
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def setup_data(config):
    transform = transforms.Compose([
        transforms.Resize(config['image_size']),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    
    return DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, num_workers=0)

def main():
    print("🔍 Debug Training Script")
    print("=" * 50)
    
    # Load config
    config = load_config('configs/config.yaml')
    print(f"✅ Config loaded: {config['dataset']}, batch_size={config['batch_size']}")
    
    # Setup device
    device = torch.device('cpu')  # Force CPU for debugging
    print(f"✅ Using device: {device}")
    
    # Setup data
    print("📊 Setting up data...")
    dataloader = setup_data(config)
    print(f"✅ DataLoader created: {len(dataloader)} batches")
    
    # Initialize model
    print("🧠 Creating model...")
    model = UNet(
        dim=config['model']['dim'],
        dim_mults=tuple(config['model']['dim_mults']),
        channels=config['channels'],
        dropout=config['model']['dropout']
    ).to(device)
    print(f"✅ Model created: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Initialize diffusion
    print("🌪️ Creating diffusion process...")
    diffusion = GaussianDiffusion(
        timesteps=config['diffusion']['timesteps'],
        beta_schedule=config['diffusion']['beta_schedule']
    )
    print(f"✅ Diffusion created: {diffusion.timesteps} timesteps")
    
    # Initialize optimizer
    print("⚡ Creating optimizer...")
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    print(f"✅ Optimizer created: lr={config['learning_rate']}")
    
    # Test single batch
    print("🧪 Testing single batch...")
    try:
        for batch_idx, (data, _) in enumerate(dataloader):
            if batch_idx >= 1:  # Only test first batch
                break
                
            print(f"📦 Batch {batch_idx}: data shape {data.shape}")
            data = data.to(device)
            
            # Random timestep
            t = torch.randint(0, diffusion.timesteps, (data.shape[0],), device=device).long()
            print(f"⏰ Timestep shape: {t.shape}")
            
            # Forward pass
            print("🔄 Running forward pass...")
            loss = diffusion.p_losses(model, data, t)
            print(f"✅ Forward pass successful: loss = {loss.item():.4f}")
            
            # Backward pass
            print("🔄 Running backward pass...")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"✅ Backward pass successful")
            
            print("🎉 Single batch training successful!")
            break
            
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("✅ Debug training completed successfully!")

if __name__ == "__main__":
    main() 