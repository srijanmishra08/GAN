import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import wandb
from tqdm import tqdm
import os
import yaml
import matplotlib.pyplot as plt
import numpy as np
from model.unet import UNet
from model.diffusion import GaussianDiffusion
from utils.losses import PerceptualLoss, FeatureMatchingLoss
from utils.metrics import calculate_fid, calculate_inception_score

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def setup_data(config):
    transform = transforms.Compose([
        transforms.Resize(config['image_size']),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Normalize to [-1, 1]
    ])
    
    if config['dataset'] == 'cifar10':
        dataset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform
        )
    elif config['dataset'] == 'mnist':
        dataset = torchvision.datasets.MNIST(
            root='./data', train=True, download=True, transform=transform
        )
    
    return DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, num_workers=2, pin_memory=True)

def save_samples(samples, epoch, save_dir='samples'):
    os.makedirs(save_dir, exist_ok=True)
    
    # Denormalize samples
    samples = (samples + 1) / 2
    samples = torch.clamp(samples, 0, 1)
    
    # Create grid
    grid = torchvision.utils.make_grid(samples[:16], nrow=4, padding=2, normalize=False)
    
    # Save image
    torchvision.utils.save_image(grid, f'{save_dir}/samples_epoch_{epoch}.png')
    
    return grid

def train_epoch(model, diffusion, dataloader, optimizer, perceptual_loss, feature_loss, device, config, epoch):
    model.train()
    total_loss = 0
    total_diff_loss = 0
    total_perc_loss = 0
    total_feat_loss = 0
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, (data, _) in enumerate(progress_bar):
        data = data.to(device)
        optimizer.zero_grad()
        
        # Random timestep
        t = torch.randint(0, diffusion.timesteps, (data.shape[0],), device=device).long()
        
        # Standard diffusion loss
        diff_loss = diffusion.p_losses(model, data, t)
        
        # Additional losses
        perc_loss = torch.tensor(0.0, device=device)
        feat_loss = torch.tensor(0.0, device=device)
        
        if config.get('use_perceptual_loss', False):
            noise = torch.randn_like(data)
            noisy_data = diffusion.q_sample(data, t, noise)
            pred_noise = model(noisy_data, t)
            
            # Reconstruct denoised image
            denoised = diffusion.predict_start_from_noise(noisy_data, t, pred_noise)
            perc_loss = perceptual_loss(denoised, data)
        
        if config.get('feature_matching', False):
            noise = torch.randn_like(data)
            noisy_data = diffusion.q_sample(data, t, noise)
            pred_noise = model(noisy_data, t)
            denoised = diffusion.predict_start_from_noise(noisy_data, t, pred_noise)
            feat_loss = feature_loss(denoised, data)
        
        # Total loss
        loss = diff_loss + config.get('perceptual_weight', 0.1) * perc_loss + config.get('feature_weight', 0.05) * feat_loss
        
        loss.backward()
        optimizer.step()
        
        # Update metrics
        total_loss += loss.item()
        total_diff_loss += diff_loss.item()
        total_perc_loss += perc_loss.item()
        total_feat_loss += feat_loss.item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Diff': f'{diff_loss.item():.4f}',
            'Perc': f'{perc_loss.item():.4f}',
            'Feat': f'{feat_loss.item():.4f}'
        })
        
        # Log to wandb
        if batch_idx % config['wandb']['log_interval'] == 0:
            wandb.log({
                'train_loss': loss.item(),
                'diffusion_loss': diff_loss.item(),
                'perceptual_loss': perc_loss.item(),
                'feature_loss': feat_loss.item(),
                'epoch': epoch,
                'batch': batch_idx,
                'learning_rate': optimizer.param_groups[0]['lr']
            })
    
    return {
        'avg_loss': total_loss / len(dataloader),
        'avg_diff_loss': total_diff_loss / len(dataloader),
        'avg_perc_loss': total_perc_loss / len(dataloader),
        'avg_feat_loss': total_feat_loss / len(dataloader)
    }

def evaluate_model(model, diffusion, dataloader, device, config, epoch):
    model.eval()
    
    with torch.no_grad():
        # Generate samples
        samples = diffusion.sample(model, config['image_size'], batch_size=16, channels=config['channels'])
        
        # Save samples
        grid = save_samples(samples, epoch)
        
        # Calculate metrics
        fid_score = calculate_fid(samples, dataloader, device, num_samples=1000)
        inception_score, inception_std = calculate_inception_score(samples, device, num_samples=1000)
        
        # Log to wandb
        wandb.log({
            'fid_score': fid_score,
            'inception_score': inception_score,
            'inception_std': inception_std,
            'samples': wandb.Image(grid),
            'epoch': epoch
        })
        
        print(f"Epoch {epoch} - FID: {fid_score:.2f}, IS: {inception_score:.2f} ± {inception_std:.2f}")
        
        return fid_score, inception_score

def main():
    # Load configuration
    config = load_config('configs/config.yaml')
    
    # Initialize wandb
    wandb.init(
        project=config['wandb']['project'],
        entity=config['wandb']['entity'],
        config=config,
        name=f"diffusion_{config['dataset']}_{config['image_size']}x{config['image_size']}"
    )
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Setup data
    dataloader = setup_data(config)
    print(f"Dataset: {config['dataset']}, Batch size: {config['batch_size']}, Total batches: {len(dataloader)}")
    
    # Initialize model and diffusion
    model = UNet(
        dim=config['model']['dim'],
        dim_mults=config['model']['dim_mults'],
        channels=config['channels'],
        dropout=config['model']['dropout']
    ).to(device)
    
    diffusion = GaussianDiffusion(
        timesteps=config['diffusion']['timesteps'],
        beta_schedule=config['diffusion']['beta_schedule']
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Initialize optimizer and losses
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    perceptual_loss = PerceptualLoss().to(device) if config.get('use_perceptual_loss', False) else None
    feature_loss = FeatureMatchingLoss().to(device) if config.get('feature_matching', False) else None
    
    # Create checkpoint directory
    os.makedirs('checkpoints', exist_ok=True)
    
    # Training loop
    best_fid = float('inf')
    
    for epoch in range(config['num_epochs']):
        # Train epoch
        metrics = train_epoch(model, diffusion, dataloader, optimizer, perceptual_loss, feature_loss, device, config, epoch)
        
        print(f"Epoch {epoch}/{config['num_epochs']}")
        print(f"  Average Loss: {metrics['avg_loss']:.4f}")
        print(f"  Diffusion Loss: {metrics['avg_diff_loss']:.4f}")
        print(f"  Perceptual Loss: {metrics['avg_perc_loss']:.4f}")
        print(f"  Feature Loss: {metrics['avg_feat_loss']:.4f}")
        
        # Evaluate and generate samples
        if epoch % 5 == 0 or epoch == config['num_epochs'] - 1:
            fid_score, inception_score = evaluate_model(model, diffusion, dataloader, device, config, epoch)
            
            # Save best model
            if fid_score < best_fid:
                best_fid = fid_score
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'fid_score': fid_score,
                    'inception_score': inception_score,
                    'config': config
                }, 'checkpoints/best_model.pth')
        
        # Save checkpoint
        if epoch % 20 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': metrics,
                'config': config
            }, f'checkpoints/checkpoint_epoch_{epoch}.pth')
    
    # Save final model
    torch.save({
        'epoch': config['num_epochs'],
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config
    }, 'checkpoints/final_model.pth')
    
    print("Training completed!")
    print(f"Best FID score: {best_fid:.2f}")

if __name__ == "__main__":
    main() 