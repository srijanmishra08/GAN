import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
from scipy import linalg

def calculate_fid(generated_images, real_dataloader, device, num_samples=1000):
    """
    Calculate Fréchet Inception Distance (FID) score
    """
    # Load pre-trained Inception model
    inception_model = models.inception_v3(pretrained=True, transform_input=False)
    inception_model.fc = nn.Identity()  # Remove classification layer
    inception_model.to(device)
    inception_model.eval()
    
    # Extract features
    real_features = []
    fake_features = []
    
    # Process real images
    with torch.no_grad():
        for i, (images, _) in enumerate(real_dataloader):
            if i * real_dataloader.batch_size >= num_samples:
                break
            images = images.to(device)
            features = inception_model(images)
            real_features.append(features.cpu().numpy())
    
    # Process generated images
    with torch.no_grad():
        for i in range(0, min(len(generated_images), num_samples), real_dataloader.batch_size):
            batch = generated_images[i:i+real_dataloader.batch_size]
            features = inception_model(batch)
            fake_features.append(features.cpu().numpy())
    
    real_features = np.concatenate(real_features, axis=0)
    fake_features = np.concatenate(fake_features, axis=0)
    
    # Calculate FID
    mu_real, sigma_real = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu_fake, sigma_fake = fake_features.mean(axis=0), np.cov(fake_features, rowvar=False)
    
    diff = mu_real - mu_fake
    covmean = linalg.sqrtm(sigma_real.dot(sigma_fake))
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = diff.dot(diff) + np.trace(sigma_real + sigma_fake - 2 * covmean)
    return fid

def calculate_inception_score(generated_images, device, num_samples=1000, num_splits=10):
    """
    Calculate Inception Score
    """
    # Load pre-trained Inception model
    inception_model = models.inception_v3(pretrained=True, transform_input=False)
    inception_model.to(device)
    inception_model.eval()
    
    # Extract predictions
    predictions = []
    
    with torch.no_grad():
        for i in range(0, min(len(generated_images), num_samples), 32):
            batch = generated_images[i:i+32]
            pred = inception_model(batch)
            predictions.append(torch.softmax(pred, dim=1).cpu().numpy())
    
    predictions = np.concatenate(predictions, axis=0)
    
    # Calculate Inception Score
    scores = []
    for i in range(num_splits):
        part = predictions[i * (num_samples // num_splits):(i + 1) * (num_samples // num_splits)]
        kl = part * (np.log(part) - np.log(np.mean(part, axis=0, keepdims=True)))
        kl = np.mean(np.sum(kl, axis=1))
        scores.append(np.exp(kl))
    
    return np.mean(scores), np.std(scores) 