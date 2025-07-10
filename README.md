# High-Resolution Diffusion Model for Image Generation

A minimal working implementation of a Denoising Diffusion Probabilistic Model (DDPM) trained on CIFAR-10 and MNIST datasets for high-resolution image generation.

## 🚀 Features

- **Custom UNet Architecture**: Enhanced with self-attention mechanisms and residual connections
- **Advanced Noise Scheduling**: Cosine noise schedule for improved training stability
- **Multiple Loss Functions**: 
  - Standard diffusion loss
  - Perceptual loss using VGG features
  - Feature matching loss for improved texture quality
- **Comprehensive Evaluation**: FID and Inception Score metrics
- **Weights & Biases Integration**: Real-time training monitoring and logging
- **Flexible Configuration**: YAML-based configuration system

## 📋 Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA-compatible GPU (recommended)
- 16GB+ RAM

## 🛠️ Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd diffusion-cifar10
```

2. **Create virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Set up Weights & Biases** (optional but recommended):
```bash
wandb login
```

## 🎯 Quick Start

### Training on CIFAR-10

1. **Configure training parameters** in `configs/config.yaml`:
```yaml
dataset: 'cifar10'
batch_size: 64
learning_rate: 2e-4
num_epochs: 100
image_size: 32
channels: 3
```

2. **Start training**:
```bash
python train.py
```

### Training on MNIST

1. **Update configuration**:
```yaml
dataset: 'mnist'
channels: 1
image_size: 28
```

2. **Run training**:
```bash
python train.py
```

## 📊 Model Architecture

### UNet with Attention
- **Base Dimensions**: 64 channels with multipliers [1, 2, 4, 8]
- **Attention Layers**: Self-attention at resolutions 16x16 and 8x8
- **Residual Connections**: GroupNorm + SiLU activation
- **Time Embedding**: Sinusoidal positional encoding

### Diffusion Process
- **Timesteps**: 1000 diffusion steps
- **Noise Schedule**: Cosine beta scheduling
- **Loss Type**: L2 loss with optional perceptual components

## 🔧 Configuration Options

### Model Configuration
```yaml
model:
  dim: 64                    # Base dimension
  dim_mults: [1, 2, 4, 8]   # Channel multipliers
  dropout: 0.1              # Dropout rate
```

### Training Configuration
```yaml
batch_size: 64              # Batch size
learning_rate: 2e-4         # Learning rate
num_epochs: 100             # Number of epochs
use_perceptual_loss: true   # Enable perceptual loss
perceptual_weight: 0.1      # Perceptual loss weight
feature_matching: true      # Enable feature matching
feature_weight: 0.05        # Feature matching weight
```

## 📈 Training Monitoring

### Weights & Biases Dashboard
- Real-time loss curves
- Generated sample grids
- FID and Inception Score tracking
- Model checkpoint management

### Local Logging
- Sample images saved to `samples/` directory
- Model checkpoints saved to `checkpoints/` directory
- Training metrics logged to console

## 🎨 Sample Generation

### Generate samples from trained model:
```python
import torch
from model.unet import UNet
from model.diffusion import GaussianDiffusion

# Load model
model = UNet(dim=64, channels=3)
checkpoint = torch.load('checkpoints/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# Initialize diffusion
diffusion = GaussianDiffusion(timesteps=1000)

# Generate samples
samples = diffusion.sample(model, image_size=32, batch_size=16, channels=3)
```

## 📊 Performance Metrics

### Expected Results (CIFAR-10)
- **FID Score**: < 50 (target)
- **Inception Score**: > 6.0
- **Training Time**: 6-12 hours on RTX 3080
- **Model Size**: ~15M parameters

### Expected Results (MNIST)
- **FID Score**: < 10 (target)
- **Inception Score**: > 8.0
- **Training Time**: 2-4 hours on RTX 3080

## 🏗️ Project Structure

```
diffusion-cifar10/
├── train.py                 # Main training script
├── model/
│   ├── __init__.py
│   ├── unet.py             # Custom UNet architecture
│   └── diffusion.py        # DDPM implementation
├── utils/
│   ├── __init__.py
│   ├── losses.py           # Custom loss functions
│   └── metrics.py          # Evaluation metrics
├── configs/
│   └── config.yaml         # Training configuration
├── requirements.txt        # Dependencies
├── samples/               # Generated samples
├── checkpoints/           # Model checkpoints
└── README.md             # This file
```

## 🔬 Advanced Features

### Custom Loss Functions
- **Perceptual Loss**: Uses VGG19 features for improved visual quality
- **Feature Matching**: Matches intermediate features for better texture preservation

### Attention Mechanisms
- **Self-Attention**: Applied at multiple resolutions
- **Efficient Implementation**: Optimized for memory usage

### Noise Scheduling
- **Cosine Schedule**: Improved over linear scheduling
- **Configurable Parameters**: Adjustable beta start/end values

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   - Reduce batch size in config
   - Use gradient accumulation
   - Enable mixed precision training

2. **Slow Training**:
   - Use GPU acceleration
   - Increase num_workers in DataLoader
   - Optimize data preprocessing

3. **Poor Sample Quality**:
   - Increase training epochs
   - Adjust learning rate
   - Enable perceptual loss

## 📚 References

- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
- [Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Original DDPM paper authors
- PyTorch and torchvision teams
- Weights & Biases for experiment tracking
- Open source community for evaluation metrics

---

**Note**: This is a minimal working implementation for educational and research purposes. For production use, consider additional optimizations and safety measures.
