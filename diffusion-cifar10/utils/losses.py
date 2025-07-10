import torch
import torch.nn as nn
import torchvision.models as models

class PerceptualLoss(nn.Module):
    def __init__(self, feature_layers=['conv_4']):
        super().__init__()
        self.vgg = models.vgg19(pretrained=True).features
        self.feature_layers = feature_layers
        
        # Freeze VGG parameters
        for param in self.vgg.parameters():
            param.requires_grad = False
    
    def forward(self, input, target):
        loss = 0
        x, y = input, target
        
        for i, layer in enumerate(self.vgg):
            x, y = layer(x), layer(y)
            if f'conv_{i}' in self.feature_layers:
                loss += nn.functional.mse_loss(x, y)
        
        return loss

class FeatureMatchingLoss(nn.Module):
    def __init__(self, feature_layers=['conv_2', 'conv_4', 'conv_8']):
        super().__init__()
        self.vgg = models.vgg19(pretrained=True).features
        self.feature_layers = feature_layers
        
        # Freeze VGG parameters
        for param in self.vgg.parameters():
            param.requires_grad = False
    
    def forward(self, input, target):
        # Extract features from multiple layers
        input_features = self.extract_features(input)
        target_features = self.extract_features(target)
        
        loss = 0
        for input_feat, target_feat in zip(input_features, target_features):
            loss += nn.functional.mse_loss(input_feat, target_feat)
        
        return loss
    
    def extract_features(self, x):
        features = []
        current_layer = 0
        
        for layer in self.vgg:
            x = layer(x)
            if isinstance(layer, nn.Conv2d):
                if f'conv_{current_layer}' in self.feature_layers:
                    features.append(x)
                current_layer += 1
        
        return features 