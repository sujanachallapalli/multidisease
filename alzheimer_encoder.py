"""
3D ResNet-18 Encoder for Alzheimer's Disease Detection from Brain MRI
Processes 128x128x128 volumetric MRI scans
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock3D(nn.Module):
    """3D Residual Block for volumetric data."""
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(
            in_channels, out_channels, kernel_size=3,
            stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(
            out_channels, out_channels, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.downsample = downsample
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        out = self.relu(out)
        
        return out


class AlzheimerEncoder(nn.Module):
    """
    3D ResNet-18 encoder for Alzheimer's disease detection from MRI.
    
    Architecture:
    - Input: 128×128×128×1 MRI volume
    - 3D ResNet-18 backbone
    - Global average pooling
    - Projection to 256-dim embedding
    
    Args:
        input_channels (int): Number of input channels (default: 1 for T1-MRI)
        output_dim (int): Output embedding dimension (default: 256)
        dropout (float): Dropout probability (default: 0.3)
    """
    
    def __init__(self, input_channels=1, output_dim=256, dropout=0.3):
        super(AlzheimerEncoder, self).__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv3d(
            input_channels, 64, kernel_size=7,
            stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        
        # ResNet layers
        self.layer1 = self._make_layer(64, 64, blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, blocks=2, stride=2)
        
        # Global pooling and projection
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.projection = nn.Sequential(
            nn.Linear(512, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self._initialize_weights()
        
    def _make_layer(self, in_channels, out_channels, blocks, stride):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv3d(
                    in_channels, out_channels, kernel_size=1,
                    stride=stride, bias=False
                ),
                nn.BatchNorm3d(out_channels)
            )
            
        layers = []
        layers.append(BasicBlock3D(in_channels, out_channels, stride, downsample))
        for _ in range(1, blocks):
            layers.append(BasicBlock3D(out_channels, out_channels))
            
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Initialize weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x):
        """
        Forward pass through 3D ResNet encoder.
        
        Args:
            x: MRI volume [batch_size, 1, 128, 128, 128]
            
        Returns:
            alzheimer_embedding: [batch_size, 256]
        """
        # Initial convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # ResNet blocks
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Global pooling
        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # [batch, 512]
        
        # Project to embedding space
        alzheimer_embedding = self.projection(x)  # [batch, 256]
        
        return alzheimer_embedding


class AlzheimerClassifier(nn.Module):
    """
    Full Alzheimer's classification model with encoder + classifier head.
    
    Classes: Cognitively Normal (0), MCI (1), Alzheimer's Disease (2)
    """
    
    def __init__(self, encoder, num_classes=3):
        super(AlzheimerClassifier, self).__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(256, num_classes)
        
    def forward(self, x):
        embedding = self.encoder(x)
        logits = self.classifier(embedding)
        return logits, embedding


def create_alzheimer_encoder(pretrained=False, checkpoint_path=None):
    """
    Factory function to create Alzheimer's encoder.
    
    Args:
        pretrained (bool): Load pretrained weights
        checkpoint_path (str): Path to checkpoint file
        
    Returns:
        model: AlzheimerEncoder instance
    """
    model = AlzheimerEncoder()
    
    if pretrained and checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded pretrained Alzheimer's encoder from {checkpoint_path}")
        
    return model


if __name__ == "__main__":
    # Test the model
    batch_size = 2
    model = AlzheimerEncoder()
    
    # Dummy MRI input
    mri_volume = torch.randn(batch_size, 1, 128, 128, 128)
    
    # Forward pass
    embedding = model(mri_volume)
    print(f"Alzheimer's embedding shape: {embedding.shape}")  # [2, 256]
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params / 1e6:.2f}M")  # ~11.2M
    
    # Test full classifier
    classifier = AlzheimerClassifier(model, num_classes=3)
    logits, emb = classifier(mri_volume)
    print(f"Logits shape: {logits.shape}")  # [2, 3]
