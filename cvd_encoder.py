"""
WaveNet Encoder for Cardiovascular Disease Detection from ECG
Processes 12-lead ECG signals using dilated causal convolutions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DilatedCausalConv1d(nn.Module):
    """Dilated causal convolution layer for WaveNet."""
    
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(DilatedCausalConv1d, self).__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=(kernel_size - 1) * dilation  # Causal padding
        )
        
    def forward(self, x):
        # Causal convolution: remove future information
        out = self.conv(x)
        # Trim the padding to make it causal
        return out[:, :, :-self.conv.padding[0]]


class WaveNetBlock(nn.Module):
    """Single WaveNet residual block with gated activation."""
    
    def __init__(self, residual_channels, gate_channels, skip_channels, 
                 kernel_size, dilation):
        super(WaveNetBlock, self).__init__()
        
        self.dilated_conv = DilatedCausalConv1d(
            residual_channels, gate_channels,
            kernel_size, dilation
        )
        
        # Gated activation: tanh and sigmoid gates
        self.conv_filter = nn.Conv1d(gate_channels, gate_channels, 1)
        self.conv_gate = nn.Conv1d(gate_channels, gate_channels, 1)
        
        # Residual and skip connections
        self.conv_residual = nn.Conv1d(gate_channels, residual_channels, 1)
        self.conv_skip = nn.Conv1d(gate_channels, skip_channels, 1)
        
    def forward(self, x):
        # Dilated convolution
        h = self.dilated_conv(x)
        
        # Gated activation unit
        filter_out = torch.tanh(self.conv_filter(h))
        gate_out = torch.sigmoid(self.conv_gate(h))
        gated = filter_out * gate_out
        
        # Residual connection
        residual = self.conv_residual(gated)
        # Ensure same length for residual connection
        if residual.size(2) != x.size(2):
            residual = F.pad(residual, (0, x.size(2) - residual.size(2)))
        residual = residual + x
        
        # Skip connection
        skip = self.conv_skip(gated)
        
        return residual, skip


class CVDEncoder(nn.Module):
    """
    WaveNet encoder for cardiovascular disease detection from ECG.
    
    Architecture:
    - Input: 12-lead ECG [batch, 12, 3600] (12 leads × 10 sec at 360 Hz)
    - 10 WaveNet blocks with exponentially increasing dilation
    - Skip connections aggregated via sum
    - Global pooling + projection to 256-dim embedding
    
    Args:
        input_channels (int): Number of ECG leads (default: 12)
        residual_channels (int): Channels in residual path (default: 64)
        gate_channels (int): Channels in gated activation (default: 64)
        skip_channels (int): Channels in skip connections (default: 64)
        num_blocks (int): Number of WaveNet blocks (default: 10)
        output_dim (int): Final embedding dimension (default: 256)
        dropout (float): Dropout probability (default: 0.3)
    """
    
    def __init__(
        self,
        input_channels=12,
        residual_channels=64,
        gate_channels=64,
        skip_channels=64,
        num_blocks=10,
        output_dim=256,
        dropout=0.3
    ):
        super(CVDEncoder, self).__init__()
        
        # Initial causal convolution
        self.start_conv = nn.Conv1d(input_channels, residual_channels, kernel_size=1)
        
        # WaveNet blocks with exponentially increasing dilation
        self.wavenet_blocks = nn.ModuleList()
        for i in range(num_blocks):
            dilation = 2 ** i  # Dilation: 1, 2, 4, 8, ..., 512
            block = WaveNetBlock(
                residual_channels, gate_channels, skip_channels,
                kernel_size=2, dilation=dilation
            )
            self.wavenet_blocks.append(block)
        
        # Final processing of skip connections
        self.end_conv1 = nn.Conv1d(skip_channels, skip_channels, kernel_size=1)
        self.end_conv2 = nn.Conv1d(skip_channels, skip_channels, kernel_size=1)
        
        # Global pooling and projection
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.projection = nn.Sequential(
            nn.Linear(skip_channels, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        """
        Forward pass through WaveNet encoder.
        
        Args:
            x: ECG signal [batch_size, 12, 3600]
            
        Returns:
            cvd_embedding: [batch_size, 256]
        """
        # Initial convolution
        x = self.start_conv(x)  # [batch, 64, 3600]
        
        # Collect skip connections
        skip_sum = 0
        
        # Pass through WaveNet blocks
        for block in self.wavenet_blocks:
            x, skip = block(x)
            # Pad skip connection if needed
            if skip.size(2) < x.size(2):
                skip = F.pad(skip, (0, x.size(2) - skip.size(2)))
            skip_sum = skip_sum + skip
        
        # Process aggregated skip connections
        out = F.relu(skip_sum)
        out = F.relu(self.end_conv1(out))
        out = self.end_conv2(out)
        
        # Global pooling
        out = self.global_pool(out)  # [batch, 64, 1]
        out = out.squeeze(-1)  # [batch, 64]
        
        # Project to embedding space
        cvd_embedding = self.projection(out)  # [batch, 256]
        
        return cvd_embedding


class CVDClassifier(nn.Module):
    """
    Full CVD classification model with encoder + classifier head.
    
    Classes: Normal Rhythm (0), Arrhythmia (1), CAD (2), Heart Failure (3)
    """
    
    def __init__(self, encoder, num_classes=4):
        super(CVDClassifier, self).__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(256, num_classes)
        
    def forward(self, x):
        embedding = self.encoder(x)
        logits = self.classifier(embedding)
        return logits, embedding


def create_cvd_encoder(pretrained=False, checkpoint_path=None):
    """
    Factory function to create CVD encoder.
    
    Args:
        pretrained (bool): Load pretrained weights
        checkpoint_path (str): Path to checkpoint file
        
    Returns:
        model: CVDEncoder instance
    """
    model = CVDEncoder()
    
    if pretrained and checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded pretrained CVD encoder from {checkpoint_path}")
        
    return model


def calculate_receptive_field(num_blocks):
    """
    Calculate receptive field of WaveNet.
    
    Receptive field = (kernel_size - 1) * sum(2^i for i in range(num_blocks)) + 1
    For kernel_size=2, num_blocks=10: (2-1) * (2^10 - 1) + 1 = 1024 samples
    At 360 Hz, this is ~2.84 seconds
    """
    rf = (2 - 1) * (2 ** num_blocks - 1) + 1
    return rf


if __name__ == "__main__":
    # Test the model
    batch_size = 4
    model = CVDEncoder()
    
    # Dummy ECG input: 12 leads × 3600 samples (10 seconds at 360 Hz)
    ecg_signal = torch.randn(batch_size, 12, 3600)
    
    # Forward pass
    embedding = model(ecg_signal)
    print(f"CVD embedding shape: {embedding.shape}")  # [4, 256]
    
    # Calculate receptive field
    rf = calculate_receptive_field(10)
    print(f"Receptive field: {rf} samples ({rf/360:.2f} seconds)")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params / 1e6:.2f}M")  # ~3.2M
    
    # Test full classifier
    classifier = CVDClassifier(model, num_classes=4)
    logits, emb = classifier(ecg_signal)
    print(f"Logits shape: {logits.shape}")  # [4, 4]
