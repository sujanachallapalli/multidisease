"""
1D-CNN Encoder for Cancer Detection from Genomic Data
Processes mutation matrix: 523 genes × 6 mutation types = 3138-dim vector
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CancerEncoder(nn.Module):
    """
    1D-CNN encoder for cancer detection from genomic mutation data.
    
    Architecture:
    - Input: 3138-dim binary mutation matrix (523 genes × 6 types)
    - 1D Convolutional layers for pattern extraction
    - Additional features: TMB, MSI, CNB
    - Output: 256-dim embedding
    
    Args:
        mutation_dim (int): Dimension of mutation matrix (default: 3138)
        additional_features_dim (int): TMB, MSI, CNB features (default: 3)
        hidden_dim (int): Hidden layer dimension (default: 128)
        output_dim (int): Final embedding dimension (default: 256)
        dropout (float): Dropout probability (default: 0.3)
    """
    
    def __init__(
        self,
        mutation_dim=3138,
        additional_features_dim=3,
        hidden_dim=128,
        output_dim=256,
        dropout=0.3
    ):
        super(CancerEncoder, self).__init__()
        
        # Reshape mutation vector for conv1d: [batch, channels, length]
        # We'll use 523 genes as length, 6 mutation types as channels
        self.num_genes = 523
        self.num_mutation_types = 6
        
        # 1D Convolutional pathway for mutation patterns
        self.conv_blocks = nn.Sequential(
            # Conv block 1
            nn.Conv1d(6, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),  # 523 -> 261
            
            # Conv block 2
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),  # 261 -> 130
            
            # Conv block 3
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # Global pooling -> [batch, 128, 1]
        )
        
        # Fully connected pathway for additional features
        self.fc_additional = nn.Sequential(
            nn.Linear(additional_features_dim, 16),
            nn.ReLU()
        )
        
        # Fusion and projection
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim + 16, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, mutation_matrix, additional_features):
        """
        Forward pass through cancer encoder.
        
        Args:
            mutation_matrix: Binary mutation matrix [batch_size, 3138]
            additional_features: TMB, MSI, CNB [batch_size, 3]
            
        Returns:
            cancer_embedding: [batch_size, 256]
        """
        batch_size = mutation_matrix.size(0)
        
        # Reshape mutation matrix: [batch, 3138] -> [batch, 6, 523]
        mutation_reshaped = mutation_matrix.view(
            batch_size, self.num_mutation_types, self.num_genes
        )
        
        # Extract features via convolutions
        conv_features = self.conv_blocks(mutation_reshaped)  # [batch, 128, 1]
        conv_features = conv_features.squeeze(-1)  # [batch, 128]
        
        # Process additional features
        additional_emb = self.fc_additional(additional_features)  # [batch, 16]
        
        # Concatenate and fuse
        combined = torch.cat([conv_features, additional_emb], dim=1)  # [batch, 144]
        cancer_embedding = self.fusion(combined)  # [batch, 256]
        
        return cancer_embedding


class CancerClassifier(nn.Module):
    """
    Full cancer classification model with encoder + classifier head.
    
    Classes: No Mutations (0), Benign Variants (1), Malignant Mutations (2)
    """
    
    def __init__(self, encoder, num_classes=3):
        super(CancerClassifier, self).__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(256, num_classes)
        
    def forward(self, mutation_matrix, additional_features):
        embedding = self.encoder(mutation_matrix, additional_features)
        logits = self.classifier(embedding)
        return logits, embedding


def create_cancer_encoder(pretrained=False, checkpoint_path=None):
    """
    Factory function to create cancer encoder.
    
    Args:
        pretrained (bool): Load pretrained weights
        checkpoint_path (str): Path to checkpoint file
        
    Returns:
        model: CancerEncoder instance
    """
    model = CancerEncoder()
    
    if pretrained and checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded pretrained cancer encoder from {checkpoint_path}")
        
    return model


# Utility function to create mutation matrix from VCF
def create_mutation_matrix(variants, gene_list, mutation_types):
    """
    Create binary mutation matrix from variant list.
    
    Args:
        variants (list): List of variant dictionaries with 'gene' and 'type'
        gene_list (list): List of 523 COSMIC cancer genes
        mutation_types (list): ['missense', 'nonsense', 'frameshift', 
                                 'splice', 'indel', 'amplification']
    
    Returns:
        mutation_matrix: Binary tensor [3138] or [batch, 3138]
    """
    matrix = torch.zeros(len(gene_list), len(mutation_types))
    
    gene_to_idx = {gene: idx for idx, gene in enumerate(gene_list)}
    type_to_idx = {mtype: idx for idx, mtype in enumerate(mutation_types)}
    
    for variant in variants:
        gene = variant.get('gene')
        mut_type = variant.get('type')
        
        if gene in gene_to_idx and mut_type in type_to_idx:
            gene_idx = gene_to_idx[gene]
            type_idx = type_to_idx[mut_type]
            matrix[gene_idx, type_idx] = 1
            
    # Flatten to 3138-dim vector
    return matrix.flatten()


if __name__ == "__main__":
    # Test the model
    batch_size = 4
    model = CancerEncoder()
    
    # Dummy inputs
    mutation_matrix = torch.randint(0, 2, (batch_size, 3138)).float()
    additional_features = torch.randn(batch_size, 3)  # TMB, MSI, CNB
    
    # Forward pass
    embedding = model(mutation_matrix, additional_features)
    print(f"Cancer embedding shape: {embedding.shape}")  # [4, 256]
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params / 1e6:.2f}M")  # ~1.8M
    
    # Test full classifier
    classifier = CancerClassifier(model, num_classes=3)
    logits, emb = classifier(mutation_matrix, additional_features)
    print(f"Logits shape: {logits.shape}")  # [4, 3]
