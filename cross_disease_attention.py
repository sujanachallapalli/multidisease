"""
Cross-Disease Transformer Attention for Comorbidity Modeling
Learns relationships between diseases through multi-head self-attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention for cross-disease modeling."""
    
    def __init__(self, embed_dim=256, num_heads=4, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert embed_dim % num_heads == 0
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = math.sqrt(self.head_dim)
        
        # Query, Key, Value projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        
        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, return_attention=False):
        """
        Args:
            x: [batch_size, num_diseases, embed_dim]
            return_attention: Whether to return attention weights
            
        Returns:
            attended: [batch_size, num_diseases, embed_dim]
            attention_weights (optional): [batch_size, num_heads, num_diseases, num_diseases]
        """
        batch_size, num_diseases, embed_dim = x.size()
        
        # Linear projections and reshape for multi-head
        Q = self.q_proj(x).view(batch_size, num_diseases, self.num_heads, self.head_dim)
        K = self.k_proj(x).view(batch_size, num_diseases, self.num_heads, self.head_dim)
        V = self.v_proj(x).view(batch_size, num_diseases, self.num_heads, self.head_dim)
        
        # Transpose for attention: [batch, heads, diseases, head_dim]
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attention_weights = F.softmax(scores, dim=-1)  # [batch, heads, diseases, diseases]
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attended = torch.matmul(attention_weights, V)  # [batch, heads, diseases, head_dim]
        
        # Concatenate heads
        attended = attended.transpose(1, 2).contiguous()
        attended = attended.view(batch_size, num_diseases, embed_dim)
        
        # Output projection
        attended = self.out_proj(attended)
        
        if return_attention:
            return attended, attention_weights
        return attended


class CrossDiseaseAttention(nn.Module):
    """
    Cross-disease attention module for learning comorbidity patterns.
    
    Architecture:
    - Multi-head self-attention (4 heads)
    - Feed-forward network
    - Layer normalization and residual connections
    
    Args:
        embed_dim (int): Embedding dimension (default: 256)
        num_heads (int): Number of attention heads (default: 4)
        ffn_dim (int): Feed-forward network dimension (default: 1024)
        dropout (float): Dropout probability (default: 0.1)
    """
    
    def __init__(self, embed_dim=256, num_heads=4, ffn_dim=1024, dropout=0.1):
        super(CrossDiseaseAttention, self).__init__()
        
        # Multi-head attention
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
    def forward(self, disease_embeddings, return_attention=False):
        """
        Forward pass through cross-disease attention.
        
        Args:
            disease_embeddings: Dict with keys ['diabetes', 'alzheimer', 'cancer', 'cvd']
                               Each value is [batch_size, 256]
            return_attention: Whether to return attention weights
            
        Returns:
            fused_embeddings: Dict with same keys, values [batch_size, 256]
            attention_weights (optional): [batch_size, num_heads, 4, 4]
        """
        # Stack disease embeddings: [batch, 4, 256]
        disease_order = ['diabetes', 'alzheimer', 'cancer', 'cvd']
        x = torch.stack([disease_embeddings[d] for d in disease_order], dim=1)
        
        # Multi-head attention with residual connection
        if return_attention:
            attended, attention_weights = self.attention(x, return_attention=True)
        else:
            attended = self.attention(x)
            
        x = self.norm1(x + attended)
        
        # Feed-forward with residual connection
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        # Split back into dictionary
        fused_embeddings = {
            disease: x[:, i, :] 
            for i, disease in enumerate(disease_order)
        }
        
        if return_attention:
            return fused_embeddings, attention_weights
        return fused_embeddings


def analyze_attention_patterns(attention_weights, disease_names):
    """
    Analyze learned comorbidity patterns from attention weights.
    
    Args:
        attention_weights: [batch, num_heads, 4, 4]
        disease_names: List of disease names
        
    Returns:
        attention_matrix: [4, 4] average attention across batch and heads
    """
    # Average across batch and heads
    avg_attention = attention_weights.mean(dim=(0, 1))  # [4, 4]
    
    print("\nCross-Disease Attention Matrix:")
    print(f"{'':12s}", end='')
    for name in disease_names:
        print(f"{name:12s}", end='')
    print()
    
    for i, from_disease in enumerate(disease_names):
        print(f"{from_disease:12s}", end='')
        for j in range(len(disease_names)):
            print(f"{avg_attention[i, j].item():12.3f}", end='')
        print()
    
    return avg_attention


if __name__ == "__main__":
    # Test the module
    batch_size = 8
    
    # Create dummy disease embeddings
    disease_embeddings = {
        'diabetes': torch.randn(batch_size, 256),
        'alzheimer': torch.randn(batch_size, 256),
        'cancer': torch.randn(batch_size, 256),
        'cvd': torch.randn(batch_size, 256)
    }
    
    # Create cross-disease attention module
    cross_attention = CrossDiseaseAttention()
    
    # Forward pass with attention weights
    fused_embeddings, attention_weights = cross_attention(
        disease_embeddings, return_attention=True
    )
    
    print("Input embeddings:")
    for disease, emb in disease_embeddings.items():
        print(f"  {disease}: {emb.shape}")
    
    print("\nFused embeddings:")
    for disease, emb in fused_embeddings.items():
        print(f"  {disease}: {emb.shape}")
    
    print(f"\nAttention weights shape: {attention_weights.shape}")
    
    # Analyze attention patterns
    disease_names = ['Diabetes', 'Alzheimer', 'Cancer', 'CVD']
    avg_attention = analyze_attention_patterns(attention_weights, disease_names)
    
    # Expected patterns:
    # - High diabetes-CVD attention (metabolic syndrome)
    # - High cancer-CVD attention (cardiotoxicity)
    # - High Alzheimer-genomics attention (APOE4)
