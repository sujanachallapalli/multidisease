"""
Complete Multi-Modal Multi-Disease Framework
Integrates all encoders, cross-disease attention, and prediction heads
"""

import torch
import torch.nn as nn
from .diabetes_encoder import DiabetesEncoder
from .alzheimer_encoder import AlzheimerEncoder
from .cancer_encoder import CancerEncoder
from .cvd_encoder import CVDEncoder
from .cross_disease_attention import CrossDiseaseAttention


class PredictionHeads(nn.Module):
    """Multi-task prediction heads for individual diseases and joint tasks."""
    
    def __init__(self, embed_dim=256):
        super(PredictionHeads, self).__init__()
        
        # Individual disease classification heads
        self.diabetes_head = nn.Linear(embed_dim, 3)  # Healthy, Pre-diabetes, T2DM
        self.alzheimer_head = nn.Linear(embed_dim, 3)  # CN, MCI, AD
        self.cancer_head = nn.Linear(embed_dim, 3)  # No mut, Benign, Malignant
        self.cvd_head = nn.Linear(embed_dim, 4)  # Normal, Arrhythmia, CAD, HF
        
        # Joint comorbidity score
        self.comorbidity_head = nn.Sequential(
            nn.Linear(embed_dim * 4, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()  # Score between 0-100 (multiply by 100)
        )
        
    def forward(self, fused_embeddings):
        """
        Args:
            fused_embeddings: Dict with disease embeddings [batch, 256]
            
        Returns:
            predictions: Dict with all predictions
        """
        predictions = {
            'diabetes': self.diabetes_head(fused_embeddings['diabetes']),
            'alzheimer': self.alzheimer_head(fused_embeddings['alzheimer']),
            'cancer': self.cancer_head(fused_embeddings['cancer']),
            'cvd': self.cvd_head(fused_embeddings['cvd'])
        }
        
        # Comorbidity score from all embeddings
        all_embeddings = torch.cat([
            fused_embeddings['diabetes'],
            fused_embeddings['alzheimer'],
            fused_embeddings['cancer'],
            fused_embeddings['cvd']
        ], dim=1)
        
        comorbidity_score = self.comorbidity_head(all_embeddings) * 100
        predictions['comorbidity_score'] = comorbidity_score
        
        return predictions


class MultiModalFramework(nn.Module):
    """
    Complete multi-modal multi-disease detection framework.
    
    Components:
    1. Disease-specific encoders (BioBERT, 3D ResNet, 1D-CNN, WaveNet)
    2. Cross-disease attention
    3. Multi-task prediction heads
    
    Total parameters: ~127.3M
    """
    
    def __init__(
        self,
        diabetes_encoder=None,
        alzheimer_encoder=None,
        cancer_encoder=None,
        cvd_encoder=None,
        use_pretrained_encoders=True
    ):
        super(MultiModalFramework, self).__init__()
        
        # Initialize encoders
        self.diabetes_encoder = diabetes_encoder if diabetes_encoder else DiabetesEncoder()
        self.alzheimer_encoder = alzheimer_encoder if alzheimer_encoder else AlzheimerEncoder()
        self.cancer_encoder = cancer_encoder if cancer_encoder else CancerEncoder()
        self.cvd_encoder = cvd_encoder if cvd_encoder else CVDEncoder()
        
        # Cross-disease attention
        self.cross_disease_attention = CrossDiseaseAttention()
        
        # Prediction heads
        self.prediction_heads = PredictionHeads()
        
    def forward(self, batch, return_attention=False):
        """
        Forward pass through full framework.
        
        Args:
            batch: Dict with keys:
                'ehr_input_ids': [batch, 512]
                'ehr_attention_mask': [batch, 512]
                'ehr_structured': [batch, 15]
                'mri': [batch, 1, 128, 128, 128]
                'genomics_mutation': [batch, 3138]
                'genomics_additional': [batch, 3]
                'ecg': [batch, 12, 3600]
            return_attention: Whether to return attention weights
            
        Returns:
            predictions: Dict with all disease predictions
            attention_weights (optional): Cross-disease attention weights
        """
        # Extract individual disease embeddings
        diabetes_emb = self.diabetes_encoder(
            batch['ehr_input_ids'],
            batch['ehr_attention_mask'],
            batch['ehr_structured']
        )
        
        alzheimer_emb = self.alzheimer_encoder(batch['mri'])
        
        cancer_emb = self.cancer_encoder(
            batch['genomics_mutation'],
            batch['genomics_additional']
        )
        
        cvd_emb = self.cvd_encoder(batch['ecg'])
        
        # Package into dictionary
        disease_embeddings = {
            'diabetes': diabetes_emb,
            'alzheimer': alzheimer_emb,
            'cancer': cancer_emb,
            'cvd': cvd_emb
        }
        
        # Cross-disease fusion
        if return_attention:
            fused_embeddings, attention_weights = self.cross_disease_attention(
                disease_embeddings, return_attention=True
            )
        else:
            fused_embeddings = self.cross_disease_attention(disease_embeddings)
        
        # Multi-task predictions
        predictions = self.prediction_heads(fused_embeddings)
        
        if return_attention:
            return predictions, attention_weights, fused_embeddings
        return predictions
    
    def load_pretrained_encoders(self, checkpoints):
        """
        Load pretrained weights for individual encoders.
        
        Args:
            checkpoints: Dict with paths to encoder checkpoints
        """
        if 'diabetes' in checkpoints:
            self.diabetes_encoder.load_state_dict(
                torch.load(checkpoints['diabetes'])['model_state_dict']
            )
        if 'alzheimer' in checkpoints:
            self.alzheimer_encoder.load_state_dict(
                torch.load(checkpoints['alzheimer'])['model_state_dict']
            )
        if 'cancer' in checkpoints:
            self.cancer_encoder.load_state_dict(
                torch.load(checkpoints['cancer'])['model_state_dict']
            )
        if 'cvd' in checkpoints:
            self.cvd_encoder.load_state_dict(
                torch.load(checkpoints['cvd'])['model_state_dict']
            )
        print("Loaded all pretrained encoders")


def count_parameters(model):
    """Count trainable parameters."""
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Count per component
    diabetes_params = sum(p.numel() for p in model.diabetes_encoder.parameters())
    alzheimer_params = sum(p.numel() for p in model.alzheimer_encoder.parameters())
    cancer_params = sum(p.numel() for p in model.cancer_encoder.parameters())
    cvd_params = sum(p.numel() for p in model.cvd_encoder.parameters())
    attention_params = sum(p.numel() for p in model.cross_disease_attention.parameters())
    heads_params = sum(p.numel() for p in model.prediction_heads.parameters())
    
    print(f"\nParameter Count:")
    print(f"  DiabetesEncoder (BioBERT):  {diabetes_params/1e6:6.1f}M")
    print(f"  AlzheimerEncoder (3D ResNet):{alzheimer_params/1e6:6.1f}M")
    print(f"  CancerEncoder (1D-CNN):      {cancer_params/1e6:6.1f}M")
    print(f"  CVDEncoder (WaveNet):        {cvd_params/1e6:6.1f}M")
    print(f"  Cross-Disease Attention:     {attention_params/1e6:6.1f}M")
    print(f"  Prediction Heads:            {heads_params/1e6:6.1f}M")
    print(f"  {'='*40}")
    print(f"  Total:                       {total/1e6:6.1f}M")
    
    return total


if __name__ == "__main__":
    # Test complete framework
    batch_size = 2
    
    # Create dummy batch
    batch = {
        'ehr_input_ids': torch.randint(0, 30522, (batch_size, 512)),
        'ehr_attention_mask': torch.ones(batch_size, 512),
        'ehr_structured': torch.randn(batch_size, 15),
        'mri': torch.randn(batch_size, 1, 128, 128, 128),
        'genomics_mutation': torch.randint(0, 2, (batch_size, 3138)).float(),
        'genomics_additional': torch.randn(batch_size, 3),
        'ecg': torch.randn(batch_size, 12, 3600)
    }
    
    # Create model
    model = MultiModalFramework()
    
    # Count parameters
    total_params = count_parameters(model)
    
    # Forward pass
    predictions, attention_weights, embeddings = model(batch, return_attention=True)
    
    print("\n\nPredictions:")
    for disease, pred in predictions.items():
        if disease != 'comorbidity_score':
            print(f"  {disease}: {pred.shape}")
        else:
            print(f"  {disease}: {pred.shape} (range 0-100)")
    
    print(f"\nAttention weights: {attention_weights.shape}")
    print(f"Example attention (patient 0, head 0):")
    print(attention_weights[0, 0])
