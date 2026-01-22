"""
Inference script for making predictions on new patient data
"""

import torch
import argparse
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from models.multimodal_framework import MultiModalFramework
from data_preprocessing.preprocess_mimic import preprocess_ehr_sample
from data_preprocessing.preprocess_adni import preprocess_mri_sample
from data_preprocessing.preprocess_tcga import preprocess_genomic_sample
from data_preprocessing.preprocess_ecg import preprocess_ecg_sample


class MultiDiseasePredictor:
    """Wrapper for easy inference."""
    
    def __init__(self, checkpoint_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.model = MultiModalFramework()
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"Loaded model from {checkpoint_path}")
        print(f"Using device: {self.device}")
    
    def predict(self, ehr_data, mri_path, genomic_path, ecg_path):
        """
        Make predictions for a patient.
        
        Args:
            ehr_data: Dict with clinical notes and lab values
            mri_path: Path to MRI file (.nii.gz)
            genomic_path: Path to VCF file
            ecg_path: Path to ECG file (.wfdb)
            
        Returns:
            results: Dict with disease predictions and risk scores
        """
        # Preprocess inputs
        ehr_batch = preprocess_ehr_sample(ehr_data)
        mri_batch = preprocess_mri_sample(mri_path)
        genomic_batch = preprocess_genomic_sample(genomic_path)
        ecg_batch = preprocess_ecg_sample(ecg_path)
        
        # Create batch
        batch = {
            'ehr_input_ids': ehr_batch['input_ids'].unsqueeze(0).to(self.device),
            'ehr_attention_mask': ehr_batch['attention_mask'].unsqueeze(0).to(self.device),
            'ehr_structured': ehr_batch['structured'].unsqueeze(0).to(self.device),
            'mri': mri_batch.unsqueeze(0).to(self.device),
            'genomics_mutation': genomic_batch['mutation'].unsqueeze(0).to(self.device),
            'genomics_additional': genomic_batch['additional'].unsqueeze(0).to(self.device),
            'ecg': ecg_batch.unsqueeze(0).to(self.device)
        }
        
        # Run inference
        with torch.no_grad():
            predictions, attention_weights, embeddings = self.model(
                batch, return_attention=True
            )
        
        # Process results
        results = self._process_predictions(predictions, attention_weights)
        
        return results
    
    def _process_predictions(self, predictions, attention_weights):
        """Convert model outputs to human-readable results."""
        
        # Disease class names
        disease_classes = {
            'diabetes': ['Healthy', 'Pre-diabetes', 'Type 2 Diabetes'],
            'alzheimer': ['Cognitively Normal', 'MCI', "Alzheimer's Disease"],
            'cancer': ['No Mutations', 'Benign Variants', 'Malignant Mutations'],
            'cvd': ['Normal Rhythm', 'Arrhythmia', 'CAD', 'Heart Failure']
        }
        
        results = {}
        
        # Process each disease
        for disease in ['diabetes', 'alzheimer', 'cancer', 'cvd']:
            probs = torch.softmax(predictions[disease], dim=1)[0]
            pred_class = torch.argmax(probs).item()
            
            results[disease] = {
                'predicted_class': disease_classes[disease][pred_class],
                'confidence': probs[pred_class].item(),
                'probabilities': {
                    cls: prob.item() 
                    for cls, prob in zip(disease_classes[disease], probs)
                }
            }
        
        # Comorbidity score
        results['comorbidity_score'] = predictions['comorbidity_score'][0].item()
        
        # Attention patterns (average across heads)
        avg_attention = attention_weights[0].mean(dim=0).cpu().numpy()
        results['attention_matrix'] = {
            'diabetes_to': {
                'diabetes': avg_attention[0, 0],
                'alzheimer': avg_attention[0, 1],
                'cancer': avg_attention[0, 2],
                'cvd': avg_attention[0, 3]
            },
            'alzheimer_to': {
                'diabetes': avg_attention[1, 0],
                'alzheimer': avg_attention[1, 1],
                'cancer': avg_attention[1, 2],
                'cvd': avg_attention[1, 3]
            },
            'cancer_to': {
                'diabetes': avg_attention[2, 0],
                'alzheimer': avg_attention[2, 1],
                'cancer': avg_attention[2, 2],
                'cvd': avg_attention[2, 3]
            },
            'cvd_to': {
                'diabetes': avg_attention[3, 0],
                'alzheimer': avg_attention[3, 1],
                'cancer': avg_attention[3, 2],
                'cvd': avg_attention[3, 3]
            }
        }
        
        return results
    
    def print_results(self, results):
        """Pretty print prediction results."""
        print("\n" + "="*70)
        print("MULTI-DISEASE RISK ASSESSMENT")
        print("="*70)
        
        for disease in ['diabetes', 'alzheimer', 'cancer', 'cvd']:
            print(f"\n{disease.upper()}:")
            print(f"  Prediction: {results[disease]['predicted_class']}")
            print(f"  Confidence: {results[disease]['confidence']:.2%}")
            print(f"  Probabilities:")
            for cls, prob in results[disease]['probabilities'].items():
                print(f"    {cls}: {prob:.2%}")
        
        print(f"\nCOMORBIDITY RISK SCORE: {results['comorbidity_score']:.1f}/100")
        
        print("\n" + "="*70)


def main(args):
    # Create predictor
    predictor = MultiDiseasePredictor(args.checkpoint, args.device)
    
    # Example EHR data (in practice, load from file)
    ehr_data = {
        'clinical_notes': "Patient presents with elevated fasting glucose...",
        'lab_values': {
            'fasting_glucose': 126,
            'hba1c': 6.8,
            # ... other lab values
        }
    }
    
    # Make prediction
    results = predictor.predict(
        ehr_data=ehr_data,
        mri_path=args.mri,
        genomic_path=args.genomics,
        ecg_path=args.ecg
    )
    
    # Print results
    predictor.print_results(results)
    
    # Save results if specified
    if args.output:
        import json
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=float)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multi-Disease Multi-Modal Prediction"
    )
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--mri', type=str, required=True,
                       help='Path to MRI file (.nii.gz)')
    parser.add_argument('--genomics', type=str, required=True,
                       help='Path to VCF file')
    parser.add_argument('--ecg', type=str, required=True,
                       help='Path to ECG file (.wfdb)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save results JSON')
    
    args = parser.parse_args()
    main(args)
