"""
Setup Breast Cancer Model in Django Database
Run this script to register the Breast Cancer model in MLModelVersion
"""

import os
import sys
import django
import pickle

# Setup Django
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'mediCare.settings')
django.setup()

from predictions.models import MLModelVersion

def setup_breast_cancer_model():
    """Register Breast Cancer model in database"""
    
    model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                             'ml_models', 'Breast_Cancer')
    
    with open(os.path.join(model_dir, 'features.pkl'), 'rb') as f:
        feature_names = pickle.load(f)
    
    with open(os.path.join(model_dir, 'metrics.pkl'), 'rb') as f:
        metrics = pickle.load(f)
    
    # Define feature types
    feature_types = {f: 'numeric' for f in feature_names}
    # Mark computed features
    computed_features = ['area_perimeter_ratio', 'shape_score', 'size_score', 
                        'texture_irregularity', 'size_variation', 'concavity_severity',
                        'symmetry_deviation', 'fractal_complexity', 'malignancy_score', 
                        'uniformity_score']
    for f in computed_features:
        if f in feature_types:
            feature_types[f] = 'computed'
    
    # Create or update model version
    model_version, created = MLModelVersion.objects.update_or_create(
        name='breast_cancer_stacking_v1',
        defaults={
            'disease': 'breast_cancer',
            'version': 'v1.0',
            'accuracy': metrics['accuracy'],
            'description': f'Breast Cancer Stacking Ensemble - Accuracy: {metrics["accuracy"]*100:.2f}%, AUC-ROC: {metrics["auc_roc"]:.4f}',
            'file_path': 'ml_models/Breast_Cancer/model.pkl',
            'feature_schema': feature_names,
            'feature_types': feature_types,
            'feature_options': {},
            'is_active': True,
        }
    )
    
    action = 'Created' if created else 'Updated'
    print(f"{action} Breast Cancer model: {model_version}")
    print(f"  - Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"  - AUC-ROC: {metrics['auc_roc']:.4f}")
    print(f"  - Features: {len(feature_names)}")
    print(f"  - Active: {model_version.is_active}")


if __name__ == '__main__':
    setup_breast_cancer_model()
    print("\nBreast Cancer model setup complete!")
