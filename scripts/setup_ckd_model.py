"""
Setup CKD Model in Django Database
Run this script to register the CKD model in MLModelVersion
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

def setup_ckd_model():
    """Register CKD model in database"""
    
    # Load feature names from trained model
    model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                             'ml_models', 'Chronic Kidney Disease (CKD)')
    
    with open(os.path.join(model_dir, 'features.pkl'), 'rb') as f:
        feature_names = pickle.load(f)
    
    with open(os.path.join(model_dir, 'metrics.pkl'), 'rb') as f:
        metrics = pickle.load(f)
    
    # Define feature types
    feature_types = {
        # Numeric features
        'age': 'numeric',
        'bp': 'numeric',
        'sg': 'numeric',
        'al': 'numeric',
        'su': 'numeric',
        'bgr': 'numeric',
        'bu': 'numeric',
        'sc': 'numeric',
        'sod': 'numeric',
        'pot': 'numeric',
        'hemo': 'numeric',
        'pcv': 'numeric',
        'wc': 'numeric',
        'rc': 'numeric',
        # Categorical features
        'rbc': 'categorical',
        'pc': 'categorical',
        'pcc': 'categorical',
        'ba': 'categorical',
        'htn': 'categorical',
        'dm': 'categorical',
        'cad': 'categorical',
        'appet': 'categorical',
        'pe': 'categorical',
        'ane': 'categorical',
        # Computed features
        'egfr_estimate': 'computed',
        'anemia_score': 'computed',
        'bp_category': 'computed',
        'albumin_creatinine_ratio': 'computed',
        'urea_creatinine_ratio': 'computed',
        'electrolyte_score': 'computed',
        'comorbidity_count': 'computed',
        'age_risk': 'computed',
        'sg_abnormal': 'computed',
        'ckd_risk_score': 'computed',
    }
    
    # Define feature options for categorical
    feature_options = {
        'rbc': ['normal', 'abnormal'],
        'pc': ['normal', 'abnormal'],
        'pcc': ['notpresent', 'present'],
        'ba': ['notpresent', 'present'],
        'htn': ['no', 'yes'],
        'dm': ['no', 'yes'],
        'cad': ['no', 'yes'],
        'appet': ['good', 'poor'],
        'pe': ['no', 'yes'],
        'ane': ['no', 'yes'],
        'al': ['0', '1', '2', '3', '4', '5'],
        'su': ['0', '1', '2', '3', '4', '5'],
    }
    
    # Create or update model version
    model_version, created = MLModelVersion.objects.update_or_create(
        name='ckd_stacking_v1',
        defaults={
            'disease': 'kidney',
            'version': 'v1.0',
            'accuracy': metrics['accuracy'],
            'description': f'CKD Stacking Ensemble Model - Accuracy: {metrics["accuracy"]*100:.2f}%, AUC-ROC: {metrics["auc_roc"]:.4f}',
            'file_path': 'ml_models/Chronic Kidney Disease (CKD)/model.pkl',
            'feature_schema': feature_names,
            'feature_types': feature_types,
            'feature_options': feature_options,
            'is_active': True,
        }
    )
    
    action = 'Created' if created else 'Updated'
    print(f"{action} CKD model version: {model_version}")
    print(f"  - Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"  - AUC-ROC: {metrics['auc_roc']:.4f}")
    print(f"  - Features: {len(feature_names)}")
    print(f"  - Active: {model_version.is_active}")


if __name__ == '__main__':
    setup_ckd_model()
    print("\nCKD model setup complete!")
