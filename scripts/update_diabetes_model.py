"""
Script to update the diabetes model in Django database
Run with: python manage.py shell < scripts/update_diabetes_model.py
Or: python manage.py runscript update_diabetes_model (if django-extensions installed)
"""

import os
import sys
import django

# Setup Django
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'mediCare.settings')
django.setup()

from predictions.models import MLModelVersion

# New feature schema for expanded diabetes model (25 features)
NEW_FEATURE_SCHEMA = [
    'pregnancies', 'glucose', 'bloodpressure', 'skinthickness', 'insulin', 
    'bmi', 'diabetespedigreefunction', 'age', 'hba1c_estimated', 'homa_ir', 
    'homa_b', 'whtr_estimate', 'metabolic_age', 'cv_risk_score', 
    'insulin_sensitivity', 'triglyceride_est', 'hdl_est', 'ldl_est', 
    'fbs_category', 'bmi_category', 'bp_category', 'age_category', 
    'pregnancy_risk', 'family_risk', 'diabetes_risk_score'
]

# Feature types - only base features are required from user input
NEW_FEATURE_TYPES = {
    'pregnancies': 'numeric',
    'glucose': 'numeric',
    'bloodpressure': 'numeric',
    'skinthickness': 'numeric',
    'insulin': 'numeric',
    'bmi': 'numeric',
    'diabetespedigreefunction': 'numeric',
    'age': 'numeric',
    # All computed features
    'hba1c_estimated': 'computed',
    'homa_ir': 'computed',
    'homa_b': 'computed',
    'whtr_estimate': 'computed',
    'metabolic_age': 'computed',
    'cv_risk_score': 'computed',
    'insulin_sensitivity': 'computed',
    'triglyceride_est': 'computed',
    'hdl_est': 'computed',
    'ldl_est': 'computed',
    'fbs_category': 'computed',
    'bmi_category': 'computed',
    'bp_category': 'computed',
    'age_category': 'computed',
    'pregnancy_risk': 'computed',
    'family_risk': 'computed',
    'diabetes_risk_score': 'computed'
}

def update_diabetes_model():
    """Update or create the diabetes model entry"""
    
    # Try to get existing active model
    model = MLModelVersion.objects.filter(disease='diabetes', is_active=True).first()
    
    if model:
        print(f"Updating existing model: {model.name}")
        model.feature_schema = NEW_FEATURE_SCHEMA
        model.feature_types = NEW_FEATURE_TYPES
        model.version = 'v4.0'
        model.accuracy = 0.9941  # 99.41%
        model.description = 'High Accuracy Stacking Ensemble (99.41% accuracy, 25 features)'
        model.save()
        print(f"✓ Updated model: {model.name}")
    else:
        # Create new model entry
        model = MLModelVersion.objects.create(
            name='diabetes_ensemble_v4',
            disease='diabetes',
            version='v4.0',
            accuracy=0.9941,
            description='High Accuracy Stacking Ensemble with 25 medical features (99.41% accuracy)',
            file_path='ml_models/diabetes/model.pkl',
            feature_schema=NEW_FEATURE_SCHEMA,
            feature_types=NEW_FEATURE_TYPES,
            is_active=True
        )
        print(f"✓ Created new model: {model.name}")
    
    print(f"\nModel Details:")
    print(f"  Name: {model.name}")
    print(f"  Version: {model.version}")
    print(f"  Accuracy: {model.accuracy * 100:.2f}%")
    print(f"  Features: {len(model.feature_schema)}")
    print(f"  Active: {model.is_active}")


if __name__ == '__main__':
    update_diabetes_model()
