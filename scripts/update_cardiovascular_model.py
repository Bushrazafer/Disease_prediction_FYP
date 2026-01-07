"""
Update MLModelVersion for Cardiovascular Disease Model
"""

import os
import sys
import django

# Setup Django
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'mediCare.settings')
django.setup()

from predictions.models import MLModelVersion
import pickle

def update_cardiovascular_model():
    """Register the cardiovascular model in database"""
    
    # Load feature names from model
    features_path = 'ml_models/cardiovascular/features.pkl'
    with open(features_path, 'rb') as f:
        feature_names = pickle.load(f)
    
    print(f"Features ({len(feature_names)}): {feature_names}")
    
    # Define feature types
    feature_types = {
        'age': 'numeric',
        'sex': 'categorical',
        'chest_pain_type': 'categorical',
        'resting_bp': 'numeric',
        'cholesterol': 'numeric',
        'fasting_bs': 'boolean',
        'resting_ecg': 'categorical',
        'max_hr': 'numeric',
        'exercise_angina': 'boolean',
        'oldpeak': 'numeric',
        'st_slope': 'categorical',
        'bmi': 'numeric',
        'smoking': 'boolean',
        'diabetes': 'boolean',
        'family_history': 'boolean',
        'physical_activity': 'categorical',
        'alcohol': 'categorical',
        'triglycerides': 'numeric',
        'hdl': 'numeric',
        'ldl': 'numeric',
        'serum_creatinine': 'numeric',
        'ejection_fraction': 'numeric',
        'platelets': 'numeric',
        'serum_sodium': 'numeric',
        'anaemia': 'boolean',
        'cv_risk_score': 'computed'
    }
    
    # Define feature options for categorical fields
    feature_options = {
        'sex': {'options': [{'value': 0, 'label': 'Female'}, {'value': 1, 'label': 'Male'}]},
        'chest_pain_type': {
            'options': [
                {'value': 0, 'label': 'Typical Angina (TA)'},
                {'value': 1, 'label': 'Atypical Angina (ATA)'},
                {'value': 2, 'label': 'Non-Anginal Pain (NAP)'},
                {'value': 3, 'label': 'Asymptomatic (ASY)'}
            ]
        },
        'resting_ecg': {
            'options': [
                {'value': 0, 'label': 'Normal'},
                {'value': 1, 'label': 'ST-T Wave Abnormality'},
                {'value': 2, 'label': 'Left Ventricular Hypertrophy'}
            ]
        },
        'st_slope': {
            'options': [
                {'value': 0, 'label': 'Upsloping'},
                {'value': 1, 'label': 'Flat'},
                {'value': 2, 'label': 'Downsloping'}
            ]
        },
        'physical_activity': {
            'options': [
                {'value': 0, 'label': 'Sedentary'},
                {'value': 1, 'label': 'Light Activity'},
                {'value': 2, 'label': 'Moderate Activity'},
                {'value': 3, 'label': 'High Activity'}
            ]
        },
        'alcohol': {
            'options': [
                {'value': 0, 'label': 'None'},
                {'value': 1, 'label': 'Occasional'},
                {'value': 2, 'label': 'Moderate'},
                {'value': 3, 'label': 'Heavy'}
            ]
        }
    }
    
    # Create or update model version
    model, created = MLModelVersion.objects.update_or_create(
        disease='cardiovascular',
        is_active=True,
        defaults={
            'name': 'Heart Disease Stacking Ensemble',
            'version': '2.0.0',
            'file_path': 'ml_models/cardiovascular/model.pkl',
            'feature_schema': feature_names,
            'feature_types': feature_types,
            'feature_options': feature_options,
            'accuracy': 99.50,
            'description': 'High-accuracy stacking ensemble model for heart disease prediction. '
                          'Uses 9 base models (RF, ET, GB, AdaBoost, KNN, SVM, MLP, XGBoost, LightGBM) '
                          'with logistic regression meta-learner. Trained on 18,918 records with 26 features. '
                          'AUC-ROC: 0.9998'
        }
    )
    
    action = 'Created' if created else 'Updated'
    print(f"\n{action} MLModelVersion:")
    print(f"  Name: {model.name}")
    print(f"  Version: {model.version}")
    print(f"  Disease: {model.disease}")
    print(f"  Features: {len(model.feature_schema)}")
    print(f"  Accuracy: {model.accuracy}%")
    print(f"  File: {model.file_path}")
    
    return model

if __name__ == '__main__':
    update_cardiovascular_model()
    print("\nCardiovascular model registered successfully!")
