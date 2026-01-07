"""
Setup Obesity Level Prediction Model in Database
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


def setup_obesity_model():
    """Register Obesity model in database"""
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.join(base_dir, 'ml_models', 'Obesity Level Prediction')
    
    # Load feature names
    with open(os.path.join(model_dir, 'features.pkl'), 'rb') as f:
        features = pickle.load(f)
    
    # Load metrics
    with open(os.path.join(model_dir, 'metrics.pkl'), 'rb') as f:
        metrics = pickle.load(f)
    
    # Feature types - base input features
    feature_types = {
        'gender': 'categorical',
        'age': 'numeric',
        'height': 'numeric',
        'weight': 'numeric',
        'calc': 'categorical',
        'favc': 'boolean',
        'fcvc': 'numeric',
        'ncp': 'numeric',
        'scc': 'boolean',
        'smoke': 'boolean',
        'ch2o': 'numeric',
        'family_history_with_overweight': 'boolean',
        'faf': 'numeric',
        'tue': 'numeric',
        'caec': 'categorical',
        'mtrans': 'categorical',
        # Computed features
        'bmi': 'computed',
        'bmi_underweight': 'computed',
        'bmi_normal': 'computed',
        'bmi_overweight': 'computed',
        'bmi_obese': 'computed',
        'bmi_severely_obese': 'computed',
        'bmi_morbidly_obese': 'computed',
        'activity_score': 'computed',
        'sedentary': 'computed',
        'diet_score': 'computed',
        'unhealthy_eating': 'computed',
        'genetic_risk': 'computed',
        'age_young': 'computed',
        'age_adult': 'computed',
        'age_middle': 'computed',
        'age_senior': 'computed',
        'lifestyle_risk': 'computed',
        'bmi_age_interaction': 'computed',
        'weight_height_ratio': 'computed',
        'caloric_balance': 'computed',
        'bmi_squared': 'computed',
        'weight_squared': 'computed',
        'age_squared': 'computed',
        'log_bmi': 'computed',
        'log_weight': 'computed',
        'active_transport': 'computed',
        'passive_transport': 'computed',
        'low_water': 'computed',
        'adequate_water': 'computed',
        'high_water': 'computed',
        'activity_diet_interaction': 'computed',
        'obesity_risk_score': 'computed',
        'health_index': 'computed',
    }
    
    # Feature options for form dropdowns
    feature_options = {
        'gender': {
            'options': [
                {'value': 'male', 'label': 'Male'},
                {'value': 'female', 'label': 'Female'}
            ]
        },
        'favc': {
            'options': [
                {'value': 'yes', 'label': 'Yes'},
                {'value': 'no', 'label': 'No'}
            ]
        },
        'scc': {
            'options': [
                {'value': 'yes', 'label': 'Yes'},
                {'value': 'no', 'label': 'No'}
            ]
        },
        'smoke': {
            'options': [
                {'value': 'yes', 'label': 'Yes'},
                {'value': 'no', 'label': 'No'}
            ]
        },
        'family_history_with_overweight': {
            'options': [
                {'value': 'yes', 'label': 'Yes'},
                {'value': 'no', 'label': 'No'}
            ]
        },
        'calc': {
            'options': [
                {'value': 'no', 'label': 'Never'},
                {'value': 'sometimes', 'label': 'Sometimes'},
                {'value': 'frequently', 'label': 'Frequently'},
                {'value': 'always', 'label': 'Always'}
            ]
        },
        'caec': {
            'options': [
                {'value': 'no', 'label': 'Never'},
                {'value': 'sometimes', 'label': 'Sometimes'},
                {'value': 'frequently', 'label': 'Frequently'},
                {'value': 'always', 'label': 'Always'}
            ]
        },
        'mtrans': {
            'options': [
                {'value': 'walking', 'label': 'Walking'},
                {'value': 'bike', 'label': 'Bicycle'},
                {'value': 'motorbike', 'label': 'Motorbike'},
                {'value': 'public_transportation', 'label': 'Public Transportation'},
                {'value': 'automobile', 'label': 'Automobile/Car'}
            ]
        },
    }
    
    # Deactivate existing models
    MLModelVersion.objects.filter(disease='obesity').update(is_active=False)
    
    # Create new model version
    model, created = MLModelVersion.objects.update_or_create(
        disease='obesity',
        version='1.0.0',
        defaults={
            'name': 'Obesity Level Prediction Model v1.0',
            'file_path': 'ml_models/Obesity Level Prediction/model.pkl',
            'feature_schema': features,
            'feature_types': feature_types,
            'feature_options': feature_options,
            'accuracy': metrics['accuracy'],
            'is_active': True,
        }
    )
    
    status = 'Created' if created else 'Updated'
    print(f'{status}: {model.name}')
    print(f'Accuracy: {model.accuracy}%')
    print(f'Features: {len(features)}')
    print(f'Classes: {metrics.get("classes", [])}')
    print(f'Active: {model.is_active}')
    
    return model


if __name__ == '__main__':
    setup_obesity_model()
