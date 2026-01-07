"""
Setup Depression Model in Database
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


def setup_depression_model():
    """Register Depression model in database"""
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.join(base_dir, 'ml_models', 'Depression')
    
    # Load feature names
    with open(os.path.join(model_dir, 'features.pkl'), 'rb') as f:
        features = pickle.load(f)
    
    # Load metrics
    with open(os.path.join(model_dir, 'metrics.pkl'), 'rb') as f:
        metrics = pickle.load(f)
    
    # Feature types
    feature_types = {
        'gender': 'categorical',
        'age': 'numeric',
        'academic_pressure': 'numeric',
        'work_pressure': 'numeric',
        'cgpa': 'numeric',
        'study_satisfaction': 'numeric',
        'job_satisfaction': 'numeric',
        'sleep_duration': 'categorical',
        'dietary_habits': 'categorical',
        'suicidal_thoughts': 'boolean',
        'work_study_hours': 'numeric',
        'financial_stress': 'numeric',
        'family_history': 'boolean',
        'sleep_risk': 'computed',
        'total_pressure': 'computed',
        'satisfaction_score': 'computed',
        'life_balance': 'computed',
        'high_risk_age': 'computed',
        'overwork': 'computed',
        'high_financial_stress': 'computed',
        'risk_factor_count': 'computed',
        'age_pressure_interaction': 'computed',
        'sleep_stress_interaction': 'computed',
        'academic_risk': 'computed',
        'depression_risk_score': 'computed',
        'protective_factors': 'computed',
        'vulnerability_index': 'computed',
        'age_squared': 'computed',
        'pressure_squared': 'computed',
        'cgpa_squared': 'computed',
        'log_work_hours': 'computed',
        'log_financial_stress': 'computed',
        'satisfaction_pressure_ratio': 'computed',
        'sleep_work_ratio': 'computed',
        'cgpa_pressure_ratio': 'computed',
        'age_sleep_interaction': 'computed',
        'cgpa_satisfaction_interaction': 'computed',
        'pressure_financial_interaction': 'computed',
        'severe_sleep_deprivation': 'computed',
        'high_pressure_low_satisfaction': 'computed',
        'multiple_risk_factors': 'computed',
        'mental_health_index': 'computed',
        'critical_risk': 'computed',
        'age_group_teen': 'computed',
        'age_group_young_adult': 'computed',
        'age_group_adult': 'computed',
        'age_group_middle': 'computed',
        'low_stress': 'computed',
        'moderate_stress': 'computed',
        'high_stress': 'computed',
        'poor_sleep': 'computed',
        'adequate_sleep': 'computed',
        'good_sleep': 'computed',
        'lifestyle_score': 'computed',
    }
    
    # Feature options
    feature_options = {
        'gender': {
            'options': [
                {'value': 'male', 'label': 'Male'},
                {'value': 'female', 'label': 'Female'}
            ]
        },
        'sleep_duration': {
            'options': [
                {'value': '0', 'label': 'Less than 5 hours'},
                {'value': '1', 'label': '5-6 hours'},
                {'value': '2', 'label': '7-8 hours'},
                {'value': '3', 'label': 'More than 8 hours'}
            ]
        },
        'dietary_habits': {
            'options': [
                {'value': '0', 'label': 'Unhealthy'},
                {'value': '1', 'label': 'Moderate'},
                {'value': '2', 'label': 'Healthy'}
            ]
        },
    }
    
    # Deactivate existing models
    MLModelVersion.objects.filter(disease='depression').update(is_active=False)
    
    # Create new model version
    model, created = MLModelVersion.objects.update_or_create(
        disease='depression',
        version='1.0.0',
        defaults={
            'name': 'Depression Prediction Model v1.0',
            'file_path': 'ml_models/Depression/model.pkl',
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
    print(f'Active: {model.is_active}')
    
    return model


if __name__ == '__main__':
    setup_depression_model()
