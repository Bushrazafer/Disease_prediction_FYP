"""
Register Diabetes ML Model in Database
Run this script to connect your trained diabetes model to the Django project
"""

import os
import django

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'mediCare.settings')
django.setup()

from predictions.models import MLModelVersion

# Define the diabetes model configuration
diabetes_model = {
    'name': 'diabetes_rf_v1',
    'disease': 'diabetes',
    'version': 'v1.0',
    'accuracy': 0.85,  # Update with your actual accuracy
    'description': 'Random Forest model for diabetes prediction trained on Pima Indians dataset',
    'file_path': 'ml_models/1️⃣ Diabetes/model.pkl',
    'feature_schema': [
        'pregnancies',
        'glucose',
        'blood_pressure',
        'skin_thickness',
        'insulin',
        'bmi',
        'diabetes_pedigree',
        'age',
        'age_bmi_interaction',
        'glucose_bmi_interaction',
        'is_high_risk_age',
        'is_obese',
        'is_prediabetic',
        'is_diabetic_glucose'
    ],
    'feature_types': {
        'pregnancies': 'numeric',
        'glucose': 'numeric',
        'blood_pressure': 'numeric',
        'skin_thickness': 'numeric',
        'insulin': 'numeric',
        'bmi': 'numeric',
        'diabetes_pedigree': 'numeric',
        'age': 'numeric',
        'age_bmi_interaction': 'computed',
        'glucose_bmi_interaction': 'computed',
        'is_high_risk_age': 'computed',
        'is_obese': 'computed',
        'is_prediabetic': 'computed',
        'is_diabetic_glucose': 'computed'
    },
    'is_active': True
}

# Create or update the model
model, created = MLModelVersion.objects.update_or_create(
    name=diabetes_model['name'],
    defaults=diabetes_model
)

if created:
    print(f"✓ Successfully registered diabetes model: {model.name}")
else:
    print(f"✓ Updated existing diabetes model: {model.name}")

print(f"\nModel Details:")
print(f"  Disease: {model.disease}")
print(f"  Version: {model.version}")
print(f"  Accuracy: {model.accuracy * 100}%")
print(f"  File Path: {model.file_path}")
print(f"  Active: {model.is_active}")
print(f"  Features: {len(model.feature_schema)}")
print(f"\nFeature Schema:")
for feature in model.feature_schema:
    print(f"  - {feature}")
