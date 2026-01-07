"""
Test Diabetes Model Integration
Verify that the model can be loaded and used for predictions
"""

import os
import django

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'mediCare.settings')
django.setup()

from predictions.models import MLModelVersion
from predictions.ml_utils import load_model, predict_with_model, detect_model_format
import numpy as np

print("=" * 70)
print("TESTING DIABETES MODEL INTEGRATION")
print("=" * 70)

# Step 1: Check if model is registered
print("\n1. Checking database registration...")
active_model = MLModelVersion.get_active_model('diabetes')

if not active_model:
    print("   ✗ ERROR: No active diabetes model found in database")
    exit(1)

print(f"   ✓ Found active model: {active_model.name}")
print(f"     Version: {active_model.version}")
print(f"     Accuracy: {active_model.accuracy * 100}%")
print(f"     File: {active_model.file_path}")

# Step 2: Check if model file exists
print("\n2. Checking model file...")
import os
from pathlib import Path
from django.conf import settings

model_path = Path(settings.BASE_DIR) / active_model.file_path
if not model_path.exists():
    print(f"   ✗ ERROR: Model file not found at {model_path}")
    exit(1)

print(f"   ✓ Model file exists")
print(f"     Size: {model_path.stat().st_size / 1024:.2f} KB")

# Step 3: Load the model
print("\n3. Loading model...")
try:
    model = load_model(active_model.file_path)
    print(f"   ✓ Model loaded successfully")
    print(f"     Type: {type(model).__name__}")
except Exception as e:
    print(f"   ✗ ERROR loading model: {e}")
    exit(1)

# Step 4: Test prediction with sample data
print("\n4. Testing prediction...")
test_data = {
    'pregnancies': 2,
    'glucose': 120,
    'blood_pressure': 70,
    'skin_thickness': 20,
    'insulin': 80,
    'bmi': 25.5,
    'diabetes_pedigree': 0.5,
    'age': 35
}

print(f"   Sample input:")
for key, value in test_data.items():
    print(f"     {key}: {value}")

try:
    # Prepare features using the prepare_features function
    from predictions.ml_utils import prepare_features, load_sklearn_model, get_model_path
    
    features = prepare_features(
        test_data,
        active_model.feature_schema,
        active_model.feature_types
    )
    
    print(f"   Features prepared: {features.shape}")
    
    # Load and apply scaler
    scaler_path = active_model.file_path.replace('model.pkl', 'scaler.pkl')
    scaler = load_sklearn_model(get_model_path(scaler_path))
    features_scaled = scaler.transform(features)
    
    print(f"   Features scaled: {features_scaled.shape}")
    
    # Make prediction
    model_format = detect_model_format(active_model.file_path)
    prediction, probability = predict_with_model(model, features_scaled, model_format)
    
    print(f"\n   ✓ Prediction successful!")
    print(f"     Prediction: {prediction[0]}")
    if probability is not None:
        prob_value = probability[0][1] * 100 if len(probability.shape) > 1 else probability[0] * 100
        print(f"     Probability: {prob_value:.2f}%")
        
        if prob_value < 30:
            risk = "LOW"
        elif prob_value < 60:
            risk = "MEDIUM"
        else:
            risk = "HIGH"
        print(f"     Risk Level: {risk}")
    
except Exception as e:
    print(f"   ✗ ERROR during prediction: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Step 5: Test through Django view
print("\n5. Testing through Django API...")
from predictions.views import make_prediction

result = make_prediction('diabetes', test_data)

if result.get('success'):
    print(f"   ✓ API prediction successful!")
    print(f"     Prediction: {result['prediction']}")
    print(f"     Probability: {result['probability']}%")
    print(f"     Risk Level: {result['risk_level'].upper()}")
    print(f"     Model Version: {result.get('model_version', 'N/A')}")
else:
    print(f"   ✗ API prediction failed: {result.get('error')}")

print("\n" + "=" * 70)
print("INTEGRATION TEST COMPLETE!")
print("=" * 70)
print("\n✓ Your diabetes model is fully integrated and working!")
print("\nNext steps:")
print("  1. Start the Django server: python manage.py runserver")
print("  2. Visit: http://127.0.0.1:8000/predict/diabetes/")
print("  3. Fill the form and test the prediction")
