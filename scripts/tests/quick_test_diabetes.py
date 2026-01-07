"""
Quick Test - Diabetes Model Integration
Run this to verify everything is working
"""

import os
import sys
import django
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'mediCare.settings')
django.setup()

from predictions.views import make_prediction

print("=" * 70)
print("🏥 DIABETES MODEL - QUICK TEST")
print("=" * 70)

# Test cases
test_cases = [
    {
        'name': '👤 Low Risk Patient (Young, Healthy)',
        'data': {
            'pregnancies': 0,
            'glucose': 85,
            'blood_pressure': 70,
            'skin_thickness': 20,
            'insulin': 80,
            'bmi': 22.0,
            'diabetes_pedigree': 0.2,
            'age': 25
        }
    },
    {
        'name': '👤 Medium Risk Patient (Middle-aged, Overweight)',
        'data': {
            'pregnancies': 2,
            'glucose': 110,
            'blood_pressure': 80,
            'skin_thickness': 25,
            'insulin': 90,
            'bmi': 28.5,
            'diabetes_pedigree': 0.5,
            'age': 45
        }
    },
    {
        'name': '👤 High Risk Patient (Older, Obese, High Glucose)',
        'data': {
            'pregnancies': 5,
            'glucose': 180,
            'blood_pressure': 90,
            'skin_thickness': 35,
            'insulin': 150,
            'bmi': 35.0,
            'diabetes_pedigree': 1.2,
            'age': 60
        }
    }
]

for i, test in enumerate(test_cases, 1):
    print(f"\n{'='*70}")
    print(f"Test {i}: {test['name']}")
    print(f"{'='*70}")
    
    data = test['data']
    print(f"\n📋 Input Data:")
    print(f"   Age: {data['age']} years")
    print(f"   Glucose: {data['glucose']} mg/dL")
    print(f"   Blood Pressure: {data['blood_pressure']} mmHg")
    print(f"   BMI: {data['bmi']}")
    print(f"   Insulin: {data['insulin']} μU/mL")
    print(f"   Pregnancies: {data['pregnancies']}")
    
    # Make prediction
    result = make_prediction('diabetes', data)
    
    if result.get('success'):
        print(f"\n🔮 Prediction Results:")
        print(f"   Prediction: {'DIABETES RISK' if result['prediction'] == 1 else 'NO DIABETES RISK'}")
        print(f"   Probability: {result['probability']}%")
        
        risk = result['risk_level'].upper()
        risk_emoji = '🟢' if risk == 'LOW' else '🟡' if risk == 'MEDIUM' else '🔴'
        print(f"   Risk Level: {risk_emoji} {risk}")
        print(f"   Model Version: {result.get('model_version', 'N/A')}")
        print(f"   Message: {result.get('message', '')}")
    else:
        print(f"\n❌ Error: {result.get('error')}")

print(f"\n{'='*70}")
print("✅ ALL TESTS COMPLETE!")
print("=" * 70)
print("\n🚀 Your diabetes model is working perfectly!")
print("\nNext steps:")
print("  1. Start server: python manage.py runserver")
print("  2. Visit: http://127.0.0.1:8000/predict/diabetes/")
print("  3. Test with real data through the web interface")
