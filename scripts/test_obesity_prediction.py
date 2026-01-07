"""Test Obesity Prediction"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'mediCare.settings')
import django
django.setup()

from predictions.views import make_prediction

# Test different cases
test_cases = [
    {'name': 'Normal Weight', 'height': 1.75, 'weight': 70},
    {'name': 'Underweight', 'height': 1.75, 'weight': 50},
    {'name': 'Obese', 'height': 1.75, 'weight': 120},
]

print("=" * 60)
print("OBESITY LEVEL PREDICTION TEST")
print("=" * 60)

for case in test_cases:
    test_data = {
        'gender': 'Male', 'age': 25,
        'height': case['height'], 'weight': case['weight'],
        'favc': 'no', 'fcvc': 3, 'ncp': 3, 'caec': 'Sometimes',
        'smoke': 'no', 'ch2o': 2, 'scc': 'no', 'faf': 2, 'tue': 1,
        'calc': 'no', 'mtrans': 'walking', 'family_history_with_overweight': 'no'
    }
    result = make_prediction('obesity', test_data)
    bmi = case['weight'] / (case['height'] ** 2)
    print(f"\n{case['name']} (BMI: {bmi:.1f}):")
    print(f"  Prediction: {result.get('display_name')}")
    print(f"  Confidence: {result.get('probability')}%")
    print(f"  Risk Level: {result.get('risk_level')}")
