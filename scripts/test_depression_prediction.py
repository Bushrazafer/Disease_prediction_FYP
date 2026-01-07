"""
Test Depression Prediction
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'mediCare.settings')

import django
django.setup()

from predictions.views import make_prediction


def test_depression():
    # Test data - student with moderate risk factors
    test_data = {
        'gender': 'male',
        'age': 22,
        'academic_pressure': 4,
        'work_pressure': 0,
        'cgpa': 7.5,
        'study_satisfaction': 3,
        'job_satisfaction': 0,
        'sleep_duration': 1,  # 5-6 hours
        'dietary_habits': 1,  # Moderate
        'suicidal_thoughts': 0,
        'work_study_hours': 8,
        'financial_stress': 3,
        'family_history': 0
    }
    
    print("Testing Depression Prediction...")
    print("-" * 50)
    
    result = make_prediction('depression', test_data)
    
    print(f"Success: {result.get('success')}")
    
    if result.get('success'):
        print(f"Prediction: {'Depression Risk' if result['prediction'] == 1 else 'No Depression Risk'}")
        print(f"Probability: {result['probability']}%")
        print(f"Risk Level: {result['risk_level']}")
        print(f"Model Version: {result.get('model_version', 'N/A')}")
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")
    
    # Test high risk case
    print("\n" + "=" * 50)
    print("Testing High Risk Case...")
    print("-" * 50)
    
    high_risk_data = {
        'gender': 'female',
        'age': 20,
        'academic_pressure': 5,
        'work_pressure': 0,
        'cgpa': 5.5,
        'study_satisfaction': 1,
        'job_satisfaction': 0,
        'sleep_duration': 0,  # Less than 5 hours
        'dietary_habits': 0,  # Unhealthy
        'suicidal_thoughts': 1,  # Yes
        'work_study_hours': 12,
        'financial_stress': 5,
        'family_history': 1  # Yes
    }
    
    result2 = make_prediction('depression', high_risk_data)
    
    print(f"Success: {result2.get('success')}")
    
    if result2.get('success'):
        print(f"Prediction: {'Depression Risk' if result2['prediction'] == 1 else 'No Depression Risk'}")
        print(f"Probability: {result2['probability']}%")
        print(f"Risk Level: {result2['risk_level']}")
    else:
        print(f"Error: {result2.get('error', 'Unknown error')}")


if __name__ == '__main__':
    test_depression()
