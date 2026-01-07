"""
RoyalSoft ML Intelligence Engine - Test Suite
Comprehensive validation and testing
"""

from predict import DiabetesPredictionEngine
import json

def print_result(title, result):
    """Pretty print test results"""
    print("\n" + "=" * 60)
    print(f"TEST: {title}")
    print("=" * 60)
    print(json.dumps(result, indent=2))

def test_valid_predictions():
    """Test valid patient scenarios"""
    engine = DiabetesPredictionEngine()
    
    # Test Case 1: High Risk Patient
    high_risk = {
        "age": 55,
        "glucose": 180,
        "blood_pressure": 90,
        "skin_thickness": 30,
        "insulin": 150,
        "bmi": 35.5,
        "diabetes_pedigree": 1.5,
        "pregnancies": 5
    }
    result1 = engine.predict(high_risk)
    print_result("High Risk Patient", result1)
    
    # Test Case 2: Low Risk Patient
    low_risk = {
        "age": 25,
        "glucose": 85,
        "blood_pressure": 80,
        "skin_thickness": 20,
        "insulin": 80,
        "bmi": 22.0,
        "diabetes_pedigree": 0.2,
        "pregnancies": 0
    }
    result2 = engine.predict(low_risk)
    print_result("Low Risk Patient", result2)
    
    # Test Case 3: Medium Risk Patient
    medium_risk = {
        "age": 40,
        "glucose": 110,
        "blood_pressure": 85,
        "skin_thickness": 25,
        "insulin": 100,
        "bmi": 28.0,
        "diabetes_pedigree": 0.5,
        "pregnancies": 2
    }
    result3 = engine.predict(medium_risk)
    print_result("Medium Risk Patient", result3)

def test_validation_errors():
    """Test input validation"""
    engine = DiabetesPredictionEngine()
    
    # Test Case 4: Missing field
    invalid1 = {
        "age": 45,
        "glucose": 140
        # Missing other fields
    }
    result4 = engine.predict(invalid1)
    print_result("Missing Fields Error", result4)
    
    # Test Case 5: Invalid glucose range
    invalid2 = {
        "age": 45,
        "glucose": 500,  # Out of range
        "blood_pressure": 85,
        "skin_thickness": 25,
        "insulin": 100,
        "bmi": 32.5,
        "diabetes_pedigree": 0.8,
        "pregnancies": 2
    }
    result5 = engine.predict(invalid2)
    print_result("Invalid Glucose Range", result5)
    
    # Test Case 6: Invalid BMI
    invalid3 = {
        "age": 45,
        "glucose": 140,
        "blood_pressure": 85,
        "skin_thickness": 25,
        "insulin": 100,
        "bmi": 70,  # Out of range
        "diabetes_pedigree": 0.8,
        "pregnancies": 2
    }
    result6 = engine.predict(invalid3)
    print_result("Invalid BMI Range", result6)
    
    # Test Case 7: Non-numeric value
    invalid4 = {
        "age": "forty-five",  # String instead of number
        "glucose": 140,
        "blood_pressure": 85,
        "skin_thickness": 25,
        "insulin": 100,
        "bmi": 32.5,
        "diabetes_pedigree": 0.8,
        "pregnancies": 2
    }
    result7 = engine.predict(invalid4)
    print_result("Non-Numeric Value Error", result7)

def test_edge_cases():
    """Test edge cases"""
    engine = DiabetesPredictionEngine()
    
    # Test Case 8: Minimum valid values
    min_values = {
        "age": 18,
        "glucose": 50,
        "blood_pressure": 80,
        "skin_thickness": 0,
        "insulin": 0,
        "bmi": 10,
        "diabetes_pedigree": 0,
        "pregnancies": 0
    }
    result8 = engine.predict(min_values)
    print_result("Minimum Valid Values", result8)
    
    # Test Case 9: Maximum valid values
    max_values = {
        "age": 120,
        "glucose": 400,
        "blood_pressure": 200,
        "skin_thickness": 100,
        "insulin": 300,
        "bmi": 60,
        "diabetes_pedigree": 3,
        "pregnancies": 20
    }
    result9 = engine.predict(max_values)
    print_result("Maximum Valid Values", result9)

if __name__ == "__main__":
    print("=" * 60)
    print("RoyalSoft ML Intelligence Engine - Test Suite")
    print("=" * 60)
    
    print("\n\n### VALID PREDICTIONS ###")
    test_valid_predictions()
    
    print("\n\n### VALIDATION ERRORS ###")
    test_validation_errors()
    
    print("\n\n### EDGE CASES ###")
    test_edge_cases()
    
    print("\n\n" + "=" * 60)
    print("✅ All tests completed!")
    print("=" * 60)
