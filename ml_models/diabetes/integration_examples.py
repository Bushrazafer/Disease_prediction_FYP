"""
RoyalSoft ML Intelligence Engine - Integration Examples
Shows how to integrate with Django, Mobile Apps, and ERP systems
"""

from predict import DiabetesPredictionEngine
import json

# Initialize engine once (singleton pattern recommended)
engine = DiabetesPredictionEngine()

# ============================================================
# EXAMPLE 1: Django REST Framework Integration
# ============================================================

def django_view_example():
    """
    Example Django view for diabetes prediction
    
    # In your Django views.py:
    from rest_framework.decorators import api_view
    from rest_framework.response import Response
    from predict import DiabetesPredictionEngine
    
    engine = DiabetesPredictionEngine()
    
    @api_view(['POST'])
    def predict_diabetes(request):
        result = engine.predict(request.data)
        status_code = 200 if result['success'] else 400
        return Response(result, status=status_code)
    """
    print("=" * 60)
    print("DJANGO INTEGRATION EXAMPLE")
    print("=" * 60)
    
    # Simulated Django request data
    request_data = {
        "age": 50,
        "glucose": 160,
        "blood_pressure": 90,
        "skin_thickness": 28,
        "insulin": 120,
        "bmi": 34.0,
        "diabetes_pedigree": 1.2,
        "pregnancies": 3
    }
    
    result = engine.predict(request_data)
    print(json.dumps(result, indent=2))
    return result

# ============================================================
# EXAMPLE 2: Mobile App Integration (JSON API)
# ============================================================

def mobile_app_example():
    """
    Example for mobile app integration
    Mobile apps can send HTTP POST requests to the API
    """
    print("\n" + "=" * 60)
    print("MOBILE APP INTEGRATION EXAMPLE")
    print("=" * 60)
    
    # Simulated mobile app data
    mobile_request = {
        "age": 35,
        "glucose": 95,
        "blood_pressure": 82,
        "skin_thickness": 22,
        "insulin": 85,
        "bmi": 24.5,
        "diabetes_pedigree": 0.3,
        "pregnancies": 1
    }
    
    result = engine.predict(mobile_request)
    
    # Mobile-friendly response
    mobile_response = {
        "status": "success" if result["success"] else "error",
        "risk_detected": result.get("prediction", 0) == 1,
        "risk_percentage": result.get("probability", 0),
        "risk_level": result.get("risk_level", "unknown"),
        "recommendations": result.get("recommendations", []),
        "message": result.get("message", "")
    }
    
    print(json.dumps(mobile_response, indent=2))
    return mobile_response

# ============================================================
# EXAMPLE 3: ERP Dashboard Integration
# ============================================================

def erp_dashboard_example():
    """
    Example for ERP system integration
    Batch processing multiple patients
    """
    print("\n" + "=" * 60)
    print("ERP DASHBOARD INTEGRATION EXAMPLE")
    print("=" * 60)
    
    # Simulated batch of patients from ERP database
    patients = [
        {
            "patient_id": "P001",
            "name": "Patient A",
            "age": 45,
            "glucose": 140,
            "blood_pressure": 85,
            "skin_thickness": 25,
            "insulin": 100,
            "bmi": 32.5,
            "diabetes_pedigree": 0.8,
            "pregnancies": 2
        },
        {
            "patient_id": "P002",
            "name": "Patient B",
            "age": 30,
            "glucose": 90,
            "blood_pressure": 80,
            "skin_thickness": 20,
            "insulin": 75,
            "bmi": 23.0,
            "diabetes_pedigree": 0.3,
            "pregnancies": 0
        },
        {
            "patient_id": "P003",
            "name": "Patient C",
            "age": 60,
            "glucose": 190,
            "blood_pressure": 95,
            "skin_thickness": 32,
            "insulin": 180,
            "bmi": 38.0,
            "diabetes_pedigree": 1.8,
            "pregnancies": 5
        }
    ]
    
    # Process batch
    results = []
    for patient in patients:
        patient_id = patient.pop("patient_id")
        patient_name = patient.pop("name")
        
        prediction = engine.predict(patient)
        
        # ERP-friendly format
        erp_record = {
            "patient_id": patient_id,
            "patient_name": patient_name,
            "risk_detected": prediction.get("prediction", 0) == 1,
            "risk_level": prediction.get("risk_level", "unknown"),
            "probability": prediction.get("probability", 0),
            "confidence": prediction.get("confidence", 0),
            "requires_followup": prediction.get("prediction", 0) == 1,
            "priority": "high" if prediction.get("probability", 0) > 70 else "medium" if prediction.get("probability", 0) > 50 else "low"
        }
        
        results.append(erp_record)
    
    print(json.dumps(results, indent=2))
    return results

# ============================================================
# EXAMPLE 4: Batch CSV Processing
# ============================================================

def batch_csv_processing():
    """
    Process multiple patients from CSV file
    """
    print("\n" + "=" * 60)
    print("BATCH CSV PROCESSING EXAMPLE")
    print("=" * 60)
    
    import pandas as pd
    
    # Load patient data
    df = pd.read_csv('diabetes.csv')
    df.columns = ['pregnancies', 'glucose', 'blood_pressure', 'skin_thickness', 
                  'insulin', 'bmi', 'diabetes_pedigree', 'age', 'outcome']
    
    # Process first 5 patients
    results = []
    for idx, row in df.head(5).iterrows():
        patient_data = {
            "age": int(row['age']),
            "glucose": int(row['glucose']),
            "blood_pressure": int(row['blood_pressure']),
            "skin_thickness": int(row['skin_thickness']),
            "insulin": int(row['insulin']),
            "bmi": float(row['bmi']),
            "diabetes_pedigree": float(row['diabetes_pedigree']),
            "pregnancies": int(row['pregnancies'])
        }
        
        prediction = engine.predict(patient_data)
        
        results.append({
            "patient_index": idx,
            "actual_outcome": int(row['outcome']),
            "predicted_outcome": prediction.get("prediction", 0),
            "probability": prediction.get("probability", 0),
            "risk_level": prediction.get("risk_level", "unknown"),
            "correct": int(row['outcome']) == prediction.get("prediction", 0)
        })
    
    print(json.dumps(results, indent=2))
    
    # Calculate accuracy
    correct = sum(1 for r in results if r['correct'])
    accuracy = (correct / len(results)) * 100
    print(f"\nBatch Accuracy: {accuracy}%")
    
    return results

# ============================================================
# EXAMPLE 5: Error Handling Best Practices
# ============================================================

def error_handling_example():
    """
    Demonstrate proper error handling
    """
    print("\n" + "=" * 60)
    print("ERROR HANDLING EXAMPLE")
    print("=" * 60)
    
    # Invalid data scenarios
    test_cases = [
        {
            "name": "Missing Field",
            "data": {"age": 45, "glucose": 140}
        },
        {
            "name": "Invalid Range",
            "data": {
                "age": 45,
                "glucose": 500,  # Invalid
                "blood_pressure": 85,
                "skin_thickness": 25,
                "insulin": 100,
                "bmi": 32.5,
                "diabetes_pedigree": 0.8,
                "pregnancies": 2
            }
        },
        {
            "name": "Non-Numeric",
            "data": {
                "age": "forty-five",  # Invalid
                "glucose": 140,
                "blood_pressure": 85,
                "skin_thickness": 25,
                "insulin": 100,
                "bmi": 32.5,
                "diabetes_pedigree": 0.8,
                "pregnancies": 2
            }
        }
    ]
    
    for test in test_cases:
        print(f"\nTest: {test['name']}")
        result = engine.predict(test['data'])
        
        if not result['success']:
            print(f"❌ Error: {result['error']}")
            # In production, log this error and return appropriate HTTP status
        else:
            print(f"✅ Success: {result['message']}")

# ============================================================
# RUN ALL EXAMPLES
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("RoyalSoft ML Intelligence Engine")
    print("Integration Examples")
    print("=" * 60)
    
    # Run all examples
    django_view_example()
    mobile_app_example()
    erp_dashboard_example()
    batch_csv_processing()
    error_handling_example()
    
    print("\n" + "=" * 60)
    print("✅ All integration examples completed!")
    print("=" * 60)
