"""
RoyalSoft ML Intelligence Engine - System Verification
Comprehensive system check before deployment
"""

import os
import pickle
import json
from predict import DiabetesPredictionEngine

def check_files():
    """Verify all required files exist"""
    print("=" * 60)
    print("1. CHECKING FILES")
    print("=" * 60)
    
    required_files = [
        'diabetes.csv',
        'train_model.py',
        'predict.py',
        'api.py',
        'test_predictions.py',
        'model.pkl',
        'scaler.pkl',
        'features.pkl',
        'metrics.pkl',
        'requirements.txt',
        'README.md'
    ]
    
    all_exist = True
    for file in required_files:
        exists = os.path.exists(file)
        status = "✅" if exists else "❌"
        print(f"{status} {file}")
        if not exists:
            all_exist = False
    
    return all_exist

def check_model_artifacts():
    """Verify model artifacts can be loaded"""
    print("\n" + "=" * 60)
    print("2. CHECKING MODEL ARTIFACTS")
    print("=" * 60)
    
    try:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        print("✅ model.pkl loaded successfully")
        
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        print("✅ scaler.pkl loaded successfully")
        
        with open('features.pkl', 'rb') as f:
            features = pickle.load(f)
        print(f"✅ features.pkl loaded successfully ({len(features)} features)")
        
        with open('metrics.pkl', 'rb') as f:
            metrics = pickle.load(f)
        print(f"✅ metrics.pkl loaded successfully")
        print(f"   - Accuracy: {metrics['accuracy']}%")
        print(f"   - AUC-ROC: {metrics['auc_roc']}")
        
        return True
    except Exception as e:
        print(f"❌ Error loading artifacts: {e}")
        return False

def check_prediction_engine():
    """Verify prediction engine works"""
    print("\n" + "=" * 60)
    print("3. CHECKING PREDICTION ENGINE")
    print("=" * 60)
    
    try:
        engine = DiabetesPredictionEngine()
        print("✅ Engine initialized successfully")
        
        # Test valid prediction
        test_data = {
            "age": 45,
            "glucose": 140,
            "blood_pressure": 85,
            "skin_thickness": 25,
            "insulin": 100,
            "bmi": 32.5,
            "diabetes_pedigree": 0.8,
            "pregnancies": 2
        }
        
        result = engine.predict(test_data)
        
        if result['success']:
            print("✅ Valid prediction works")
            print(f"   - Prediction: {result['prediction']}")
            print(f"   - Probability: {result['probability']}%")
            print(f"   - Risk Level: {result['risk_level']}")
        else:
            print(f"❌ Prediction failed: {result.get('error')}")
            return False
        
        # Test validation
        invalid_data = {"age": 45}  # Missing fields
        result = engine.predict(invalid_data)
        
        if not result['success']:
            print("✅ Input validation works")
            print(f"   - Error caught: {result['error']}")
        else:
            print("❌ Validation should have failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def check_feature_ranges():
    """Verify feature validation ranges"""
    print("\n" + "=" * 60)
    print("4. CHECKING FEATURE VALIDATION")
    print("=" * 60)
    
    engine = DiabetesPredictionEngine()
    
    test_cases = [
        ("Age too low", {"age": 10}, False),
        ("Age too high", {"age": 150}, False),
        ("Glucose too low", {"glucose": 30}, False),
        ("Glucose too high", {"glucose": 500}, False),
        ("BMI too low", {"bmi": 5}, False),
        ("BMI too high", {"bmi": 70}, False),
    ]
    
    all_passed = True
    for name, partial_data, should_pass in test_cases:
        # Complete the data
        full_data = {
            "age": 45,
            "glucose": 140,
            "blood_pressure": 85,
            "skin_thickness": 25,
            "insulin": 100,
            "bmi": 32.5,
            "diabetes_pedigree": 0.8,
            "pregnancies": 2
        }
        full_data.update(partial_data)
        
        result = engine.predict(full_data)
        
        if should_pass and result['success']:
            print(f"✅ {name}: Passed as expected")
        elif not should_pass and not result['success']:
            print(f"✅ {name}: Rejected as expected")
        else:
            print(f"❌ {name}: Unexpected result")
            all_passed = False
    
    return all_passed

def check_output_format():
    """Verify output format is correct"""
    print("\n" + "=" * 60)
    print("5. CHECKING OUTPUT FORMAT")
    print("=" * 60)
    
    engine = DiabetesPredictionEngine()
    
    test_data = {
        "age": 45,
        "glucose": 140,
        "blood_pressure": 85,
        "skin_thickness": 25,
        "insulin": 100,
        "bmi": 32.5,
        "diabetes_pedigree": 0.8,
        "pregnancies": 2
    }
    
    result = engine.predict(test_data)
    
    required_fields = [
        'success',
        'prediction',
        'probability',
        'confidence',
        'risk_level',
        'message',
        'feature_importance',
        'recommendations',
        'model_info'
    ]
    
    all_present = True
    for field in required_fields:
        if field in result:
            print(f"✅ {field}: Present")
        else:
            print(f"❌ {field}: Missing")
            all_present = False
    
    # Verify types
    if result['success']:
        assert isinstance(result['prediction'], int), "prediction should be int"
        assert isinstance(result['probability'], (int, float)), "probability should be numeric"
        assert isinstance(result['confidence'], (int, float)), "confidence should be numeric"
        assert isinstance(result['risk_level'], str), "risk_level should be string"
        assert isinstance(result['feature_importance'], list), "feature_importance should be list"
        assert isinstance(result['recommendations'], list), "recommendations should be list"
        assert isinstance(result['model_info'], dict), "model_info should be dict"
        print("✅ All field types correct")
    
    return all_present

def check_performance():
    """Check prediction performance"""
    print("\n" + "=" * 60)
    print("6. CHECKING PERFORMANCE")
    print("=" * 60)
    
    import time
    
    engine = DiabetesPredictionEngine()
    
    test_data = {
        "age": 45,
        "glucose": 140,
        "blood_pressure": 85,
        "skin_thickness": 25,
        "insulin": 100,
        "bmi": 32.5,
        "diabetes_pedigree": 0.8,
        "pregnancies": 2
    }
    
    # Warm-up
    engine.predict(test_data)
    
    # Measure performance
    times = []
    for _ in range(10):
        start = time.time()
        engine.predict(test_data)
        end = time.time()
        times.append((end - start) * 1000)  # Convert to ms
    
    avg_time = sum(times) / len(times)
    max_time = max(times)
    min_time = min(times)
    
    print(f"Average prediction time: {avg_time:.2f}ms")
    print(f"Min: {min_time:.2f}ms, Max: {max_time:.2f}ms")
    
    if avg_time < 100:
        print("✅ Performance is excellent (< 100ms)")
        return True
    elif avg_time < 500:
        print("⚠️ Performance is acceptable (< 500ms)")
        return True
    else:
        print("❌ Performance is slow (> 500ms)")
        return False

def generate_report():
    """Generate final verification report"""
    print("\n" + "=" * 60)
    print("SYSTEM VERIFICATION REPORT")
    print("=" * 60)
    
    checks = [
        ("Files", check_files()),
        ("Model Artifacts", check_model_artifacts()),
        ("Prediction Engine", check_prediction_engine()),
        ("Feature Validation", check_feature_ranges()),
        ("Output Format", check_output_format()),
        ("Performance", check_performance())
    ]
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for name, passed in checks:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {name}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ALL CHECKS PASSED - SYSTEM READY FOR PRODUCTION")
    else:
        print("⚠️ SOME CHECKS FAILED - REVIEW ISSUES BEFORE DEPLOYMENT")
    print("=" * 60)
    
    return all_passed

if __name__ == "__main__":
    print("=" * 60)
    print("RoyalSoft ML Intelligence Engine")
    print("System Verification")
    print("=" * 60)
    print()
    
    success = generate_report()
    
    exit(0 if success else 1)
