"""
Fix Model Feature Mismatch Issues
Updates MLModelVersion records to match actual trained models
"""

import os
import sys
import django
import pickle
import joblib

# Setup Django
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'mediCare.settings')
django.setup()

from predictions.models import MLModelVersion

def fix_ckd_model():
    """Fix CKD model feature schema"""
    print("Fixing CKD Model...")
    
    # Load actual features from trained model
    ckd_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                           'ml_models', 'Chronic Kidney Disease (CKD)')
    
    with open(os.path.join(ckd_dir, 'features.pkl'), 'rb') as f:
        actual_features = pickle.load(f)
    
    # Keep all features as they are in the trained model
    # Note: 'id' feature is included in the trained model
    
    print(f"Actual CKD features: {len(actual_features)}")
    
    # Define correct feature types for 25 features (including id)
    feature_types = {
        # ID feature (will be auto-generated)
        'id': 'computed',
        # Numeric features
        'age': 'numeric',
        'bp': 'numeric',
        'sg': 'numeric',
        'al': 'numeric',
        'su': 'numeric',
        'bgr': 'numeric',
        'bu': 'numeric',
        'sc': 'numeric',
        'sod': 'numeric',
        'pot': 'numeric',
        'hemo': 'numeric',
        'pcv': 'numeric',
        'wc': 'numeric',
        'rc': 'numeric',
        # Categorical features
        'rbc': 'categorical',
        'pc': 'categorical',
        'pcc': 'categorical',
        'ba': 'categorical',
        'htn': 'categorical',
        'dm': 'categorical',
        'cad': 'categorical',
        'appet': 'categorical',
        'pe': 'categorical',
        'ane': 'categorical',
    }
    
    # Define feature options for categorical
    feature_options = {
        'rbc': ['normal', 'abnormal'],
        'pc': ['normal', 'abnormal'],
        'pcc': ['notpresent', 'present'],
        'ba': ['notpresent', 'present'],
        'htn': ['no', 'yes'],
        'dm': ['no', 'yes'],
        'cad': ['no', 'yes'],
        'appet': ['good', 'poor'],
        'pe': ['no', 'yes'],
        'ane': ['no', 'yes'],
        'al': ['0', '1', '2', '3', '4', '5'],
        'su': ['0', '1', '2', '3', '4', '5'],
    }
    
    # Update model version
    model = MLModelVersion.objects.get(disease='kidney', is_active=True)
    model.feature_schema = actual_features
    model.feature_types = feature_types
    model.feature_options = feature_options
    model.save()
    
    print(f"✓ Updated CKD model: {len(actual_features)} features")
    return True

def fix_breast_cancer_model():
    """Fix Breast Cancer model feature schema"""
    print("Fixing Breast Cancer Model...")
    
    # Load actual features from trained model
    bc_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                          'ml_models', 'Breast_Cancer')
    
    with open(os.path.join(bc_dir, 'features.pkl'), 'rb') as f:
        actual_features = pickle.load(f)
    
    print(f"Actual Breast Cancer features: {len(actual_features)}")
    
    # Define feature types for 30 features (all numeric for breast cancer)
    feature_types = {feature: 'numeric' for feature in actual_features}
    
    # Update model version
    model = MLModelVersion.objects.get(disease='breast_cancer', is_active=True)
    model.feature_schema = actual_features
    model.feature_types = feature_types
    model.feature_options = {}  # No categorical options for breast cancer
    model.save()
    
    print(f"✓ Updated Breast Cancer model: {len(actual_features)} features")
    return True

def verify_all_models():
    """Verify all models have correct feature counts"""
    print("\nVerifying all models...")
    
    models = MLModelVersion.objects.filter(is_active=True)
    all_good = True
    
    for model in models:
        try:
            model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), model.file_path)
            if os.path.exists(model_path):
                loaded_model = joblib.load(model_path)
                if hasattr(loaded_model, 'n_features_in_'):
                    db_features = len(model.feature_schema)
                    model_features = loaded_model.n_features_in_
                    
                    if db_features == model_features:
                        print(f"✓ {model.disease}: {db_features} features (OK)")
                    else:
                        print(f"✗ {model.disease}: DB={db_features}, Model={model_features} (MISMATCH)")
                        all_good = False
                else:
                    print(f"? {model.disease}: Cannot determine model features")
            else:
                print(f"✗ {model.disease}: Model file not found - {model.file_path}")
                all_good = False
        except Exception as e:
            print(f"✗ {model.disease}: Error loading model - {str(e)[:50]}")
            all_good = False
    
    return all_good

def main():
    """Main function to fix all model feature mismatches"""
    print("=" * 60)
    print("  FIXING MODEL FEATURE MISMATCHES")
    print("=" * 60)
    
    try:
        # Fix CKD model
        fix_ckd_model()
        
        # Fix Breast Cancer model
        fix_breast_cancer_model()
        
        # Verify all models
        if verify_all_models():
            print("\n🎉 All models have correct feature counts!")
        else:
            print("\n⚠️  Some models still have mismatches")
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("  FEATURE FIX COMPLETE")
    print("=" * 60)
    return True

if __name__ == '__main__':
    main()