"""
Heart/Cardiovascular Disease Prediction Module
For Django integration
"""

import os
import pickle
import numpy as np
import pandas as pd

# Get the directory where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load model and preprocessing objects
def load_model():
    """Load the trained model and preprocessing objects"""
    model_path = os.path.join(BASE_DIR, 'model.pkl')
    scaler_path = os.path.join(BASE_DIR, 'scaler.pkl')
    imputer_path = os.path.join(BASE_DIR, 'imputer.pkl')
    features_path = os.path.join(BASE_DIR, 'features.pkl')
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    with open(scaler_path, 'rb') as f:
        scaler_dict = pickle.load(f)
    
    with open(imputer_path, 'rb') as f:
        imputer = pickle.load(f)
    
    with open(features_path, 'rb') as f:
        feature_names = pickle.load(f)
    
    return model, scaler_dict, imputer, feature_names

# Load on module import
try:
    MODEL, SCALER_DICT, IMPUTER, FEATURE_NAMES = load_model()
    MODEL_LOADED = True
except Exception as e:
    print(f"Warning: Could not load heart disease model: {e}")
    MODEL_LOADED = False
    MODEL = None
    SCALER_DICT = None
    IMPUTER = None
    FEATURE_NAMES = None

# Required basic features from frontend
BASIC_FEATURES = [
    'age', 'sex', 'chest_pain_type', 'resting_bp', 'cholesterol',
    'fasting_bs', 'resting_ecg', 'max_hr', 'exercise_angina', 
    'oldpeak', 'st_slope'
]

# Feature aliases for frontend compatibility
FEATURE_ALIASES = {
    'blood_pressure': 'resting_bp',
    'restingbp': 'resting_bp',
    'bp': 'resting_bp',
    'chestpaintype': 'chest_pain_type',
    'chest_pain': 'chest_pain_type',
    'fastingbs': 'fasting_bs',
    'fasting_blood_sugar': 'fasting_bs',
    'restingecg': 'resting_ecg',
    'resting_electrocardiogram': 'resting_ecg',
    'maxhr': 'max_hr',
    'max_heart_rate': 'max_hr',
    'exerciseangina': 'exercise_angina',
    'exercise_induced_angina': 'exercise_angina',
    'old_peak': 'oldpeak',
    'stslope': 'st_slope',
    'st_segment_slope': 'st_slope',
    'gender': 'sex'
}

def prepare_features(input_data):
    """
    Prepare input features for prediction.
    Handles basic input and generates derived features.
    """
    # Normalize field names
    normalized_data = {}
    for key, value in input_data.items():
        key_lower = key.lower()
        if key_lower in FEATURE_ALIASES:
            normalized_data[FEATURE_ALIASES[key_lower]] = value
        else:
            normalized_data[key_lower] = value
    
    # Handle sex/gender encoding
    if 'sex' in normalized_data:
        sex_val = normalized_data['sex']
        if isinstance(sex_val, str):
            normalized_data['sex'] = 1 if sex_val.lower() in ['male', 'm', '1'] else 0
    
    # Handle chest pain type encoding
    if 'chest_pain_type' in normalized_data:
        cpt = normalized_data['chest_pain_type']
        if isinstance(cpt, str):
            cpt_map = {'ta': 0, 'ata': 1, 'nap': 2, 'asy': 3, 
                       'typical': 0, 'atypical': 1, 'non-anginal': 2, 'asymptomatic': 3}
            normalized_data['chest_pain_type'] = cpt_map.get(cpt.lower(), 3)
    
    # Handle resting ECG encoding
    if 'resting_ecg' in normalized_data:
        ecg = normalized_data['resting_ecg']
        if isinstance(ecg, str):
            ecg_map = {'normal': 0, 'st': 1, 'lvh': 2}
            normalized_data['resting_ecg'] = ecg_map.get(ecg.lower(), 0)
    
    # Handle ST slope encoding
    if 'st_slope' in normalized_data:
        slope = normalized_data['st_slope']
        if isinstance(slope, str):
            slope_map = {'up': 0, 'flat': 1, 'down': 2, 'upsloping': 0, 'downsloping': 2}
            normalized_data['st_slope'] = slope_map.get(slope.lower(), 1)
    
    # Handle exercise angina encoding
    if 'exercise_angina' in normalized_data:
        ea = normalized_data['exercise_angina']
        if isinstance(ea, str):
            normalized_data['exercise_angina'] = 1 if ea.lower() in ['yes', 'y', '1', 'true'] else 0
    
    # Handle fasting blood sugar encoding
    if 'fasting_bs' in normalized_data:
        fbs = normalized_data['fasting_bs']
        if isinstance(fbs, str):
            normalized_data['fasting_bs'] = 1 if fbs.lower() in ['yes', 'y', '1', 'true', '>120'] else 0
        elif isinstance(fbs, (int, float)) and fbs > 1:
            # If actual glucose value provided, convert to binary
            normalized_data['fasting_bs'] = 1 if fbs > 120 else 0
    
    # Extract basic values
    age = float(normalized_data.get('age', 50))
    sex = int(normalized_data.get('sex', 1))
    chest_pain_type = int(normalized_data.get('chest_pain_type', 0))
    resting_bp = float(normalized_data.get('resting_bp', 120))
    cholesterol = float(normalized_data.get('cholesterol', 200))
    fasting_bs = int(normalized_data.get('fasting_bs', 0))
    resting_ecg = int(normalized_data.get('resting_ecg', 0))
    max_hr = float(normalized_data.get('max_hr', 150))
    exercise_angina = int(normalized_data.get('exercise_angina', 0))
    oldpeak = float(normalized_data.get('oldpeak', 0))
    st_slope = int(normalized_data.get('st_slope', 0))
    
    # Get or calculate derived features
    bmi = float(normalized_data.get('bmi', 25.0))
    if 'height' in normalized_data and 'weight' in normalized_data:
        height_m = float(normalized_data['height']) / 100
        weight = float(normalized_data['weight'])
        if height_m > 0:
            bmi = weight / (height_m ** 2)
    
    smoking = int(normalized_data.get('smoking', 0))
    if isinstance(normalized_data.get('smoking'), str):
        smoking = 1 if normalized_data['smoking'].lower() in ['yes', 'y', '1', 'true'] else 0
    
    diabetes = int(normalized_data.get('diabetes', 0))
    if isinstance(normalized_data.get('diabetes'), str):
        diabetes = 1 if normalized_data['diabetes'].lower() in ['yes', 'y', '1', 'true'] else 0
    
    family_history = int(normalized_data.get('family_history', 0))
    if isinstance(normalized_data.get('family_history'), str):
        family_history = 1 if normalized_data['family_history'].lower() in ['yes', 'y', '1', 'true'] else 0
    
    physical_activity = int(normalized_data.get('physical_activity', 2))
    alcohol = int(normalized_data.get('alcohol', 1))
    
    # Lipid profile - estimate if not provided
    triglycerides = float(normalized_data.get('triglycerides', 150))
    hdl = float(normalized_data.get('hdl', 50))
    ldl = float(normalized_data.get('ldl', 100))
    
    # Clinical markers - estimate based on other factors
    serum_creatinine = float(normalized_data.get('serum_creatinine', 1.0))
    ejection_fraction = float(normalized_data.get('ejection_fraction', 55))
    platelets = float(normalized_data.get('platelets', 250000))
    serum_sodium = float(normalized_data.get('serum_sodium', 137))
    anaemia = int(normalized_data.get('anaemia', 0))
    
    # Calculate CV Risk Score
    cv_risk_score = (
        (age / 100) * 0.15 +
        sex * 0.10 +
        (chest_pain_type == 3) * 0.15 +
        (resting_bp > 140) * 0.10 +
        (cholesterol > 240) * 0.10 +
        fasting_bs * 0.08 +
        (resting_ecg > 0) * 0.07 +
        (max_hr < 120) * 0.08 +
        exercise_angina * 0.12 +
        (oldpeak > 1) * 0.10 +
        (st_slope > 0) * 0.08 +
        smoking * 0.07 +
        diabetes * 0.08 +
        family_history * 0.10 +
        (bmi > 30) * 0.07
    )
    
    # Build feature array in correct order
    features = {
        'age': age,
        'sex': sex,
        'chest_pain_type': chest_pain_type,
        'resting_bp': resting_bp,
        'cholesterol': cholesterol,
        'fasting_bs': fasting_bs,
        'resting_ecg': resting_ecg,
        'max_hr': max_hr,
        'exercise_angina': exercise_angina,
        'oldpeak': oldpeak,
        'st_slope': st_slope,
        'bmi': bmi,
        'smoking': smoking,
        'diabetes': diabetes,
        'family_history': family_history,
        'physical_activity': physical_activity,
        'alcohol': alcohol,
        'triglycerides': triglycerides,
        'hdl': hdl,
        'ldl': ldl,
        'serum_creatinine': serum_creatinine,
        'ejection_fraction': ejection_fraction,
        'platelets': platelets,
        'serum_sodium': serum_sodium,
        'anaemia': anaemia,
        'cv_risk_score': cv_risk_score
    }
    
    return features

def predict(input_data):
    """
    Make a heart disease prediction.
    
    Args:
        input_data: dict with patient features
        
    Returns:
        dict with prediction, probability, risk_level, and recommendations
    """
    if not MODEL_LOADED:
        return {
            'success': False,
            'error': 'Model not loaded'
        }
    
    try:
        # Prepare features
        features = prepare_features(input_data)
        
        # Create feature array in correct order
        feature_array = np.array([[features[f] for f in FEATURE_NAMES]])
        
        # Handle missing cholesterol (0 values)
        if features['cholesterol'] == 0:
            feature_array[0, FEATURE_NAMES.index('cholesterol')] = np.nan
        
        # Impute missing values
        feature_array = IMPUTER.transform(feature_array)
        
        # Scale features
        scaler = SCALER_DICT['scaler']
        power_transformer = SCALER_DICT['power_transformer']
        
        feature_array = scaler.transform(feature_array)
        feature_array = power_transformer.transform(feature_array)
        
        # Make prediction
        prediction = MODEL.predict(feature_array)[0]
        probability = MODEL.predict_proba(feature_array)[0][1] * 100
        
        # Determine risk level
        if probability < 30:
            risk_level = 'low'
        elif probability < 60:
            risk_level = 'medium'
        else:
            risk_level = 'high'
        
        # Generate recommendations
        recommendations = get_recommendations(features, risk_level, probability)
        
        return {
            'success': True,
            'prediction': int(prediction),
            'probability': round(probability, 1),
            'risk_level': risk_level,
            'recommendations': recommendations,
            'risk_factors': get_risk_factors(features)
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def get_risk_factors(features):
    """Identify key risk factors from patient data"""
    risk_factors = []
    
    if features['age'] > 55:
        risk_factors.append({'factor': 'Age', 'status': 'elevated', 'value': f"{features['age']} years"})
    
    if features['resting_bp'] > 140:
        risk_factors.append({'factor': 'Blood Pressure', 'status': 'high', 'value': f"{features['resting_bp']} mmHg"})
    elif features['resting_bp'] > 120:
        risk_factors.append({'factor': 'Blood Pressure', 'status': 'elevated', 'value': f"{features['resting_bp']} mmHg"})
    
    if features['cholesterol'] > 240:
        risk_factors.append({'factor': 'Cholesterol', 'status': 'high', 'value': f"{features['cholesterol']} mg/dL"})
    elif features['cholesterol'] > 200:
        risk_factors.append({'factor': 'Cholesterol', 'status': 'elevated', 'value': f"{features['cholesterol']} mg/dL"})
    
    if features['bmi'] > 30:
        risk_factors.append({'factor': 'BMI', 'status': 'high', 'value': f"{features['bmi']:.1f}"})
    elif features['bmi'] > 25:
        risk_factors.append({'factor': 'BMI', 'status': 'elevated', 'value': f"{features['bmi']:.1f}"})
    
    if features['smoking'] == 1:
        risk_factors.append({'factor': 'Smoking', 'status': 'high', 'value': 'Yes'})
    
    if features['diabetes'] == 1:
        risk_factors.append({'factor': 'Diabetes', 'status': 'elevated', 'value': 'Yes'})
    
    if features['family_history'] == 1:
        risk_factors.append({'factor': 'Family History', 'status': 'elevated', 'value': 'Yes'})
    
    if features['exercise_angina'] == 1:
        risk_factors.append({'factor': 'Exercise Angina', 'status': 'high', 'value': 'Yes'})
    
    return risk_factors

def get_recommendations(features, risk_level, probability):
    """Generate personalized recommendations based on risk factors"""
    recommendations = []
    
    if risk_level == 'low':
        recommendations.append("Maintain your healthy lifestyle with regular exercise and balanced diet.")
        recommendations.append("Continue annual health check-ups to monitor cardiovascular health.")
    elif risk_level == 'medium':
        recommendations.append("Schedule a consultation with a cardiologist within 1-2 months.")
        recommendations.append("Consider lifestyle modifications to reduce risk factors.")
    else:
        recommendations.append("URGENT: Consult a cardiologist immediately for comprehensive evaluation.")
        recommendations.append("Get an ECG, echocardiogram, and stress test as soon as possible.")
    
    # Specific recommendations based on risk factors
    if features['resting_bp'] > 140:
        recommendations.append("Monitor blood pressure daily and consider medication if consistently high.")
    
    if features['cholesterol'] > 240:
        recommendations.append("Discuss cholesterol-lowering medications with your doctor.")
    
    if features['smoking'] == 1:
        recommendations.append("Quit smoking - this is the single most important step to reduce heart disease risk.")
    
    if features['bmi'] > 30:
        recommendations.append("Work on weight management through diet and exercise.")
    
    if features['physical_activity'] < 2:
        recommendations.append("Increase physical activity to at least 150 minutes per week.")
    
    return recommendations

# For testing
if __name__ == "__main__":
    # Test prediction
    test_data = {
        'age': 55,
        'sex': 'male',
        'chest_pain_type': 'ASY',
        'resting_bp': 140,
        'cholesterol': 250,
        'fasting_bs': 1,
        'resting_ecg': 'ST',
        'max_hr': 130,
        'exercise_angina': 'yes',
        'oldpeak': 2.0,
        'st_slope': 'flat',
        'smoking': 'yes',
        'diabetes': 'yes',
        'family_history': 'yes'
    }
    
    result = predict(test_data)
    print("\nTest Prediction Result:")
    print(f"Success: {result['success']}")
    if result['success']:
        print(f"Prediction: {'Heart Disease' if result['prediction'] == 1 else 'No Heart Disease'}")
        print(f"Probability: {result['probability']}%")
        print(f"Risk Level: {result['risk_level']}")
        print(f"Risk Factors: {result['risk_factors']}")
        print(f"Recommendations: {result['recommendations']}")
