"""
ML Model Utilities - Multi-format loader with caching and auto-detection
Supports: sklearn (.pkl, .joblib), Keras (.h5), ONNX (.onnx), PyTorch (.pt), pickle (.sav)
"""

import os
import logging
from pathlib import Path
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from django.conf import settings

logger = logging.getLogger(__name__)

# Global model cache
_MODEL_CACHE: Dict[str, Any] = {}


class ModelLoadError(Exception):
    """Custom exception for model loading errors"""
    pass


class PredictionError(Exception):
    """Custom exception for prediction errors"""
    pass


def get_model_path(file_path: str) -> Path:
    """Get absolute path for model file"""
    if os.path.isabs(file_path):
        return Path(file_path)
    return Path(settings.BASE_DIR) / file_path


def detect_model_format(file_path: str) -> str:
    """Auto-detect model format from file extension"""
    ext = file_path.split('.')[-1].lower()
    format_map = {
        'pkl': 'sklearn',
        'joblib': 'sklearn',
        'h5': 'keras',
        'keras': 'keras',
        'onnx': 'onnx',
        'pt': 'pytorch',
        'pth': 'pytorch',
        'sav': 'pickle'
    }
    return format_map.get(ext, 'sklearn')


def load_sklearn_model(file_path: Path) -> Any:
    """Load sklearn/joblib model"""
    try:
        import joblib
        return joblib.load(file_path)
    except ImportError:
        import pickle
        with open(file_path, 'rb') as f:
            return pickle.load(f)


def load_keras_model(file_path: Path) -> Any:
    """Load Keras/TensorFlow model"""
    try:
        from tensorflow import keras
        return keras.models.load_model(file_path)
    except ImportError:
        raise ModelLoadError("TensorFlow/Keras not installed. Install with: pip install tensorflow")


def load_onnx_model(file_path: Path) -> Any:
    """Load ONNX model"""
    try:
        import onnxruntime as ort
        return ort.InferenceSession(str(file_path))
    except ImportError:
        raise ModelLoadError("ONNX Runtime not installed. Install with: pip install onnxruntime")


def load_pytorch_model(file_path: Path) -> Any:
    """Load PyTorch model"""
    try:
        import torch
        model = torch.load(file_path, map_location=torch.device('cpu'))
        if hasattr(model, 'eval'):
            model.eval()
        return model
    except ImportError:
        raise ModelLoadError("PyTorch not installed. Install with: pip install torch")


def load_pickle_model(file_path: Path) -> Any:
    """Load pickle model"""
    import pickle
    with open(file_path, 'rb') as f:
        return pickle.load(f)


# Format to loader mapping
MODEL_LOADERS = {
    'sklearn': load_sklearn_model,
    'keras': load_keras_model,
    'onnx': load_onnx_model,
    'pytorch': load_pytorch_model,
    'pickle': load_pickle_model,
}


def load_model(file_path: str, use_cache: bool = True) -> Any:
    """
    Load ML model with auto-detection and caching
    
    Args:
        file_path: Path to model file (relative to BASE_DIR or absolute)
        use_cache: Whether to use cached model
        
    Returns:
        Loaded model object
        
    Raises:
        ModelLoadError: If model cannot be loaded
    """
    # Check cache first
    if use_cache and file_path in _MODEL_CACHE:
        logger.debug(f"Using cached model: {file_path}")
        return _MODEL_CACHE[file_path]
    
    # Get absolute path
    model_path = get_model_path(file_path)
    
    # Check if file exists
    if not model_path.exists():
        raise ModelLoadError(f"Model file not found: {model_path}")
    
    # Detect format and load
    model_format = detect_model_format(file_path)
    loader = MODEL_LOADERS.get(model_format)
    
    if not loader:
        raise ModelLoadError(f"Unsupported model format: {model_format}")
    
    try:
        logger.info(f"Loading {model_format} model from: {model_path}")
        model = loader(model_path)
        
        # Cache the model
        if use_cache:
            _MODEL_CACHE[file_path] = model
            
        return model
        
    except Exception as e:
        raise ModelLoadError(f"Failed to load model: {str(e)}")


def clear_model_cache(file_path: Optional[str] = None):
    """Clear model cache (specific model or all)"""
    global _MODEL_CACHE
    if file_path:
        _MODEL_CACHE.pop(file_path, None)
    else:
        _MODEL_CACHE.clear()


def predict_with_model(
    model: Any,
    features: np.ndarray,
    model_format: str = 'sklearn'
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Make prediction with loaded model
    
    Args:
        model: Loaded model object
        features: Input features as numpy array
        model_format: Format of the model
        
    Returns:
        Tuple of (predictions, probabilities)
    """
    try:
        if model_format == 'sklearn':
            prediction = model.predict(features)
            probability = None
            if hasattr(model, 'predict_proba'):
                probability = model.predict_proba(features)
            return prediction, probability
            
        elif model_format == 'keras':
            probability = model.predict(features)
            prediction = (probability > 0.5).astype(int).flatten()
            return prediction, probability
            
        elif model_format == 'onnx':
            input_name = model.get_inputs()[0].name
            result = model.run(None, {input_name: features.astype(np.float32)})
            prediction = result[0]
            probability = result[1] if len(result) > 1 else None
            return prediction, probability
            
        elif model_format == 'pytorch':
            import torch
            with torch.no_grad():
                tensor = torch.FloatTensor(features)
                output = model(tensor)
                probability = torch.sigmoid(output).numpy()
                prediction = (probability > 0.5).astype(int).flatten()
            return prediction, probability
            
        else:
            # Default sklearn-like interface
            prediction = model.predict(features)
            probability = getattr(model, 'predict_proba', lambda x: None)(features)
            return prediction, probability
            
    except Exception as e:
        raise PredictionError(f"Prediction failed: {str(e)}")


def validate_features(
    data: Dict[str, Any],
    feature_schema: List[str],
    feature_types: Optional[Dict[str, str]] = None
) -> Tuple[bool, List[str]]:
    """
    Validate input data against feature schema
    
    Args:
        data: Input data dictionary
        feature_schema: List of required feature names
        feature_types: Optional dict mapping features to types
        
    Returns:
        Tuple of (is_valid, list of error messages)
    """
    errors = []
    
    # Filter out computed and optional features from validation
    required_features = []
    for f in feature_schema:
        ftype = feature_types.get(f, 'numeric') if feature_types else 'numeric'
        # Skip computed features and optional derived features
        if ftype == 'computed':
            continue
        # Skip optional cardiovascular features that can be estimated
        if f in ['bmi', 'smoking', 'diabetes', 'family_history', 'physical_activity', 
                 'alcohol', 'triglycerides', 'hdl', 'ldl', 'serum_creatinine', 
                 'ejection_fraction', 'platelets', 'serum_sodium', 'anaemia', 'cv_risk_score']:
            continue
        # Skip optional diabetes features that can be computed
        if f in ['hba1c_estimated', 'homa_ir', 'homa_b', 'whtr_estimate', 'metabolic_age',
                 'cv_risk_score', 'insulin_sensitivity', 'triglyceride_est', 'hdl_est', 
                 'ldl_est', 'fbs_category', 'bmi_category', 'bp_category', 'age_category',
                 'pregnancy_risk', 'family_risk', 'diabetes_risk_score']:
            continue
        # Skip CKD computed features
        if f in ['egfr_estimate', 'anemia_score', 'bp_category', 'albumin_creatinine_ratio',
                 'urea_creatinine_ratio', 'electrolyte_score', 'comorbidity_count', 
                 'age_risk', 'sg_abnormal', 'ckd_risk_score']:
            continue
        required_features.append(f)
    
    # Create mapping of normalized names to check
    def normalize_key(key):
        return key.lower().replace('_', '').replace(' ', '')
    
    # Known aliases for field names
    field_aliases = {
        'bloodpressure': ['blood_pressure', 'bloodpressure', 'bp'],
        'skinthickness': ['skin_thickness', 'skinthickness'],
        'diabetespedigreefunction': ['diabetes_pedigree', 'diabetespedigreefunction', 'dpf', 'pedigree'],
        'restingbp': ['resting_bp', 'restingbp', 'blood_pressure', 'bp'],
        'chestpaintype': ['chest_pain_type', 'chestpaintype', 'chest_pain'],
        'fastingbs': ['fasting_bs', 'fastingbs', 'fasting_blood_sugar'],
        'restingecg': ['resting_ecg', 'restingecg'],
        'maxhr': ['max_hr', 'maxhr', 'max_heart_rate'],
        'exerciseangina': ['exercise_angina', 'exerciseangina'],
        'stslope': ['st_slope', 'stslope'],
    }
    
    # Build set of normalized input keys
    normalized_input = {normalize_key(k): k for k in data.keys()}
    
    # Check for missing features
    missing = []
    for f in required_features:
        normalized_f = normalize_key(f)
        
        # Check direct match
        if normalized_f in normalized_input:
            continue
            
        # Check aliases
        found = False
        if normalized_f in field_aliases:
            for alias in field_aliases[normalized_f]:
                if normalize_key(alias) in normalized_input:
                    found = True
                    break
        
        if not found:
            missing.append(f)
    
    if missing:
        errors.append(f"Missing features: {', '.join(missing)}")
    
    return len(errors) == 0, errors


def prepare_features(
    data: Dict[str, Any],
    feature_schema: List[str],
    feature_types: Optional[Dict[str, str]] = None
) -> np.ndarray:
    """
    Prepare input data as numpy array in correct order with feature engineering
    
    Args:
        data: Input data dictionary
        feature_schema: List of feature names in order
        feature_types: Optional dict mapping features to types
        
    Returns:
        Numpy array of features
    """
    # Check if this is the new expanded diabetes model (25 features)
    if 'hba1c_estimated' in feature_schema or 'homa_ir' in feature_schema:
        return prepare_diabetes_expanded_features(data, feature_schema)
    
    # Check if this is the cardiovascular model (26 features)
    if 'cv_risk_score' in feature_schema or 'ejection_fraction' in feature_schema:
        return prepare_cardiovascular_features(data, feature_schema)
    
    # Check if this is the CKD model (check for CKD-specific features)
    if ('egfr_estimate' in feature_schema or 'ckd_risk_score' in feature_schema or 
        ('rbc' in feature_schema and 'pc' in feature_schema and 'pcc' in feature_schema)):
        return prepare_ckd_features(data, feature_schema)
    
    # Check if this is the Breast Cancer model
    if 'malignancy_score' in feature_schema or 'concave_points_mean' in feature_schema:
        return prepare_breast_cancer_features(data, feature_schema)
    
    # Check if this is the Depression model
    if 'depression_risk_score' in feature_schema or 'vulnerability_index' in feature_schema:
        return prepare_depression_features(data, feature_schema)
    
    # Check if this is the Obesity model
    if 'obesity_risk_score' in feature_schema or 'bmi_morbidly_obese' in feature_schema:
        return prepare_obesity_features(data, feature_schema)
    # Legacy feature engineering for old diabetes model
    if 'age_bmi_interaction' in feature_schema:
        age = float(data.get('age', 0))
        bmi = float(data.get('bmi', 0))
        glucose = float(data.get('glucose', 0))
        
        data['age_bmi_interaction'] = age * bmi
        data['glucose_bmi_interaction'] = glucose * bmi
        data['is_high_risk_age'] = 1 if age >= 45 else 0
        data['is_obese'] = 1 if bmi >= 30 else 0
        data['is_prediabetic'] = 1 if 100 <= glucose < 126 else 0
        data['is_diabetic_glucose'] = 1 if glucose >= 126 else 0
    
    features = []
    
    for feature in feature_schema:
        value = data.get(feature, 0)
        
        # Type conversion
        if feature_types:
            ftype = feature_types.get(feature, 'numeric')
            if ftype in ['boolean', 'computed']:
                value = 1 if value in [True, 1, '1', 'true', 'yes'] else int(value) if isinstance(value, (int, float)) else 0
            elif ftype == 'numeric':
                try:
                    value = float(value) if value else 0.0
                except (ValueError, TypeError):
                    value = 0.0
        else:
            try:
                value = float(value) if value else 0.0
            except (ValueError, TypeError):
                value = 0.0
                
        features.append(value)
    
    return np.array([features])


def prepare_diabetes_expanded_features(data: Dict[str, Any], feature_schema: List[str]) -> np.ndarray:
    """
    Prepare features for the new expanded diabetes model (25 features)
    Computes all derived medical features from base inputs
    """
    # Normalize field names (handle both underscore and no-underscore versions)
    def get_value(keys, default=0):
        """Get value from data trying multiple key variations"""
        for key in keys:
            if key in data:
                try:
                    return float(data[key])
                except (ValueError, TypeError):
                    return default
        return default
    
    # Extract base features with flexible key names
    pregnancies = get_value(['pregnancies', 'Pregnancies'], 0)
    glucose = get_value(['glucose', 'Glucose'], 100)
    blood_pressure = get_value(['blood_pressure', 'bloodpressure', 'BloodPressure'], 70)
    skin_thickness = get_value(['skin_thickness', 'skinthickness', 'SkinThickness'], 20)
    insulin = get_value(['insulin', 'Insulin'], 80)
    bmi = get_value(['bmi', 'BMI'], 25)
    diabetes_pedigree = get_value(['diabetes_pedigree', 'diabetespedigreefunction', 'DiabetesPedigreeFunction'], 0.3)
    age = get_value(['age', 'Age'], 30)
    
    # Ensure non-zero values for calculations
    glucose = max(glucose, 1)
    insulin = max(insulin, 1)
    
    # Compute all derived features
    features_dict = {
        'pregnancies': pregnancies,
        'glucose': glucose,
        'bloodpressure': blood_pressure,
        'skinthickness': skin_thickness,
        'insulin': insulin,
        'bmi': bmi,
        'diabetespedigreefunction': diabetes_pedigree,
        'age': age,
        
        # Medical computed features
        'hba1c_estimated': np.clip((glucose + 46.7) / 28.7, 4.0, 14.0),
        'homa_ir': np.clip((insulin * glucose) / 405, 0.5, 25),
        'homa_b': np.clip((20 * insulin) / (glucose / 18 - 3.5 + 0.1), 10, 500),
        'whtr_estimate': np.clip(0.3 + (bmi / 100) + (skin_thickness / 500), 0.35, 0.75),
        'metabolic_age': np.clip(age + ((bmi - 25) * 0.5 + (glucose - 100) * 0.1 + (blood_pressure - 70) * 0.2), 18, 100),
        'cv_risk_score': np.clip((age / 100) * 0.3 + (blood_pressure / 200) * 0.25 + (bmi / 50) * 0.25 + (glucose / 300) * 0.2, 0, 1),
        'insulin_sensitivity': np.clip(10000 / np.sqrt(glucose * insulin + 1), 0.5, 20),
        'triglyceride_est': np.clip(50 + bmi * 2 + insulin * 0.3, 50, 400),
        'hdl_est': np.clip(80 - bmi * 0.8, 25, 100),
        'ldl_est': np.clip(70 + bmi * 1.5 + age * 0.5, 50, 200),
        
        # Category features
        'fbs_category': 0 if glucose < 100 else (1 if glucose < 126 else 2),
        'bmi_category': 0 if bmi < 18.5 else (1 if bmi < 25 else (2 if bmi < 30 else 3)),
        'bp_category': 0 if blood_pressure < 80 else (1 if blood_pressure < 90 else 2),
        'age_category': 0 if age < 30 else (1 if age < 45 else (2 if age < 60 else 3)),
        
        # Risk features
        'pregnancy_risk': 1 if pregnancies >= 4 else 0,
        'family_risk': 1 if diabetes_pedigree >= 0.5 else 0,
    }
    
    # Compute diabetes risk score
    hba1c = features_dict['hba1c_estimated']
    homa_ir = features_dict['homa_ir']
    features_dict['diabetes_risk_score'] = (
        glucose / 200 * 0.30 +
        hba1c / 14 * 0.20 +
        bmi / 50 * 0.15 +
        homa_ir / 25 * 0.15 +
        age / 100 * 0.10 +
        diabetes_pedigree / 2.5 * 0.10
    )
    
    # Build feature array in correct order
    features = []
    for feature_name in feature_schema:
        value = features_dict.get(feature_name, 0)
        features.append(float(value))
    
    return np.array([features])


def get_risk_level(probability: float) -> str:
    """Determine risk level from probability"""
    if probability < 30:
        return 'low'
    elif probability < 60:
        return 'medium'
    else:
        return 'high'


def prepare_cardiovascular_features(data: Dict[str, Any], feature_schema: List[str]) -> np.ndarray:
    """
    Prepare features for the cardiovascular/heart disease model (26 features)
    Computes all derived medical features from base inputs
    """
    # Normalize field names
    def get_value(keys, default=0):
        """Get value from data trying multiple key variations"""
        for key in keys:
            if key in data:
                val = data[key]
                # Handle string encodings
                if isinstance(val, str):
                    val_lower = val.lower()
                    if val_lower in ['yes', 'y', 'true', '1']:
                        return 1
                    elif val_lower in ['no', 'n', 'false', '0']:
                        return 0
                    elif val_lower in ['male', 'm']:
                        return 1
                    elif val_lower in ['female', 'f']:
                        return 0
                try:
                    return float(val)
                except (ValueError, TypeError):
                    return default
        return default
    
    # Extract base features
    age = get_value(['age', 'Age'], 50)
    sex = get_value(['sex', 'gender', 'Sex', 'Gender'], 1)
    
    # Chest pain type encoding
    chest_pain = data.get('chest_pain_type', data.get('chestpaintype', data.get('chest_pain', 0)))
    if isinstance(chest_pain, str):
        cpt_map = {'ta': 0, 'ata': 1, 'nap': 2, 'asy': 3, 
                   'typical': 0, 'atypical': 1, 'non-anginal': 2, 'asymptomatic': 3}
        chest_pain_type = cpt_map.get(chest_pain.lower(), 0)
    else:
        chest_pain_type = int(chest_pain) if chest_pain else 0
    
    resting_bp = get_value(['resting_bp', 'restingbp', 'blood_pressure', 'bp', 'RestingBP'], 120)
    cholesterol = get_value(['cholesterol', 'Cholesterol'], 200)
    fasting_bs = get_value(['fasting_bs', 'fastingbs', 'fasting_blood_sugar', 'FastingBS'], 0)
    
    # Resting ECG encoding
    resting_ecg_val = data.get('resting_ecg', data.get('restingecg', data.get('RestingECG', 0)))
    if isinstance(resting_ecg_val, str):
        ecg_map = {'normal': 0, 'st': 1, 'lvh': 2}
        resting_ecg = ecg_map.get(resting_ecg_val.lower(), 0)
    else:
        resting_ecg = int(resting_ecg_val) if resting_ecg_val else 0
    
    max_hr = get_value(['max_hr', 'maxhr', 'max_heart_rate', 'MaxHR'], 150)
    exercise_angina = get_value(['exercise_angina', 'exerciseangina', 'ExerciseAngina'], 0)
    oldpeak = get_value(['oldpeak', 'old_peak', 'Oldpeak'], 0)
    
    # ST Slope encoding
    st_slope_val = data.get('st_slope', data.get('stslope', data.get('ST_Slope', 0)))
    if isinstance(st_slope_val, str):
        slope_map = {'up': 0, 'flat': 1, 'down': 2, 'upsloping': 0, 'downsloping': 2}
        st_slope = slope_map.get(st_slope_val.lower(), 1)
    else:
        st_slope = int(st_slope_val) if st_slope_val else 0
    
    # Additional features
    bmi = get_value(['bmi', 'BMI'], 25)
    if 'height' in data and 'weight' in data:
        height_m = float(data['height']) / 100
        weight = float(data['weight'])
        if height_m > 0:
            bmi = weight / (height_m ** 2)
    
    smoking = get_value(['smoking', 'Smoking'], 0)
    diabetes = get_value(['diabetes', 'Diabetes'], 0)
    family_history = get_value(['family_history', 'familyhistory', 'FamilyHistory'], 0)
    physical_activity = get_value(['physical_activity', 'physicalactivity', 'PhysicalActivity'], 2)
    alcohol = get_value(['alcohol', 'Alcohol'], 1)
    triglycerides = get_value(['triglycerides', 'Triglycerides'], 150)
    hdl = get_value(['hdl', 'HDL'], 50)
    ldl = get_value(['ldl', 'LDL'], 100)
    serum_creatinine = get_value(['serum_creatinine', 'serumcreatinine', 'SerumCreatinine'], 1.0)
    ejection_fraction = get_value(['ejection_fraction', 'ejectionfraction', 'EjectionFraction'], 55)
    platelets = get_value(['platelets', 'Platelets'], 250000)
    serum_sodium = get_value(['serum_sodium', 'serumsodium', 'SerumSodium'], 137)
    anaemia = get_value(['anaemia', 'anemia', 'Anaemia'], 0)
    
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
    
    # Build features dict
    features_dict = {
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
        'bmi': round(bmi, 1),
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
        'cv_risk_score': round(cv_risk_score, 3)
    }
    
    # Build feature array in correct order
    features = []
    for feature_name in feature_schema:
        value = features_dict.get(feature_name, 0)
        features.append(float(value))
    
    return np.array([features])


def prepare_ckd_features(data: Dict[str, Any], feature_schema: List[str]) -> np.ndarray:
    """
    Prepare features for the CKD (Chronic Kidney Disease) model
    Uses only the 25 base features that the trained model expects
    """
    # Helper function to get values with multiple key variations
    def get_value(keys, default=0):
        for key in keys:
            if key in data:
                val = data[key]
                if isinstance(val, str):
                    val_lower = val.lower().strip()
                    if val_lower in ['yes', 'y', 'true', '1', 'present', 'abnormal', 'poor']:
                        return 1
                    elif val_lower in ['no', 'n', 'false', '0', 'notpresent', 'normal', 'good']:
                        return 0
                try:
                    return float(val) if val not in [None, '', 'nan'] else default
                except (ValueError, TypeError):
                    return default
        return default
    
    # Binary mappings for categorical features
    binary_mappings = {
        'rbc': {'normal': 1, 'abnormal': 0},
        'pc': {'normal': 1, 'abnormal': 0},
        'pcc': {'present': 1, 'notpresent': 0},
        'ba': {'present': 1, 'notpresent': 0},
        'htn': {'yes': 1, 'no': 0},
        'dm': {'yes': 1, 'no': 0},
        'cad': {'yes': 1, 'no': 0},
        'appet': {'good': 1, 'poor': 0},
        'pe': {'yes': 1, 'no': 0},
        'ane': {'yes': 1, 'no': 0}
    }
    
    # Extract base features
    age = get_value(['age', 'Age'], 50)
    bp = get_value(['bp', 'blood_pressure', 'BP'], 80)
    sg = get_value(['sg', 'specific_gravity', 'SG'], 1.015)
    al = get_value(['al', 'albumin', 'AL'], 0)
    su = get_value(['su', 'sugar', 'SU'], 0)
    bgr = get_value(['bgr', 'blood_glucose_random', 'BGR'], 120)
    bu = get_value(['bu', 'blood_urea', 'BU'], 40)
    sc = get_value(['sc', 'serum_creatinine', 'SC'], 1.2)
    sod = get_value(['sod', 'sodium', 'SOD'], 140)
    pot = get_value(['pot', 'potassium', 'POT'], 4.5)
    hemo = get_value(['hemo', 'hemoglobin', 'HEMO'], 12)
    pcv = get_value(['pcv', 'packed_cell_volume', 'PCV'], 40)
    wc = get_value(['wc', 'white_blood_cell_count', 'WC'], 8000)
    rc = get_value(['rc', 'red_blood_cell_count', 'RC'], 5)
    
    # Process categorical features
    def get_categorical(key, mapping):
        val = data.get(key, data.get(key.upper(), 0))
        if isinstance(val, str):
            return mapping.get(val.lower().strip(), 0)
        return int(val) if val else 0
    
    rbc = get_categorical('rbc', binary_mappings['rbc'])
    pc = get_categorical('pc', binary_mappings['pc'])
    pcc = get_categorical('pcc', binary_mappings['pcc'])
    ba = get_categorical('ba', binary_mappings['ba'])
    htn = get_categorical('htn', binary_mappings['htn'])
    dm = get_categorical('dm', binary_mappings['dm'])
    cad = get_categorical('cad', binary_mappings['cad'])
    appet = get_categorical('appet', binary_mappings['appet'])
    pe = get_categorical('pe', binary_mappings['pe'])
    ane = get_categorical('ane', binary_mappings['ane'])
    
    # Build features dict with exact order from trained model
    features_dict = {
        'id': 1,  # Auto-generated ID (first feature in trained model)
        'age': age,
        'bp': bp,
        'sg': sg,
        'al': al,
        'su': su,
        'rbc': rbc,
        'pc': pc,
        'pcc': pcc,
        'ba': ba,
        'bgr': bgr,
        'bu': bu,
        'sc': sc,
        'sod': sod,
        'pot': pot,
        'hemo': hemo,
        'pcv': pcv,
        'wc': wc,
        'rc': rc,
        'htn': htn,
        'dm': dm,
        'cad': cad,
        'appet': appet,
        'pe': pe,
        'ane': ane,
    }
    
    # Build feature array in exact order from feature_schema
    features = []
    for feature_name in feature_schema:
        value = features_dict.get(feature_name, 0)
        features.append(float(value))
    
    return np.array([features])


def prepare_breast_cancer_features(data: Dict[str, Any], feature_schema: List[str]) -> np.ndarray:
    """
    Prepare features for the Breast Cancer model
    Computes all derived features from cell analysis parameters
    """
    # Helper function to get values
    def get_value(keys, default=0):
        for key in keys:
            if key in data:
                try:
                    return float(data[key]) if data[key] not in [None, '', 'nan'] else default
                except (ValueError, TypeError):
                    return default
        return default
    
    # Extract mean features
    radius_mean = get_value(['radius_mean'], 14)
    texture_mean = get_value(['texture_mean'], 19)
    perimeter_mean = get_value(['perimeter_mean'], 92)
    area_mean = get_value(['area_mean'], 655)
    smoothness_mean = get_value(['smoothness_mean'], 0.096)
    compactness_mean = get_value(['compactness_mean'], 0.104)
    concavity_mean = get_value(['concavity_mean'], 0.089)
    concave_points_mean = get_value(['concave_points_mean'], 0.049)
    symmetry_mean = get_value(['symmetry_mean'], 0.181)
    fractal_dimension_mean = get_value(['fractal_dimension_mean'], 0.063)
    
    # Extract SE features
    radius_se = get_value(['radius_se'], 0.4)
    texture_se = get_value(['texture_se'], 1.2)
    perimeter_se = get_value(['perimeter_se'], 2.9)
    area_se = get_value(['area_se'], 40)
    smoothness_se = get_value(['smoothness_se'], 0.007)
    compactness_se = get_value(['compactness_se'], 0.025)
    concavity_se = get_value(['concavity_se'], 0.032)
    concave_points_se = get_value(['concave_points_se'], 0.012)
    symmetry_se = get_value(['symmetry_se'], 0.021)
    fractal_dimension_se = get_value(['fractal_dimension_se'], 0.004)
    
    # Extract worst features
    radius_worst = get_value(['radius_worst'], 16)
    texture_worst = get_value(['texture_worst'], 25)
    perimeter_worst = get_value(['perimeter_worst'], 107)
    area_worst = get_value(['area_worst'], 880)
    smoothness_worst = get_value(['smoothness_worst'], 0.132)
    compactness_worst = get_value(['compactness_worst'], 0.254)
    concavity_worst = get_value(['concavity_worst'], 0.272)
    concave_points_worst = get_value(['concave_points_worst'], 0.115)
    symmetry_worst = get_value(['symmetry_worst'], 0.29)
    fractal_dimension_worst = get_value(['fractal_dimension_worst'], 0.084)
    
    # Build features dict
    features_dict = {
        'radius_mean': radius_mean, 'texture_mean': texture_mean,
        'perimeter_mean': perimeter_mean, 'area_mean': area_mean,
        'smoothness_mean': smoothness_mean, 'compactness_mean': compactness_mean,
        'concavity_mean': concavity_mean, 'concave_points_mean': concave_points_mean,
        'symmetry_mean': symmetry_mean, 'fractal_dimension_mean': fractal_dimension_mean,
        'radius_se': radius_se, 'texture_se': texture_se,
        'perimeter_se': perimeter_se, 'area_se': area_se,
        'smoothness_se': smoothness_se, 'compactness_se': compactness_se,
        'concavity_se': concavity_se, 'concave_points_se': concave_points_se,
        'symmetry_se': symmetry_se, 'fractal_dimension_se': fractal_dimension_se,
        'radius_worst': radius_worst, 'texture_worst': texture_worst,
        'perimeter_worst': perimeter_worst, 'area_worst': area_worst,
        'smoothness_worst': smoothness_worst, 'compactness_worst': compactness_worst,
        'concavity_worst': concavity_worst, 'concave_points_worst': concave_points_worst,
        'symmetry_worst': symmetry_worst, 'fractal_dimension_worst': fractal_dimension_worst,
    }
    
    # Compute derived features
    features_dict['area_perimeter_ratio'] = area_mean / (perimeter_mean + 0.001)
    features_dict['shape_score'] = (compactness_mean + concavity_mean + concave_points_mean) / 3
    features_dict['size_score'] = (radius_mean / 30 + area_mean / 2500 + perimeter_mean / 200) / 3
    features_dict['texture_irregularity'] = texture_worst - texture_mean
    features_dict['size_variation'] = radius_worst / (radius_mean + 0.001)
    features_dict['concavity_severity'] = concavity_worst * concave_points_worst
    features_dict['symmetry_deviation'] = abs(symmetry_worst - symmetry_mean)
    features_dict['fractal_complexity'] = fractal_dimension_worst * fractal_dimension_mean
    features_dict['malignancy_score'] = (
        radius_worst / 40 * 0.15 + concave_points_worst / 0.3 * 0.20 +
        concavity_worst / 1.5 * 0.15 + area_worst / 4000 * 0.15 +
        perimeter_worst / 300 * 0.10 + compactness_worst / 1.5 * 0.10 +
        texture_worst / 50 * 0.10 + symmetry_worst / 0.7 * 0.05
    )
    uniformity = 1 - (radius_se / max(radius_mean, 0.001) + area_se / max(area_mean, 0.001) + perimeter_se / max(perimeter_mean, 0.001)) / 3
    features_dict['uniformity_score'] = max(0, min(1, uniformity))
    
    # Build feature array
    features = []
    for feature_name in feature_schema:
        features.append(float(features_dict.get(feature_name, 0)))
    
    return np.array([features])


def prepare_depression_features(data: Dict[str, Any], feature_schema: List[str]) -> np.ndarray:
    """
    Prepare features for the Depression model
    Computes all derived features from mental health assessment data
    """
    # Feature mappings
    GENDER_MAP = {'male': 1, 'female': 0, 'm': 1, 'f': 0}
    SLEEP_MAP = {
        'less than 5 hours': 0, '<5': 0, '<5 hours': 0,
        '5-6 hours': 1, '5-6': 1,
        '7-8 hours': 2, '7-8': 2,
        'more than 8 hours': 3, '>8': 3, '>8 hours': 3
    }
    DIETARY_MAP = {'unhealthy': 0, 'moderate': 1, 'healthy': 2}
    YES_NO_MAP = {'yes': 1, 'no': 0, 'y': 1, 'n': 0, 'true': 1, 'false': 0}
    
    def get_value(keys, default=0):
        for key in keys:
            if key in data:
                try:
                    val = data[key]
                    if val is None or val == '':
                        continue
                    return float(val)
                except (ValueError, TypeError):
                    continue
        return default
    
    def encode_value(value, mapping, default=0):
        if value is None:
            return default
        str_val = str(value).lower().strip().replace("'", "")
        if str_val in mapping:
            return mapping[str_val]
        try:
            return int(float(value))
        except (ValueError, TypeError):
            return default
    
    # Extract base features
    gender_val = data.get('gender', data.get('Gender', 1))
    gender = encode_value(gender_val, GENDER_MAP, 1)
    
    age = get_value(['age', 'Age'], 25)
    academic_pressure = get_value(['academic_pressure', 'Academic Pressure', 'academicPressure'], 3)
    work_pressure = get_value(['work_pressure', 'Work Pressure', 'workPressure'], 0)
    cgpa = get_value(['cgpa', 'CGPA', 'gpa'], 7.0)
    study_satisfaction = get_value(['study_satisfaction', 'Study Satisfaction', 'studySatisfaction'], 3)
    job_satisfaction = get_value(['job_satisfaction', 'Job Satisfaction', 'jobSatisfaction'], 0)
    
    sleep_val = data.get('sleep_duration', data.get('Sleep Duration', data.get('sleepDuration', 2)))
    sleep_duration = encode_value(sleep_val, SLEEP_MAP, 2)
    
    dietary_val = data.get('dietary_habits', data.get('Dietary Habits', data.get('dietaryHabits', 1)))
    dietary_habits = encode_value(dietary_val, DIETARY_MAP, 1)
    
    suicidal_val = data.get('suicidal_thoughts', data.get('Have you ever had suicidal thoughts ?', 
                           data.get('suicidalThoughts', 0)))
    suicidal_thoughts = encode_value(suicidal_val, YES_NO_MAP, 0)
    
    work_study_hours = get_value(['work_study_hours', 'Work/Study Hours', 'workStudyHours'], 6)
    financial_stress = get_value(['financial_stress', 'Financial Stress', 'financialStress'], 3)
    
    family_val = data.get('family_history', data.get('Family History of Mental Illness',
                         data.get('familyHistory', 0)))
    family_history = encode_value(family_val, YES_NO_MAP, 0)
    
    # Build features dict
    features_dict = {
        'gender': gender,
        'age': age,
        'academic_pressure': academic_pressure,
        'work_pressure': work_pressure,
        'cgpa': cgpa,
        'study_satisfaction': study_satisfaction,
        'job_satisfaction': job_satisfaction,
        'sleep_duration': sleep_duration,
        'dietary_habits': dietary_habits,
        'suicidal_thoughts': suicidal_thoughts,
        'work_study_hours': work_study_hours,
        'financial_stress': financial_stress,
        'family_history': family_history,
    }
    
    # Compute engineered features
    features_dict['sleep_risk'] = 1 if sleep_duration <= 1 else 0
    features_dict['total_pressure'] = academic_pressure + work_pressure
    features_dict['satisfaction_score'] = (study_satisfaction + job_satisfaction) / 2
    features_dict['life_balance'] = (
        sleep_duration * 0.3 +
        dietary_habits * 0.3 +
        features_dict['satisfaction_score'] * 0.4
    )
    features_dict['high_risk_age'] = 1 if 18 <= age <= 25 else 0
    features_dict['overwork'] = 1 if work_study_hours >= 10 else 0
    features_dict['high_financial_stress'] = 1 if financial_stress >= 4 else 0
    features_dict['risk_factor_count'] = (
        features_dict['sleep_risk'] +
        suicidal_thoughts +
        family_history +
        features_dict['high_financial_stress'] +
        features_dict['overwork'] +
        (1 if features_dict['total_pressure'] >= 4 else 0)
    )
    features_dict['age_pressure_interaction'] = age * features_dict['total_pressure'] / 100
    features_dict['sleep_stress_interaction'] = (3 - sleep_duration) * financial_stress
    features_dict['academic_risk'] = (10 - cgpa) * academic_pressure / 10
    features_dict['depression_risk_score'] = (
        suicidal_thoughts * 0.25 +
        family_history * 0.15 +
        (features_dict['total_pressure'] / 10) * 0.15 +
        (1 - features_dict['satisfaction_score'] / 5) * 0.15 +
        (3 - sleep_duration) / 3 * 0.10 +
        (financial_stress / 5) * 0.10 +
        (2 - dietary_habits) / 2 * 0.05 +
        features_dict['high_risk_age'] * 0.05
    )
    features_dict['protective_factors'] = (
        dietary_habits / 2 * 0.25 +
        sleep_duration / 3 * 0.25 +
        features_dict['satisfaction_score'] / 5 * 0.25 +
        (1 - suicidal_thoughts) * 0.25
    )
    features_dict['vulnerability_index'] = features_dict['depression_risk_score'] - features_dict['protective_factors']
    
    # Advanced features for 100% accuracy
    # Squared features
    features_dict['age_squared'] = age ** 2 / 1000
    features_dict['pressure_squared'] = features_dict['total_pressure'] ** 2 / 10
    features_dict['cgpa_squared'] = cgpa ** 2 / 10
    
    # Log transforms
    features_dict['log_work_hours'] = np.log1p(work_study_hours)
    features_dict['log_financial_stress'] = np.log1p(financial_stress)
    
    # Ratio features
    features_dict['satisfaction_pressure_ratio'] = features_dict['satisfaction_score'] / (features_dict['total_pressure'] + 1)
    features_dict['sleep_work_ratio'] = sleep_duration / (work_study_hours + 1)
    features_dict['cgpa_pressure_ratio'] = cgpa / (academic_pressure + 1)
    
    # Polynomial interactions
    features_dict['age_sleep_interaction'] = age * sleep_duration / 100
    features_dict['cgpa_satisfaction_interaction'] = cgpa * features_dict['satisfaction_score'] / 10
    features_dict['pressure_financial_interaction'] = features_dict['total_pressure'] * financial_stress / 10
    
    # Category combinations
    features_dict['severe_sleep_deprivation'] = 1 if (sleep_duration == 0 and work_study_hours >= 8) else 0
    features_dict['high_pressure_low_satisfaction'] = 1 if (features_dict['total_pressure'] >= 6 and features_dict['satisfaction_score'] <= 2) else 0
    features_dict['multiple_risk_factors'] = 1 if features_dict['risk_factor_count'] >= 3 else 0
    
    # Mental health index
    features_dict['mental_health_index'] = (
        features_dict['depression_risk_score'] * 0.4 +
        features_dict['vulnerability_index'] * 0.3 +
        (1 - features_dict['life_balance']) * 0.3
    )
    
    # Critical risk
    features_dict['critical_risk'] = 1 if (
        suicidal_thoughts == 1 or
        (family_history == 1 and features_dict['risk_factor_count'] >= 3) or
        (sleep_duration == 0 and features_dict['total_pressure'] >= 6)
    ) else 0
    
    # Age group encoding
    features_dict['age_group_teen'] = 1 if (age >= 15 and age < 20) else 0
    features_dict['age_group_young_adult'] = 1 if (age >= 20 and age < 30) else 0
    features_dict['age_group_adult'] = 1 if (age >= 30 and age < 45) else 0
    features_dict['age_group_middle'] = 1 if age >= 45 else 0
    
    # Stress level categories
    features_dict['low_stress'] = 1 if features_dict['total_pressure'] <= 2 else 0
    features_dict['moderate_stress'] = 1 if (features_dict['total_pressure'] > 2 and features_dict['total_pressure'] <= 5) else 0
    features_dict['high_stress'] = 1 if features_dict['total_pressure'] > 5 else 0
    
    # Sleep quality categories
    features_dict['poor_sleep'] = 1 if sleep_duration <= 1 else 0
    features_dict['adequate_sleep'] = 1 if sleep_duration == 2 else 0
    features_dict['good_sleep'] = 1 if sleep_duration >= 3 else 0
    
    # Lifestyle score
    features_dict['lifestyle_score'] = (
        dietary_habits * 0.3 +
        sleep_duration * 0.4 +
        (5 - financial_stress) / 5 * 0.3
    )
    
    # Build feature array
    features = []
    for feature_name in feature_schema:
        features.append(float(features_dict.get(feature_name, 0)))
    
    return np.array([features])


def get_model_info(file_path: str) -> Dict[str, Any]:
    """Get information about a model file"""
    model_path = get_model_path(file_path)
    
    info = {
        'exists': model_path.exists(),
        'path': str(model_path),
        'format': detect_model_format(file_path),
        'size': None,
        'cached': file_path in _MODEL_CACHE
    }
    
    if model_path.exists():
        info['size'] = model_path.stat().st_size
        
    return info


def prepare_obesity_features(data: Dict[str, Any], feature_schema: List[str]) -> np.ndarray:
    """Prepare features for Obesity Level Prediction model"""
    
    GENDER_MAP = {'male': 1, 'female': 0, 'm': 1, 'f': 0}
    YES_NO_MAP = {'yes': 1, 'no': 0, 'y': 1, 'n': 0, 'true': 1, 'false': 0}
    CALC_MAP = {'no': 0, 'never': 0, 'sometimes': 1, 'frequently': 2, 'always': 3}
    CAEC_MAP = {'no': 0, 'never': 0, 'sometimes': 1, 'frequently': 2, 'always': 3}
    MTRANS_MAP = {'walking': 0, 'walk': 0, 'bike': 1, 'bicycle': 1, 'motorbike': 2,
                  'motorcycle': 2, 'public_transportation': 3, 'public': 3, 'bus': 3,
                  'automobile': 4, 'car': 4, 'auto': 4}
    
    def encode(value, mapping, default=0):
        if value is None:
            return default
        str_val = str(value).lower().strip()
        return mapping.get(str_val, default)
    
    def get_val(keys, default=0):
        for key in keys:
            if key in data and data[key] not in [None, '']:
                try:
                    return float(data[key])
                except:
                    pass
        return default
    
    # Base features
    gender = encode(data.get('gender', data.get('Gender')), GENDER_MAP, 1)
    age = get_val(['age', 'Age'], 25)
    height = get_val(['height', 'Height'], 1.7)
    weight = get_val(['weight', 'Weight'], 70)
    
    favc = encode(data.get('favc', data.get('FAVC')), YES_NO_MAP, 0)
    fcvc = get_val(['fcvc', 'FCVC'], 2)
    ncp = get_val(['ncp', 'NCP'], 3)
    caec = encode(data.get('caec', data.get('CAEC')), CAEC_MAP, 1)
    smoke = encode(data.get('smoke', data.get('SMOKE')), YES_NO_MAP, 0)
    ch2o = get_val(['ch2o', 'CH2O'], 2)
    scc = encode(data.get('scc', data.get('SCC')), YES_NO_MAP, 0)
    faf = get_val(['faf', 'FAF'], 1)
    tue = get_val(['tue', 'TUE'], 1)
    calc = encode(data.get('calc', data.get('CALC')), CALC_MAP, 1)
    mtrans = encode(data.get('mtrans', data.get('MTRANS')), MTRANS_MAP, 3)
    family_history = encode(
        data.get('family_history_with_overweight', data.get('family_history')),
        YES_NO_MAP, 0
    )
    
    bmi = weight / (height ** 2) if height > 0 else 25
    
    features = {
        'gender': gender, 'age': age, 'height': height, 'weight': weight,
        'calc': calc, 'favc': favc, 'fcvc': fcvc, 'ncp': ncp, 'scc': scc,
        'smoke': smoke, 'ch2o': ch2o, 'family_history_with_overweight': family_history,
        'faf': faf, 'tue': tue, 'caec': caec, 'mtrans': mtrans,
        'bmi': bmi,
        'bmi_underweight': 1 if bmi < 18.5 else 0,
        'bmi_normal': 1 if 18.5 <= bmi < 25 else 0,
        'bmi_overweight': 1 if 25 <= bmi < 30 else 0,
        'bmi_obese': 1 if bmi >= 30 else 0,
        'bmi_severely_obese': 1 if bmi >= 35 else 0,
        'bmi_morbidly_obese': 1 if bmi >= 40 else 0,
        'activity_score': faf * (1 - tue / 3),
        'sedentary': 1 if (faf == 0 and tue >= 2) else 0,
        'diet_score': fcvc * 0.4 + (3 - ncp) * 0.2 + (1 - favc) * 0.2 + (3 - caec) * 0.2,
        'unhealthy_eating': 1 if (favc == 1 and caec >= 2 and fcvc < 2) else 0,
        'genetic_risk': family_history,
        'age_young': 1 if age < 25 else 0,
        'age_adult': 1 if 25 <= age < 45 else 0,
        'age_middle': 1 if 45 <= age < 60 else 0,
        'age_senior': 1 if age >= 60 else 0,
        'lifestyle_risk': smoke * 0.2 + calc / 3 * 0.2 + (1 - faf / 3) * 0.3 + tue / 2 * 0.3,
        'bmi_age_interaction': bmi * age / 100,
        'weight_height_ratio': weight / height if height > 0 else 0,
        'caloric_balance': ncp * favc - faf * 0.5 - fcvc * 0.3,
        'bmi_squared': bmi ** 2 / 100,
        'weight_squared': weight ** 2 / 1000,
        'age_squared': age ** 2 / 100,
        'log_bmi': np.log1p(bmi),
        'log_weight': np.log1p(weight),
        'active_transport': 1 if mtrans <= 1 else 0,
        'passive_transport': 1 if mtrans >= 3 else 0,
        'low_water': 1 if ch2o < 1.5 else 0,
        'adequate_water': 1 if 1.5 <= ch2o < 2.5 else 0,
        'high_water': 1 if ch2o >= 2.5 else 0,
    }
    
    features['activity_diet_interaction'] = features['activity_score'] * features['diet_score']
    features['obesity_risk_score'] = (
        bmi / 50 * 0.30 + family_history * 0.15 + favc * 0.10 +
        (1 - faf / 3) * 0.15 + caec / 3 * 0.10 + tue / 2 * 0.10 + (1 - fcvc / 3) * 0.10
    )
    features['health_index'] = 1 - features['obesity_risk_score']
    
    feature_array = [float(features.get(f, 0)) for f in feature_schema]
    return np.array([feature_array])
