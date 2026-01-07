"""
CKD Prediction Module - Production Ready
Provides prediction functionality for the trained CKD model
"""

import pickle
import numpy as np
import os
from pathlib import Path


class CKDPredictor:
    """CKD Prediction class for production use"""
    
    def __init__(self, model_dir=None):
        if model_dir is None:
            model_dir = Path(__file__).parent
        else:
            model_dir = Path(model_dir)
        
        self.model_dir = model_dir
        self.model = None
        self.scaler = None
        self.power_transformer = None
        self.imputer = None
        self.feature_names = None
        self.label_encoders = None
        
        self._load_model()
    
    def _load_model(self):
        """Load all model components"""
        # Load model
        with open(self.model_dir / 'model.pkl', 'rb') as f:
            self.model = pickle.load(f)
        
        # Load scaler
        with open(self.model_dir / 'scaler.pkl', 'rb') as f:
            scaler_dict = pickle.load(f)
            self.scaler = scaler_dict['scaler']
            self.power_transformer = scaler_dict.get('power_transformer')
        
        # Load imputer
        with open(self.model_dir / 'imputer.pkl', 'rb') as f:
            self.imputer = pickle.load(f)
        
        # Load feature names
        with open(self.model_dir / 'features.pkl', 'rb') as f:
            self.feature_names = pickle.load(f)
        
        # Load label encoders if exists
        label_encoders_path = self.model_dir / 'label_encoders.pkl'
        if label_encoders_path.exists():
            with open(label_encoders_path, 'rb') as f:
                self.label_encoders = pickle.load(f)
    
    def preprocess_input(self, data: dict) -> np.ndarray:
        """Preprocess input data for prediction"""
        
        # Binary mappings for categorical features
        binary_mappings = {
            'rbc': {'normal': 1, 'abnormal': 0},
            'pc': {'normal': 1, 'abnormal': 0},
            'pcc': {'present': 1, 'notpresent': 0},
            'ba': {'present': 1, 'notpresent': 0},
            'htn': {'yes': 1, 'no': 0, 1: 1, 0: 0, '1': 1, '0': 0},
            'dm': {'yes': 1, 'no': 0, 1: 1, 0: 0, '1': 1, '0': 0},
            'cad': {'yes': 1, 'no': 0, 1: 1, 0: 0, '1': 1, '0': 0},
            'appet': {'good': 1, 'poor': 0},
            'pe': {'yes': 1, 'no': 0, 1: 1, 0: 0, '1': 1, '0': 0},
            'ane': {'yes': 1, 'no': 0, 1: 1, 0: 0, '1': 1, '0': 0}
        }
        
        # Process input data
        processed = {}
        
        for key, value in data.items():
            key_lower = key.lower().replace(' ', '_')
            
            # Handle categorical features
            if key_lower in binary_mappings:
                if isinstance(value, str):
                    value = value.lower().strip()
                processed[key_lower] = binary_mappings[key_lower].get(value, 0)
            else:
                # Numeric features
                try:
                    processed[key_lower] = float(value) if value not in [None, '', 'nan'] else 0
                except (ValueError, TypeError):
                    processed[key_lower] = 0
        
        # Engineer features (same as training)
        sc = processed.get('sc', 1.0)
        age = processed.get('age', 50)
        hemo = processed.get('hemo', 12)
        bp = processed.get('bp', 80)
        al = processed.get('al', 0)
        bu = processed.get('bu', 20)
        sod = processed.get('sod', 140)
        pot = processed.get('pot', 4.0)
        sg = processed.get('sg', 1.015)
        
        # eGFR estimate
        sc_safe = max(sc, 0.1)
        processed['egfr_estimate'] = min(150, 141 * np.power(sc_safe / 0.9, -1.209) * np.power(0.993, age))
        
        # Anemia score
        if hemo < 7:
            processed['anemia_score'] = 3
        elif hemo < 10:
            processed['anemia_score'] = 2
        elif hemo < 12:
            processed['anemia_score'] = 1
        else:
            processed['anemia_score'] = 0
        
        # BP category
        if bp < 80:
            processed['bp_category'] = 0
        elif bp < 90:
            processed['bp_category'] = 1
        elif bp < 120:
            processed['bp_category'] = 2
        else:
            processed['bp_category'] = 3
        
        # Albumin-Creatinine ratio
        processed['albumin_creatinine_ratio'] = al / sc_safe
        
        # Urea-Creatinine ratio
        processed['urea_creatinine_ratio'] = bu / sc_safe
        
        # Electrolyte score
        processed['electrolyte_score'] = (
            (1 if sod < 135 else 0) + 
            (1 if sod > 145 else 0) +
            (1 if pot < 3.5 else 0) + 
            (1 if pot > 5.0 else 0)
        )
        
        # Comorbidity count
        processed['comorbidity_count'] = (
            processed.get('htn', 0) + 
            processed.get('dm', 0) + 
            processed.get('cad', 0) + 
            processed.get('ane', 0)
        )
        
        # Age risk
        if age < 40:
            processed['age_risk'] = 0
        elif age < 60:
            processed['age_risk'] = 1
        else:
            processed['age_risk'] = 2
        
        # SG abnormal
        processed['sg_abnormal'] = 1 if (sg < 1.010 or sg > 1.025) else 0
        
        # CKD risk score
        processed['ckd_risk_score'] = (
            (0.25 if sc > 1.2 else 0) +
            (0.15 if bu > 20 else 0) +
            (0.15 if hemo < 12 else 0) +
            (0.20 if al > 0 else 0) +
            (0.10 if processed.get('htn', 0) == 1 else 0) +
            (0.10 if processed.get('dm', 0) == 1 else 0) +
            (0.05 if age > 60 else 0)
        )
        
        # Build feature array in correct order
        features = []
        for feature_name in self.feature_names:
            features.append(processed.get(feature_name, 0))
        
        return np.array([features])
    
    def predict(self, data: dict) -> dict:
        """Make prediction for input data"""
        
        # Preprocess
        features = self.preprocess_input(data)
        
        # Impute
        features = self.imputer.transform(features)
        
        # Scale
        features = self.scaler.transform(features)
        
        # Power transform
        if self.power_transformer is not None:
            features = self.power_transformer.transform(features)
        
        # Predict
        prediction = self.model.predict(features)[0]
        probability = self.model.predict_proba(features)[0]
        
        # Get probability of CKD (class 1)
        ckd_probability = probability[1] * 100
        
        # Determine risk level
        if ckd_probability < 30:
            risk_level = 'low'
        elif ckd_probability < 60:
            risk_level = 'medium'
        else:
            risk_level = 'high'
        
        return {
            'prediction': int(prediction),
            'probability': round(ckd_probability, 2),
            'risk_level': risk_level,
            'message': f'CKD Risk Assessment: {risk_level.title()} Risk ({ckd_probability:.1f}%)'
        }


# Singleton instance for quick access
_predictor = None

def get_predictor():
    """Get or create predictor instance"""
    global _predictor
    if _predictor is None:
        _predictor = CKDPredictor()
    return _predictor


def predict(data: dict) -> dict:
    """Quick prediction function"""
    return get_predictor().predict(data)


if __name__ == "__main__":
    # Test prediction
    test_data = {
        'age': 60,
        'bp': 90,
        'sg': 1.015,
        'al': 2,
        'su': 0,
        'rbc': 'normal',
        'pc': 'abnormal',
        'pcc': 'present',
        'ba': 'notpresent',
        'bgr': 150,
        'bu': 50,
        'sc': 2.5,
        'sod': 135,
        'pot': 4.5,
        'hemo': 10,
        'pcv': 32,
        'wc': 8000,
        'rc': 4.0,
        'htn': 'yes',
        'dm': 'yes',
        'cad': 'no',
        'appet': 'poor',
        'pe': 'yes',
        'ane': 'yes'
    }
    
    result = predict(test_data)
    print(f"Prediction: {result}")
