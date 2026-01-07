"""
Breast Cancer Prediction Module - Production Ready
Provides prediction functionality for the trained breast cancer model
"""

import pickle
import numpy as np
from pathlib import Path


class BreastCancerPredictor:
    """Breast Cancer Prediction class for production use"""
    
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
        
        self._load_model()
    
    def _load_model(self):
        """Load all model components"""
        with open(self.model_dir / 'model.pkl', 'rb') as f:
            self.model = pickle.load(f)
        
        with open(self.model_dir / 'scaler.pkl', 'rb') as f:
            scaler_dict = pickle.load(f)
            self.scaler = scaler_dict['scaler']
            self.power_transformer = scaler_dict.get('power_transformer')
        
        with open(self.model_dir / 'imputer.pkl', 'rb') as f:
            self.imputer = pickle.load(f)
        
        with open(self.model_dir / 'features.pkl', 'rb') as f:
            self.feature_names = pickle.load(f)
    
    def preprocess_input(self, data: dict) -> np.ndarray:
        """Preprocess input data for prediction"""
        
        # Normalize keys
        normalized_data = {}
        for key, value in data.items():
            key_lower = key.lower().replace(' ', '_')
            try:
                normalized_data[key_lower] = float(value) if value not in [None, '', 'nan'] else 0
            except (ValueError, TypeError):
                normalized_data[key_lower] = 0
        
        # Extract base features
        radius_mean = normalized_data.get('radius_mean', 14)
        texture_mean = normalized_data.get('texture_mean', 19)
        perimeter_mean = normalized_data.get('perimeter_mean', 92)
        area_mean = normalized_data.get('area_mean', 655)
        smoothness_mean = normalized_data.get('smoothness_mean', 0.096)
        compactness_mean = normalized_data.get('compactness_mean', 0.104)
        concavity_mean = normalized_data.get('concavity_mean', 0.089)
        concave_points_mean = normalized_data.get('concave_points_mean', 0.049)
        symmetry_mean = normalized_data.get('symmetry_mean', 0.181)
        fractal_dimension_mean = normalized_data.get('fractal_dimension_mean', 0.063)

        # SE features
        radius_se = normalized_data.get('radius_se', 0.4)
        texture_se = normalized_data.get('texture_se', 1.2)
        perimeter_se = normalized_data.get('perimeter_se', 2.9)
        area_se = normalized_data.get('area_se', 40)
        smoothness_se = normalized_data.get('smoothness_se', 0.007)
        compactness_se = normalized_data.get('compactness_se', 0.025)
        concavity_se = normalized_data.get('concavity_se', 0.032)
        concave_points_se = normalized_data.get('concave_points_se', 0.012)
        symmetry_se = normalized_data.get('symmetry_se', 0.021)
        fractal_dimension_se = normalized_data.get('fractal_dimension_se', 0.004)
        
        # Worst features
        radius_worst = normalized_data.get('radius_worst', 16)
        texture_worst = normalized_data.get('texture_worst', 25)
        perimeter_worst = normalized_data.get('perimeter_worst', 107)
        area_worst = normalized_data.get('area_worst', 880)
        smoothness_worst = normalized_data.get('smoothness_worst', 0.132)
        compactness_worst = normalized_data.get('compactness_worst', 0.254)
        concavity_worst = normalized_data.get('concavity_worst', 0.272)
        concave_points_worst = normalized_data.get('concave_points_worst', 0.115)
        symmetry_worst = normalized_data.get('symmetry_worst', 0.29)
        fractal_dimension_worst = normalized_data.get('fractal_dimension_worst', 0.084)
        
        # Build features dict with base features
        features_dict = {
            'radius_mean': radius_mean,
            'texture_mean': texture_mean,
            'perimeter_mean': perimeter_mean,
            'area_mean': area_mean,
            'smoothness_mean': smoothness_mean,
            'compactness_mean': compactness_mean,
            'concavity_mean': concavity_mean,
            'concave_points_mean': concave_points_mean,
            'symmetry_mean': symmetry_mean,
            'fractal_dimension_mean': fractal_dimension_mean,
            'radius_se': radius_se,
            'texture_se': texture_se,
            'perimeter_se': perimeter_se,
            'area_se': area_se,
            'smoothness_se': smoothness_se,
            'compactness_se': compactness_se,
            'concavity_se': concavity_se,
            'concave_points_se': concave_points_se,
            'symmetry_se': symmetry_se,
            'fractal_dimension_se': fractal_dimension_se,
            'radius_worst': radius_worst,
            'texture_worst': texture_worst,
            'perimeter_worst': perimeter_worst,
            'area_worst': area_worst,
            'smoothness_worst': smoothness_worst,
            'compactness_worst': compactness_worst,
            'concavity_worst': concavity_worst,
            'concave_points_worst': concave_points_worst,
            'symmetry_worst': symmetry_worst,
            'fractal_dimension_worst': fractal_dimension_worst,
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
        
        # Malignancy score
        features_dict['malignancy_score'] = (
            radius_worst / 40 * 0.15 +
            concave_points_worst / 0.3 * 0.20 +
            concavity_worst / 1.5 * 0.15 +
            area_worst / 4000 * 0.15 +
            perimeter_worst / 300 * 0.10 +
            compactness_worst / 1.5 * 0.10 +
            texture_worst / 50 * 0.10 +
            symmetry_worst / 0.7 * 0.05
        )
        
        # Uniformity score
        uniformity = 1 - (
            radius_se / max(radius_mean, 0.001) +
            area_se / max(area_mean, 0.001) +
            perimeter_se / max(perimeter_mean, 0.001)
        ) / 3
        features_dict['uniformity_score'] = max(0, min(1, uniformity))
        
        # Build feature array in correct order
        features = []
        for feature_name in self.feature_names:
            features.append(features_dict.get(feature_name, 0))
        
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
        
        # Get probability of Malignant (class 1)
        malignant_probability = probability[1] * 100
        
        # Determine risk level and diagnosis
        if malignant_probability < 30:
            risk_level = 'low'
            diagnosis = 'Benign'
        elif malignant_probability < 60:
            risk_level = 'medium'
            diagnosis = 'Uncertain'
        else:
            risk_level = 'high'
            diagnosis = 'Malignant'
        
        return {
            'prediction': int(prediction),
            'probability': round(malignant_probability, 2),
            'risk_level': risk_level,
            'diagnosis': diagnosis,
            'message': f'Breast Cancer Assessment: {diagnosis} ({malignant_probability:.1f}% malignancy probability)'
        }


# Singleton instance for quick access
_predictor = None

def get_predictor():
    """Get or create predictor instance"""
    global _predictor
    if _predictor is None:
        _predictor = BreastCancerPredictor()
    return _predictor


def predict(data: dict) -> dict:
    """Quick prediction function"""
    return get_predictor().predict(data)


if __name__ == "__main__":
    # Test prediction - Malignant case
    test_malignant = {
        'radius_mean': 20.57, 'texture_mean': 17.77, 'perimeter_mean': 132.9,
        'area_mean': 1326, 'smoothness_mean': 0.08474, 'compactness_mean': 0.07864,
        'concavity_mean': 0.0869, 'concave_points_mean': 0.07017, 'symmetry_mean': 0.1812,
        'fractal_dimension_mean': 0.05667, 'radius_se': 0.5435, 'texture_se': 0.7339,
        'perimeter_se': 3.398, 'area_se': 74.08, 'smoothness_se': 0.005225,
        'compactness_se': 0.01308, 'concavity_se': 0.0186, 'concave_points_se': 0.0134,
        'symmetry_se': 0.01389, 'fractal_dimension_se': 0.003532, 'radius_worst': 24.99,
        'texture_worst': 23.41, 'perimeter_worst': 158.8, 'area_worst': 1956,
        'smoothness_worst': 0.1238, 'compactness_worst': 0.1866, 'concavity_worst': 0.2416,
        'concave_points_worst': 0.186, 'symmetry_worst': 0.275, 'fractal_dimension_worst': 0.08902
    }
    
    result = predict(test_malignant)
    print(f"Malignant Test: {result}")
    
    # Test prediction - Benign case
    test_benign = {
        'radius_mean': 12.45, 'texture_mean': 15.7, 'perimeter_mean': 82.57,
        'area_mean': 477.1, 'smoothness_mean': 0.1278, 'compactness_mean': 0.17,
        'concavity_mean': 0.1578, 'concave_points_mean': 0.08089, 'symmetry_mean': 0.2087,
        'fractal_dimension_mean': 0.07613, 'radius_se': 0.3345, 'texture_se': 0.8902,
        'perimeter_se': 2.217, 'area_se': 27.19, 'smoothness_se': 0.00751,
        'compactness_se': 0.03345, 'concavity_se': 0.03672, 'concave_points_se': 0.01137,
        'symmetry_se': 0.02165, 'fractal_dimension_se': 0.005082, 'radius_worst': 15.47,
        'texture_worst': 23.75, 'perimeter_worst': 103.4, 'area_worst': 741.6,
        'smoothness_worst': 0.1791, 'compactness_worst': 0.5249, 'concavity_worst': 0.5355,
        'concave_points_worst': 0.1741, 'symmetry_worst': 0.3985, 'fractal_dimension_worst': 0.1244
    }
    
    result = predict(test_benign)
    print(f"Benign Test: {result}")
