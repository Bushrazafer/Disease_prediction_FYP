"""
Feature Signature Validation
Ensures input features match model's expected feature order and types.
"""

import logging
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class FeatureSignatureError(Exception):
    """Raised when feature signature doesn't match model expectations"""
    pass


class FeatureSignatureValidator:
    """
    Validates that input features match the model's expected signature.
    Prevents silent errors from feature order mismatches.
    """
    
    @classmethod
    def validate_signature(
        cls,
        input_features: List[str],
        model_features: List[str],
        strict: bool = True,
    ) -> Tuple[bool, List[str]]:
        """
        Validate that input features match model's expected features.
        
        Args:
            input_features: List of feature names from input data
            model_features: List of feature names expected by model (in order)
            strict: If True, requires exact match. If False, allows subset.
            
        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []
        
        # Normalize feature names
        input_normalized = [cls._normalize(f) for f in input_features]
        model_normalized = [cls._normalize(f) for f in model_features]
        
        if strict:
            # Check for exact match
            if len(input_normalized) != len(model_normalized):
                errors.append(
                    f"Feature count mismatch: got {len(input_normalized)}, "
                    f"expected {len(model_normalized)}"
                )
            
            # Check order
            for i, (inp, mod) in enumerate(zip(input_normalized, model_normalized)):
                if inp != mod:
                    errors.append(
                        f"Feature order mismatch at position {i}: "
                        f"got '{input_features[i]}', expected '{model_features[i]}'"
                    )
        else:
            # Check that all model features are present
            missing = set(model_normalized) - set(input_normalized)
            if missing:
                # Map back to original names
                missing_original = [
                    model_features[model_normalized.index(m)] 
                    for m in missing if m in model_normalized
                ]
                errors.append(f"Missing required features: {', '.join(missing_original)}")
        
        return len(errors) == 0, errors
    
    @classmethod
    def reorder_features(
        cls,
        data: Dict[str, Any],
        expected_order: List[str],
    ) -> List[Any]:
        """
        Reorder input data to match expected feature order.
        
        Args:
            data: Input data dictionary
            expected_order: List of feature names in expected order
            
        Returns:
            List of values in correct order
        """
        # Create normalized lookup
        normalized_data = {cls._normalize(k): v for k, v in data.items()}
        
        result = []
        for feature in expected_order:
            norm_feature = cls._normalize(feature)
            if norm_feature in normalized_data:
                result.append(normalized_data[norm_feature])
            else:
                # Use default value
                result.append(0)
                logger.warning(f"Feature '{feature}' not found in input, using default 0")
        
        return result
    
    @classmethod
    def load_model_signature(cls, model_path: str) -> Optional[List[str]]:
        """
        Load feature signature from model's features.pkl or metadata.
        
        Args:
            model_path: Path to model file
            
        Returns:
            List of feature names or None if not found
        """
        model_dir = Path(model_path).parent
        
        # Try features.pkl
        features_path = model_dir / 'features.pkl'
        if features_path.exists():
            try:
                import joblib
                features = joblib.load(features_path)
                if isinstance(features, list):
                    return features
                elif isinstance(features, dict) and 'features' in features:
                    return features['features']
            except Exception as e:
                logger.warning(f"Could not load features.pkl: {e}")
        
        # Try metadata.json
        metadata_path = model_dir / 'metadata.json'
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                return metadata.get('feature_order', metadata.get('features'))
            except Exception as e:
                logger.warning(f"Could not load metadata.json: {e}")
        
        return None
    
    @classmethod
    def create_signature_file(
        cls,
        model_path: str,
        features: List[str],
        feature_types: Optional[Dict[str, str]] = None,
    ):
        """
        Create a feature signature file for a model.
        
        Args:
            model_path: Path to model file
            features: List of feature names in order
            feature_types: Optional dict mapping features to types
        """
        model_dir = Path(model_path).parent
        
        signature = {
            'features': features,
            'feature_count': len(features),
            'feature_types': feature_types or {},
        }
        
        signature_path = model_dir / 'feature_signature.json'
        with open(signature_path, 'w') as f:
            json.dump(signature, f, indent=2)
        
        logger.info(f"Created feature signature at {signature_path}")
    
    @classmethod
    def _normalize(cls, name: str) -> str:
        """Normalize feature name for comparison"""
        return name.lower().replace(' ', '_').replace('-', '_')
    
    @classmethod
    def get_feature_diff(
        cls,
        input_features: List[str],
        model_features: List[str],
    ) -> Dict[str, List[str]]:
        """
        Get detailed diff between input and model features.
        
        Returns:
            Dict with 'missing', 'extra', and 'common' feature lists
        """
        input_set = set(cls._normalize(f) for f in input_features)
        model_set = set(cls._normalize(f) for f in model_features)
        
        return {
            'missing': list(model_set - input_set),
            'extra': list(input_set - model_set),
            'common': list(input_set & model_set),
        }
