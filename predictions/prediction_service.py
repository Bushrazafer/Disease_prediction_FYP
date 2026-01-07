"""
Enhanced Prediction Service with Tier Detection and Validation
Replaces the legacy make_prediction function with proper safety checks.
"""

import logging
import hashlib
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
import numpy as np

from django.utils import timezone

from .models import Prediction, MLModelVersion
from .ml_utils import (
    load_model, predict_with_model, detect_model_format,
    prepare_features, load_sklearn_model, get_model_path,
    ModelLoadError, PredictionError
)
from .inference import InferenceEngine, TierDetector, InputValidator
from .inference.feature_validator import FeatureSignatureValidator

logger = logging.getLogger(__name__)


# Obesity class to risk level mapping
OBESITY_RISK_MAP = {
    'Insufficient_Weight': 'low',
    'Normal_Weight': 'low',
    'Overweight_Level_I': 'medium',
    'Overweight_Level_II': 'medium',
    'Obesity_Type_I': 'high',
    'Obesity_Type_II': 'high',
    'Obesity_Type_III': 'critical'
}

# Result disclaimers by disease
RESULT_DISCLAIMERS = {
    'diabetes': 'This is a risk estimate, not a diagnosis. Please consult your doctor for HbA1c testing.',
    'cardiovascular': 'This is a risk estimate. Please consult a cardiologist for proper evaluation.',
    'kidney': 'This is a risk estimate. Please consult a nephrologist for renal function testing.',
    'breast_cancer': 'This analysis requires interpretation by a qualified oncologist or pathologist.',
    'depression': 'This is a screening tool. Please consult a mental health professional.',
    'obesity': 'This classification is based on lifestyle factors. Consult a healthcare provider.',
}


def make_enhanced_prediction(
    disease_name: str, 
    data: Dict[str, Any], 
    user=None,
    session_id: Optional[str] = None,
    ip_address: Optional[str] = None,
    disclaimer_acknowledged: bool = False,
) -> Dict[str, Any]:
    """
    Enhanced prediction function with tier detection, validation, and audit trail.
    
    Args:
        disease_name: Name of the disease
        data: Input data dictionary
        user: Django user object (optional)
        session_id: Session ID for anonymous users
        ip_address: IP address for audit (will be hashed)
        disclaimer_acknowledged: Whether user acknowledged disclaimer
        
    Returns:
        Dictionary with prediction results and metadata
    """
    # Get active model
    active_model = MLModelVersion.get_active_model(disease_name)
    
    if not active_model:
        # Return clear error instead of mock prediction
        return {
            'success': False,
            'error': f'The {disease_name.replace("_", " ")} prediction model is currently unavailable. '
                    'Please try again later or contact support.',
            'model_unavailable': True,
        }
    
    # Validate feature signature against model schema
    if active_model.feature_schema:
        input_features = list(data.keys())
        is_valid, signature_errors = FeatureSignatureValidator.validate_signature(
            input_features, 
            active_model.feature_schema,
            strict=False  # Allow subset matching
        )
        if not is_valid:
            logger.warning(f"Feature signature warnings for {disease_name}: {signature_errors}")
    try:
        # Step 1: Detect tier based on input completeness
        tier_result = TierDetector.detect_tier(disease_name, data)
        
        # Step 2: Validate input with medical range checking
        tier_reqs = TierDetector.get_tier_requirements(disease_name)
        required_fields = []
        if tier_result.tier in tier_reqs:
            required_fields = tier_reqs[tier_result.tier].get('required', [])
        
        validation = InputValidator.validate(data, required_fields, disease_name)
        
        if not validation.is_valid:
            return {
                'success': False,
                'error': f"Validation failed: {'; '.join(validation.errors)}",
                'errors': validation.errors,
                'warnings': validation.warnings,
            }
        
        # Step 3: Prepare features
        features = prepare_features(
            validation.sanitized_data,
            active_model.feature_schema,
            active_model.feature_types
        )
        
        # Step 4: Load model and preprocessing artifacts
        model = load_model(active_model.file_path)
        model_format = detect_model_format(active_model.file_path)
        
        # Apply preprocessing with error handling
        try:
            features = _apply_preprocessing(disease_name, active_model, features)
        except Exception as e:
            logger.error(f"Preprocessing error for {disease_name}: {e}")
            return {
                'success': False,
                'error': f'Preprocessing failed: {str(e)}',
                'preprocessing_error': True,
            }
        
        # Step 5: Make prediction with error handling
        try:
            prediction, probability_array = predict_with_model(model, features, model_format)
        except Exception as e:
            logger.error(f"Model prediction error for {disease_name}: {e}")
            return {
                'success': False,
                'error': f'Model prediction failed: {str(e)}',
                'model_error': True,
            }
        
        # Step 6: Handle multi-class (obesity) vs binary classification
        if disease_name == 'obesity':
            result = _handle_obesity_prediction(
                active_model, prediction, probability_array, 
                data, tier_result, validation
            )
        else:
            result = _handle_binary_prediction(
                disease_name, active_model, prediction, probability_array,
                data, tier_result, validation
            )
        
        # Step 7: Save prediction with audit trail
        if result['success']:
            _save_prediction(
                disease_name=disease_name,
                data=data,
                result=result,
                user=user,
                active_model=active_model,
                tier_result=tier_result,
                validation=validation,
                session_id=session_id,
                ip_address=ip_address,
                disclaimer_acknowledged=disclaimer_acknowledged,
            )
        
        return result
        
    except ModelLoadError as e:
        logger.error(f"Model load error for {disease_name}: {e}")
        return {
            'success': False,
            'error': f'Failed to load the {disease_name.replace("_", " ")} model. Please try again later.',
            'model_unavailable': True,
        }
        
    except PredictionError as e:
        logger.error(f"Prediction error for {disease_name}: {e}")
        return {
            'success': False,
            'error': f'Prediction failed: {str(e)}',
        }
        
    except Exception as e:
        logger.exception(f"Unexpected error in prediction for {disease_name}: {e}")
        return {
            'success': False,
            'error': 'An unexpected error occurred. Please try again.',
        }


def _apply_preprocessing(
    disease_name: str, 
    active_model: MLModelVersion, 
    features: np.ndarray
) -> np.ndarray:
    """Apply preprocessing (scaler, imputer, power transformer) to features."""
    
    # Special handling for CKD model - features are already properly encoded
    if disease_name == 'kidney':
        # CKD features are already encoded and ready for prediction
        # Skip preprocessing as it causes issues with categorical data
        logger.info("Skipping preprocessing for CKD model - features already prepared")
        return features
    
    scaler = None
    power_transformer = None
    imputer = None
    
    scaler_path = active_model.file_path.replace('model.pkl', 'scaler.pkl')
    imputer_path = active_model.file_path.replace('model.pkl', 'imputer.pkl')
    
    try:
        scaler_data = load_sklearn_model(get_model_path(scaler_path))
        
        if isinstance(scaler_data, dict):
            scaler = scaler_data.get('scaler')
            power_transformer = scaler_data.get('power_transformer')
        else:
            scaler = scaler_data
        
        try:
            imputer = load_sklearn_model(get_model_path(imputer_path))
        except:
            pass
        
        if imputer is not None:
            features = imputer.transform(features)
        
        if scaler is not None:
            features = scaler.transform(features)
        if power_transformer is not None:
            features = power_transformer.transform(features)
            
    except Exception as e:
        logger.warning(f"Could not load preprocessing artifacts for {disease_name}: {e}")
    
    return features


def _apply_ckd_preprocessing(active_model: MLModelVersion, features: np.ndarray) -> np.ndarray:
    """Apply CKD-specific preprocessing - features are already encoded by prepare_ckd_features."""
    try:
        # Load imputer and scaler (features are already properly encoded)
        imputer_path = active_model.file_path.replace('model.pkl', 'imputer.pkl')
        scaler_path = active_model.file_path.replace('model.pkl', 'scaler.pkl')
        
        imputer = load_sklearn_model(get_model_path(imputer_path))
        scaler_data = load_sklearn_model(get_model_path(scaler_path))
        
        # Handle scaler format (dict or direct object)
        if isinstance(scaler_data, dict):
            scaler = scaler_data.get('scaler')
            power_transformer = scaler_data.get('power_transformer')
        else:
            scaler = scaler_data
            power_transformer = None
        
        # Apply preprocessing in correct order
        features = imputer.transform(features)
        
        if scaler is not None:
            features = scaler.transform(features)
        if power_transformer is not None:
            features = power_transformer.transform(features)
        
        return features
        
    except Exception as e:
        logger.warning(f"Could not load CKD preprocessing artifacts: {e}")
        return features


def _handle_binary_prediction(
    disease_name: str,
    active_model: MLModelVersion,
    prediction: np.ndarray,
    probability_array: Optional[np.ndarray],
    data: Dict[str, Any],
    tier_result,
    validation,
) -> Dict[str, Any]:
    """Handle binary classification prediction."""
    # Extract probability
    if probability_array is not None:
        if len(probability_array.shape) > 1:
            probability = float(probability_array[0][1]) * 100
        else:
            probability = float(probability_array[0]) * 100
    else:
        probability = float(prediction[0]) * 100
    
    # Calculate confidence based on tier
    confidence = TierDetector.calculate_confidence(
        disease_name, data, tier_result, probability
    )
    
    # Determine risk level
    risk_level = _get_risk_level(probability)
    
    return {
        'success': True,
        'prediction': int(prediction[0]),
        'probability': round(probability, 2),
        'risk_level': risk_level,
        
        # Tier info
        'tier': tier_result.tier.value,
        'tier_description': tier_result.tier_description,
        'confidence': round(confidence, 1),
        
        # Validation info
        'validation_warnings': validation.warnings if validation.warnings else None,
        'out_of_range_fields': validation.out_of_range_fields if validation.out_of_range_fields else None,
        
        # Upgrade suggestions
        'missing_for_upgrade': tier_result.missing_for_upgrade if tier_result.missing_for_upgrade else None,
        
        # Disclaimers
        'tier_disclaimer': tier_result.tier_disclaimer,
        'result_disclaimer': RESULT_DISCLAIMERS.get(disease_name),
        
        # Model info
        'model_version': active_model.version,
        
        # Message
        'message': f'Risk assessment completed ({tier_result.tier.value} analysis). Risk level: {risk_level.title()}'
    }


def _handle_obesity_prediction(
    active_model: MLModelVersion,
    prediction: np.ndarray,
    probability_array: Optional[np.ndarray],
    data: Dict[str, Any],
    tier_result,
    validation,
) -> Dict[str, Any]:
    """Handle multi-class obesity prediction."""
    try:
        encoder_path = active_model.file_path.replace('model.pkl', 'label_encoder.pkl')
        label_encoder = load_sklearn_model(get_model_path(encoder_path))
        
        class_name = label_encoder.inverse_transform([prediction[0]])[0]
        
        if probability_array is not None:
            probability = float(max(probability_array[0])) * 100
        else:
            probability = 100.0
        
        risk_level = OBESITY_RISK_MAP.get(class_name, 'medium')
        
        confidence = TierDetector.calculate_confidence(
            'obesity', data, tier_result, probability
        )
        
        return {
            'success': True,
            'prediction': int(prediction[0]),
            'class_name': class_name,
            'display_name': class_name.replace('_', ' '),
            'probability': round(probability, 2),
            'risk_level': risk_level,
            
            # Tier info
            'tier': tier_result.tier.value,
            'tier_description': tier_result.tier_description,
            'confidence': round(confidence, 1),
            
            # Validation info
            'validation_warnings': validation.warnings if validation.warnings else None,
            
            # Disclaimers
            'tier_disclaimer': tier_result.tier_disclaimer,
            'result_disclaimer': RESULT_DISCLAIMERS.get('obesity'),
            
            # Model info
            'model_version': active_model.version,
            
            # Message
            'message': f'Obesity Level: {class_name.replace("_", " ")}'
        }
    except Exception as e:
        logger.error(f"Error in obesity prediction: {e}")
        raise PredictionError(f"Failed to process obesity prediction: {e}")


def _get_risk_level(probability: float) -> str:
    """Determine risk level from probability."""
    if probability < 30:
        return 'low'
    elif probability < 60:
        return 'medium'
    elif probability < 85:
        return 'high'
    else:
        return 'critical'


def _save_prediction(
    disease_name: str,
    data: Dict[str, Any],
    result: Dict[str, Any],
    user,
    active_model: MLModelVersion,
    tier_result,
    validation,
    session_id: Optional[str],
    ip_address: Optional[str],
    disclaimer_acknowledged: bool,
    request=None,
):
    """Save prediction to database with audit trail and activity logging."""
    try:
        # Hash IP address for privacy
        ip_hash = None
        if ip_address:
            ip_hash = hashlib.sha256(ip_address.encode()).hexdigest()[:32]
        
        prediction_obj = Prediction.objects.create(
            user=user if user and user.is_authenticated else None,
            prediction_type=disease_name,
            input_data=data,
            result={
                'prediction': result.get('prediction'),
                'probability': result.get('probability'),
                'risk_level': result.get('risk_level'),
                'class_name': result.get('class_name'),
                'tier': result.get('tier'),
                'confidence': result.get('confidence'),
            },
            probability=result.get('probability'),
            risk_level=result.get('risk_level'),
            model_version=active_model.name,
            
            # New audit fields
            prediction_tier=tier_result.tier.value,
            confidence_score=result.get('confidence'),
            disclaimer_acknowledged=disclaimer_acknowledged,
            disclaimer_version='1.0',
            consent_timestamp=timezone.now() if disclaimer_acknowledged else None,
            validation_warnings=validation.warnings if validation.warnings else None,
            session_id=session_id,
            ip_hash=ip_hash,
        )
        
        # Log prediction activity if user is authenticated and request is available
        if user and user.is_authenticated and request:
            try:
                from .activity_tracker import log_prediction
                log_prediction(user, request, prediction_obj, disease_name)
            except Exception as e:
                logger.warning(f"Failed to log prediction activity: {e}")
        
        return prediction_obj
        
    except Exception as e:
        logger.error(f"Failed to save prediction: {e}")
        # Don't fail the prediction if saving fails
        return None


# Legacy compatibility wrapper
def make_prediction(disease_name: str, data: Dict[str, Any], user=None) -> Dict[str, Any]:
    """
    Legacy wrapper for backward compatibility.
    Calls the enhanced prediction function.
    """
    return make_enhanced_prediction(disease_name, data, user)
