"""
Unified Inference Engine for MediCare AI
Provides standardized prediction interface with tier detection, validation, and confidence scoring.
"""

from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import logging
import numpy as np

from .validator import InputValidator, ValidationResult
from .tier_detector import TierDetector, TierResult, PredictionTier

logger = logging.getLogger(__name__)


@dataclass
class PredictionOutput:
    """Standardized prediction output format"""
    success: bool
    
    # Core prediction results
    prediction: Optional[int] = None
    probability: Optional[float] = None
    risk_level: Optional[str] = None
    
    # For multi-class predictions (e.g., obesity)
    class_name: Optional[str] = None
    display_name: Optional[str] = None
    
    # Tier and confidence
    tier: Optional[str] = None
    tier_description: Optional[str] = None
    confidence: Optional[float] = None
    
    # Validation info
    validation_warnings: Optional[list] = None
    out_of_range_fields: Optional[list] = None
    
    # Upgrade suggestions
    missing_for_upgrade: Optional[list] = None
    
    # Disclaimers
    tier_disclaimer: Optional[str] = None
    result_disclaimer: Optional[str] = None
    
    # Model info
    model_version: Optional[str] = None
    
    # Error info
    error: Optional[str] = None
    errors: Optional[list] = None
    
    # Message for display
    message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values"""
        return {k: v for k, v in asdict(self).items() if v is not None}


class InferenceEngine:
    """
    Unified inference engine with:
    - Input validation with medical range checking
    - Tier detection based on input completeness
    - Confidence scoring
    - Standardized output format
    """
    
    # Risk level thresholds
    RISK_THRESHOLDS = {
        'low_max': 30,
        'medium_max': 60,
        'high_max': 85,
    }
    
    # Disease-specific result disclaimers
    RESULT_DISCLAIMERS = {
        'diabetes': 'This is a risk estimate, not a diagnosis. Please consult your doctor for HbA1c testing and proper evaluation.',
        'cardiovascular': 'This is a risk estimate, not a diagnosis. Please consult a cardiologist for proper cardiac evaluation.',
        'kidney': 'This is a risk estimate, not a diagnosis. Please consult a nephrologist for proper renal function testing.',
        'breast_cancer': 'This analysis is based on FNA cell measurements and must be interpreted by a qualified pathologist or oncologist.',
        'depression': 'This is a screening tool, not a clinical diagnosis. Please consult a mental health professional for proper evaluation.',
        'obesity': 'This is a classification based on lifestyle factors. Please consult a healthcare provider for personalized advice.',
    }
    
    @classmethod
    def predict(
        cls,
        disease_name: str,
        data: Dict[str, Any],
        model_predict_fn,
        required_fields: Optional[list] = None,
        model_version: Optional[str] = None,
    ) -> PredictionOutput:
        """
        Make a prediction with full validation and tier detection.
        
        Args:
            disease_name: Name of the disease
            data: Input data dictionary
            model_predict_fn: Function that takes data and returns (prediction, probability)
            required_fields: Optional list of required fields (uses tier detection if not provided)
            model_version: Optional model version string
            
        Returns:
            PredictionOutput with standardized results
        """
        try:
            # Step 1: Detect tier
            tier_result = TierDetector.detect_tier(disease_name, data)
            
            # Step 2: Validate input
            if required_fields is None:
                # Use tier-based requirements
                tier_reqs = TierDetector.get_tier_requirements(disease_name)
                if tier_result.tier in tier_reqs:
                    required_fields = tier_reqs[tier_result.tier].get('required', [])
                else:
                    required_fields = []
            
            validation = InputValidator.validate(data, required_fields, disease_name)
            
            if not validation.is_valid:
                return PredictionOutput(
                    success=False,
                    error='Validation failed',
                    errors=validation.errors,
                    validation_warnings=validation.warnings,
                )
            
            # Step 3: Make prediction
            try:
                prediction, probability = model_predict_fn(validation.sanitized_data)
            except Exception as e:
                logger.error(f"Model prediction failed for {disease_name}: {e}")
                return PredictionOutput(
                    success=False,
                    error=f'Prediction failed: {str(e)}',
                )
            
            # Step 4: Calculate confidence
            confidence = TierDetector.calculate_confidence(
                disease_name, data, tier_result, probability
            )
            
            # Step 5: Determine risk level
            risk_level = cls._get_risk_level(probability)
            
            # Step 6: Build output
            return PredictionOutput(
                success=True,
                prediction=int(prediction) if prediction is not None else None,
                probability=round(float(probability), 2) if probability is not None else None,
                risk_level=risk_level,
                tier=tier_result.tier.value,
                tier_description=tier_result.tier_description,
                confidence=round(confidence, 1),
                validation_warnings=validation.warnings if validation.warnings else None,
                out_of_range_fields=validation.out_of_range_fields if validation.out_of_range_fields else None,
                missing_for_upgrade=tier_result.missing_for_upgrade if tier_result.missing_for_upgrade else None,
                tier_disclaimer=tier_result.tier_disclaimer,
                result_disclaimer=cls.RESULT_DISCLAIMERS.get(disease_name),
                model_version=model_version,
                message=cls._build_message(risk_level, tier_result.tier),
            )
            
        except Exception as e:
            logger.exception(f"Inference engine error for {disease_name}: {e}")
            return PredictionOutput(
                success=False,
                error=f'An unexpected error occurred: {str(e)}',
            )
    
    @classmethod
    def predict_multiclass(
        cls,
        disease_name: str,
        data: Dict[str, Any],
        model_predict_fn,
        class_names: list,
        class_risk_map: Dict[str, str],
        required_fields: Optional[list] = None,
        model_version: Optional[str] = None,
    ) -> PredictionOutput:
        """
        Make a multi-class prediction (e.g., obesity levels).
        
        Args:
            disease_name: Name of the disease
            data: Input data dictionary
            model_predict_fn: Function that takes data and returns (class_index, probabilities)
            class_names: List of class names
            class_risk_map: Mapping of class names to risk levels
            required_fields: Optional list of required fields
            model_version: Optional model version string
            
        Returns:
            PredictionOutput with class name and risk level
        """
        try:
            # Detect tier and validate
            tier_result = TierDetector.detect_tier(disease_name, data)
            
            if required_fields is None:
                tier_reqs = TierDetector.get_tier_requirements(disease_name)
                if tier_result.tier in tier_reqs:
                    required_fields = tier_reqs[tier_result.tier].get('required', [])
                else:
                    required_fields = []
            
            validation = InputValidator.validate(data, required_fields, disease_name)
            
            if not validation.is_valid:
                return PredictionOutput(
                    success=False,
                    error='Validation failed',
                    errors=validation.errors,
                    validation_warnings=validation.warnings,
                )
            
            # Make prediction
            try:
                class_index, probabilities = model_predict_fn(validation.sanitized_data)
            except Exception as e:
                logger.error(f"Model prediction failed for {disease_name}: {e}")
                return PredictionOutput(
                    success=False,
                    error=f'Prediction failed: {str(e)}',
                )
            
            # Get class name and probability
            class_name = class_names[class_index] if class_index < len(class_names) else 'Unknown'
            probability = float(max(probabilities)) * 100 if probabilities is not None else 100.0
            
            # Get risk level from class
            risk_level = class_risk_map.get(class_name, 'medium')
            
            # Calculate confidence
            confidence = TierDetector.calculate_confidence(
                disease_name, data, tier_result, probability
            )
            
            return PredictionOutput(
                success=True,
                prediction=int(class_index),
                class_name=class_name,
                display_name=class_name.replace('_', ' '),
                probability=round(probability, 2),
                risk_level=risk_level,
                tier=tier_result.tier.value,
                tier_description=tier_result.tier_description,
                confidence=round(confidence, 1),
                validation_warnings=validation.warnings if validation.warnings else None,
                out_of_range_fields=validation.out_of_range_fields if validation.out_of_range_fields else None,
                missing_for_upgrade=tier_result.missing_for_upgrade if tier_result.missing_for_upgrade else None,
                tier_disclaimer=tier_result.tier_disclaimer,
                result_disclaimer=cls.RESULT_DISCLAIMERS.get(disease_name),
                model_version=model_version,
                message=f'Classification: {class_name.replace("_", " ")}',
            )
            
        except Exception as e:
            logger.exception(f"Inference engine error for {disease_name}: {e}")
            return PredictionOutput(
                success=False,
                error=f'An unexpected error occurred: {str(e)}',
            )
    
    @classmethod
    def _get_risk_level(cls, probability: float) -> str:
        """Determine risk level from probability"""
        if probability < cls.RISK_THRESHOLDS['low_max']:
            return 'low'
        elif probability < cls.RISK_THRESHOLDS['medium_max']:
            return 'medium'
        elif probability < cls.RISK_THRESHOLDS['high_max']:
            return 'high'
        else:
            return 'critical'
    
    @classmethod
    def _build_message(cls, risk_level: str, tier: PredictionTier) -> str:
        """Build user-friendly message"""
        tier_label = {
            PredictionTier.SCREENING: 'screening',
            PredictionTier.STANDARD: 'standard',
            PredictionTier.CONFIRMATION: 'comprehensive',
        }.get(tier, 'standard')
        
        return f'Risk assessment completed ({tier_label} analysis). Risk level: {risk_level.title()}'
    
    @classmethod
    def create_error_response(cls, error_message: str) -> PredictionOutput:
        """Create a standardized error response"""
        return PredictionOutput(
            success=False,
            error=error_message,
        )
    
    @classmethod
    def create_unavailable_response(cls, disease_name: str) -> PredictionOutput:
        """Create response when model is unavailable (replaces mock prediction)"""
        return PredictionOutput(
            success=False,
            error=f'The {disease_name} prediction model is currently unavailable. Please try again later or contact support.',
            message='Model unavailable',
        )
