"""
Base classes for disease metadata configuration.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum


class FeatureType(Enum):
    """Types of input features"""
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"
    COMPUTED = "computed"  # Derived features, not user-input


class InputTier(Enum):
    """Prediction tiers based on input completeness"""
    SCREENING = "screening"  # Basic inputs only - lower confidence
    STANDARD = "standard"    # Standard inputs - moderate confidence
    CONFIRMATION = "confirmation"  # Full lab data - high confidence


class RiskLevel(Enum):
    """Risk level classifications"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class FeatureSpec:
    """Specification for a single input feature"""
    name: str
    display_name: str
    description: str
    feature_type: FeatureType
    required: bool = True
    tier: InputTier = InputTier.SCREENING  # Which tier this feature belongs to
    
    # Validation
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    medical_min: Optional[float] = None  # Medically plausible minimum
    medical_max: Optional[float] = None  # Medically plausible maximum
    
    # For categorical features
    options: Optional[List[Dict[str, Any]]] = None
    
    # Default value for imputation
    default_value: Optional[Any] = None
    
    # Feature importance weight (0-1) for confidence calculation
    importance_weight: float = 0.5
    
    # Is this a high-signal feature for the disease?
    high_signal: bool = False
    
    # Unit of measurement
    unit: Optional[str] = None
    
    # Help text for users
    help_text: Optional[str] = None
    
    # Warning thresholds
    warning_low: Optional[float] = None
    warning_high: Optional[float] = None


@dataclass
class TierConfig:
    """Configuration for prediction tiers"""
    tier: InputTier
    required_features: List[str]
    optional_features: List[str]
    min_confidence: float  # Minimum confidence for this tier
    max_confidence: float  # Maximum confidence for this tier
    description: str
    disclaimer: str


@dataclass
class ThresholdConfig:
    """Threshold configuration for risk classification"""
    low_max: float = 30.0
    medium_max: float = 60.0
    high_max: float = 85.0
    # Above high_max is critical
    
    # Sensitivity-optimized thresholds (for screening)
    screening_threshold: float = 0.3  # Lower threshold for higher sensitivity
    
    # Specificity-optimized thresholds (for confirmation)
    confirmation_threshold: float = 0.5  # Standard threshold


@dataclass 
class DiseaseMetadata:
    """Complete metadata for a disease prediction model"""
    disease_name: str
    display_name: str
    description: str
    
    # Feature specifications
    features: Dict[str, FeatureSpec] = field(default_factory=dict)
    
    # Feature order for model input
    feature_order: List[str] = field(default_factory=list)
    
    # Tier configurations
    tiers: Dict[InputTier, TierConfig] = field(default_factory=dict)
    
    # Threshold configuration
    thresholds: ThresholdConfig = field(default_factory=ThresholdConfig)
    
    # Is this a clinician-only tool?
    clinician_only: bool = False
    clinician_warning: Optional[str] = None
    
    # Medical disclaimers
    general_disclaimer: str = ""
    result_disclaimer: str = ""
    
    # Crisis resources (for mental health)
    crisis_resources: Optional[List[Dict[str, str]]] = None
    
    # Output classes for multi-class prediction
    output_classes: Optional[List[str]] = None
    
    # Model version this metadata is compatible with
    compatible_model_version: Optional[str] = None
    
    def get_required_features(self, tier: InputTier = InputTier.SCREENING) -> List[str]:
        """Get required features for a specific tier"""
        if tier in self.tiers:
            return self.tiers[tier].required_features
        return [f.name for f in self.features.values() if f.required]
    
    def get_high_signal_features(self) -> List[str]:
        """Get list of high-signal features"""
        return [f.name for f in self.features.values() if f.high_signal]
    
    def validate_input(self, data: Dict[str, Any]) -> Tuple[bool, List[str], List[str]]:
        """
        Validate input data against feature specs.
        Returns: (is_valid, errors, warnings)
        """
        errors = []
        warnings = []
        
        for feature_name, spec in self.features.items():
            value = data.get(feature_name)
            
            # Check required fields
            if spec.required and spec.feature_type != FeatureType.COMPUTED:
                if value is None or value == '':
                    errors.append(f"Missing required field: {spec.display_name}")
                    continue
            
            if value is None or value == '':
                continue
                
            # Type validation
            if spec.feature_type == FeatureType.NUMERIC:
                try:
                    num_value = float(value)
                    
                    # Range validation
                    if spec.min_value is not None and num_value < spec.min_value:
                        errors.append(f"{spec.display_name} must be at least {spec.min_value}")
                    if spec.max_value is not None and num_value > spec.max_value:
                        errors.append(f"{spec.display_name} must be at most {spec.max_value}")
                    
                    # Medical plausibility warnings
                    if spec.medical_min is not None and num_value < spec.medical_min:
                        warnings.append(f"{spec.display_name} ({num_value}) is below typical medical range ({spec.medical_min})")
                    if spec.medical_max is not None and num_value > spec.medical_max:
                        warnings.append(f"{spec.display_name} ({num_value}) is above typical medical range ({spec.medical_max})")
                        
                except (ValueError, TypeError):
                    errors.append(f"{spec.display_name} must be a number")
                    
            elif spec.feature_type == FeatureType.CATEGORICAL:
                if spec.options:
                    valid_values = [opt.get('value') for opt in spec.options]
                    if str(value).lower() not in [str(v).lower() for v in valid_values]:
                        errors.append(f"{spec.display_name} must be one of: {', '.join(map(str, valid_values))}")
        
        return len(errors) == 0, errors, warnings
    
    def detect_tier(self, data: Dict[str, Any]) -> InputTier:
        """Detect which tier the input data qualifies for"""
        high_signal_features = self.get_high_signal_features()
        provided_high_signal = sum(1 for f in high_signal_features if data.get(f) not in [None, ''])
        
        # Check confirmation tier first
        if InputTier.CONFIRMATION in self.tiers:
            conf_config = self.tiers[InputTier.CONFIRMATION]
            if all(data.get(f) not in [None, ''] for f in conf_config.required_features):
                return InputTier.CONFIRMATION
        
        # Check standard tier
        if InputTier.STANDARD in self.tiers:
            std_config = self.tiers[InputTier.STANDARD]
            if all(data.get(f) not in [None, ''] for f in std_config.required_features):
                return InputTier.STANDARD
        
        # Default to screening
        return InputTier.SCREENING
    
    def calculate_confidence(self, data: Dict[str, Any], tier: InputTier) -> float:
        """Calculate confidence score based on input completeness and tier"""
        if tier not in self.tiers:
            return 50.0
            
        tier_config = self.tiers[tier]
        
        # Base confidence from tier
        base_confidence = (tier_config.min_confidence + tier_config.max_confidence) / 2
        
        # Adjust based on high-signal feature presence
        high_signal = self.get_high_signal_features()
        if high_signal:
            provided = sum(1 for f in high_signal if data.get(f) not in [None, ''])
            signal_ratio = provided / len(high_signal)
            confidence_boost = (tier_config.max_confidence - base_confidence) * signal_ratio
            base_confidence += confidence_boost
        
        return min(base_confidence, tier_config.max_confidence)
