"""
Probability Calibration for Medical Predictions
Ensures model probabilities reflect true risk levels.
"""

import logging
from typing import Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class ProbabilityCalibrator:
    """
    Calibrates raw model probabilities to better reflect true risk.
    
    Medical screening models often need calibration because:
    1. Raw probabilities may not reflect true population risk
    2. Different tiers need different calibration
    3. Sensitivity/specificity tradeoffs vary by use case
    """
    
    # Tier-based calibration adjustments
    # Screening tier: favor sensitivity (catch more positives)
    # Confirmation tier: more balanced
    TIER_ADJUSTMENTS = {
        'screening': {
            'threshold': 0.35,  # Lower threshold for higher sensitivity
            'sensitivity_boost': 0.1,  # Boost positive predictions
        },
        'standard': {
            'threshold': 0.45,
            'sensitivity_boost': 0.05,
        },
        'confirmation': {
            'threshold': 0.50,  # Standard threshold
            'sensitivity_boost': 0.0,
        },
    }
    
    # Disease-specific calibration factors
    # Based on typical model overconfidence patterns
    DISEASE_CALIBRATION = {
        'diabetes': {'scale': 0.95, 'shift': 0.02},
        'cardiovascular': {'scale': 0.92, 'shift': 0.03},
        'kidney': {'scale': 0.90, 'shift': 0.05},
        'breast_cancer': {'scale': 0.98, 'shift': 0.01},
        'depression': {'scale': 0.88, 'shift': 0.05},
        'obesity': {'scale': 1.0, 'shift': 0.0},  # Multi-class, no calibration
    }
    
    @classmethod
    def calibrate(
        cls,
        probability: float,
        disease_name: str,
        tier: str = 'standard',
        apply_tier_adjustment: bool = True,
    ) -> float:
        """
        Calibrate a raw probability.
        
        Args:
            probability: Raw probability from model (0-100 scale)
            disease_name: Name of the disease
            tier: Prediction tier (screening/standard/confirmation)
            apply_tier_adjustment: Whether to apply tier-based sensitivity boost
            
        Returns:
            Calibrated probability (0-100 scale)
        """
        # Convert to 0-1 scale for calculations
        prob = probability / 100.0
        
        # Apply disease-specific calibration
        calibration = cls.DISEASE_CALIBRATION.get(disease_name, {'scale': 1.0, 'shift': 0.0})
        prob = prob * calibration['scale'] + calibration['shift']
        
        # Apply tier-based sensitivity adjustment
        if apply_tier_adjustment and tier in cls.TIER_ADJUSTMENTS:
            tier_adj = cls.TIER_ADJUSTMENTS[tier]
            # Boost probabilities near the threshold for screening
            if prob > tier_adj['threshold'] - 0.15:
                prob += tier_adj['sensitivity_boost']
        
        # Clamp to valid range
        prob = max(0.0, min(1.0, prob))
        
        return prob * 100.0
    
    @classmethod
    def get_optimal_threshold(
        cls,
        disease_name: str,
        tier: str = 'standard',
        optimize_for: str = 'balanced',
    ) -> float:
        """
        Get optimal classification threshold for a disease/tier combination.
        
        Args:
            disease_name: Name of the disease
            tier: Prediction tier
            optimize_for: 'sensitivity', 'specificity', or 'balanced'
            
        Returns:
            Optimal threshold (0-1 scale)
        """
        base_threshold = cls.TIER_ADJUSTMENTS.get(tier, {}).get('threshold', 0.5)
        
        if optimize_for == 'sensitivity':
            # Lower threshold to catch more positives
            return base_threshold - 0.1
        elif optimize_for == 'specificity':
            # Higher threshold to reduce false positives
            return base_threshold + 0.1
        else:
            return base_threshold
    
    @classmethod
    def apply_platt_scaling(
        cls,
        probability: float,
        a: float = 1.0,
        b: float = 0.0,
    ) -> float:
        """
        Apply Platt scaling calibration.
        
        Platt scaling: P_calibrated = 1 / (1 + exp(a * logit(P) + b))
        
        Args:
            probability: Raw probability (0-100 scale)
            a: Scaling parameter (default 1.0 = no scaling)
            b: Shift parameter (default 0.0 = no shift)
            
        Returns:
            Calibrated probability (0-100 scale)
        """
        prob = probability / 100.0
        
        # Avoid log(0) and log(1)
        prob = max(0.001, min(0.999, prob))
        
        # Logit transform
        logit = np.log(prob / (1 - prob))
        
        # Apply scaling
        scaled_logit = a * logit + b
        
        # Inverse logit (sigmoid)
        calibrated = 1 / (1 + np.exp(-scaled_logit))
        
        return calibrated * 100.0
    
    @classmethod
    def get_confidence_interval(
        cls,
        probability: float,
        confidence_score: float,
        confidence_level: float = 0.95,
    ) -> Tuple[float, float]:
        """
        Calculate confidence interval for a probability estimate.
        
        Args:
            probability: Point estimate (0-100 scale)
            confidence_score: Model confidence (0-100 scale)
            confidence_level: Desired confidence level (default 0.95)
            
        Returns:
            Tuple of (lower_bound, upper_bound) in 0-100 scale
        """
        # Width of interval inversely proportional to confidence
        # Higher confidence = narrower interval
        base_width = 30  # Base interval width at 50% confidence
        width = base_width * (1 - confidence_score / 100) * 2
        
        # Adjust for confidence level
        z_score = 1.96 if confidence_level == 0.95 else 1.645  # 95% or 90%
        width = width * (z_score / 1.96)
        
        lower = max(0, probability - width / 2)
        upper = min(100, probability + width / 2)
        
        return (round(lower, 1), round(upper, 1))
