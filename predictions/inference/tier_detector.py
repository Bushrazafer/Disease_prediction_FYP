"""
Tier Detection System for Prediction Quality Assessment
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class PredictionTier(Enum):
    """Prediction quality tiers based on input completeness"""
    SCREENING = "screening"      # Basic inputs - lower confidence
    STANDARD = "standard"        # Standard inputs - moderate confidence  
    CONFIRMATION = "confirmation" # Full lab data - high confidence


@dataclass
class TierResult:
    """Result of tier detection"""
    tier: PredictionTier
    confidence_range: Tuple[float, float]  # (min, max) confidence
    missing_for_upgrade: List[str]  # Features needed to upgrade tier
    provided_high_signal: List[str]  # High-signal features that were provided
    tier_description: str
    tier_disclaimer: str


# High-signal features by disease - these significantly impact prediction quality
HIGH_SIGNAL_FEATURES = {
    'diabetes': ['glucose', 'insulin', 'hba1c', 'homa_ir'],
    'cardiovascular': ['cholesterol', 'max_hr', 'oldpeak', 'st_slope', 'exercise_angina', 'ejection_fraction'],
    'kidney': ['sc', 'bu', 'egfr', 'hemo', 'al'],  # serum creatinine, blood urea, eGFR, hemoglobin, albumin
    'breast_cancer': ['concave_points_worst', 'radius_worst', 'perimeter_worst', 'area_worst', 'concavity_worst'],
    'depression': ['suicidal_thoughts', 'family_history', 'sleep_duration', 'financial_stress'],
    'obesity': ['bmi', 'faf', 'fcvc', 'family_history_with_overweight'],
}

# Tier requirements by disease
TIER_REQUIREMENTS = {
    'diabetes': {
        PredictionTier.SCREENING: {
            'required': ['age', 'bmi'],
            'optional': ['blood_pressure', 'pregnancies'],
            'confidence': (40, 60),
            'description': 'Basic screening based on demographics',
            'disclaimer': 'This is a basic screening. Blood glucose testing is recommended for accurate assessment.',
        },
        PredictionTier.STANDARD: {
            'required': ['age', 'bmi', 'glucose', 'blood_pressure'],
            'optional': ['diabetes_pedigree', 'skin_thickness'],
            'confidence': (60, 80),
            'description': 'Standard assessment with glucose data',
            'disclaimer': 'Results based on standard metabolic indicators. Full lab panel recommended for confirmation.',
        },
        PredictionTier.CONFIRMATION: {
            'required': ['age', 'bmi', 'glucose', 'blood_pressure', 'insulin'],
            'optional': ['diabetes_pedigree', 'skin_thickness', 'pregnancies'],
            'confidence': (80, 95),
            'description': 'Comprehensive assessment with full metabolic panel',
            'disclaimer': 'High-confidence assessment based on complete lab data.',
        },
    },
    'cardiovascular': {
        PredictionTier.SCREENING: {
            'required': ['age', 'sex'],
            'optional': ['bmi', 'smoking', 'family_history'],
            'confidence': (35, 55),
            'description': 'Basic risk screening',
            'disclaimer': 'Basic screening only. ECG and blood tests recommended.',
        },
        PredictionTier.STANDARD: {
            'required': ['age', 'sex', 'resting_bp', 'cholesterol', 'chest_pain_type'],
            'optional': ['max_hr', 'fasting_bs', 'bmi'],
            'confidence': (55, 75),
            'description': 'Standard cardiac risk assessment',
            'disclaimer': 'Standard assessment. Stress test recommended for confirmation.',
        },
        PredictionTier.CONFIRMATION: {
            'required': ['age', 'sex', 'resting_bp', 'cholesterol', 'chest_pain_type', 
                        'max_hr', 'exercise_angina', 'oldpeak', 'st_slope'],
            'optional': ['ejection_fraction', 'triglycerides', 'hdl', 'ldl'],
            'confidence': (75, 92),
            'description': 'Comprehensive cardiac assessment with ECG data',
            'disclaimer': 'High-confidence assessment based on clinical and ECG data.',
        },
    },
    'kidney': {
        PredictionTier.SCREENING: {
            'required': ['age', 'bp'],
            'optional': ['htn', 'dm'],
            'confidence': (35, 55),
            'description': 'Basic CKD risk screening',
            'disclaimer': 'Basic screening. Blood and urine tests required for diagnosis.',
        },
        PredictionTier.STANDARD: {
            'required': ['age', 'bp', 'sc', 'bu', 'hemo'],
            'optional': ['sg', 'al', 'htn', 'dm'],
            'confidence': (55, 75),
            'description': 'Standard renal function assessment',
            'disclaimer': 'Standard assessment based on basic renal markers.',
        },
        PredictionTier.CONFIRMATION: {
            'required': ['age', 'bp', 'sc', 'bu', 'hemo', 'sg', 'al', 'bgr', 'sod', 'pot'],
            'optional': ['pcv', 'wc', 'rc', 'pcc', 'ba'],
            'confidence': (75, 92),
            'description': 'Comprehensive renal panel assessment',
            'disclaimer': 'High-confidence assessment based on complete renal panel.',
        },
    },
    'breast_cancer': {
        # Breast cancer is clinician-only - all tiers require FNA data
        PredictionTier.SCREENING: {
            'required': ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean'],
            'optional': [],
            'confidence': (50, 70),
            'description': 'Basic FNA cell analysis',
            'disclaimer': 'Preliminary analysis. Full cell panel recommended.',
        },
        PredictionTier.STANDARD: {
            'required': ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
                        'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave_points_mean'],
            'optional': ['symmetry_mean', 'fractal_dimension_mean'],
            'confidence': (70, 85),
            'description': 'Standard FNA cell analysis',
            'disclaimer': 'Standard analysis. Worst-value features improve accuracy.',
        },
        PredictionTier.CONFIRMATION: {
            'required': ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
                        'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave_points_mean',
                        'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst',
                        'concavity_worst', 'concave_points_worst'],
            'optional': ['symmetry_worst', 'fractal_dimension_worst'],
            'confidence': (85, 97),
            'description': 'Comprehensive FNA cell analysis with worst-value features',
            'disclaimer': 'High-confidence analysis based on complete cell measurements.',
        },
    },
    'depression': {
        PredictionTier.SCREENING: {
            'required': ['age', 'gender', 'sleep_duration'],
            'optional': ['dietary_habits'],
            'confidence': (40, 60),
            'description': 'Basic mental health screening',
            'disclaimer': 'Basic screening only. Professional evaluation recommended.',
        },
        PredictionTier.STANDARD: {
            'required': ['age', 'gender', 'sleep_duration', 'academic_pressure', 
                        'work_study_hours', 'financial_stress'],
            'optional': ['dietary_habits', 'cgpa', 'study_satisfaction'],
            'confidence': (60, 78),
            'description': 'Standard mental health assessment',
            'disclaimer': 'Standard assessment. Clinical evaluation recommended for high-risk results.',
        },
        PredictionTier.CONFIRMATION: {
            'required': ['age', 'gender', 'sleep_duration', 'academic_pressure',
                        'work_study_hours', 'financial_stress', 'family_history', 'suicidal_thoughts'],
            'optional': ['dietary_habits', 'cgpa', 'study_satisfaction', 'job_satisfaction'],
            'confidence': (78, 90),
            'description': 'Comprehensive mental health assessment',
            'disclaimer': 'Comprehensive assessment. Always consult a mental health professional.',
        },
    },
    'obesity': {
        PredictionTier.SCREENING: {
            'required': ['height', 'weight', 'age', 'gender'],
            'optional': [],
            'confidence': (50, 70),
            'description': 'Basic BMI-based assessment',
            'disclaimer': 'Basic assessment based on BMI. Lifestyle factors improve accuracy.',
        },
        PredictionTier.STANDARD: {
            'required': ['height', 'weight', 'age', 'gender', 'faf', 'fcvc', 'ncp'],
            'optional': ['ch2o', 'tue', 'smoke'],
            'confidence': (70, 85),
            'description': 'Standard lifestyle assessment',
            'disclaimer': 'Standard assessment including activity and diet factors.',
        },
        PredictionTier.CONFIRMATION: {
            'required': ['height', 'weight', 'age', 'gender', 'faf', 'fcvc', 'ncp',
                        'favc', 'caec', 'calc', 'mtrans', 'family_history_with_overweight'],
            'optional': ['ch2o', 'tue', 'smoke', 'scc'],
            'confidence': (85, 95),
            'description': 'Comprehensive lifestyle and genetic assessment',
            'disclaimer': 'High-confidence assessment based on complete lifestyle profile.',
        },
    },
}


class TierDetector:
    """Detects prediction tier based on input completeness"""
    
    @classmethod
    def detect_tier(cls, disease_name: str, data: Dict[str, Any]) -> TierResult:
        """
        Detect the appropriate prediction tier based on provided inputs.
        
        Args:
            disease_name: Name of the disease
            data: Input data dictionary
            
        Returns:
            TierResult with tier info and upgrade suggestions
        """
        requirements = TIER_REQUIREMENTS.get(disease_name, {})
        high_signal = HIGH_SIGNAL_FEATURES.get(disease_name, [])
        
        # Normalize data keys
        normalized_data = {k.lower().replace(' ', '_'): v for k, v in data.items()}
        
        # Check which features are provided (not None or empty)
        def is_provided(key: str) -> bool:
            return normalized_data.get(key) not in [None, '', 'nan']
        
        # Find provided high-signal features
        provided_high_signal = [f for f in high_signal if is_provided(f)]
        
        # Check tiers from highest to lowest
        detected_tier = PredictionTier.SCREENING
        
        for tier in [PredictionTier.CONFIRMATION, PredictionTier.STANDARD, PredictionTier.SCREENING]:
            if tier not in requirements:
                continue
                
            tier_req = requirements[tier]
            required = tier_req['required']
            
            if all(is_provided(f) for f in required):
                detected_tier = tier
                break
        
        # Get tier config
        tier_config = requirements.get(detected_tier, {
            'required': [],
            'optional': [],
            'confidence': (50, 70),
            'description': 'Standard assessment',
            'disclaimer': 'Please consult a healthcare provider.',
        })
        
        # Find features needed for upgrade
        missing_for_upgrade = []
        if detected_tier != PredictionTier.CONFIRMATION:
            next_tier = PredictionTier.STANDARD if detected_tier == PredictionTier.SCREENING else PredictionTier.CONFIRMATION
            if next_tier in requirements:
                next_required = requirements[next_tier]['required']
                missing_for_upgrade = [f for f in next_required if not is_provided(f)]
        
        return TierResult(
            tier=detected_tier,
            confidence_range=tier_config['confidence'],
            missing_for_upgrade=missing_for_upgrade,
            provided_high_signal=provided_high_signal,
            tier_description=tier_config['description'],
            tier_disclaimer=tier_config['disclaimer'],
        )
    
    @classmethod
    def calculate_confidence(
        cls, 
        disease_name: str, 
        data: Dict[str, Any],
        tier_result: TierResult,
        model_probability: float
    ) -> float:
        """
        Calculate confidence score based on tier and input quality.
        
        Args:
            disease_name: Name of the disease
            data: Input data dictionary
            tier_result: Result from detect_tier
            model_probability: Raw probability from model
            
        Returns:
            Confidence score (0-100)
        """
        min_conf, max_conf = tier_result.confidence_range
        
        # Base confidence from tier
        base_confidence = (min_conf + max_conf) / 2
        
        # Adjust based on high-signal features
        high_signal = HIGH_SIGNAL_FEATURES.get(disease_name, [])
        if high_signal:
            signal_ratio = len(tier_result.provided_high_signal) / len(high_signal)
            confidence_boost = (max_conf - base_confidence) * signal_ratio * 0.5
            base_confidence += confidence_boost
        
        # Adjust based on model certainty (how far from 50%)
        certainty = abs(model_probability - 50) / 50  # 0 to 1
        certainty_boost = (max_conf - base_confidence) * certainty * 0.3
        base_confidence += certainty_boost
        
        return min(max(base_confidence, min_conf), max_conf)
    
    @classmethod
    def get_tier_requirements(cls, disease_name: str) -> Dict[str, Any]:
        """Get tier requirements for a disease"""
        return TIER_REQUIREMENTS.get(disease_name, {})
    
    @classmethod
    def get_high_signal_features(cls, disease_name: str) -> List[str]:
        """Get high-signal features for a disease"""
        return HIGH_SIGNAL_FEATURES.get(disease_name, [])
