"""
Prediction Explainer for Medical Risk Assessments
Generates human-readable explanations for predictions.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass


@dataclass
class RiskFactor:
    """A single risk factor contributing to the prediction"""
    name: str
    display_name: str
    value: Any
    status: str  # 'normal', 'elevated', 'high', 'critical'
    contribution: str  # 'increases', 'decreases', 'neutral'
    explanation: str


@dataclass
class PredictionExplanation:
    """Complete explanation for a prediction"""
    summary: str
    risk_factors: List[RiskFactor]
    protective_factors: List[RiskFactor]
    missing_factors: List[str]
    recommendations: List[str]
    tier_explanation: str
    confidence_explanation: str


# Risk factor thresholds by disease
RISK_THRESHOLDS = {
    'diabetes': {
        'glucose': {'normal': (70, 100), 'elevated': (100, 126), 'high': (126, 200), 'critical': (200, 999)},
        'bmi': {'normal': (18.5, 25), 'elevated': (25, 30), 'high': (30, 40), 'critical': (40, 100)},
        'blood_pressure': {'normal': (60, 80), 'elevated': (80, 90), 'high': (90, 120), 'critical': (120, 200)},
        'age': {'normal': (0, 45), 'elevated': (45, 60), 'high': (60, 100)},
        'insulin': {'normal': (16, 166), 'elevated': (166, 300), 'high': (300, 999)},
    },
    'cardiovascular': {
        'cholesterol': {'normal': (0, 200), 'elevated': (200, 240), 'high': (240, 300), 'critical': (300, 999)},
        'resting_bp': {'normal': (90, 120), 'elevated': (120, 140), 'high': (140, 180), 'critical': (180, 999)},
        'max_hr': {'normal': (100, 180), 'elevated': (60, 100), 'high': (0, 60)},  # Low is bad
        'oldpeak': {'normal': (0, 1), 'elevated': (1, 2), 'high': (2, 4), 'critical': (4, 10)},
        'age': {'normal': (0, 50), 'elevated': (50, 65), 'high': (65, 100)},
    },
    'kidney': {
        'sc': {'normal': (0.7, 1.3), 'elevated': (1.3, 2.0), 'high': (2.0, 5.0), 'critical': (5.0, 100)},
        'bu': {'normal': (7, 20), 'elevated': (20, 40), 'high': (40, 100), 'critical': (100, 500)},
        'hemo': {'normal': (12, 17), 'elevated': (10, 12), 'high': (7, 10), 'critical': (0, 7)},  # Low is bad
        'al': {'normal': (0, 0), 'elevated': (1, 2), 'high': (3, 4), 'critical': (5, 5)},
    },
    'depression': {
        'sleep_duration': {'normal': (2, 3), 'elevated': (1, 2), 'high': (0, 1)},  # Low is bad
        'financial_stress': {'normal': (1, 2), 'elevated': (3, 3), 'high': (4, 5)},
        'academic_pressure': {'normal': (1, 2), 'elevated': (3, 3), 'high': (4, 5)},
        'work_study_hours': {'normal': (4, 8), 'elevated': (8, 12), 'high': (12, 24)},
    },
}

# Display names for features
FEATURE_DISPLAY_NAMES = {
    'glucose': 'Blood Glucose',
    'bmi': 'Body Mass Index',
    'blood_pressure': 'Blood Pressure',
    'insulin': 'Insulin Level',
    'cholesterol': 'Cholesterol',
    'resting_bp': 'Resting Blood Pressure',
    'max_hr': 'Maximum Heart Rate',
    'oldpeak': 'ST Depression',
    'sc': 'Serum Creatinine',
    'bu': 'Blood Urea',
    'hemo': 'Hemoglobin',
    'al': 'Albumin (Urine)',
    'sleep_duration': 'Sleep Duration',
    'financial_stress': 'Financial Stress',
    'academic_pressure': 'Academic Pressure',
    'work_study_hours': 'Work/Study Hours',
    'age': 'Age',
    'family_history': 'Family History',
    'suicidal_thoughts': 'Thoughts of Self-Harm',
}


class PredictionExplainerService:
    """Generates explanations for predictions"""
    
    @classmethod
    def explain(
        cls,
        disease_name: str,
        data: Dict[str, Any],
        prediction_result: Dict[str, Any],
    ) -> PredictionExplanation:
        """
        Generate explanation for a prediction.
        
        Args:
            disease_name: Name of the disease
            data: Input data used for prediction
            prediction_result: Result from prediction service
            
        Returns:
            PredictionExplanation with detailed breakdown
        """
        risk_factors = []
        protective_factors = []
        missing_factors = []
        
        thresholds = RISK_THRESHOLDS.get(disease_name, {})
        
        # Analyze each feature
        for feature, ranges in thresholds.items():
            value = data.get(feature)
            display_name = FEATURE_DISPLAY_NAMES.get(feature, feature.replace('_', ' ').title())
            
            if value is None or value == '':
                missing_factors.append(display_name)
                continue
            
            try:
                num_value = float(value)
            except (ValueError, TypeError):
                continue
            
            # Determine status
            status = cls._get_status(num_value, ranges)
            
            # Create risk factor
            factor = RiskFactor(
                name=feature,
                display_name=display_name,
                value=num_value,
                status=status,
                contribution='increases' if status in ['elevated', 'high', 'critical'] else 'neutral',
                explanation=cls._get_factor_explanation(feature, num_value, status, disease_name),
            )
            
            if status in ['elevated', 'high', 'critical']:
                risk_factors.append(factor)
            elif status == 'normal':
                factor.contribution = 'decreases'
                protective_factors.append(factor)
        
        # Generate summary
        risk_level = prediction_result.get('risk_level', 'unknown')
        probability = prediction_result.get('probability', 0)
        summary = cls._generate_summary(disease_name, risk_level, probability, len(risk_factors))
        
        # Generate recommendations
        recommendations = cls._generate_recommendations(disease_name, risk_factors, risk_level)
        
        # Tier explanation
        tier = prediction_result.get('tier', 'standard')
        confidence = prediction_result.get('confidence', 50)
        tier_explanation = cls._get_tier_explanation(tier, confidence)
        confidence_explanation = cls._get_confidence_explanation(confidence, len(missing_factors))
        
        return PredictionExplanation(
            summary=summary,
            risk_factors=risk_factors,
            protective_factors=protective_factors,
            missing_factors=missing_factors,
            recommendations=recommendations,
            tier_explanation=tier_explanation,
            confidence_explanation=confidence_explanation,
        )
    
    @classmethod
    def _get_status(cls, value: float, ranges: Dict[str, tuple]) -> str:
        """Determine status based on value and ranges"""
        for status, (low, high) in ranges.items():
            if low <= value <= high:
                return status
        return 'normal'
    
    @classmethod
    def _get_factor_explanation(
        cls, 
        feature: str, 
        value: float, 
        status: str, 
        disease_name: str
    ) -> str:
        """Generate explanation for a specific factor"""
        explanations = {
            'glucose': {
                'elevated': f'Your blood glucose ({value:.0f} mg/dL) is in the pre-diabetic range.',
                'high': f'Your blood glucose ({value:.0f} mg/dL) is in the diabetic range.',
                'critical': f'Your blood glucose ({value:.0f} mg/dL) is very high and needs immediate attention.',
            },
            'bmi': {
                'elevated': f'Your BMI ({value:.1f}) indicates overweight.',
                'high': f'Your BMI ({value:.1f}) indicates obesity.',
                'critical': f'Your BMI ({value:.1f}) indicates severe obesity.',
            },
            'cholesterol': {
                'elevated': f'Your cholesterol ({value:.0f} mg/dL) is borderline high.',
                'high': f'Your cholesterol ({value:.0f} mg/dL) is high.',
                'critical': f'Your cholesterol ({value:.0f} mg/dL) is very high.',
            },
            'sc': {
                'elevated': f'Your serum creatinine ({value:.1f} mg/dL) is slightly elevated.',
                'high': f'Your serum creatinine ({value:.1f} mg/dL) indicates reduced kidney function.',
                'critical': f'Your serum creatinine ({value:.1f} mg/dL) indicates significant kidney impairment.',
            },
            'sleep_duration': {
                'elevated': f'Your sleep duration is below optimal.',
                'high': f'Your sleep duration is significantly below recommended levels.',
            },
        }
        
        feature_explanations = explanations.get(feature, {})
        return feature_explanations.get(status, f'{FEATURE_DISPLAY_NAMES.get(feature, feature)} is {status}.')
    
    @classmethod
    def _generate_summary(
        cls, 
        disease_name: str, 
        risk_level: str, 
        probability: float,
        num_risk_factors: int
    ) -> str:
        """Generate summary text"""
        disease_display = disease_name.replace('_', ' ').title()
        
        if risk_level == 'low':
            return f'Your {disease_display} risk appears to be low ({probability:.0f}%). Continue maintaining healthy habits.'
        elif risk_level == 'medium':
            return f'Your {disease_display} risk is moderate ({probability:.0f}%). {num_risk_factors} risk factor(s) identified. Consider lifestyle modifications.'
        elif risk_level == 'high':
            return f'Your {disease_display} risk is elevated ({probability:.0f}%). {num_risk_factors} risk factor(s) identified. Please consult a healthcare provider.'
        else:
            return f'Your {disease_display} risk is significant ({probability:.0f}%). Immediate medical consultation is recommended.'
    
    @classmethod
    def _generate_recommendations(
        cls, 
        disease_name: str, 
        risk_factors: List[RiskFactor],
        risk_level: str
    ) -> List[str]:
        """Generate personalized recommendations"""
        recommendations = []
        
        # General recommendation based on risk level
        if risk_level in ['high', 'critical']:
            recommendations.append('Schedule an appointment with your healthcare provider for proper evaluation.')
        
        # Factor-specific recommendations
        factor_names = [f.name for f in risk_factors]
        
        if 'glucose' in factor_names or 'bmi' in factor_names:
            recommendations.append('Consider reducing sugar and refined carbohydrate intake.')
            recommendations.append('Aim for at least 150 minutes of moderate exercise per week.')
        
        if 'cholesterol' in factor_names:
            recommendations.append('Reduce saturated fat intake and increase fiber consumption.')
        
        if 'blood_pressure' in factor_names or 'resting_bp' in factor_names:
            recommendations.append('Reduce sodium intake and manage stress levels.')
        
        if 'sleep_duration' in factor_names:
            recommendations.append('Prioritize getting 7-8 hours of quality sleep per night.')
        
        if 'financial_stress' in factor_names or 'academic_pressure' in factor_names:
            recommendations.append('Consider stress management techniques like meditation or counseling.')
        
        # Always add disclaimer
        recommendations.append('This is not medical advice. Please consult a healthcare professional.')
        
        return recommendations[:5]  # Limit to 5 recommendations
    
    @classmethod
    def _get_tier_explanation(cls, tier: str, confidence: float) -> str:
        """Explain what the tier means"""
        explanations = {
            'screening': 'This is a basic screening based on limited information. More data would improve accuracy.',
            'standard': 'This assessment is based on standard health indicators.',
            'confirmation': 'This is a comprehensive assessment based on detailed health data.',
        }
        return explanations.get(tier, 'Standard assessment.')
    
    @classmethod
    def _get_confidence_explanation(cls, confidence: float, num_missing: int) -> str:
        """Explain the confidence score"""
        if confidence >= 80:
            return 'High confidence: Assessment based on comprehensive data.'
        elif confidence >= 60:
            if num_missing > 0:
                return f'Moderate confidence: {num_missing} additional data point(s) would improve accuracy.'
            return 'Moderate confidence: Assessment based on standard data.'
        else:
            return f'Lower confidence: Limited data available. {num_missing} key data point(s) missing.'
