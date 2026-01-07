"""
Input Validator with Medical Range Checking
"""

from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of input validation"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    sanitized_data: Dict[str, Any]
    out_of_range_fields: List[str]


# Medical plausibility ranges for common features
MEDICAL_RANGES = {
    # Diabetes features
    'glucose': {'min': 30, 'max': 600, 'unit': 'mg/dL'},
    'blood_pressure': {'min': 30, 'max': 250, 'unit': 'mmHg'},
    'bmi': {'min': 10, 'max': 80, 'unit': 'kg/m²'},
    'insulin': {'min': 0, 'max': 900, 'unit': 'μU/mL'},
    'age': {'min': 0, 'max': 120, 'unit': 'years'},
    'skin_thickness': {'min': 0, 'max': 100, 'unit': 'mm'},
    'pregnancies': {'min': 0, 'max': 20, 'unit': 'count'},
    
    # Cardiovascular features
    'cholesterol': {'min': 50, 'max': 600, 'unit': 'mg/dL'},
    'max_hr': {'min': 50, 'max': 250, 'unit': 'bpm'},
    'resting_bp': {'min': 50, 'max': 250, 'unit': 'mmHg'},
    'oldpeak': {'min': -5, 'max': 10, 'unit': 'mm'},
    'triglycerides': {'min': 30, 'max': 1000, 'unit': 'mg/dL'},
    'hdl': {'min': 10, 'max': 150, 'unit': 'mg/dL'},
    'ldl': {'min': 30, 'max': 400, 'unit': 'mg/dL'},
    'ejection_fraction': {'min': 10, 'max': 80, 'unit': '%'},
    
    # Kidney features
    'sc': {'min': 0.1, 'max': 20, 'unit': 'mg/dL'},  # serum creatinine
    'bu': {'min': 5, 'max': 200, 'unit': 'mg/dL'},   # blood urea
    'bgr': {'min': 30, 'max': 600, 'unit': 'mg/dL'}, # blood glucose random
    'hemo': {'min': 3, 'max': 20, 'unit': 'g/dL'},   # hemoglobin
    'sg': {'min': 1.000, 'max': 1.050, 'unit': ''},  # specific gravity
    'sod': {'min': 100, 'max': 180, 'unit': 'mEq/L'}, # sodium
    'pot': {'min': 2, 'max': 8, 'unit': 'mEq/L'},    # potassium
    
    # Breast cancer FNA features (cell measurements)
    'radius_mean': {'min': 5, 'max': 35, 'unit': 'μm'},
    'texture_mean': {'min': 5, 'max': 50, 'unit': ''},
    'perimeter_mean': {'min': 30, 'max': 250, 'unit': 'μm'},
    'area_mean': {'min': 100, 'max': 3000, 'unit': 'μm²'},
    'smoothness_mean': {'min': 0.01, 'max': 0.25, 'unit': ''},
    'compactness_mean': {'min': 0.01, 'max': 0.5, 'unit': ''},
    'concavity_mean': {'min': 0, 'max': 0.6, 'unit': ''},
    'concave_points_mean': {'min': 0, 'max': 0.3, 'unit': ''},
    
    # Depression features
    'academic_pressure': {'min': 0, 'max': 5, 'unit': 'scale'},
    'work_pressure': {'min': 0, 'max': 5, 'unit': 'scale'},
    'cgpa': {'min': 0, 'max': 10, 'unit': 'GPA'},
    'financial_stress': {'min': 0, 'max': 5, 'unit': 'scale'},
    'work_study_hours': {'min': 0, 'max': 24, 'unit': 'hours'},
    
    # Obesity features
    'height': {'min': 0.5, 'max': 2.5, 'unit': 'm'},
    'weight': {'min': 20, 'max': 300, 'unit': 'kg'},
    'fcvc': {'min': 1, 'max': 3, 'unit': 'scale'},
    'ncp': {'min': 1, 'max': 4, 'unit': 'meals'},
    'ch2o': {'min': 1, 'max': 3, 'unit': 'liters'},
    'faf': {'min': 0, 'max': 3, 'unit': 'days/week'},
    'tue': {'min': 0, 'max': 2, 'unit': 'hours'},
}


class InputValidator:
    """Validates and sanitizes input data for predictions"""
    
    @classmethod
    def validate(
        cls,
        data: Dict[str, Any],
        required_fields: List[str],
        disease_name: Optional[str] = None
    ) -> ValidationResult:
        """
        Validate input data with medical range checking.
        
        Args:
            data: Input data dictionary
            required_fields: List of required field names
            disease_name: Optional disease name for context-specific validation
            
        Returns:
            ValidationResult with validation status and sanitized data
        """
        errors = []
        warnings = []
        out_of_range = []
        sanitized = {}
        
        # Normalize field names
        normalized_data = cls._normalize_field_names(data)
        
        # Check required fields
        for field in required_fields:
            normalized_field = cls._normalize_key(field)
            if normalized_field not in normalized_data or normalized_data[normalized_field] in [None, '']:
                errors.append(f"Missing required field: {field}")
        
        # Validate and sanitize each field
        for key, value in normalized_data.items():
            if value in [None, '']:
                continue
                
            # Try to convert to appropriate type
            sanitized_value, field_errors, field_warnings = cls._validate_field(key, value)
            
            if field_errors:
                errors.extend(field_errors)
            if field_warnings:
                warnings.extend(field_warnings)
                out_of_range.append(key)
                
            sanitized[key] = sanitized_value
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            sanitized_data=sanitized,
            out_of_range_fields=out_of_range
        )
    
    @classmethod
    def _normalize_field_names(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize field names to lowercase with underscores"""
        normalized = {}
        for key, value in data.items():
            norm_key = cls._normalize_key(key)
            normalized[norm_key] = value
        return normalized
    
    @classmethod
    def _normalize_key(cls, key: str) -> str:
        """Normalize a single key"""
        return key.lower().replace(' ', '_').replace('-', '_')
    
    @classmethod
    def _validate_field(
        cls, 
        field_name: str, 
        value: Any
    ) -> Tuple[Any, List[str], List[str]]:
        """Validate a single field value"""
        errors = []
        warnings = []
        
        # Get medical range if available
        range_info = MEDICAL_RANGES.get(field_name)
        
        # Try numeric conversion
        try:
            if isinstance(value, str):
                # Handle boolean-like strings
                if value.lower() in ['yes', 'true', '1']:
                    return 1, errors, warnings
                elif value.lower() in ['no', 'false', '0']:
                    return 0, errors, warnings
                    
            num_value = float(value)
            
            # Check medical plausibility
            if range_info:
                if num_value < range_info['min']:
                    warnings.append(
                        f"{field_name} ({num_value}) is below typical range "
                        f"(min: {range_info['min']} {range_info['unit']})"
                    )
                elif num_value > range_info['max']:
                    warnings.append(
                        f"{field_name} ({num_value}) is above typical range "
                        f"(max: {range_info['max']} {range_info['unit']})"
                    )
            
            return num_value, errors, warnings
            
        except (ValueError, TypeError):
            # Keep as string for categorical values
            return value, errors, warnings
    
    @classmethod
    def get_field_info(cls, field_name: str) -> Optional[Dict[str, Any]]:
        """Get medical range info for a field"""
        return MEDICAL_RANGES.get(cls._normalize_key(field_name))
