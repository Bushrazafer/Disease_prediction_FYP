"""
Metadata loader for disease configurations.
"""

import json
from pathlib import Path
from typing import Dict, Optional
from django.conf import settings

from .base import (
    DiseaseMetadata, FeatureSpec, FeatureType, InputTier, 
    TierConfig, ThresholdConfig
)


class MetadataLoader:
    """Loads and caches disease metadata configurations"""
    
    _cache: Dict[str, DiseaseMetadata] = {}
    _metadata_dir: Optional[Path] = None
    
    @classmethod
    def get_metadata_dir(cls) -> Path:
        """Get the metadata directory path"""
        if cls._metadata_dir is None:
            cls._metadata_dir = Path(settings.BASE_DIR) / 'predictions' / 'metadata' / 'configs'
        return cls._metadata_dir
    
    @classmethod
    def load(cls, disease_name: str, use_cache: bool = True) -> Optional[DiseaseMetadata]:
        """Load metadata for a disease"""
        if use_cache and disease_name in cls._cache:
            return cls._cache[disease_name]
        
        # Try loading from JSON config
        config_path = cls.get_metadata_dir() / f'{disease_name}.json'
        if config_path.exists():
            metadata = cls._load_from_json(config_path)
        else:
            # Fall back to hardcoded defaults
            metadata = cls._get_default_metadata(disease_name)
        
        if metadata and use_cache:
            cls._cache[disease_name] = metadata
            
        return metadata
    
    @classmethod
    def _load_from_json(cls, path: Path) -> Optional[DiseaseMetadata]:
        """Load metadata from JSON file"""
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            return cls._parse_metadata(data)
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"Failed to load metadata from {path}: {e}")
            return None
    
    @classmethod
    def _parse_metadata(cls, data: dict) -> DiseaseMetadata:
        """Parse metadata from dictionary"""
        # Parse features
        features = {}
        for name, spec_data in data.get('features', {}).items():
            features[name] = FeatureSpec(
                name=name,
                display_name=spec_data.get('display_name', name.replace('_', ' ').title()),
                description=spec_data.get('description', ''),
                feature_type=FeatureType(spec_data.get('type', 'numeric')),
                required=spec_data.get('required', True),
                tier=InputTier(spec_data.get('tier', 'screening')),
                min_value=spec_data.get('min_value'),
                max_value=spec_data.get('max_value'),
                medical_min=spec_data.get('medical_min'),
                medical_max=spec_data.get('medical_max'),
                options=spec_data.get('options'),
                default_value=spec_data.get('default'),
                importance_weight=spec_data.get('importance', 0.5),
                high_signal=spec_data.get('high_signal', False),
                unit=spec_data.get('unit'),
                help_text=spec_data.get('help_text'),
                warning_low=spec_data.get('warning_low'),
                warning_high=spec_data.get('warning_high'),
            )
        
        # Parse tiers
        tiers = {}
        for tier_name, tier_data in data.get('tiers', {}).items():
            tiers[InputTier(tier_name)] = TierConfig(
                tier=InputTier(tier_name),
                required_features=tier_data.get('required_features', []),
                optional_features=tier_data.get('optional_features', []),
                min_confidence=tier_data.get('min_confidence', 50),
                max_confidence=tier_data.get('max_confidence', 70),
                description=tier_data.get('description', ''),
                disclaimer=tier_data.get('disclaimer', ''),
            )
        
        # Parse thresholds
        thresh_data = data.get('thresholds', {})
        thresholds = ThresholdConfig(
            low_max=thresh_data.get('low_max', 30),
            medium_max=thresh_data.get('medium_max', 60),
            high_max=thresh_data.get('high_max', 85),
            screening_threshold=thresh_data.get('screening_threshold', 0.3),
            confirmation_threshold=thresh_data.get('confirmation_threshold', 0.5),
        )
        
        return DiseaseMetadata(
            disease_name=data.get('disease_name', ''),
            display_name=data.get('display_name', ''),
            description=data.get('description', ''),
            features=features,
            feature_order=data.get('feature_order', list(features.keys())),
            tiers=tiers,
            thresholds=thresholds,
            clinician_only=data.get('clinician_only', False),
            clinician_warning=data.get('clinician_warning'),
            general_disclaimer=data.get('general_disclaimer', ''),
            result_disclaimer=data.get('result_disclaimer', ''),
            crisis_resources=data.get('crisis_resources'),
            output_classes=data.get('output_classes'),
            compatible_model_version=data.get('compatible_model_version'),
        )
    
    @classmethod
    def _get_default_metadata(cls, disease_name: str) -> Optional[DiseaseMetadata]:
        """Get default metadata for known diseases"""
        defaults = {
            'diabetes': cls._get_diabetes_metadata(),
            'cardiovascular': cls._get_cardiovascular_metadata(),
            'kidney': cls._get_kidney_metadata(),
            'breast_cancer': cls._get_breast_cancer_metadata(),
            'depression': cls._get_depression_metadata(),
            'obesity': cls._get_obesity_metadata(),
        }
        return defaults.get(disease_name)
    
    @classmethod
    def _get_diabetes_metadata(cls) -> DiseaseMetadata:
        """Default diabetes metadata"""
        return DiseaseMetadata(
            disease_name='diabetes',
            display_name='Diabetes',
            description='Type 2 Diabetes risk assessment based on metabolic indicators',
            features={
                'age': FeatureSpec(
                    name='age', display_name='Age', description='Age in years',
                    feature_type=FeatureType.NUMERIC, required=True,
                    min_value=1, max_value=120, medical_min=18, medical_max=100,
                    unit='years', high_signal=False
                ),
                'glucose': FeatureSpec(
                    name='glucose', display_name='Fasting Blood Glucose',
                    description='Fasting plasma glucose level',
                    feature_type=FeatureType.NUMERIC, required=True,
                    min_value=0, max_value=500, medical_min=50, medical_max=400,
                    unit='mg/dL', high_signal=True, tier=InputTier.STANDARD,
                    warning_low=70, warning_high=126
                ),
                'bmi': FeatureSpec(
                    name='bmi', display_name='Body Mass Index (BMI)',
                    description='Weight in kg divided by height in meters squared',
                    feature_type=FeatureType.NUMERIC, required=True,
                    min_value=10, max_value=70, medical_min=15, medical_max=60,
                    unit='kg/m²', high_signal=True
                ),
                'blood_pressure': FeatureSpec(
                    name='blood_pressure', display_name='Diastolic Blood Pressure',
                    description='Diastolic blood pressure',
                    feature_type=FeatureType.NUMERIC, required=True,
                    min_value=0, max_value=200, medical_min=40, medical_max=150,
                    unit='mmHg', high_signal=False
                ),
                'insulin': FeatureSpec(
                    name='insulin', display_name='Insulin Level',
                    description='2-Hour serum insulin',
                    feature_type=FeatureType.NUMERIC, required=False,
                    min_value=0, max_value=900, medical_min=0, medical_max=600,
                    unit='μU/mL', high_signal=True, tier=InputTier.CONFIRMATION
                ),
                'diabetes_pedigree': FeatureSpec(
                    name='diabetes_pedigree', display_name='Family History Score',
                    description='Diabetes pedigree function - indicates genetic predisposition',
                    feature_type=FeatureType.NUMERIC, required=False,
                    min_value=0, max_value=2.5, medical_min=0, medical_max=2.5,
                    help_text='Score based on family history of diabetes (0-2.5)',
                    high_signal=True
                ),
            },
            tiers={
                InputTier.SCREENING: TierConfig(
                    tier=InputTier.SCREENING,
                    required_features=['age', 'bmi'],
                    optional_features=['blood_pressure'],
                    min_confidence=40, max_confidence=60,
                    description='Basic screening with demographic data',
                    disclaimer='This is a basic screening. Lab tests recommended for accurate assessment.'
                ),
                InputTier.STANDARD: TierConfig(
                    tier=InputTier.STANDARD,
                    required_features=['age', 'bmi', 'glucose', 'blood_pressure'],
                    optional_features=['diabetes_pedigree'],
                    min_confidence=60, max_confidence=80,
                    description='Standard assessment with glucose data',
                    disclaimer='Results based on standard metabolic indicators.'
                ),
                InputTier.CONFIRMATION: TierConfig(
                    tier=InputTier.CONFIRMATION,
                    required_features=['age', 'bmi', 'glucose', 'blood_pressure', 'insulin'],
                    optional_features=['diabetes_pedigree', 'skin_thickness'],
                    min_confidence=80, max_confidence=95,
                    description='Comprehensive assessment with full lab data',
                    disclaimer='High-confidence assessment based on complete metabolic panel.'
                ),
            },
            general_disclaimer='This tool provides risk estimates only and is NOT a diagnosis. '
                             'Always consult a healthcare provider for proper evaluation.',
            result_disclaimer='This is a risk assessment, not a medical diagnosis. '
                            'Please consult your doctor for proper testing and evaluation.',
        )
    
    @classmethod
    def _get_cardiovascular_metadata(cls) -> DiseaseMetadata:
        """Default cardiovascular metadata"""
        return DiseaseMetadata(
            disease_name='cardiovascular',
            display_name='Cardiovascular Disease',
            description='Heart disease risk assessment',
            clinician_only=False,
            general_disclaimer='This tool estimates cardiovascular risk and is NOT a diagnosis. '
                             'Consult a cardiologist for proper evaluation.',
        )
    
    @classmethod
    def _get_kidney_metadata(cls) -> DiseaseMetadata:
        """Default kidney disease metadata"""
        return DiseaseMetadata(
            disease_name='kidney',
            display_name='Chronic Kidney Disease',
            description='CKD risk assessment based on renal function indicators',
            general_disclaimer='This tool estimates kidney disease risk. '
                             'Lab tests are required for accurate diagnosis.',
        )
    
    @classmethod
    def _get_breast_cancer_metadata(cls) -> DiseaseMetadata:
        """Default breast cancer metadata - CLINICIAN ONLY"""
        return DiseaseMetadata(
            disease_name='breast_cancer',
            display_name='Breast Cancer Cell Analysis',
            description='Analysis of Fine Needle Aspiration (FNA) biopsy cell measurements',
            clinician_only=True,
            clinician_warning='⚠️ CLINICIAN USE ONLY: This tool analyzes FNA biopsy cell measurements '
                            'from pathology lab reports. These values are NOT patient-providable. '
                            'This tool requires professional biopsy results and should only be used '
                            'by healthcare professionals.',
            general_disclaimer='This tool analyzes professional biopsy results and is NOT for self-diagnosis. '
                             'Cell measurements must come from certified pathology lab reports.',
            result_disclaimer='This analysis is based on FNA cell measurements and should be '
                            'interpreted by a qualified oncologist or pathologist.',
        )
    
    @classmethod
    def _get_depression_metadata(cls) -> DiseaseMetadata:
        """Default depression metadata with crisis resources"""
        return DiseaseMetadata(
            disease_name='depression',
            display_name='Depression Risk Assessment',
            description='Mental health risk screening based on validated indicators',
            crisis_resources=[
                {'name': 'National Helpline (India)', 'number': '1800-599-0019'},
                {'name': 'iCall', 'number': '9152987821'},
                {'name': 'Vandrevala Foundation', 'number': '1860-2662-345'},
                {'name': 'National Suicide Prevention Lifeline (US)', 'number': '988'},
            ],
            general_disclaimer='This is a screening tool only and NOT a clinical diagnosis. '
                             'If you are experiencing severe symptoms or thoughts of self-harm, '
                             'please seek immediate professional help.',
            result_disclaimer='This assessment does not replace professional mental health evaluation. '
                            'Please consult a mental health professional for proper diagnosis and treatment.',
        )
    
    @classmethod
    def _get_obesity_metadata(cls) -> DiseaseMetadata:
        """Default obesity metadata"""
        return DiseaseMetadata(
            disease_name='obesity',
            display_name='Obesity Level Prediction',
            description='Obesity classification based on lifestyle and physical indicators',
            output_classes=[
                'Insufficient_Weight', 'Normal_Weight', 
                'Overweight_Level_I', 'Overweight_Level_II',
                'Obesity_Type_I', 'Obesity_Type_II', 'Obesity_Type_III'
            ],
            general_disclaimer='This tool provides obesity level classification based on '
                             'lifestyle factors. Consult a healthcare provider for personalized advice.',
        )
    
    @classmethod
    def clear_cache(cls, disease_name: Optional[str] = None):
        """Clear metadata cache"""
        if disease_name:
            cls._cache.pop(disease_name, None)
        else:
            cls._cache.clear()
