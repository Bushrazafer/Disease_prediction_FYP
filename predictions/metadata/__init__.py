"""
Disease Metadata System for MediCare AI
Provides centralized configuration for input specs, thresholds, UI labels, and validation rules.
"""

from .base import DiseaseMetadata, FeatureSpec, TierConfig
from .loader import MetadataLoader

__all__ = ['DiseaseMetadata', 'FeatureSpec', 'TierConfig', 'MetadataLoader']
