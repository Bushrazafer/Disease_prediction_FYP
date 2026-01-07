"""
ML Model Management Script
Centralized script for managing all ML models
"""

import os
import sys
import django
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'mediCare.settings')
django.setup()

from predictions.models import MLModelVersion


class ModelManager:
    """Centralized model management"""
    
    MODELS_CONFIG = {
        'diabetes': {
            'name': 'diabetes_rf_v1',
            'version': 'v1.0',
            'accuracy': 0.85,
            'description': 'Random Forest model for diabetes prediction',
            'file_path': 'ml_models/diabetes/model.pkl',
            'feature_schema': [
                'pregnancies', 'glucose', 'blood_pressure', 'skin_thickness',
                'insulin', 'bmi', 'diabetes_pedigree', 'age',
                'age_bmi_interaction', 'glucose_bmi_interaction',
                'is_high_risk_age', 'is_obese', 'is_prediabetic', 'is_diabetic_glucose'
            ],
            'feature_types': {
                'pregnancies': 'numeric', 'glucose': 'numeric',
                'blood_pressure': 'numeric', 'skin_thickness': 'numeric',
                'insulin': 'numeric', 'bmi': 'numeric',
                'diabetes_pedigree': 'numeric', 'age': 'numeric',
                'age_bmi_interaction': 'computed', 'glucose_bmi_interaction': 'computed',
                'is_high_risk_age': 'computed', 'is_obese': 'computed',
                'is_prediabetic': 'computed', 'is_diabetic_glucose': 'computed'
            }
        }
    }
    
    @classmethod
    def register_model(cls, disease_name):
        """Register a model in database"""
        if disease_name not in cls.MODELS_CONFIG:
            print(f"❌ Unknown disease: {disease_name}")
            return False
        
        config = cls.MODELS_CONFIG[disease_name]
        config['disease'] = disease_name
        
        model, created = MLModelVersion.objects.update_or_create(
            name=config['name'],
            defaults=config
        )
        
        status = "registered" if created else "updated"
        print(f"✅ Model {status}: {model.name}")
        print(f"   Disease: {model.disease}")
        print(f"   Version: {model.version}")
        print(f"   Accuracy: {model.accuracy * 100}%")
        print(f"   Features: {len(model.feature_schema)}")
        
        return True
    
    @classmethod
    def list_models(cls):
        """List all registered models"""
        models = MLModelVersion.objects.all()
        
        if not models:
            print("No models registered")
            return
        
        print("\n" + "="*70)
        print("REGISTERED ML MODELS")
        print("="*70)
        
        for model in models:
            status = "✅ ACTIVE" if model.is_active else "⚪ INACTIVE"
            print(f"\n{status} {model.name}")
            print(f"   Disease: {model.get_disease_display()}")
            print(f"   Version: {model.version}")
            print(f"   Accuracy: {model.accuracy * 100}%")
            print(f"   Features: {len(model.feature_schema)}")
            print(f"   File: {model.file_path}")
    
    @classmethod
    def activate_model(cls, model_name):
        """Activate a specific model"""
        try:
            model = MLModelVersion.objects.get(name=model_name)
            model.is_active = True
            model.save()
            print(f"✅ Activated: {model.name}")
        except MLModelVersion.DoesNotExist:
            print(f"❌ Model not found: {model_name}")
    
    @classmethod
    def deactivate_model(cls, model_name):
        """Deactivate a specific model"""
        try:
            model = MLModelVersion.objects.get(name=model_name)
            model.is_active = False
            model.save()
            print(f"⚪ Deactivated: {model.name}")
        except MLModelVersion.DoesNotExist:
            print(f"❌ Model not found: {model_name}")
    
    @classmethod
    def delete_model(cls, model_name):
        """Delete a model from database"""
        try:
            model = MLModelVersion.objects.get(name=model_name)
            model.delete()
            print(f"🗑️ Deleted: {model_name}")
        except MLModelVersion.DoesNotExist:
            print(f"❌ Model not found: {model_name}")
    
    @classmethod
    def register_all(cls):
        """Register all configured models"""
        print("\n" + "="*70)
        print("REGISTERING ALL MODELS")
        print("="*70 + "\n")
        
        for disease in cls.MODELS_CONFIG.keys():
            cls.register_model(disease)
            print()


def main():
    """Main CLI interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description='ML Model Management')
    parser.add_argument('action', choices=[
        'register', 'list', 'activate', 'deactivate', 'delete', 'register-all'
    ], help='Action to perform')
    parser.add_argument('--model', help='Model name or disease name')
    
    args = parser.parse_args()
    
    if args.action == 'register':
        if not args.model:
            print("❌ Please specify --model <disease_name>")
            return
        ModelManager.register_model(args.model)
    
    elif args.action == 'list':
        ModelManager.list_models()
    
    elif args.action == 'activate':
        if not args.model:
            print("❌ Please specify --model <model_name>")
            return
        ModelManager.activate_model(args.model)
    
    elif args.action == 'deactivate':
        if not args.model:
            print("❌ Please specify --model <model_name>")
            return
        ModelManager.deactivate_model(args.model)
    
    elif args.action == 'delete':
        if not args.model:
            print("❌ Please specify --model <model_name>")
            return
        confirm = input(f"Are you sure you want to delete {args.model}? (yes/no): ")
        if confirm.lower() == 'yes':
            ModelManager.delete_model(args.model)
    
    elif args.action == 'register-all':
        ModelManager.register_all()


if __name__ == '__main__':
    main()
