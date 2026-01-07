"""
MediCare AI - Cardiovascular Disease Prediction Model
AUDIT-COMPLIANT Training Pipeline v5.0

Addresses all audit issues (D1-D5):
- D1: Data Leakage Prevention - SMOTE only after train/test split
- D2: Realistic Accuracy Targets - 85-92% for screening
- D3: Stratified K-Fold - Repeated stratified K-fold evaluation
- D4: Calibration Curves - Brier score and calibration analysis
- D5: Threshold Optimization - Sensitivity-optimized threshold
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, 
    ExtraTreesClassifier, StackingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from training_utils import MedicalModelTrainer, print_audit_compliance_summary

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

import warnings
warnings.filterwarnings('ignore')


class AuditCompliantCardiovascularTrainer:
    """Audit-compliant cardiovascular model trainer."""
    
    def __init__(self, model_type='screening'):
        self.model_type = model_type
        self.trainer = MedicalModelTrainer(model_type=model_type)
        self.model = None
        self.scaler = None
        self.power_transformer = None
        self.imputer = None
        self.feature_names = None
        self.metrics = {}
        
    def load_and_preprocess(self, filepath='heart_disease_expanded.csv'):
        """Load and preprocess data."""
        print("\n" + "=" * 70)
        print("📊 LOADING CARDIOVASCULAR DATA")
        print("=" * 70)
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        full_path = os.path.join(script_dir, filepath)
        
        df = pd.read_csv(full_path)
        print(f"   Loaded {len(df)} records")
        
        # Handle missing cholesterol values
        df['cholesterol'] = df['cholesterol'].replace(0, np.nan)
        
        # Impute missing values
        self.imputer = SimpleImputer(strategy='median')
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'heart_disease' in numeric_cols:
            numeric_cols.remove('heart_disease')
        
        df[numeric_cols] = self.imputer.fit_transform(df[numeric_cols])
        
        return df
    
    def build_model(self):
        """Build regularized ensemble model."""
        print("\n" + "=" * 70)
        print("🏗️ BUILDING REGULARIZED ENSEMBLE MODEL")
        print("=" * 70)
        
        estimators = []
        
        # Random Forest - regularized
        rf = RandomForestClassifier(
            n_estimators=150,
            max_depth=6,
            min_samples_split=15,
            min_samples_leaf=5,
            max_features='sqrt',
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        estimators.append(('rf', rf))
        print("   ✓ Random Forest (regularized)")
        
        # XGBoost if available
        if HAS_XGBOOST:
            xgb = XGBClassifier(
                n_estimators=150,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.7,
                colsample_bytree=0.7,
                reg_alpha=0.5,
                reg_lambda=2.0,
                random_state=42,
                n_jobs=-1,
                eval_metric='logloss',
                verbosity=0
            )
            estimators.append(('xgb', xgb))
            print("   ✓ XGBoost (regularized)")
        
        # Gradient Boosting
        gb = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
            min_samples_split=15,
            min_samples_leaf=8,
            subsample=0.6,
            random_state=42
        )
        estimators.append(('gb', gb))
        print("   ✓ Gradient Boosting (regularized)")
        
        # Logistic Regression
        lr = LogisticRegression(
            C=0.3,
            solver='lbfgs',
            max_iter=1000,
            class_weight='balanced',
            random_state=42
        )
        estimators.append(('lr', lr))
        print("   ✓ Logistic Regression (regularized)")
        
        # Meta-learner
        meta_learner = LogisticRegression(C=0.1, solver='lbfgs', max_iter=1000, random_state=42)
        
        self.model = StackingClassifier(
            estimators=estimators,
            final_estimator=meta_learner,
            cv=5,
            stack_method='predict_proba',
            n_jobs=-1,
            passthrough=False
        )
        
        return self.model
    
    def train(self, df):
        """AUDIT-COMPLIANT training pipeline."""
        print("\n" + "=" * 70)
        print("🎯 AUDIT-COMPLIANT TRAINING PIPELINE")
        print("=" * 70)
        
        X = df.drop('heart_disease', axis=1)
        y = df['heart_disease']
        self.feature_names = X.columns.tolist()
        
        # D1 FIX: Safe train/test split BEFORE any resampling
        X_train, X_test, y_train, y_test = self.trainer.safe_train_test_split(
            X.values, y.values, test_size=0.2
        )
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Power transform
        self.power_transformer = PowerTransformer(method='yeo-johnson')
        X_train_final = self.power_transformer.fit_transform(X_train_scaled)
        X_test_final = self.power_transformer.transform(X_test_scaled)
        
        # D1 FIX: Apply SMOTE only to training data
        X_train_resampled, y_train_resampled = self.trainer.apply_smote_safely(
            X_train_final, y_train
        )
        
        # Build model
        self.build_model()
        
        # D3 FIX: Evaluate with repeated stratified K-fold
        cv_results = self.trainer.evaluate_with_repeated_kfold(
            self.model, X_train_final, y_train, n_splits=5, n_repeats=3
        )
        
        # Train final model
        print("\n   Training final model...")
        self.model.fit(X_train_resampled, y_train_resampled)
        
        # D2, D4, D5: Full evaluation
        self.metrics = self.trainer.full_evaluation(self.model, X_test_final, y_test)
        self.metrics['cv_accuracy_mean'] = cv_results['accuracy']['mean']
        self.metrics['cv_accuracy_std'] = cv_results['accuracy']['std']
        
        return self.metrics
    
    def save_artifacts(self, output_dir=None):
        """Save all model artifacts."""
        if output_dir is None:
            output_dir = os.path.dirname(os.path.abspath(__file__))
            
        print("\n" + "=" * 70)
        print("💾 SAVING MODEL ARTIFACTS")
        print("=" * 70)
        
        with open(os.path.join(output_dir, 'model.pkl'), 'wb') as f:
            pickle.dump(self.model, f)
        print("   ✓ model.pkl")
        
        with open(os.path.join(output_dir, 'scaler.pkl'), 'wb') as f:
            pickle.dump({'scaler': self.scaler, 'power_transformer': self.power_transformer}, f)
        print("   ✓ scaler.pkl")
        
        with open(os.path.join(output_dir, 'features.pkl'), 'wb') as f:
            pickle.dump(self.feature_names, f)
        print("   ✓ features.pkl")
        
        with open(os.path.join(output_dir, 'imputer.pkl'), 'wb') as f:
            pickle.dump(self.imputer, f)
        print("   ✓ imputer.pkl")
        
        self.metrics['model_type'] = self.model_type
        self.metrics['audit_compliant'] = True
        self.metrics['version'] = '5.0.0-audit'
        
        with open(os.path.join(output_dir, 'metrics.pkl'), 'wb') as f:
            pickle.dump(self.metrics, f)
        print("   ✓ metrics.pkl")
        
        print(f"\n   ✅ Training complete! Accuracy: {self.metrics['accuracy']*100:.2f}%")


def main():
    print("\n" + "=" * 70)
    print("  🫀 MediCare AI - Cardiovascular Disease Prediction")
    print("  AUDIT-COMPLIANT Training Pipeline v5.0")
    print("=" * 70)
    
    print_audit_compliance_summary()
    
    trainer = AuditCompliantCardiovascularTrainer(model_type='screening')
    df = trainer.load_and_preprocess()
    trainer.train(df)
    trainer.save_artifacts()
    
    return trainer


if __name__ == "__main__":
    trainer = main()
