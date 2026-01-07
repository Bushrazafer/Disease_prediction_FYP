"""
MediCare AI - Depression Prediction Model
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
from sklearn.preprocessing import RobustScaler, PowerTransformer
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
)
from sklearn.linear_model import LogisticRegression

from training_utils import MedicalModelTrainer, print_audit_compliance_summary

try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

import warnings
warnings.filterwarnings('ignore')


class AuditCompliantDepressionTrainer:
    """Audit-compliant depression model trainer."""
    
    GENDER_MAP = {'Male': 1, 'Female': 0}
    SLEEP_MAP = {
        "'Less than 5 hours'": 0, 'Less than 5 hours': 0,
        "'5-6 hours'": 1, '5-6 hours': 1,
        "'7-8 hours'": 2, '7-8 hours': 2,
        "'More than 8 hours'": 3, 'More than 8 hours': 3
    }
    DIETARY_MAP = {'Unhealthy': 0, 'Moderate': 1, 'Healthy': 2}
    YES_NO_MAP = {'Yes': 1, 'No': 0}
    
    def __init__(self, model_type='screening'):
        self.model_type = model_type
        self.trainer = MedicalModelTrainer(model_type=model_type)
        self.model = None
        self.scaler = None
        self.power_transformer = None
        self.feature_names = None
        self.metrics = {}
        
    def load_and_preprocess(self, filepath='student_depression_dataset.csv'):
        """Load and preprocess data."""
        print("\n" + "=" * 70)
        print("📊 LOADING DEPRESSION DATA")
        print("=" * 70)
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        full_path = os.path.join(script_dir, filepath)
        
        df = pd.read_csv(full_path)
        print(f"   Loaded {len(df)} records")
        
        # Drop unnecessary columns
        cols_to_drop = ['id', 'City', 'Profession', 'Degree']
        for col in cols_to_drop:
            if col in df.columns:
                df = df.drop(col, axis=1)
        
        # Encode categorical variables
        df['Gender'] = df['Gender'].map(self.GENDER_MAP)
        df['Sleep Duration'] = df['Sleep Duration'].map(self.SLEEP_MAP)
        df['Dietary Habits'] = df['Dietary Habits'].map(self.DIETARY_MAP)
        
        yes_no_cols = ['Have you ever had suicidal thoughts ?', 'Family History of Mental Illness']
        for col in yes_no_cols:
            if col in df.columns:
                df[col] = df[col].map(self.YES_NO_MAP)
        
        if 'Financial Stress' in df.columns:
            df['Financial Stress'] = pd.to_numeric(df['Financial Stress'], errors='coerce')
            df['Financial Stress'] = df['Financial Stress'].fillna(3)
        
        # Rename columns
        df.columns = [
            'gender', 'age', 'academic_pressure', 'work_pressure', 'cgpa',
            'study_satisfaction', 'job_satisfaction', 'sleep_duration',
            'dietary_habits', 'suicidal_thoughts', 'work_study_hours',
            'financial_stress', 'family_history', 'depression'
        ]
        
        # Handle missing values
        df = df.fillna(df.median(numeric_only=True))
        
        # Feature engineering
        df['total_pressure'] = df['academic_pressure'] + df['work_pressure']
        df['satisfaction_score'] = (df['study_satisfaction'] + df['job_satisfaction']) / 2
        df['sleep_risk'] = (df['sleep_duration'] <= 1).astype(int)
        df['risk_factor_count'] = (
            df['sleep_risk'] + df['suicidal_thoughts'] + df['family_history'] +
            (df['financial_stress'] >= 4).astype(int) + (df['total_pressure'] >= 4).astype(int)
        )
        df['depression_risk_score'] = (
            df['suicidal_thoughts'] * 0.25 + df['family_history'] * 0.15 +
            (df['total_pressure'] / 10) * 0.15 + (1 - df['satisfaction_score'] / 5) * 0.15 +
            (3 - df['sleep_duration']) / 3 * 0.10 + (df['financial_stress'] / 5) * 0.10
        )
        
        return df
    
    def build_model(self):
        """Build regularized ensemble model."""
        print("\n" + "=" * 70)
        print("🏗️ BUILDING REGULARIZED ENSEMBLE MODEL")
        print("=" * 70)
        
        estimators = []
        
        rf = RandomForestClassifier(
            n_estimators=150, max_depth=6, min_samples_split=15, min_samples_leaf=5,
            class_weight='balanced', random_state=42, n_jobs=-1
        )
        estimators.append(('rf', rf))
        print("   ✓ Random Forest (regularized)")
        
        if HAS_LIGHTGBM:
            lgbm = LGBMClassifier(
                n_estimators=150, max_depth=5, learning_rate=0.05,
                num_leaves=31, subsample=0.7, colsample_bytree=0.7,
                reg_alpha=0.5, reg_lambda=2.0, class_weight='balanced',
                random_state=42, n_jobs=-1, verbose=-1
            )
            estimators.append(('lgbm', lgbm))
            print("   ✓ LightGBM (regularized)")
        elif HAS_XGBOOST:
            xgb = XGBClassifier(
                n_estimators=150, max_depth=4, learning_rate=0.05,
                subsample=0.7, colsample_bytree=0.7, reg_alpha=0.5, reg_lambda=2.0,
                random_state=42, n_jobs=-1, eval_metric='logloss', verbosity=0
            )
            estimators.append(('xgb', xgb))
            print("   ✓ XGBoost (regularized)")
        
        gb = GradientBoostingClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.05,
            min_samples_split=15, min_samples_leaf=8, subsample=0.6, random_state=42
        )
        estimators.append(('gb', gb))
        print("   ✓ Gradient Boosting (regularized)")
        
        lr = LogisticRegression(C=0.3, solver='lbfgs', max_iter=1000, class_weight='balanced', random_state=42)
        estimators.append(('lr', lr))
        print("   ✓ Logistic Regression (regularized)")
        
        meta_learner = LogisticRegression(C=0.1, solver='lbfgs', max_iter=1000, random_state=42)
        
        self.model = StackingClassifier(
            estimators=estimators, final_estimator=meta_learner,
            cv=5, stack_method='predict_proba', n_jobs=-1, passthrough=False
        )
        
        return self.model
    
    def train(self, df):
        """AUDIT-COMPLIANT training pipeline."""
        print("\n" + "=" * 70)
        print("🎯 AUDIT-COMPLIANT TRAINING PIPELINE")
        print("=" * 70)
        
        X = df.drop('depression', axis=1)
        y = df['depression']
        self.feature_names = X.columns.tolist()
        
        # D1 FIX: Safe train/test split
        X_train, X_test, y_train, y_test = self.trainer.safe_train_test_split(X.values, y.values, test_size=0.2)
        
        # Scale
        self.scaler = RobustScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.power_transformer = PowerTransformer(method='yeo-johnson')
        X_train_final = self.power_transformer.fit_transform(X_train_scaled)
        X_test_final = self.power_transformer.transform(X_test_scaled)
        
        # D1 FIX: SMOTE only on training
        X_train_resampled, y_train_resampled = self.trainer.apply_smote_safely(X_train_final, y_train)
        
        self.build_model()
        
        # D3 FIX: Repeated K-fold
        cv_results = self.trainer.evaluate_with_repeated_kfold(self.model, X_train_final, y_train)
        
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
        
        self.metrics['model_type'] = self.model_type
        self.metrics['audit_compliant'] = True
        self.metrics['version'] = '5.0.0-audit'
        
        with open(os.path.join(output_dir, 'metrics.pkl'), 'wb') as f:
            pickle.dump(self.metrics, f)
        print("   ✓ metrics.pkl")
        
        print(f"\n   ✅ Training complete! Accuracy: {self.metrics['accuracy']*100:.2f}%")


def main():
    print("\n" + "=" * 70)
    print("  🧠 MediCare AI - Depression Prediction")
    print("  AUDIT-COMPLIANT Training Pipeline v5.0")
    print("=" * 70)
    
    print_audit_compliance_summary()
    
    trainer = AuditCompliantDepressionTrainer(model_type='screening')
    df = trainer.load_and_preprocess()
    trainer.train(df)
    trainer.save_artifacts()
    
    return trainer


if __name__ == "__main__":
    trainer = main()
