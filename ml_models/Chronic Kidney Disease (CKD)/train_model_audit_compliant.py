"""
MediCare AI - Chronic Kidney Disease (CKD) Prediction Model
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
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, 
    ExtraTreesClassifier, StackingClassifier
)
from sklearn.linear_model import LogisticRegression

from training_utils import MedicalModelTrainer, print_audit_compliance_summary

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

import warnings
warnings.filterwarnings('ignore')


NUMERIC_FEATURES = ['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc']
CATEGORICAL_FEATURES = ['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']


class AuditCompliantCKDTrainer:
    """Audit-compliant CKD model trainer."""
    
    def __init__(self, model_type='screening'):
        self.model_type = model_type
        self.trainer = MedicalModelTrainer(model_type=model_type)
        self.model = None
        self.scaler = None
        self.power_transformer = None
        self.imputer = None
        self.feature_names = None
        self.label_encoders = {}
        self.metrics = {}
        
    def load_and_preprocess(self, filepath='kidney_disease.csv'):
        """Load and preprocess CKD data."""
        print("\n" + "=" * 70)
        print("📊 LOADING CKD DATA")
        print("=" * 70)
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        full_path = os.path.join(script_dir, filepath)
        
        df = pd.read_csv(full_path)
        print(f"   Loaded {len(df)} records")
        
        # Drop id column
        if 'id' in df.columns:
            df = df.drop('id', axis=1)
        
        # Clean target
        df['classification'] = df['classification'].str.strip()
        df['classification'] = df['classification'].map({'ckd': 1, 'notckd': 0})
        
        # Clean categorical columns
        binary_mappings = {
            'rbc': {'normal': 1, 'abnormal': 0},
            'pc': {'normal': 1, 'abnormal': 0},
            'pcc': {'present': 1, 'notpresent': 0},
            'ba': {'present': 1, 'notpresent': 0},
            'htn': {'yes': 1, 'no': 0},
            'dm': {'yes': 1, 'no': 0, ' yes': 1, ' no': 0},
            'cad': {'yes': 1, 'no': 0},
            'appet': {'good': 1, 'poor': 0},
            'pe': {'yes': 1, 'no': 0},
            'ane': {'yes': 1, 'no': 0}
        }
        
        for col in CATEGORICAL_FEATURES:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.lower()
                df[col] = df[col].replace({'?': np.nan, 'nan': np.nan, '': np.nan})
                if col in binary_mappings:
                    df[col] = df[col].map(binary_mappings[col])
        
        # Clean numeric columns
        for col in NUMERIC_FEATURES:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Impute missing values
        numeric_cols = [col for col in NUMERIC_FEATURES if col in df.columns]
        knn_imputer = KNNImputer(n_neighbors=5, weights='distance')
        df[numeric_cols] = knn_imputer.fit_transform(df[numeric_cols])
        
        # Impute categorical with mode
        for col in CATEGORICAL_FEATURES:
            if col in df.columns and df[col].isnull().sum() > 0:
                mode_val = df[col].mode()[0] if len(df[col].mode()) > 0 else 0
                df[col] = df[col].fillna(mode_val)
        
        df = df.fillna(0)
        
        # Feature engineering
        df['egfr_estimate'] = 141 * np.power(df['sc'].clip(lower=0.1) / 0.9, -1.209) * np.power(0.993, df['age'])
        df['egfr_estimate'] = df['egfr_estimate'].clip(0, 150)
        df['anemia_score'] = np.where(df['hemo'] < 7, 3, np.where(df['hemo'] < 10, 2, np.where(df['hemo'] < 12, 1, 0)))
        df['albumin_creatinine_ratio'] = df['al'] / (df['sc'].clip(lower=0.1))
        
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
        
        if HAS_XGBOOST:
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
        
        X = df.drop('classification', axis=1)
        y = df['classification']
        self.feature_names = X.columns.tolist()
        
        # D1 FIX: Safe train/test split
        X_train, X_test, y_train, y_test = self.trainer.safe_train_test_split(X.values, y.values, test_size=0.2)
        
        # Scale
        self.imputer = SimpleImputer(strategy='median')
        X_train_imputed = self.imputer.fit_transform(X_train)
        X_test_imputed = self.imputer.transform(X_test)
        
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train_imputed)
        X_test_scaled = self.scaler.transform(X_test_imputed)
        
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
        
        with open(os.path.join(output_dir, 'imputer.pkl'), 'wb') as f:
            pickle.dump(self.imputer, f)
        print("   ✓ imputer.pkl")
        
        with open(os.path.join(output_dir, 'label_encoders.pkl'), 'wb') as f:
            pickle.dump(self.label_encoders, f)
        print("   ✓ label_encoders.pkl")
        
        self.metrics['model_type'] = self.model_type
        self.metrics['audit_compliant'] = True
        self.metrics['version'] = '5.0.0-audit'
        
        with open(os.path.join(output_dir, 'metrics.pkl'), 'wb') as f:
            pickle.dump(self.metrics, f)
        print("   ✓ metrics.pkl")
        
        print(f"\n   ✅ Training complete! Accuracy: {self.metrics['accuracy']*100:.2f}%")


def main():
    print("\n" + "=" * 70)
    print("  🫘 MediCare AI - Chronic Kidney Disease Prediction")
    print("  AUDIT-COMPLIANT Training Pipeline v5.0")
    print("=" * 70)
    
    print_audit_compliance_summary()
    
    trainer = AuditCompliantCKDTrainer(model_type='screening')
    df = trainer.load_and_preprocess()
    trainer.train(df)
    trainer.save_artifacts()
    
    return trainer


if __name__ == "__main__":
    trainer = main()
