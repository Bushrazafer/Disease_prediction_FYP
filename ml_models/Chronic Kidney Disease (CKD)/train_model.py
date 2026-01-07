"""
Chronic Kidney Disease (CKD) Prediction Model Training
High-accuracy Stacking Ensemble with 95%+ accuracy target
"""

import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, PowerTransformer, LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.ensemble import (
    RandomForestClassifier, 
    GradientBoostingClassifier, 
    ExtraTreesClassifier,
    StackingClassifier,
    AdaBoostClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    roc_auc_score,
    classification_report,
    confusion_matrix
)

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("XGBoost not available, using alternatives")

try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    print("LightGBM not available, using alternatives")


# Feature configuration for CKD dataset
NUMERIC_FEATURES = ['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc']
CATEGORICAL_FEATURES = ['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']

# Medical reference ranges for CKD
MEDICAL_RANGES = {
    'age': {'min': 2, 'max': 90},
    'bp': {'min': 50, 'max': 180},
    'sg': {'min': 1.005, 'max': 1.025},
    'al': {'min': 0, 'max': 5},
    'su': {'min': 0, 'max': 5},
    'bgr': {'min': 22, 'max': 490},
    'bu': {'min': 1.5, 'max': 391},
    'sc': {'min': 0.4, 'max': 76},
    'sod': {'min': 4.5, 'max': 163},
    'pot': {'min': 2.5, 'max': 47},
    'hemo': {'min': 3.1, 'max': 17.8},
    'pcv': {'min': 9, 'max': 54},
    'wc': {'min': 2200, 'max': 26400},
    'rc': {'min': 2.1, 'max': 8}
}


def load_data():
    """Load the CKD dataset"""
    df = pd.read_csv('kidney_disease.csv')
    print(f"Loaded dataset: {df.shape}")
    print(f"Target distribution:")
    print(df['classification'].value_counts())
    return df


def clean_data(df):
    """Clean and preprocess the CKD dataset"""
    print("\n" + "=" * 60)
    print("DATA CLEANING")
    print("=" * 60)
    
    df = df.copy()
    
    # Drop id column
    if 'id' in df.columns:
        df = df.drop('id', axis=1)
    
    # Clean target variable
    df['classification'] = df['classification'].str.strip()
    df['classification'] = df['classification'].map({'ckd': 1, 'notckd': 0})
    
    # Clean categorical columns - strip whitespace and handle variations
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower()
            df[col] = df[col].replace({'?': np.nan, 'nan': np.nan, '': np.nan})
    
    # Clean numeric columns
    for col in NUMERIC_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    print(f"Missing values per column:")
    missing = df.isnull().sum()
    print(missing[missing > 0])
    
    return df


def encode_categorical(df, label_encoders=None):
    """Encode categorical variables"""
    print("\n" + "=" * 60)
    print("ENCODING CATEGORICAL FEATURES")
    print("=" * 60)
    
    df = df.copy()
    
    if label_encoders is None:
        label_encoders = {}
    
    # Binary mappings for categorical features
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
            if col in binary_mappings:
                df[col] = df[col].map(binary_mappings[col])
            print(f"   Encoded {col}")
    
    return df, label_encoders


def impute_missing_values(df):
    """Impute missing values using KNN for numeric and mode for categorical"""
    print("\n" + "=" * 60)
    print("IMPUTING MISSING VALUES")
    print("=" * 60)
    
    df = df.copy()
    
    # Impute numeric features with KNN
    numeric_cols = [col for col in NUMERIC_FEATURES if col in df.columns]
    if numeric_cols:
        knn_imputer = KNNImputer(n_neighbors=5, weights='distance')
        df[numeric_cols] = knn_imputer.fit_transform(df[numeric_cols])
        print(f"   KNN imputed {len(numeric_cols)} numeric features")
    
    # Impute categorical features with mode
    cat_cols = [col for col in CATEGORICAL_FEATURES if col in df.columns]
    for col in cat_cols:
        if df[col].isnull().sum() > 0:
            mode_val = df[col].mode()[0] if len(df[col].mode()) > 0 else 0
            df[col] = df[col].fillna(mode_val)
            print(f"   Mode imputed {col}")
    
    # Final check - fill any remaining NaN with 0
    df = df.fillna(0)
    
    print(f"\n   Missing values after imputation: {df.isnull().sum().sum()}")
    return df


def engineer_features(df):
    """Create derived features for better prediction"""
    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING")
    print("=" * 60)
    
    df = df.copy()
    
    # 1. eGFR estimation (simplified CKD-EPI formula approximation)
    # Using serum creatinine and age
    df['egfr_estimate'] = 141 * np.power(df['sc'].clip(lower=0.1) / 0.9, -1.209) * np.power(0.993, df['age'])
    df['egfr_estimate'] = df['egfr_estimate'].clip(0, 150)
    print("   Created: eGFR estimate")
    
    # 2. Anemia severity score (based on hemoglobin)
    df['anemia_score'] = np.where(df['hemo'] < 7, 3,
                         np.where(df['hemo'] < 10, 2,
                         np.where(df['hemo'] < 12, 1, 0)))
    print("   Created: Anemia severity score")
    
    # 3. Blood pressure category
    df['bp_category'] = np.where(df['bp'] < 80, 0,
                        np.where(df['bp'] < 90, 1,
                        np.where(df['bp'] < 120, 2, 3)))
    print("   Created: BP category")
    
    # 4. Albumin-Creatinine interaction (kidney damage indicator)
    df['albumin_creatinine_ratio'] = df['al'] / (df['sc'].clip(lower=0.1))
    print("   Created: Albumin-Creatinine ratio")
    
    # 5. Urea-Creatinine ratio (kidney function)
    df['urea_creatinine_ratio'] = df['bu'] / (df['sc'].clip(lower=0.1))
    print("   Created: Urea-Creatinine ratio")
    
    # 6. Electrolyte imbalance score
    df['electrolyte_score'] = (
        (df['sod'] < 135).astype(int) + 
        (df['sod'] > 145).astype(int) +
        (df['pot'] < 3.5).astype(int) + 
        (df['pot'] > 5.0).astype(int)
    )
    print("   Created: Electrolyte imbalance score")
    
    # 7. Comorbidity count
    comorbidity_cols = ['htn', 'dm', 'cad', 'ane']
    df['comorbidity_count'] = df[comorbidity_cols].sum(axis=1)
    print("   Created: Comorbidity count")
    
    # 8. Age risk category
    df['age_risk'] = np.where(df['age'] < 40, 0,
                    np.where(df['age'] < 60, 1, 2))
    print("   Created: Age risk category")
    
    # 9. Specific gravity abnormality
    df['sg_abnormal'] = ((df['sg'] < 1.010) | (df['sg'] > 1.025)).astype(int)
    print("   Created: SG abnormality flag")
    
    # 10. Overall CKD risk score
    df['ckd_risk_score'] = (
        (df['sc'] > 1.2).astype(int) * 0.25 +
        (df['bu'] > 20).astype(int) * 0.15 +
        (df['hemo'] < 12).astype(int) * 0.15 +
        (df['al'] > 0).astype(int) * 0.20 +
        (df['htn'] == 1).astype(int) * 0.10 +
        (df['dm'] == 1).astype(int) * 0.10 +
        (df['age'] > 60).astype(int) * 0.05
    )
    print("   Created: CKD risk score")
    
    print(f"\n   Total features: {len(df.columns) - 1}")
    return df


def preprocess_data(df):
    """Complete preprocessing pipeline"""
    
    # Clean data
    df = clean_data(df)
    
    # Encode categorical
    df, label_encoders = encode_categorical(df)
    
    # Impute missing values
    df = impute_missing_values(df)
    
    # Engineer features
    df = engineer_features(df)
    
    # Separate features and target
    X = df.drop('classification', axis=1)
    y = df['classification']
    
    feature_names = list(X.columns)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create imputer for production
    imputer = SimpleImputer(strategy='median')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)
    
    # Power transform for better distribution
    power_transformer = PowerTransformer(method='yeo-johnson', standardize=False)
    X_train_transformed = power_transformer.fit_transform(X_train_scaled)
    X_test_transformed = power_transformer.transform(X_test_scaled)
    
    return (X_train_transformed, X_test_transformed, y_train, y_test, 
            scaler, power_transformer, imputer, feature_names, label_encoders)


def build_stacking_ensemble():
    """Build a high-accuracy stacking ensemble for CKD prediction"""
    
    print("\n" + "=" * 60)
    print("BUILDING STACKING ENSEMBLE")
    print("=" * 60)
    
    # Base estimators
    estimators = [
        ('rf', RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )),
        ('et', ExtraTreesClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )),
        ('gb', GradientBoostingClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42
        )),
        ('ada', AdaBoostClassifier(
            n_estimators=150,
            learning_rate=0.1,
            random_state=42
        )),
        ('knn', KNeighborsClassifier(
            n_neighbors=5,
            weights='distance',
            n_jobs=-1
        )),
        ('svm', SVC(
            kernel='rbf',
            C=10,
            gamma='scale',
            probability=True,
            class_weight='balanced',
            random_state=42
        )),
        ('mlp', MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            max_iter=500,
            early_stopping=True,
            random_state=42
        ))
    ]
    
    # Add XGBoost if available
    if HAS_XGBOOST:
        estimators.append(('xgb', XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=1.5,
            random_state=42,
            eval_metric='logloss',
            n_jobs=-1
        )))
        print("   Added XGBoost")
    
    # Add LightGBM if available
    if HAS_LIGHTGBM:
        estimators.append(('lgbm', LGBMClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight='balanced',
            random_state=42,
            verbose=-1,
            n_jobs=-1
        )))
        print("   Added LightGBM")
    
    # Meta-learner
    meta_learner = LogisticRegression(
        C=1.0,
        max_iter=1000,
        class_weight='balanced',
        random_state=42
    )
    
    # Stacking Classifier
    stacking_clf = StackingClassifier(
        estimators=estimators,
        final_estimator=meta_learner,
        cv=5,
        stack_method='predict_proba',
        n_jobs=-1
    )
    
    print(f"   Built stacking ensemble with {len(estimators)} base models")
    return stacking_clf


def train_and_evaluate(X_train, X_test, y_train, y_test):
    """Train the model and evaluate performance"""
    
    print("\n" + "=" * 60)
    print("TRAINING MODEL")
    print("=" * 60)
    
    # Build model
    model = build_stacking_ensemble()
    
    # Train
    print("\nTraining model (this may take a few minutes)...")
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, y_pred_proba)
    
    print("\n" + "=" * 60)
    print("MODEL PERFORMANCE")
    print("=" * 60)
    print(f"\nAccuracy:  {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall:    {recall * 100:.2f}%")
    print(f"F1-Score:  {f1 * 100:.2f}%")
    print(f"AUC-ROC:   {auc_roc:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['No CKD', 'CKD']))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Cross-validation
    print("\n" + "=" * 60)
    print("CROSS-VALIDATION (5-Fold)")
    print("=" * 60)
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
    print(f"\nCV Scores: {cv_scores}")
    print(f"CV Mean:   {cv_scores.mean() * 100:.2f}%")
    print(f"CV Std:    {cv_scores.std() * 100:.2f}%")
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc_roc': auc_roc,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'confusion_matrix': cm.tolist()
    }
    
    return model, metrics


def save_model(model, scaler, power_transformer, imputer, feature_names, label_encoders, metrics):
    """Save model and preprocessing objects"""
    
    print("\n" + "=" * 60)
    print("SAVING MODEL")
    print("=" * 60)
    
    # Save model
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("Saved: model.pkl")
    
    # Save scaler (as dict like other models)
    scaler_dict = {
        'scaler': scaler,
        'power_transformer': power_transformer
    }
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler_dict, f)
    print("Saved: scaler.pkl")
    
    # Save imputer
    with open('imputer.pkl', 'wb') as f:
        pickle.dump(imputer, f)
    print("Saved: imputer.pkl")
    
    # Save feature names
    with open('features.pkl', 'wb') as f:
        pickle.dump(feature_names, f)
    print("Saved: features.pkl")
    
    # Save label encoders
    with open('label_encoders.pkl', 'wb') as f:
        pickle.dump(label_encoders, f)
    print("Saved: label_encoders.pkl")
    
    # Save metrics
    with open('metrics.pkl', 'wb') as f:
        pickle.dump(metrics, f)
    print("Saved: metrics.pkl")


def main():
    print("=" * 60)
    print("CHRONIC KIDNEY DISEASE (CKD) PREDICTION MODEL TRAINING")
    print("=" * 60)
    
    # Load data
    df = load_data()
    
    # Preprocess
    print("\nPreprocessing data...")
    (X_train, X_test, y_train, y_test, 
     scaler, power_transformer, imputer, feature_names, label_encoders) = preprocess_data(df)
    
    print(f"\nTraining set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    print(f"Features: {len(feature_names)}")
    
    # Train and evaluate
    model, metrics = train_and_evaluate(X_train, X_test, y_train, y_test)
    
    # Save
    save_model(model, scaler, power_transformer, imputer, feature_names, label_encoders, metrics)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\nFinal Accuracy: {metrics['accuracy'] * 100:.2f}%")
    print(f"Final AUC-ROC:  {metrics['auc_roc']:.4f}")


if __name__ == "__main__":
    main()
