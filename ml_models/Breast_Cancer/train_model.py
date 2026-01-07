"""
Breast Cancer Prediction Model Training
Wisconsin Breast Cancer Dataset - High-accuracy Stacking Ensemble
Target: 97%+ accuracy for Malignant/Benign classification
"""

import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, PowerTransformer, LabelEncoder
from sklearn.impute import SimpleImputer
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


# Feature groups for breast cancer dataset
MEAN_FEATURES = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
    'smoothness_mean', 'compactness_mean', 'concavity_mean', 
    'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean'
]

SE_FEATURES = [
    'radius_se', 'texture_se', 'perimeter_se', 'area_se',
    'smoothness_se', 'compactness_se', 'concavity_se',
    'concave_points_se', 'symmetry_se', 'fractal_dimension_se'
]

WORST_FEATURES = [
    'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst',
    'smoothness_worst', 'compactness_worst', 'concavity_worst',
    'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst'
]


def load_data():
    """Load the breast cancer dataset"""
    df = pd.read_csv('data.csv')
    print(f"Loaded dataset: {df.shape}")
    print(f"Target distribution:")
    print(df['diagnosis'].value_counts())
    return df


def clean_data(df):
    """Clean and preprocess the breast cancer dataset"""
    print("\n" + "=" * 60)
    print("DATA CLEANING")
    print("=" * 60)
    
    df = df.copy()
    
    # Drop id and unnamed columns
    if 'id' in df.columns:
        df = df.drop('id', axis=1)
    if 'Unnamed: 32' in df.columns:
        df = df.drop('Unnamed: 32', axis=1)
    
    # Fix column names (replace spaces with underscores)
    df.columns = df.columns.str.replace(' ', '_')
    
    # Encode target: M (Malignant) = 1, B (Benign) = 0
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
    
    print(f"After cleaning: {df.shape}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    
    return df


def engineer_features(df):
    """Create derived features for better prediction"""
    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING")
    print("=" * 60)
    
    df = df.copy()
    
    # 1. Area to Perimeter ratio (circularity indicator)
    df['area_perimeter_ratio'] = df['area_mean'] / (df['perimeter_mean'] + 0.001)
    print("   Created: Area-Perimeter ratio")
    
    # 2. Compactness score (combination of shape features)
    df['shape_score'] = (df['compactness_mean'] + df['concavity_mean'] + df['concave_points_mean']) / 3
    print("   Created: Shape score")
    
    # 3. Size score (combination of size features)
    df['size_score'] = (df['radius_mean'] / 30 + df['area_mean'] / 2500 + df['perimeter_mean'] / 200) / 3
    print("   Created: Size score")
    
    # 4. Texture irregularity
    df['texture_irregularity'] = df['texture_worst'] - df['texture_mean']
    print("   Created: Texture irregularity")
    
    # 5. Size variation (worst vs mean)
    df['size_variation'] = df['radius_worst'] / (df['radius_mean'] + 0.001)
    print("   Created: Size variation")
    
    # 6. Concavity severity
    df['concavity_severity'] = df['concavity_worst'] * df['concave_points_worst']
    print("   Created: Concavity severity")
    
    # 7. Symmetry deviation
    df['symmetry_deviation'] = abs(df['symmetry_worst'] - df['symmetry_mean'])
    print("   Created: Symmetry deviation")
    
    # 8. Fractal complexity
    df['fractal_complexity'] = df['fractal_dimension_worst'] * df['fractal_dimension_mean']
    print("   Created: Fractal complexity")
    
    # 9. Overall malignancy score (weighted combination)
    df['malignancy_score'] = (
        df['radius_worst'] / 40 * 0.15 +
        df['concave_points_worst'] / 0.3 * 0.20 +
        df['concavity_worst'] / 1.5 * 0.15 +
        df['area_worst'] / 4000 * 0.15 +
        df['perimeter_worst'] / 300 * 0.10 +
        df['compactness_worst'] / 1.5 * 0.10 +
        df['texture_worst'] / 50 * 0.10 +
        df['symmetry_worst'] / 0.7 * 0.05
    )
    print("   Created: Malignancy score")
    
    # 10. Cell uniformity score
    df['uniformity_score'] = 1 - (
        df['radius_se'] / df['radius_mean'].clip(lower=0.001) +
        df['area_se'] / df['area_mean'].clip(lower=0.001) +
        df['perimeter_se'] / df['perimeter_mean'].clip(lower=0.001)
    ) / 3
    df['uniformity_score'] = df['uniformity_score'].clip(0, 1)
    print("   Created: Uniformity score")
    
    print(f"\n   Total features: {len(df.columns) - 1}")
    return df


def preprocess_data(df):
    """Complete preprocessing pipeline"""
    
    # Clean data
    df = clean_data(df)
    
    # Engineer features
    df = engineer_features(df)
    
    # Separate features and target
    X = df.drop('diagnosis', axis=1)
    y = df['diagnosis']
    
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
            scaler, power_transformer, imputer, feature_names)


def build_stacking_ensemble():
    """Build a high-accuracy stacking ensemble for breast cancer prediction"""
    
    print("\n" + "=" * 60)
    print("BUILDING STACKING ENSEMBLE")
    print("=" * 60)
    
    # Base estimators - optimized for breast cancer classification
    estimators = [
        ('rf', RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )),
        ('et', ExtraTreesClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )),
        ('gb', GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
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
            max_depth=5,
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
            max_depth=5,
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
    print(classification_report(y_test, y_pred, target_names=['Benign', 'Malignant']))
    
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


def save_model(model, scaler, power_transformer, imputer, feature_names, metrics):
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
    
    # Save metrics
    with open('metrics.pkl', 'wb') as f:
        pickle.dump(metrics, f)
    print("Saved: metrics.pkl")


def main():
    print("=" * 60)
    print("BREAST CANCER PREDICTION MODEL TRAINING")
    print("Wisconsin Breast Cancer Dataset")
    print("=" * 60)
    
    # Load data
    df = load_data()
    
    # Preprocess
    print("\nPreprocessing data...")
    (X_train, X_test, y_train, y_test, 
     scaler, power_transformer, imputer, feature_names) = preprocess_data(df)
    
    print(f"\nTraining set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    print(f"Features: {len(feature_names)}")
    
    # Train and evaluate
    model, metrics = train_and_evaluate(X_train, X_test, y_train, y_test)
    
    # Save
    save_model(model, scaler, power_transformer, imputer, feature_names, metrics)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\nFinal Accuracy: {metrics['accuracy'] * 100:.2f}%")
    print(f"Final AUC-ROC:  {metrics['auc_roc']:.4f}")


if __name__ == "__main__":
    main()
