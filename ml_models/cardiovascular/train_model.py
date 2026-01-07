"""
Heart/Cardiovascular Disease Prediction Model Training
High-accuracy Stacking Ensemble with 98%+ accuracy target
"""

import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, PowerTransformer
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

def load_data():
    """Load the expanded heart disease dataset"""
    df = pd.read_csv('heart_disease_expanded.csv')
    print(f"Loaded dataset: {df.shape}")
    return df

def preprocess_data(df):
    """Preprocess data for training"""
    
    # Separate features and target
    X = df.drop('heart_disease', axis=1)
    y = df['heart_disease']
    
    feature_names = list(X.columns)
    
    # Handle missing values (cholesterol = 0)
    X['cholesterol'] = X['cholesterol'].replace(0, np.nan)
    
    # Impute missing values
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)
    X = pd.DataFrame(X_imputed, columns=feature_names)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Power transform for better distribution
    power_transformer = PowerTransformer(method='yeo-johnson', standardize=False)
    X_train_transformed = power_transformer.fit_transform(X_train_scaled)
    X_test_transformed = power_transformer.transform(X_test_scaled)
    
    return (X_train_transformed, X_test_transformed, y_train, y_test, 
            scaler, power_transformer, imputer, feature_names)

def build_stacking_ensemble():
    """Build a high-accuracy stacking ensemble"""
    
    # Base estimators
    estimators = [
        ('rf', RandomForestClassifier(
            n_estimators=300,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )),
        ('et', ExtraTreesClassifier(
            n_estimators=300,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )),
        ('gb', GradientBoostingClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            random_state=42
        )),
        ('ada', AdaBoostClassifier(
            n_estimators=150,
            learning_rate=0.1,
            random_state=42
        )),
        ('knn', KNeighborsClassifier(
            n_neighbors=7,
            weights='distance',
            n_jobs=-1
        )),
        ('svm', SVC(
            kernel='rbf',
            C=10,
            gamma='scale',
            probability=True,
            random_state=42
        )),
        ('mlp', MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            max_iter=500,
            random_state=42
        ))
    ]
    
    # Add XGBoost if available
    if HAS_XGBOOST:
        estimators.append(('xgb', XGBClassifier(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.1,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss',
            n_jobs=-1
        )))
    
    # Add LightGBM if available
    if HAS_LIGHTGBM:
        estimators.append(('lgbm', LGBMClassifier(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.1,
            random_state=42,
            verbose=-1,
            n_jobs=-1
        )))
    
    # Meta-learner
    meta_learner = LogisticRegression(
        C=1.0,
        max_iter=1000,
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
    
    return stacking_clf

def train_and_evaluate(X_train, X_test, y_train, y_test):
    """Train the model and evaluate performance"""
    
    print("\n" + "=" * 60)
    print("TRAINING STACKING ENSEMBLE")
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
    print(classification_report(y_test, y_pred, target_names=['No Disease', 'Heart Disease']))
    
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
        'cv_std': cv_scores.std()
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
    
    # Save scaler (as dict like diabetes model)
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
    print("HEART DISEASE PREDICTION MODEL TRAINING")
    print("=" * 60)
    
    # Load data
    df = load_data()
    
    # Preprocess
    print("\nPreprocessing data...")
    (X_train, X_test, y_train, y_test, 
     scaler, power_transformer, imputer, feature_names) = preprocess_data(df)
    
    print(f"Training set: {X_train.shape}")
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
