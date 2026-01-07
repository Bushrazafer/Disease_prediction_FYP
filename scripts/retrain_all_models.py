"""
Batch Model Retraining Script
Retrains all models to be compatible with scikit-learn 1.3.2
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, PowerTransformer, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE

import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def retrain_cardiovascular():
    """Retrain cardiovascular model"""
    print("\n" + "=" * 60)
    print("  CARDIOVASCULAR MODEL RETRAINING")
    print("=" * 60)
    
    model_dir = os.path.join(BASE_DIR, 'ml_models', 'cardiovascular')
    os.chdir(model_dir)
    
    # Load data
    df = pd.read_csv('heart_disease_expanded.csv')
    print(f"   Loaded: {len(df)} records")
    
    # Standardize column names
    df.columns = [col.lower().replace(' ', '_') for col in df.columns]
    
    # Target column
    target_col = 'heart_disease' if 'heart_disease' in df.columns else df.columns[-1]
    
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    # Handle missing
    X = X.replace(0, np.nan, regex=False)
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    feature_names = X_imputed.columns.tolist()
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_imputed, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Balance
    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
    
    # Scale
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train_bal)
    X_test_scaled = scaler.transform(X_test)
    
    power_transformer = PowerTransformer(method='yeo-johnson')
    X_train_final = power_transformer.fit_transform(X_train_scaled)
    X_test_final = power_transformer.transform(X_test_scaled)
    
    # Train
    model = RandomForestClassifier(
        n_estimators=100, max_depth=8, min_samples_split=10,
        min_samples_leaf=4, class_weight='balanced', random_state=42, n_jobs=-1
    )
    model.fit(X_train_final, y_train_bal)
    
    # Evaluate
    y_pred = model.predict(X_test_final)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"   ✅ Accuracy: {accuracy * 100:.2f}%")
    
    # Save
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('scaler.pkl', 'wb') as f:
        pickle.dump({'scaler': scaler, 'power_transformer': power_transformer}, f)
    with open('imputer.pkl', 'wb') as f:
        pickle.dump(imputer, f)
    with open('features.pkl', 'wb') as f:
        pickle.dump(feature_names, f)
    
    print("   ✓ Saved all artifacts")
    return accuracy


def retrain_kidney():
    """Retrain kidney disease model"""
    print("\n" + "=" * 60)
    print("  KIDNEY DISEASE MODEL RETRAINING")
    print("=" * 60)
    
    model_dir = os.path.join(BASE_DIR, 'ml_models', 'Chronic Kidney Disease (CKD)')
    os.chdir(model_dir)
    
    # Load data
    df = pd.read_csv('kidney_disease.csv')
    print(f"   Loaded: {len(df)} records")
    
    # Clean column names
    df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]
    
    # Target column
    if 'classification' in df.columns:
        target_col = 'classification'
        # Convert target to binary
        df[target_col] = df[target_col].apply(lambda x: 1 if 'ckd' in str(x).lower() and 'not' not in str(x).lower() else 0)
    elif 'class' in df.columns:
        target_col = 'class'
        df[target_col] = df[target_col].apply(lambda x: 1 if 'ckd' in str(x).lower() and 'not' not in str(x).lower() else 0)
    else:
        target_col = df.columns[-1]
    
    # Separate features and target
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    # Handle categorical columns
    for col in X.columns:
        if X[col].dtype == 'object':
            le = LabelEncoder()
            X[col] = X[col].fillna('unknown')
            X[col] = le.fit_transform(X[col].astype(str))
    
    # Handle missing
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    feature_names = X_imputed.columns.tolist()
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_imputed, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Balance
    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
    
    # Scale
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train_bal)
    X_test_scaled = scaler.transform(X_test)
    
    power_transformer = PowerTransformer(method='yeo-johnson')
    X_train_final = power_transformer.fit_transform(X_train_scaled)
    X_test_final = power_transformer.transform(X_test_scaled)
    
    # Train
    model = RandomForestClassifier(
        n_estimators=100, max_depth=8, min_samples_split=10,
        min_samples_leaf=4, class_weight='balanced', random_state=42, n_jobs=-1
    )
    model.fit(X_train_final, y_train_bal)
    
    # Evaluate
    y_pred = model.predict(X_test_final)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"   ✅ Accuracy: {accuracy * 100:.2f}%")
    
    # Save
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('scaler.pkl', 'wb') as f:
        pickle.dump({'scaler': scaler, 'power_transformer': power_transformer}, f)
    with open('imputer.pkl', 'wb') as f:
        pickle.dump(imputer, f)
    with open('features.pkl', 'wb') as f:
        pickle.dump(feature_names, f)
    
    print("   ✓ Saved all artifacts")
    return accuracy


def retrain_breast_cancer():
    """Retrain breast cancer model"""
    print("\n" + "=" * 60)
    print("  BREAST CANCER MODEL RETRAINING")
    print("=" * 60)
    
    model_dir = os.path.join(BASE_DIR, 'ml_models', 'Breast_Cancer')
    os.chdir(model_dir)
    
    # Load data
    df = pd.read_csv('data.csv')
    print(f"   Loaded: {len(df)} records")
    
    # Clean column names
    df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]
    
    # Drop ID column if exists
    if 'id' in df.columns:
        df = df.drop('id', axis=1)
    if 'unnamed:_32' in df.columns:
        df = df.drop('unnamed:_32', axis=1)
    
    # Target column
    target_col = 'diagnosis' if 'diagnosis' in df.columns else df.columns[-1]
    
    # Convert target to binary
    if df[target_col].dtype == 'object':
        df[target_col] = df[target_col].apply(lambda x: 1 if x == 'M' else 0)
    
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    # Handle missing
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    feature_names = X_imputed.columns.tolist()
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_imputed, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Balance
    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
    
    # Scale
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train_bal)
    X_test_scaled = scaler.transform(X_test)
    
    power_transformer = PowerTransformer(method='yeo-johnson')
    X_train_final = power_transformer.fit_transform(X_train_scaled)
    X_test_final = power_transformer.transform(X_test_scaled)
    
    # Train
    model = RandomForestClassifier(
        n_estimators=100, max_depth=8, min_samples_split=10,
        min_samples_leaf=4, class_weight='balanced', random_state=42, n_jobs=-1
    )
    model.fit(X_train_final, y_train_bal)
    
    # Evaluate
    y_pred = model.predict(X_test_final)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"   ✅ Accuracy: {accuracy * 100:.2f}%")
    
    # Save
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('scaler.pkl', 'wb') as f:
        pickle.dump({'scaler': scaler, 'power_transformer': power_transformer}, f)
    with open('imputer.pkl', 'wb') as f:
        pickle.dump(imputer, f)
    with open('features.pkl', 'wb') as f:
        pickle.dump(feature_names, f)
    
    print("   ✓ Saved all artifacts")
    return accuracy


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  BATCH MODEL RETRAINING")
    print("  Compatible with scikit-learn 1.3.2")
    print("=" * 60)
    
    results = {}
    
    try:
        results['cardiovascular'] = retrain_cardiovascular()
    except Exception as e:
        print(f"   ❌ Cardiovascular failed: {e}")
        results['cardiovascular'] = None
    
    try:
        results['kidney'] = retrain_kidney()
    except Exception as e:
        print(f"   ❌ Kidney failed: {e}")
        results['kidney'] = None
    
    try:
        results['breast_cancer'] = retrain_breast_cancer()
    except Exception as e:
        print(f"   ❌ Breast Cancer failed: {e}")
        results['breast_cancer'] = None
    
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    for model, acc in results.items():
        if acc:
            print(f"   {model}: {acc * 100:.2f}%")
        else:
            print(f"   {model}: FAILED")
    print("=" * 60)
