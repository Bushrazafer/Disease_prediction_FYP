"""
Diabetes Model Retraining Script - Compatible with scikit-learn 1.3.2
This script retrains the model to be compatible with the current environment.
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import RobustScaler, PowerTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("  DIABETES MODEL RETRAINING - Compatible Version")
print("  scikit-learn 1.3.2 Compatible")
print("=" * 70)

# Get script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Load data
print("\n📊 Loading data...")
if os.path.exists('diabetes_expanded.csv'):
    df = pd.read_csv('diabetes_expanded.csv')
    print(f"   Loaded expanded dataset: {len(df)} records")
else:
    df = pd.read_csv('diabetes.csv')
    print(f"   Loaded original dataset: {len(df)} records")

# Standardize column names
df.columns = [col.lower().replace(' ', '_') for col in df.columns]

# Handle target column name
if 'outcome' in df.columns:
    target_col = 'outcome'
elif 'Outcome' in df.columns:
    target_col = 'Outcome'
else:
    target_col = df.columns[-1]

print(f"   Target column: {target_col}")

# Separate features and target
X = df.drop(target_col, axis=1)
y = df[target_col]

# Handle missing values
print("\n🔧 Preprocessing...")
# Replace 0s with NaN for columns where 0 is invalid
zero_invalid = ['glucose', 'bloodpressure', 'skinthickness', 'insulin', 'bmi']
for col in zero_invalid:
    if col in X.columns:
        X[col] = X[col].replace(0, np.nan)

# Impute missing values
imputer = SimpleImputer(strategy='median')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Feature names
feature_names = X_imputed.columns.tolist()
print(f"   Features: {len(feature_names)}")

# Split data
print("\n📊 Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X_imputed, y, test_size=0.2, random_state=42, stratify=y
)
print(f"   Train: {len(X_train)}, Test: {len(X_test)}")

# Balance with SMOTE
print("\n⚖️ Balancing with SMOTE...")
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
print(f"   After SMOTE: {len(X_train_balanced)} samples")

# Scale features
print("\n📏 Scaling features...")
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train_balanced)
X_test_scaled = scaler.transform(X_test)

# Power transform
power_transformer = PowerTransformer(method='yeo-johnson')
X_train_final = power_transformer.fit_transform(X_train_scaled)
X_test_final = power_transformer.transform(X_test_scaled)

# Build model - Simple but effective ensemble
print("\n🏗️ Building model...")

# Base estimators (compatible with sklearn 1.3.2)
estimators = [
    ('rf', RandomForestClassifier(
        n_estimators=150,
        max_depth=6,
        min_samples_split=10,
        min_samples_leaf=4,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )),
    ('gb', GradientBoostingClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        min_samples_split=10,
        subsample=0.8,
        random_state=42
    )),
    ('lr', LogisticRegression(
        C=0.5,
        max_iter=1000,
        class_weight='balanced',
        random_state=42
    ))
]

# Meta-learner
meta_learner = LogisticRegression(C=0.1, max_iter=1000, random_state=42)

# Stacking classifier
model = StackingClassifier(
    estimators=estimators,
    final_estimator=meta_learner,
    cv=5,
    stack_method='predict_proba',
    n_jobs=-1
)

# Train
print("\n🎯 Training model...")
model.fit(X_train_final, y_train_balanced)

# Evaluate
print("\n📊 Evaluating...")
y_pred = model.predict(X_test_final)
y_pred_proba = model.predict_proba(X_test_final)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
print(f"\n   ✅ Test Accuracy: {accuracy * 100:.2f}%")

# Cross-validation
cv_scores = cross_val_score(model, X_train_final, y_train_balanced, cv=5)
print(f"   ✅ CV Accuracy: {cv_scores.mean() * 100:.2f}% (+/- {cv_scores.std() * 100:.2f}%)")

# Classification report
print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred, target_names=['No Diabetes', 'Diabetes']))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(f"\n📊 Confusion Matrix:")
print(f"   TN: {cm[0][0]}, FP: {cm[0][1]}")
print(f"   FN: {cm[1][0]}, TP: {cm[1][1]}")

# Save artifacts
print("\n💾 Saving model artifacts...")

# Save model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("   ✓ model.pkl")

# Save scaler (combined with power transformer)
with open('scaler.pkl', 'wb') as f:
    pickle.dump({
        'scaler': scaler,
        'power_transformer': power_transformer
    }, f)
print("   ✓ scaler.pkl")

# Save imputer
with open('imputer.pkl', 'wb') as f:
    pickle.dump(imputer, f)
print("   ✓ imputer.pkl")

# Save features
with open('features.pkl', 'wb') as f:
    pickle.dump(feature_names, f)
print("   ✓ features.pkl")

# Save metrics
metrics = {
    'accuracy': round(accuracy * 100, 2),
    'cv_accuracy_mean': round(cv_scores.mean() * 100, 2),
    'cv_accuracy_std': round(cv_scores.std() * 100, 2),
    'confusion_matrix': cm.tolist(),
    'sklearn_version': '1.3.2',
    'model_type': 'StackingClassifier',
    'compatible': True
}

with open('metrics.pkl', 'wb') as f:
    pickle.dump(metrics, f)
print("   ✓ metrics.pkl")

print("\n" + "=" * 70)
print("  ✅ MODEL RETRAINED SUCCESSFULLY!")
print(f"  Accuracy: {accuracy * 100:.2f}%")
print("=" * 70)
