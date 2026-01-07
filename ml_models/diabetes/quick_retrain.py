"""
Quick Diabetes Model Retraining - No CV for speed
Compatible with scikit-learn 1.3.2
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, PowerTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE

import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("  QUICK DIABETES MODEL RETRAINING")
print("=" * 60)

# Get script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Load data
print("\n📊 Loading data...")
df = pd.read_csv('diabetes_expanded.csv')
print(f"   Loaded: {len(df)} records")

# Standardize column names
df.columns = [col.lower().replace(' ', '_') for col in df.columns]

# Separate features and target
X = df.drop('outcome', axis=1)
y = df['outcome']

# Handle missing values
zero_invalid = ['glucose', 'bloodpressure', 'skinthickness', 'insulin', 'bmi']
for col in zero_invalid:
    if col in X.columns:
        X[col] = X[col].replace(0, np.nan)

# Impute
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

# Power transform
power_transformer = PowerTransformer(method='yeo-johnson')
X_train_final = power_transformer.fit_transform(X_train_scaled)
X_test_final = power_transformer.transform(X_test_scaled)

# Simple but effective model - Random Forest only (fast)
print("\n🏗️ Training Random Forest...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=8,
    min_samples_split=10,
    min_samples_leaf=4,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
model.fit(X_train_final, y_train_bal)

# Evaluate
y_pred = model.predict(X_test_final)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n   ✅ Test Accuracy: {accuracy * 100:.2f}%")

print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred, target_names=['No Diabetes', 'Diabetes']))

# Save
print("\n💾 Saving...")
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("   ✓ model.pkl")

with open('scaler.pkl', 'wb') as f:
    pickle.dump({'scaler': scaler, 'power_transformer': power_transformer}, f)
print("   ✓ scaler.pkl")

with open('imputer.pkl', 'wb') as f:
    pickle.dump(imputer, f)
print("   ✓ imputer.pkl")

with open('features.pkl', 'wb') as f:
    pickle.dump(feature_names, f)
print("   ✓ features.pkl")

metrics = {
    'accuracy': round(accuracy * 100, 2),
    'sklearn_version': '1.3.2',
    'model_type': 'RandomForestClassifier'
}
with open('metrics.pkl', 'wb') as f:
    pickle.dump(metrics, f)
print("   ✓ metrics.pkl")

print("\n" + "=" * 60)
print(f"  ✅ DONE! Accuracy: {accuracy * 100:.2f}%")
print("=" * 60)
