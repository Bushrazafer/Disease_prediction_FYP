"""
MediCare AI - Medical-Grade Training Utilities
Addresses audit issues D1-D5 for all disease prediction models

D1: Data Leakage Prevention - SMOTE only after train/test split
D2: Realistic Accuracy Targets - 85-92% for screening models
D3: Stratified K-Fold - Repeated stratified K-fold for evaluation
D4: Calibration Curves - Calibration plotting and Brier score
D5: Threshold Optimization - Sensitivity/specificity optimization
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, RepeatedStratifiedKFold, cross_val_score
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, brier_score_loss, roc_curve
)
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
import warnings
warnings.filterwarnings('ignore')


# Medical-grade accuracy targets (D2)
ACCURACY_TARGETS = {
    'screening': {'min': 0.80, 'max': 0.88, 'target': 0.85},
    'standard': {'min': 0.85, 'max': 0.92, 'target': 0.88},
    'confirmation': {'min': 0.88, 'max': 0.95, 'target': 0.92}
}


class MedicalModelTrainer:
    """
    Medical-grade model training with audit compliance.
    Addresses D1-D5 issues from the audit document.
    """
    
    def __init__(self, model_type='screening', random_state=42):
        """
        Initialize trainer with model type.
        
        Args:
            model_type: 'screening', 'standard', or 'confirmation'
            random_state: Random seed for reproducibility
        """
        self.model_type = model_type
        self.random_state = random_state
        self.targets = ACCURACY_TARGETS.get(model_type, ACCURACY_TARGETS['screening'])
        self.optimal_threshold = 0.5
        self.calibration_info = {}
        
    def safe_train_test_split(self, X, y, test_size=0.2):
        """
        D1 FIX: Proper train/test split BEFORE any resampling.
        
        This prevents data leakage by ensuring test data is never seen
        during SMOTE or any other preprocessing.
        """
        print("\n" + "=" * 60)
        print("📊 SAFE TRAIN/TEST SPLIT (D1 Fix)")
        print("=" * 60)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=self.random_state, 
            stratify=y
        )
        
        print(f"   Training set: {len(X_train)} samples")
        print(f"   Test set:     {len(X_test)} samples (HELD OUT)")
        print(f"   Class distribution (train):")
        print(f"      - Class 0: {(y_train == 0).sum()} ({(y_train == 0).sum()/len(y_train)*100:.1f}%)")
        print(f"      - Class 1: {(y_train == 1).sum()} ({(y_train == 1).sum()/len(y_train)*100:.1f}%)")
        
        return X_train, X_test, y_train, y_test
    
    def apply_smote_safely(self, X_train, y_train, sampling_strategy='auto'):
        """
        D1 FIX: Apply SMOTE only to training data.
        
        CRITICAL: This must be called AFTER train_test_split, never before.
        """
        print("\n" + "=" * 60)
        print("⚖️ SAFE SMOTE APPLICATION (D1 Fix)")
        print("=" * 60)
        
        print(f"   Before SMOTE:")
        print(f"      - Class 0: {(y_train == 0).sum()}")
        print(f"      - Class 1: {(y_train == 1).sum()}")
        
        # Use BorderlineSMOTE for better generalization
        smote = BorderlineSMOTE(
            sampling_strategy=sampling_strategy,
            k_neighbors=5,
            m_neighbors=10,
            random_state=self.random_state
        )
        
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        
        print(f"   After SMOTE:")
        print(f"      - Class 0: {(y_resampled == 0).sum()}")
        print(f"      - Class 1: {(y_resampled == 1).sum()}")
        print(f"   ⚠️ SMOTE applied ONLY to training data (test data untouched)")
        
        return X_resampled, y_resampled
    
    def evaluate_with_repeated_kfold(self, model, X, y, n_splits=5, n_repeats=3):
        """
        D3 FIX: Use Repeated Stratified K-Fold for robust evaluation.
        
        This provides more reliable accuracy estimates than single K-fold.
        """
        print("\n" + "=" * 60)
        print("🔄 REPEATED STRATIFIED K-FOLD EVALUATION (D3 Fix)")
        print("=" * 60)
        
        rskf = RepeatedStratifiedKFold(
            n_splits=n_splits, 
            n_repeats=n_repeats, 
            random_state=self.random_state
        )
        
        # Collect metrics across all folds
        accuracies = []
        precisions = []
        recalls = []
        f1_scores = []
        aucs = []
        
        fold_num = 0
        for train_idx, val_idx in rskf.split(X, y):
            fold_num += 1
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]
            
            # Apply SMOTE to training fold only (D1 compliance)
            smote = BorderlineSMOTE(random_state=self.random_state)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train_fold, y_train_fold)
            
            # Train and evaluate
            model.fit(X_train_resampled, y_train_resampled)
            y_pred = model.predict(X_val_fold)
            y_proba = model.predict_proba(X_val_fold)[:, 1] if hasattr(model, 'predict_proba') else y_pred
            
            accuracies.append(accuracy_score(y_val_fold, y_pred))
            precisions.append(precision_score(y_val_fold, y_pred, zero_division=0))
            recalls.append(recall_score(y_val_fold, y_pred, zero_division=0))
            f1_scores.append(f1_score(y_val_fold, y_pred, zero_division=0))
            aucs.append(roc_auc_score(y_val_fold, y_proba))
        
        results = {
            'accuracy': {'mean': np.mean(accuracies), 'std': np.std(accuracies)},
            'precision': {'mean': np.mean(precisions), 'std': np.std(precisions)},
            'recall': {'mean': np.mean(recalls), 'std': np.std(recalls)},
            'f1_score': {'mean': np.mean(f1_scores), 'std': np.std(f1_scores)},
            'auc_roc': {'mean': np.mean(aucs), 'std': np.std(aucs)}
        }
        
        print(f"   {n_splits}-Fold x {n_repeats} Repeats = {n_splits * n_repeats} total evaluations")
        print(f"\n   📊 Cross-Validation Results:")
        print(f"   ┌────────────────────────────────────────────┐")
        print(f"   │  Accuracy:  {results['accuracy']['mean']*100:5.2f}% (+/- {results['accuracy']['std']*100:4.2f}%)  │")
        print(f"   │  Precision: {results['precision']['mean']*100:5.2f}% (+/- {results['precision']['std']*100:4.2f}%)  │")
        print(f"   │  Recall:    {results['recall']['mean']*100:5.2f}% (+/- {results['recall']['std']*100:4.2f}%)  │")
        print(f"   │  F1-Score:  {results['f1_score']['mean']*100:5.2f}% (+/- {results['f1_score']['std']*100:4.2f}%)  │")
        print(f"   │  AUC-ROC:   {results['auc_roc']['mean']:6.4f} (+/- {results['auc_roc']['std']:5.4f})  │")
        print(f"   └────────────────────────────────────────────┘")
        
        return results
    
    def compute_calibration_metrics(self, y_true, y_proba, n_bins=10):
        """
        D4 FIX: Compute calibration curve and Brier score.
        
        Well-calibrated probabilities are essential for medical predictions.
        """
        print("\n" + "=" * 60)
        print("📈 CALIBRATION ANALYSIS (D4 Fix)")
        print("=" * 60)
        
        # Brier score (lower is better, 0 is perfect)
        brier = brier_score_loss(y_true, y_proba)
        
        # Calibration curve
        prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=n_bins, strategy='uniform')
        
        # Expected Calibration Error (ECE)
        bin_counts = np.histogram(y_proba, bins=n_bins, range=(0, 1))[0]
        ece = np.sum(np.abs(prob_true - prob_pred) * (bin_counts / len(y_proba)))
        
        self.calibration_info = {
            'brier_score': brier,
            'ece': ece,
            'prob_true': prob_true,
            'prob_pred': prob_pred
        }
        
        print(f"   Brier Score:                  {brier:.4f} (lower is better)")
        print(f"   Expected Calibration Error:   {ece:.4f} (lower is better)")
        
        # Interpretation
        if brier < 0.1:
            print(f"   ✅ Excellent calibration")
        elif brier < 0.2:
            print(f"   ⚠️ Good calibration, consider Platt scaling")
        else:
            print(f"   ❌ Poor calibration, Platt scaling recommended")
        
        return self.calibration_info
    
    def calibrate_model(self, model, X_train, y_train, method='sigmoid'):
        """
        D4 FIX: Apply probability calibration (Platt scaling or isotonic).
        
        Returns a calibrated classifier wrapper.
        """
        print(f"\n   Applying {method} calibration...")
        
        calibrated = CalibratedClassifierCV(
            model, 
            method=method, 
            cv=5
        )
        calibrated.fit(X_train, y_train)
        
        print(f"   ✅ Model calibrated using {method} method")
        return calibrated
    
    def optimize_threshold(self, y_true, y_proba, optimize_for='sensitivity'):
        """
        D5 FIX: Optimize classification threshold for medical use case.
        
        For medical screening, we typically want high sensitivity (catch all positives)
        even at the cost of some specificity (more false positives).
        
        Args:
            optimize_for: 'sensitivity', 'specificity', 'f1', or 'youden'
        """
        print("\n" + "=" * 60)
        print("🎯 THRESHOLD OPTIMIZATION (D5 Fix)")
        print("=" * 60)
        
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        
        # Calculate metrics at each threshold
        sensitivities = tpr  # True Positive Rate = Sensitivity
        specificities = 1 - fpr  # True Negative Rate = Specificity
        
        if optimize_for == 'sensitivity':
            # Find threshold that gives at least 90% sensitivity
            target_sensitivity = 0.90
            valid_idx = np.where(sensitivities >= target_sensitivity)[0]
            if len(valid_idx) > 0:
                # Among valid thresholds, pick one with best specificity
                best_idx = valid_idx[np.argmax(specificities[valid_idx])]
            else:
                best_idx = np.argmax(sensitivities)
            
        elif optimize_for == 'specificity':
            # Find threshold that gives at least 85% specificity
            target_specificity = 0.85
            valid_idx = np.where(specificities >= target_specificity)[0]
            if len(valid_idx) > 0:
                best_idx = valid_idx[np.argmax(sensitivities[valid_idx])]
            else:
                best_idx = np.argmax(specificities)
                
        elif optimize_for == 'youden':
            # Youden's J statistic: maximize (sensitivity + specificity - 1)
            j_scores = sensitivities + specificities - 1
            best_idx = np.argmax(j_scores)
            
        else:  # f1
            # Calculate F1 at each threshold
            f1_scores = []
            for thresh in thresholds:
                y_pred = (y_proba >= thresh).astype(int)
                f1_scores.append(f1_score(y_true, y_pred, zero_division=0))
            best_idx = np.argmax(f1_scores)
        
        self.optimal_threshold = thresholds[best_idx]
        
        print(f"   Optimization target: {optimize_for}")
        print(f"   Optimal threshold:   {self.optimal_threshold:.4f}")
        print(f"   At this threshold:")
        print(f"      - Sensitivity: {sensitivities[best_idx]*100:.1f}%")
        print(f"      - Specificity: {specificities[best_idx]*100:.1f}%")
        
        # Medical recommendation
        if optimize_for == 'sensitivity':
            print(f"\n   💡 High sensitivity prioritized for screening")
            print(f"      (Minimizes missed diagnoses, may have more false positives)")
        
        return self.optimal_threshold
    
    def check_accuracy_target(self, accuracy):
        """
        D2 FIX: Check if accuracy is within realistic medical targets.
        
        Flags suspiciously high accuracy that may indicate overfitting.
        """
        print("\n" + "=" * 60)
        print("🎯 ACCURACY TARGET CHECK (D2 Fix)")
        print("=" * 60)
        
        print(f"   Model type:     {self.model_type}")
        print(f"   Target range:   {self.targets['min']*100:.0f}% - {self.targets['max']*100:.0f}%")
        print(f"   Actual:         {accuracy*100:.2f}%")
        
        if accuracy > self.targets['max']:
            print(f"\n   ⚠️ WARNING: Accuracy exceeds realistic target!")
            print(f"      This may indicate:")
            print(f"      - Data leakage (check SMOTE application)")
            print(f"      - Overfitting (check regularization)")
            print(f"      - Test data contamination")
            return False
        elif accuracy < self.targets['min']:
            print(f"\n   ⚠️ WARNING: Accuracy below minimum target")
            print(f"      Consider:")
            print(f"      - More feature engineering")
            print(f"      - Hyperparameter tuning")
            print(f"      - More training data")
            return False
        else:
            print(f"\n   ✅ Accuracy within realistic medical target range")
            return True
    
    def full_evaluation(self, model, X_test, y_test, X_train=None, y_train=None):
        """
        Complete evaluation with all D1-D5 fixes applied.
        """
        print("\n" + "=" * 70)
        print("📊 COMPREHENSIVE MODEL EVALUATION")
        print("=" * 70)
        
        # Get predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
        
        # Basic metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_proba)
        
        print(f"\n   📊 Test Set Metrics:")
        print(f"   ┌────────────────────────────────────────┐")
        print(f"   │  Accuracy:     {accuracy*100:6.2f}%               │")
        print(f"   │  Precision:    {precision*100:6.2f}%               │")
        print(f"   │  Recall:       {recall*100:6.2f}%               │")
        print(f"   │  F1-Score:     {f1*100:6.2f}%               │")
        print(f"   │  AUC-ROC:      {auc:6.4f}                │")
        print(f"   └────────────────────────────────────────┘")
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        print(f"\n   📊 Confusion Matrix:")
        print(f"   ┌─────────────────┬─────────────────┐")
        print(f"   │   TN: {tn:5d}     │   FP: {fp:5d}     │")
        print(f"   ├─────────────────┼─────────────────┤")
        print(f"   │   FN: {fn:5d}     │   TP: {tp:5d}     │")
        print(f"   └─────────────────┴─────────────────┘")
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        print(f"\n   Sensitivity (Recall): {sensitivity*100:.2f}%")
        print(f"   Specificity:          {specificity*100:.2f}%")
        
        # D2: Check accuracy target
        self.check_accuracy_target(accuracy)
        
        # D4: Calibration analysis
        self.compute_calibration_metrics(y_test, y_proba)
        
        # D5: Threshold optimization
        self.optimize_threshold(y_test, y_proba, optimize_for='sensitivity')
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_roc': auc,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'brier_score': self.calibration_info.get('brier_score'),
            'optimal_threshold': self.optimal_threshold,
            'confusion_matrix': cm.tolist()
        }


def print_audit_compliance_summary():
    """Print summary of audit compliance fixes."""
    print("\n" + "=" * 70)
    print("✅ AUDIT COMPLIANCE SUMMARY")
    print("=" * 70)
    print("""
    D1: Data Leakage Prevention
        ✓ SMOTE applied ONLY after train/test split
        ✓ Test data never seen during resampling
        
    D2: Realistic Accuracy Targets
        ✓ Screening models: 80-88% accuracy
        ✓ Standard models: 85-92% accuracy
        ✓ Confirmation models: 88-95% accuracy
        ✓ Flags suspiciously high accuracy
        
    D3: Stratified K-Fold Evaluation
        ✓ Repeated Stratified K-Fold (5x3 = 15 evaluations)
        ✓ SMOTE applied within each fold
        ✓ Reports mean ± std for all metrics
        
    D4: Calibration Analysis
        ✓ Brier score computation
        ✓ Expected Calibration Error (ECE)
        ✓ Platt scaling / isotonic calibration
        
    D5: Threshold Optimization
        ✓ Sensitivity-optimized threshold for screening
        ✓ Youden's J statistic option
        ✓ Reports sensitivity/specificity at optimal threshold
    """)
    print("=" * 70)
