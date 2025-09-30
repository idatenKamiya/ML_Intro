import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load the UCI Breast Cancer dataset
print("Loading UCI Breast Cancer Dataset...")
data = load_breast_cancer()
X, y = data.data, data.target

# Dataset info
print(f"Dataset shape: {X.shape}")
print(f"Class distribution:")
print(f"  Benign (class 1): {np.sum(y == 1)} ({np.mean(y == 1)*100:.1f}%)")
print(f"  Malignant (class 0): {np.sum(y == 0)} ({np.mean(y == 0)*100:.1f}%)")
print()

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("=" * 60)
print("PART (A): STANDARD LOGISTIC REGRESSION")
print("=" * 60)

# Train standard logistic regression
lr_standard = LogisticRegression(random_state=42, max_iter=1000)
lr_standard.fit(X_train_scaled, y_train)

# Predictions
y_train_pred_std = lr_standard.predict(X_train_scaled)
y_test_pred_std = lr_standard.predict(X_test_scaled)

# Prediction probabilities for log loss
y_train_proba_std = lr_standard.predict_proba(X_train_scaled)
y_test_proba_std = lr_standard.predict_proba(X_test_scaled)

# Calculate metrics
train_acc_std = accuracy_score(y_train, y_train_pred_std)
test_acc_std = accuracy_score(y_test, y_test_pred_std)
train_logloss_std = log_loss(y_train, y_train_proba_std)
test_logloss_std = log_loss(y_test, y_test_proba_std)

print(f"Training Accuracy: {train_acc_std:.4f}")
print(f"Test Accuracy: {test_acc_std:.4f}")
print(f"Training Log Loss: {train_logloss_std:.4f}")
print(f"Test Log Loss: {test_logloss_std:.4f}")
print()

print("Classification Report (Test Set):")
print(classification_report(y_test, y_test_pred_std, 
                          target_names=['Malignant', 'Benign']))

print("Confusion Matrix (Test Set):")
cm_std = confusion_matrix(y_test, y_test_pred_std)
print(cm_std)
print()

print("=" * 60)
print("PART (B): WEIGHTED LOGISTIC REGRESSION")
print("=" * 60)

# Train weighted logistic regression with class_weight={0:1, 1:20}
lr_weighted = LogisticRegression(
    class_weight={0: 1, 1: 20}, 
    random_state=42, 
    max_iter=1000
)
lr_weighted.fit(X_train_scaled, y_train)

# Predictions
y_train_pred_wt = lr_weighted.predict(X_train_scaled)
y_test_pred_wt = lr_weighted.predict(X_test_scaled)

# Prediction probabilities for log loss
y_train_proba_wt = lr_weighted.predict_proba(X_train_scaled)
y_test_proba_wt = lr_weighted.predict_proba(X_test_scaled)

# Calculate metrics
train_acc_wt = accuracy_score(y_train, y_train_pred_wt)
test_acc_wt = accuracy_score(y_test, y_test_pred_wt)
train_logloss_wt = log_loss(y_train, y_train_proba_wt)
test_logloss_wt = log_loss(y_test, y_test_proba_wt)

print(f"Training Accuracy: {train_acc_wt:.4f}")
print(f"Test Accuracy: {test_acc_wt:.4f}")
print(f"Training Log Loss: {train_logloss_wt:.4f}")
print(f"Test Log Loss: {test_logloss_wt:.4f}")
print()

print("Classification Report (Test Set):")
print(classification_report(y_test, y_test_pred_wt, 
                          target_names=['Malignant', 'Benign']))

print("Confusion Matrix (Test Set):")
cm_wt = confusion_matrix(y_test, y_test_pred_wt)
print(cm_wt)
print()

print("=" * 60)
print("PART (C): COMPARISON AND ANALYSIS")
print("=" * 60)

# Create comparison summary
comparison_df = pd.DataFrame({
    'Metric': ['Training Accuracy', 'Test Accuracy', 'Training Log Loss', 'Test Log Loss'],
    'Standard LR': [train_acc_std, test_acc_std, train_logloss_std, test_logloss_std],
    'Weighted LR': [train_acc_wt, test_acc_wt, train_logloss_wt, test_logloss_wt],
    'Difference': [train_acc_wt - train_acc_std, test_acc_wt - test_acc_std, 
                   train_logloss_wt - train_logloss_std, test_logloss_wt - test_logloss_std]
})

print("Performance Comparison:")
print(comparison_df.round(4))
print()

# Detailed classification metrics comparison
from sklearn.metrics import precision_score, recall_score, f1_score

def get_detailed_metrics(y_true, y_pred):
    """Calculate detailed metrics for each class"""
    precision_0 = precision_score(y_true, y_pred, pos_label=0)
    recall_0 = recall_score(y_true, y_pred, pos_label=0)
    f1_0 = f1_score(y_true, y_pred, pos_label=0)
    
    precision_1 = precision_score(y_true, y_pred, pos_label=1)
    recall_1 = recall_score(y_true, y_pred, pos_label=1)
    f1_1 = f1_score(y_true, y_pred, pos_label=1)
    
    return {
        'malignant_precision': precision_0,
        'malignant_recall': recall_0,
        'malignant_f1': f1_0,
        'benign_precision': precision_1,
        'benign_recall': recall_1,
        'benign_f1': f1_1
    }

metrics_std = get_detailed_metrics(y_test, y_test_pred_std)
metrics_wt = get_detailed_metrics(y_test, y_test_pred_wt)

detailed_comparison = pd.DataFrame({
    'Standard LR': metrics_std,
    'Weighted LR': metrics_wt
}).T

print("Detailed Metrics Comparison (Test Set):")
print(detailed_comparison.round(4))
print()

# Analysis of prediction changes
print("Analysis of Prediction Changes:")
print(f"Total test samples: {len(y_test)}")

# Count prediction changes
same_predictions = np.sum(y_test_pred_std == y_test_pred_wt)
different_predictions = len(y_test) - same_predictions

print(f"Samples with same predictions: {same_predictions}")
print(f"Samples with different predictions: {different_predictions}")
print()

# Analyze where predictions changed
if different_predictions > 0:
    changed_indices = np.where(y_test_pred_std != y_test_pred_wt)[0]
    print("Prediction changes:")
    
    for idx in changed_indices[:10]:  # Show first 10 changes
        true_label = "Malignant" if y_test[idx] == 0 else "Benign"
        std_pred = "Malignant" if y_test_pred_std[idx] == 0 else "Benign"
        wt_pred = "Malignant" if y_test_pred_wt[idx] == 0 else "Benign"
        
        print(f"  Sample {idx}: True={true_label}, Standard={std_pred}, Weighted={wt_pred}")
        print(f"    Probabilities - Standard: [{y_test_proba_std[idx][0]:.3f}, {y_test_proba_std[idx][1]:.3f}]")
        print(f"    Probabilities - Weighted: [{y_test_proba_wt[idx][0]:.3f}, {y_test_proba_wt[idx][1]:.3f}]")
        print()

# Cost analysis (assuming malignant misclassification is 20x more costly)
def calculate_cost(y_true, y_pred, fn_cost=20, fp_cost=1):
    """Calculate total cost based on confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    total_cost = fn * fn_cost + fp * fp_cost
    return total_cost, {'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp}

cost_std, breakdown_std = calculate_cost(y_test, y_test_pred_std)
cost_wt, breakdown_wt = calculate_cost(y_test, y_test_pred_wt)

print("Cost Analysis (FN cost = 20, FP cost = 1):")
print(f"Standard LR total cost: {cost_std}")
print(f"  Breakdown: {breakdown_std}")
print(f"Weighted LR total cost: {cost_wt}")
print(f"  Breakdown: {breakdown_wt}")
print(f"Cost reduction with weighting: {cost_std - cost_wt}")
print()

print("=" * 60)
print("KEY FINDINGS")
print("=" * 60)

print("1. ACCURACY CHANGES:")
print(f"   - Standard LR test accuracy: {test_acc_std:.4f}")
print(f"   - Weighted LR test accuracy: {test_acc_wt:.4f}")
print(f"   - Change: {test_acc_wt - test_acc_std:.4f}")

print("\n2. RECALL FOR MALIGNANT (Class 0) - Most Important:")
print(f"   - Standard LR recall: {metrics_std['malignant_recall']:.4f}")
print(f"   - Weighted LR recall: {metrics_wt['malignant_recall']:.4f}")
print(f"   - Improvement: {metrics_wt['malignant_recall'] - metrics_std['malignant_recall']:.4f}")

print("\n3. PRECISION FOR BENIGN (Class 1):")
print(f"   - Standard LR precision: {metrics_std['benign_precision']:.4f}")
print(f"   - Weighted LR precision: {metrics_wt['benign_precision']:.4f}")
print(f"   - Change: {metrics_wt['benign_precision'] - metrics_std['benign_precision']:.4f}")

print("\n4. BUSINESS IMPACT:")
print(f"   - Cost reduction: {cost_std - cost_wt} units")
print(f"   - Percentage cost reduction: {((cost_std - cost_wt) / cost_std * 100):.1f}%")

print("\n5. TRADE-OFFS:")
if test_acc_wt < test_acc_std:
    print("   - Lower overall accuracy but better at catching malignant cases")
else:
    print("   - Improved accuracy while better at catching malignant cases")
    
if metrics_wt['benign_precision'] < metrics_std['benign_precision']:
    print("   - More false positives (benign cases classified as malignant)")
    print("   - This trade-off is acceptable given the 20:1 cost ratio")