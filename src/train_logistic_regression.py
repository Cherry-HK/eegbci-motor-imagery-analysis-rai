"""
Logistic Regression Model for Motor Imagery EEG Classification

This script trains a Logistic Regression classifier on CSP features
for binary motor imagery classification (left vs right hand).

Dataset: EEG Motor Movement/Imagery Dataset (PhysioNet)
Features: CSP (Common Spatial Patterns)
Task: Left hand vs Right hand imagery
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, roc_auc_score
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
import pickle
import os

print("="*70)
print("Motor Imagery BCI - Logistic Regression Classifier")
print("="*70)

# ============================================================================
# 1. Load Preprocessed CSP Features
# ============================================================================
print("\n[Step 1/6] Loading preprocessed CSP features...")

try:
    X_train = np.load('X_train_csp.npy')
    X_test = np.load('X_test_csp.npy')
    y_train = np.load('y_train.npy')
    y_test = np.load('y_test.npy')

    print(f"   Training data: {X_train.shape}")
    print(f"   Test data: {X_test.shape}")
    print(f"   Training labels: {y_train.shape}")
    print(f"   Test labels: {y_test.shape}")

except FileNotFoundError:
    print("\n   ERROR: Preprocessed data files not found!")
    print("  Please run preprocessing first:")
    print("    python src/preprocessing.py")
    exit(1)

# ============================================================================
# 2. Data Distribution Analysis
# ============================================================================
print("\n[Step 2/6] Analyzing data distribution...")

train_left = np.sum(y_train == 0)
train_right = np.sum(y_train == 1)
test_left = np.sum(y_test == 0)
test_right = np.sum(y_test == 1)

print(f"\n  Training set: {len(y_train)} samples")
print(f"    Left hand (0):  {train_left} ({train_left/len(y_train)*100:.1f}%)")
print(f"    Right hand (1): {train_right} ({train_right/len(y_train)*100:.1f}%)")

print(f"\n  Test set: {len(y_test)} samples")
print(f"    Left hand (0):  {test_left} ({test_left/len(y_test)*100:.1f}%)")
print(f"    Right hand (1): {test_right} ({test_right/len(y_test)*100:.1f}%)")

# ============================================================================
# 3. Train Logistic Regression Model
# ============================================================================
print("\n[Step 3/6] Training Logistic Regression classifier...")

# Initialize model
# Parameters optimized for BCI applications
model = LogisticRegression(
    penalty='l2',              # L2 regularization (Ridge)
    C=1.0,                     # Regularization strength
    solver='lbfgs',            # Optimization algorithm
    max_iter=1000,             # Maximum iterations
    random_state=42,           # For reproducibility
    class_weight='balanced'    # Handle any class imbalance
)

# Train the model
model.fit(X_train, y_train)

print(f"   Model trained successfully")
print(f"   Converged in {model.n_iter_[0]} iterations")

# ============================================================================
# 4. Cross-Validation
# ============================================================================
print("\n[Step 4/6] Performing 5-fold cross-validation...")

# Stratified K-Fold to preserve class distribution
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')

print(f"  CV Scores: {[f'{s*100:.1f}%' for s in cv_scores]}")
print(f"  Mean: {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")

# ============================================================================
# 5. Evaluation on Test Set
# ============================================================================
print("\n[Step 5/6] Evaluating on test set...")

# Predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
y_test_proba = model.predict_proba(X_test)[:, 1]

# Metrics
train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)
precision = precision_score(y_test, y_test_pred)
recall = recall_score(y_test, y_test_pred)
f1 = f1_score(y_test, y_test_pred)
roc_auc = roc_auc_score(y_test, y_test_proba)

print(f"\n  Training Accuracy: {train_acc*100:.2f}%")
print(f"  Test Accuracy:     {test_acc*100:.2f}%")
print(f"  Precision:         {precision*100:.2f}%")
print(f"  Recall:            {recall*100:.2f}%")
print(f"  F1-Score:          {f1*100:.2f}%")
print(f"  ROC AUC:           {roc_auc:.3f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_test_pred)
print(f"\n  Confusion Matrix:")
print(f"                 Predicted")
print(f"                 Left  Right")
print(f"    Actual Left   {cm[0,0]:3d}    {cm[0,1]:3d}")
print(f"          Right   {cm[1,0]:3d}    {cm[1,1]:3d}")

# Detailed Report
print(f"\n  Classification Report:")
print(classification_report(y_test, y_test_pred,
                          target_names=['Left Hand', 'Right Hand']))

# ============================================================================
# 6. Save Model and Results
# ============================================================================
print("\n[Step 6/6] Saving model and results...")

# Create models directory
os.makedirs('models', exist_ok=True)

# Save model
model_path = 'models/logistic_regression_model.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(model, f)
print(f"Model saved: {model_path}")

# Save results
results = {
    'train_accuracy': train_acc,
    'test_accuracy': test_acc,
    'cv_mean': cv_scores.mean(),
    'cv_std': cv_scores.std(),
    'precision': precision,
    'recall': recall,
    'f1_score': f1,
    'roc_auc': roc_auc,
    'confusion_matrix': cm
}

results_path = 'models/logistic_regression_results.pkl'
with open(results_path, 'wb') as f:
    pickle.dump(results, f)
print(f"   Results saved: {results_path}")

# ============================================================================
# 7. Visualizations
# ============================================================================
print("\n Creating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. Confusion Matrix
ax = axes[0, 0]
im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
ax.figure.colorbar(im, ax=ax)
ax.set(xticks=[0, 1], yticks=[0, 1],
       xticklabels=['Left', 'Right'], yticklabels=['Left', 'Right'],
       xlabel='Predicted', ylabel='Actual',
       title='Confusion Matrix')

# Add text annotations
for i in range(2):
    for j in range(2):
        text = ax.text(j, i, cm[i, j], ha="center", va="center",
                      color="white" if cm[i, j] > cm.max()/2 else "black",
                      fontsize=20, fontweight='bold')

# 2. ROC Curve
ax = axes[0, 1]
fpr, tpr, _ = roc_curve(y_test, y_test_proba)
ax.plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {roc_auc:.3f})')
ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Chance')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve')
ax.legend()
ax.grid(alpha=0.3)

# 3. Feature Coefficients
ax = axes[1, 0]
features = [f'CSP{i+1}' for i in range(X_train.shape[1])]
coeffs = model.coef_[0]
colors = ['red' if c < 0 else 'blue' for c in coeffs]
ax.barh(features, np.abs(coeffs), color=colors, alpha=0.7)
ax.set_xlabel('Absolute Coefficient Value')
ax.set_title('Feature Importance (LR Coefficients)')
ax.grid(axis='x', alpha=0.3)

# 4. Cross-Validation Scores
ax = axes[1, 1]
folds = range(1, 6)
ax.bar(folds, cv_scores, color='steelblue', alpha=0.7)
ax.axhline(cv_scores.mean(), color='red', linestyle='--', linewidth=2,
          label=f'Mean: {cv_scores.mean()*100:.2f}%')
ax.set_xlabel('Fold')
ax.set_ylabel('Accuracy')
ax.set_title('Cross-Validation Scores')
ax.set_ylim([0, 1])
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
fig_path = 'models/logistic_regression_visualization.png'
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
print(f"   Visualization saved: {fig_path}")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*70)
print("TRAINING SUMMARY")
print("="*70)
print(f"Model:            Logistic Regression")
print(f"Features:         {X_train.shape[1]} CSP components")
print(f"Training samples: {len(y_train)}")
print(f"Test samples:     {len(y_test)}")
print(f"CV Accuracy:      {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")
print(f"Test Accuracy:    {test_acc*100:.2f}%")
print(f"Test F1-Score:    {f1*100:.2f}%")
print(f"ROC AUC:          {roc_auc:.3f}")
print("="*70)

print("\n Training complete!")
print("\nTo use the trained model:")
print("  import pickle")
print("  with open('models/logistic_regression_model.pkl', 'rb') as f:")
print("      model = pickle.load(f)")
print("  predictions = model.predict(new_csp_features)")

plt.show()
