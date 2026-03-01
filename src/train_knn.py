"""
KNN Model for Motor Imagery EEG Classification

This script trains K-Nearest Neighbors classifiers on CSP and raw EEG features
using four approaches (CSP-only, Raw-only, Combined, Combined+PCA) and
selects the best via GridSearchCV.

Dataset: EEG Motor Movement/Imagery Dataset (PhysioNet)
Features: CSP (Common Spatial Patterns) + Raw EEG
Task: Left hand vs Right hand imagery
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

print("=" * 70)
print("Motor Imagery BCI - KNN Classifier")
print("=" * 70)

# ============================================================================
# 1. Load Preprocessed Features
# ============================================================================
print("\n[Step 1/7] Loading preprocessed features...")

try:
    X_train_csp = np.load('X_train_csp.npy')
    X_test_csp  = np.load('X_test_csp.npy')
    X_train_raw = np.load('X_train.npy')
    X_test_raw  = np.load('X_test.npy')
    y_train     = np.load('y_train.npy')
    y_test      = np.load('y_test.npy')

    print(f"   CSP features  - X_train: {X_train_csp.shape} | X_test: {X_test_csp.shape}")
    print(f"   Raw features  - X_train: {X_train_raw.shape} | X_test: {X_test_raw.shape}")
    print(f"   Labels        - Train: {np.unique(y_train, return_counts=True)}")
    print(f"                   Test:  {np.unique(y_test, return_counts=True)}")

except FileNotFoundError:
    print("\n   ERROR: Preprocessed data files not found!")
    print("   Please run preprocessing first:")
    print("     python src/preprocessing.py")
    exit(1)

# ============================================================================
# 2. Feature Preparation
# ============================================================================
print("\n[Step 2/7] Preparing features...")

if X_train_raw.ndim == 3:
    print(f"   Raw features are 3D {X_train_raw.shape} -> flattening...")
    X_train_raw = X_train_raw.reshape(X_train_raw.shape[0], -1)
    X_test_raw  = X_test_raw.reshape(X_test_raw.shape[0], -1)
    print(f"   After flatten: {X_train_raw.shape}")

X_train_combined = np.hstack([X_train_csp, X_train_raw])
X_test_combined  = np.hstack([X_test_csp,  X_test_raw])
print(f"   Combined features: {X_train_combined.shape}")

# ============================================================================
# 3. Scale Features
# ============================================================================
print("\n[Step 3/7] Scaling features...")

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train_combined)
X_test_s  = scaler.transform(X_test_combined)
print("   Features scaled.")

# ============================================================================
# 4. Train & Compare Four Approaches
# ============================================================================
print("\n[Step 4/7] Training KNN with four feature approaches...")

results = {}

param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11, 15, 21],
    'weights':     ['uniform', 'distance'],
    'metric':      ['euclidean', 'manhattan']
}

# --- Approach A: CSP only ---
print("\n   Approach A: CSP features only...")
scaler_a = StandardScaler()
Xtr_a = scaler_a.fit_transform(X_train_csp)
Xte_a = scaler_a.transform(X_test_csp)

gs_a = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
gs_a.fit(Xtr_a, y_train)
y_pred_a = gs_a.best_estimator_.predict(Xte_a)
results['CSP Only'] = accuracy_score(y_test, y_pred_a)
print(f"     Best params: {gs_a.best_params_}")
print(f"     Accuracy: {results['CSP Only']:.4f}")

# --- Approach B: Raw features only ---
print("\n   Approach B: Raw features only...")
scaler_b = StandardScaler()
Xtr_b = scaler_b.fit_transform(X_train_raw)
Xte_b = scaler_b.transform(X_test_raw)

gs_b = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
gs_b.fit(Xtr_b, y_train)
y_pred_b = gs_b.best_estimator_.predict(Xte_b)
results['Raw Only'] = accuracy_score(y_test, y_pred_b)
print(f"     Best params: {gs_b.best_params_}")
print(f"     Accuracy: {results['Raw Only']:.4f}")

# --- Approach C: Combined features ---
print("\n   Approach C: Combined CSP + Raw features...")
gs_c = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
gs_c.fit(X_train_s, y_train)
y_pred_c = gs_c.best_estimator_.predict(X_test_s)
results['Combined'] = accuracy_score(y_test, y_pred_c)
print(f"     Best params: {gs_c.best_params_}")
print(f"     Accuracy: {results['Combined']:.4f}")

# --- Approach D: Combined + PCA dimensionality reduction ---
print("\n   Approach D: Combined + PCA...")
pca = PCA(n_components=0.95)
Xtr_pca = pca.fit_transform(X_train_s)
Xte_pca = pca.transform(X_test_s)
print(f"     PCA reduced to {Xtr_pca.shape[1]} components")

gs_d = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
gs_d.fit(Xtr_pca, y_train)
y_pred_d = gs_d.best_estimator_.predict(Xte_pca)
results['Combined + PCA'] = accuracy_score(y_test, y_pred_d)
print(f"     Best params: {gs_d.best_params_}")
print(f"     Accuracy: {results['Combined + PCA']:.4f}")

# ============================================================================
# 5. Best Model Report
# ============================================================================
print("\n[Step 5/7] Evaluating best model...")

best_approach = max(results, key=results.get)
best_acc      = results[best_approach]

print(f"\n{'=' * 50}")
print(f"  Best Approach: {best_approach}")
print(f"  Best Accuracy: {best_acc:.4f} ({best_acc:.2%})")
print(f"{'=' * 50}")

best_preds = {
    'CSP Only': y_pred_a, 'Raw Only': y_pred_b,
    'Combined': y_pred_c, 'Combined + PCA': y_pred_d
}
y_pred_best = best_preds[best_approach]

best_estimators = {
    'CSP Only': gs_a.best_estimator_, 'Raw Only': gs_b.best_estimator_,
    'Combined': gs_c.best_estimator_, 'Combined + PCA': gs_d.best_estimator_
}
best_model = best_estimators[best_approach]

print("\n  Classification Report (Best Model):")
print(classification_report(y_test, y_pred_best,
      target_names=['Left Hand', 'Right Hand'], zero_division=0))

# ============================================================================
# 6. Save Model and Results
# ============================================================================
print("\n[Step 6/7] Saving model and results...")

os.makedirs('models', exist_ok=True)

model_path = 'models/knn_model.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(best_model, f)
print(f"   Model saved: {model_path}")

results_data = {
    'best_approach': best_approach,
    'best_accuracy': best_acc,
    'all_results': results,
    'confusion_matrix': confusion_matrix(y_test, y_pred_best),
}
results_path = 'models/knn_results.pkl'
with open(results_path, 'wb') as f:
    pickle.dump(results_data, f)
print(f"   Results saved: {results_path}")

# ============================================================================
# 7. Visualizations
# ============================================================================
print("\n[Step 7/7] Creating visualizations...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_best)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Left', 'Right'],
            yticklabels=['Left', 'Right'], ax=axes[0])
axes[0].set_xlabel('Predicted', fontsize=12)
axes[0].set_ylabel('True', fontsize=12)
axes[0].set_title(f'Best KNN Confusion Matrix ({best_approach})\nAccuracy: {best_acc:.2%}',
                  fontsize=13, fontweight='bold')

# Accuracy Comparison
bars = axes[1].bar(results.keys(), results.values(),
                   color=['#3498db', '#2ecc71', '#9b59b6', '#e67e22'])
axes[1].set_ylabel('Accuracy', fontsize=12)
axes[1].set_title('KNN Accuracy Comparison Across Feature Sets',
                  fontsize=13, fontweight='bold')
axes[1].set_ylim(0, 1.0)
axes[1].axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Baseline (50%)')
for bar, val in zip(bars, results.values()):
    axes[1].text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                 f'{val:.2%}', ha='center', fontsize=11, fontweight='bold')
axes[1].legend()

plt.tight_layout()
fig_path = 'models/knn_visualization.png'
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
print(f"   Visualization saved: {fig_path}")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 70)
print("TRAINING SUMMARY")
print("=" * 70)
print(f"Model:            KNN ({best_approach})")
print(f"Training samples: {len(y_train)}")
print(f"Test samples:     {len(y_test)}")
print(f"Test Accuracy:    {best_acc * 100:.2f}%")
for approach, acc in results.items():
    marker = " <-- best" if approach == best_approach else ""
    print(f"  {approach:20s}: {acc * 100:.2f}%{marker}")
print("=" * 70)

print("\nTraining complete!")

plt.show()
