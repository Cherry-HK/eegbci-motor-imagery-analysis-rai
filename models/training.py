import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GroupKFold, GridSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve
)

# ==========================================================
# 1. LOAD DATA
# ==========================================================
DATA_DIR = "experiments/version5"

X_train = np.load(os.path.join(DATA_DIR, "X_train_csp.npy"))
X_test = np.load(os.path.join(DATA_DIR, "X_test_csp.npy"))
y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))
train_subjects = np.load(os.path.join(DATA_DIR, "train_subjects.npy"))
test_subjects = np.load(os.path.join(DATA_DIR, "test_subjects.npy"))

print("="*70)
print("EEG MOTOR IMAGERY CLASSIFICATION (LEFT vs RIGHT)")
print("="*70)
print(f"Train shape: {X_train.shape}")
print(f"Test shape: {X_test.shape}")
print(f"Train subjects: {len(np.unique(train_subjects))}")
print(f"Test subjects: {len(np.unique(test_subjects))}")

print("\nClass distribution:")
print(f"Train → Left={np.sum(y_train==0)}, Right={np.sum(y_train==1)}")
print(f"Test  → Left={np.sum(y_test==0)}, Right={np.sum(y_test==1)}")

# ==========================================================
# 2. BASELINE MODEL (PIPELINE)
# ==========================================================
print("\n" + "="*70)
print("BASELINE SVM (PIPELINE)")
print("="*70)

baseline_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(
        kernel='rbf',
        C=1,
        gamma='scale',
        class_weight='balanced',
        probability=True,
        random_state=42
    ))
])

baseline_pipeline.fit(X_train, y_train)
y_pred_baseline = baseline_pipeline.predict(X_test)
y_proba_baseline = baseline_pipeline.predict_proba(X_test)[:, 1]

print(f"Baseline Accuracy: {accuracy_score(y_test, y_pred_baseline)*100:.2f}%")

# ==========================================================
# 3. SUBJECT-AWARE CROSS VALIDATION
# ==========================================================
print("\n" + "="*70)
print("SUBJECT-AWARE CROSS VALIDATION")
print("="*70)

gkf = GroupKFold(n_splits=5)

cv_scores = cross_val_score(
    baseline_pipeline,
    X_train,
    y_train,
    groups=train_subjects,
    cv=gkf,
    scoring='balanced_accuracy',
    n_jobs=-1
)

print("CV Balanced Accuracy Scores:")
print([f"{s:.3f}" for s in cv_scores])
print(f"Mean CV Score: {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")

# ==========================================================
# 4. HYPERPARAMETER TUNING (PIPELINE SAFE)
# ==========================================================
print("\n" + "="*70)
print("GRID SEARCH (SUBJECT-INDEPENDENT)")
print("="*70)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(probability=True, random_state=42))
])

param_grid = {
    'svm__C': [0.01, 0.1, 1, 10, 50, 100, 500],
    'svm__gamma': ['scale', 1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
    'svm__class_weight': ['balanced']
}

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=gkf,
    scoring='balanced_accuracy',
    n_jobs=-1,
    verbose=2
)

grid_search.fit(X_train, y_train, groups=train_subjects)

print("\nBest Parameters:", grid_search.best_params_)
print(f"Best CV Balanced Accuracy: {grid_search.best_score_*100:.2f}%")

best_model = grid_search.best_estimator_

print("\n=== MODEL CHECK ===")
print("Baseline model:")
print(baseline_pipeline.named_steps['svm'])

print("\nTuned model:")
print(best_model.named_steps['svm'])

# ==========================================================
# 5. FINAL TEST EVALUATION
# ==========================================================
print("\n" + "="*70)
print("FINAL TEST EVALUATION")
print("="*70)

y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
balanced_acc = balanced_accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

# Sensitivity & specificity (biomedical standard)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)

print(f"Accuracy:           {accuracy*100:.2f}%")
print(f"Balanced Accuracy:  {balanced_acc*100:.2f}%")
print(f"F1 Score:           {f1:.3f}")
print(f"ROC-AUC:            {roc_auc:.3f}")
print(f"Sensitivity:        {sensitivity:.3f}")
print(f"Specificity:        {specificity:.3f}")

print("\nClassification Report:")
print(classification_report(
    y_test, y_pred,
    target_names=["Left Hand", "Right Hand"],
    digits=3
))

# ==========================================================
# 6. VISUALIZATION
# ==========================================================
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Left','Right'],
            yticklabels=['Left','Right'])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")

plt.subplot(1,2,2)
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
plt.plot([0,1],[0,1],'--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(DATA_DIR, "svm_evaluation.png"), dpi=300)
plt.show()

print("Saved plot: svm_evaluation.png")

# ==========================================================
# 7. RESULTS TABLE (FOR PAPER / REPORT)
# ==========================================================
results_df = pd.DataFrame({
    "Model": ["Baseline SVM", "Tuned SVM"],
    "Accuracy (%)": [
        accuracy_score(y_test, y_pred_baseline)*100,
        accuracy*100
    ],
    "Balanced Acc (%)": [
        balanced_accuracy_score(y_test, y_pred_baseline)*100,
        balanced_acc*100
    ],
    "F1 Score": [
        f1_score(y_test, y_pred_baseline),
        f1
    ],
    "ROC-AUC": [
        roc_auc_score(y_test, y_proba_baseline),
        roc_auc
    ]
})

print("\nResults Comparison:")
print(results_df)

results_df.to_csv(os.path.join(DATA_DIR, "svm_results.csv"), index=False)
print("Saved: svm_results.csv")

# ==========================================================
# 8. SAVE FINAL MODEL
# ==========================================================
joblib.dump(best_model, os.path.join(DATA_DIR, "svm_motor_imagery_pipeline.pkl"))
print("Saved model: svm_motor_imagery_pipeline.pkl")

print("\n" + "="*70)
print("PIPELINE COMPLETE")
print("="*70)