import csv
import os

import matplotlib
import numpy as np
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline


class CovRegularizer(BaseEstimator, TransformerMixin):
    """Add small regularization to covariance matrices to ensure positive definiteness."""
    def __init__(self, reg=1e-7):
        self.reg = reg

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        n = X.shape[-1]
        return X + self.reg * np.eye(n)

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ==========================================================
# 1. LOAD PREPROCESSED DATA
# ==========================================================
BASE_DIR = "models/output"
DATA_DIR = os.path.join(BASE_DIR, "preprocessing_result")
RESULTS_DIR = os.path.join(BASE_DIR, "riemann_parameter_study")

X = np.load(os.path.join(DATA_DIR, "X.npy"))
y = np.load(os.path.join(DATA_DIR, "y.npy"))
subjects = np.load(os.path.join(DATA_DIR, "subjects.npy"))

os.makedirs(RESULTS_DIR, exist_ok=True)

print("Dataset shape:", X.shape)
print("Number of subjects:", len(np.unique(subjects)))
print("Results directory:", RESULTS_DIR)


def run_loso_riemann(
    X,
    y,
    subjects,
    *,
    cov_estimator="lwf",
    ts_metric="riemann",
    c_value=1.0,
    solver="lbfgs",
    class_weight="balanced",
    progress_label="",
):
    """Run LOSO with a Riemannian pipeline and return summary metrics."""
    logo = LeaveOneGroupOut()

    accuracies = []
    f1_scores = []
    auc_scores = []
    conf_matrices = []
    fold_rows = []
    all_y_true = []
    all_y_pred = []

    total_folds = len(np.unique(subjects))

    for fold, (train_idx, test_idx) in enumerate(logo.split(X, y, groups=subjects), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        test_subject = int(subjects[test_idx][0])

        print(
            f"[{progress_label}] Fold {fold}/{total_folds} "
            f"(test subject {test_subject})"
        )

        model = Pipeline(
            [
                ("cov", Covariances(estimator=cov_estimator)),
                ("reg", CovRegularizer(reg=1e-7)),
                ("ts", TangentSpace(metric=ts_metric)),
                (
                    "lr",
                    LogisticRegression(
                        C=c_value,
                        solver=solver,
                        class_weight=class_weight,
                        max_iter=1000,
                        random_state=42,
                    ),
                ),
            ]
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_score = model.decision_function(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_score)
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

        accuracies.append(acc)
        f1_scores.append(f1)
        auc_scores.append(auc)
        conf_matrices.append(cm)
        all_y_true.extend(y_test.tolist())
        all_y_pred.extend(y_pred.tolist())
        fold_rows.append(
            {
                "fold": fold,
                "subject": test_subject,
                "accuracy": acc,
                "f1_score": f1,
                "roc_auc": auc,
            }
        )

    return {
        "cov_estimator": cov_estimator,
        "ts_metric": ts_metric,
        "C": c_value,
        "solver": solver,
        "class_weight": class_weight,
        "mean_accuracy": float(np.mean(accuracies)),
        "std_accuracy": float(np.std(accuracies)),
        "mean_f1": float(np.mean(f1_scores)),
        "mean_auc": float(np.mean(auc_scores)),
        "avg_confusion_matrix": np.mean(conf_matrices, axis=0),
        "overall_confusion_matrix": confusion_matrix(all_y_true, all_y_pred, labels=[0, 1]),
        "fold_rows": fold_rows,
    }


def write_summary_csv(path, rows):
    fieldnames = [
        "parameter_name",
        "parameter_value",
        "cov_estimator",
        "ts_metric",
        "C",
        "solver",
        "class_weight",
        "mean_accuracy",
        "std_accuracy",
        "mean_f1",
        "mean_auc",
    ]

    with open(path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_fold_csv(path, rows):
    fieldnames = [
        "parameter_name",
        "parameter_value",
        "cov_estimator",
        "ts_metric",
        "C",
        "solver",
        "class_weight",
        "fold",
        "subject",
        "accuracy",
        "f1_score",
        "roc_auc",
    ]

    with open(path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_parameter_results(path, parameter_name, parameter_values, metric_values):
    plt.figure(figsize=(8, 5))
    plt.plot(parameter_values, metric_values, marker="o", linewidth=2)
    plt.xlabel(parameter_name)
    plt.ylabel("Mean Accuracy")
    plt.title(f"Riemann Parameter Study: {parameter_name}")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def save_confusion_matrix_csv(path, matrix):
    np.savetxt(path, matrix, delimiter=",", fmt="%d")


def save_confusion_matrix_plot(path, matrix, title):
    plt.figure(figsize=(5, 4))
    plt.imshow(matrix, interpolation="nearest", cmap="Blues")
    plt.title(title)
    plt.colorbar()
    tick_labels = ["Left", "Right"]
    tick_positions = np.arange(len(tick_labels))
    plt.xticks(tick_positions, tick_labels)
    plt.yticks(tick_positions, tick_labels)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")

    threshold = matrix.max() / 2.0 if matrix.size else 0.0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            plt.text(
                j,
                i,
                f"{int(matrix[i, j])}",
                ha="center",
                va="center",
                color="white" if matrix[i, j] > threshold else "black",
            )

    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def make_safe_filename(value):
    safe_value = str(value)
    for old, new in [(" ", "_"), (".", "p"), ("-", "neg"), ("/", "_")]:
        safe_value = safe_value.replace(old, new)
    return safe_value


def run_parameter_study(parameter_name, values, base_config):
    summary_rows = []
    fold_rows = []
    plot_labels = []
    plot_scores = []
    parameter_dir = os.path.join(RESULTS_DIR, parameter_name)

    os.makedirs(parameter_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print(f"Testing parameter: {parameter_name}")
    print("=" * 70)

    total_values = len(values)

    for value_index, value in enumerate(values, 1):
        config = base_config.copy()

        if parameter_name == "cov_estimator":
            config["cov_estimator"] = value
        elif parameter_name == "ts_metric":
            config["ts_metric"] = value
        elif parameter_name == "C":
            config["c_value"] = value
        elif parameter_name == "solver":
            config["solver"] = value
        elif parameter_name == "class_weight":
            config["class_weight"] = value
        else:
            raise ValueError(f"Unsupported parameter: {parameter_name}")

        progress_label = f"{parameter_name}={value} [{value_index}/{total_values}]"
        print(f"\nRunning {progress_label}")
        result = run_loso_riemann(X, y, subjects, progress_label=progress_label, **config)

        summary_row = {
            "parameter_name": parameter_name,
            "parameter_value": value,
            "cov_estimator": result["cov_estimator"],
            "ts_metric": result["ts_metric"],
            "C": result["C"],
            "solver": result["solver"],
            "class_weight": result["class_weight"],
            "mean_accuracy": result["mean_accuracy"],
            "std_accuracy": result["std_accuracy"],
            "mean_f1": result["mean_f1"],
            "mean_auc": result["mean_auc"],
        }
        summary_rows.append(summary_row)

        for fold_row in result["fold_rows"]:
            fold_rows.append(
                {
                    "parameter_name": parameter_name,
                    "parameter_value": value,
                    "cov_estimator": result["cov_estimator"],
                    "ts_metric": result["ts_metric"],
                    "C": result["C"],
                    "solver": result["solver"],
                    "class_weight": result["class_weight"],
                    **fold_row,
                }
            )

        plot_labels.append(str(value))
        plot_scores.append(result["mean_accuracy"])

        print(f"Mean Accuracy: {result['mean_accuracy'] * 100:.2f}% +/- {result['std_accuracy'] * 100:.2f}%")
        print(f"Mean F1 Score: {result['mean_f1']:.3f}")
        print(f"Mean ROC-AUC: {result['mean_auc']:.3f}")

        safe_value = make_safe_filename(value)
        confusion_csv_path = os.path.join(
            parameter_dir, f"confusion_{parameter_name}_{safe_value}.csv"
        )
        confusion_plot_path = os.path.join(
            parameter_dir, f"confusion_{parameter_name}_{safe_value}.png"
        )
        save_confusion_matrix_csv(confusion_csv_path, result["overall_confusion_matrix"])
        save_confusion_matrix_plot(
            confusion_plot_path,
            result["overall_confusion_matrix"],
            f"Overall LOSO Confusion Matrix: {parameter_name}={value}",
        )

    summary_csv_path = os.path.join(parameter_dir, f"summary_{parameter_name}.csv")
    fold_csv_path = os.path.join(parameter_dir, f"fold_results_{parameter_name}.csv")
    plot_path = os.path.join(parameter_dir, f"plot_{parameter_name}.png")

    write_summary_csv(summary_csv_path, summary_rows)
    write_fold_csv(fold_csv_path, fold_rows)
    plot_parameter_results(plot_path, parameter_name, plot_labels, plot_scores)

    best_row = max(summary_rows, key=lambda row: row["mean_accuracy"])
    print("\nBest result for", parameter_name)
    print(best_row)
    print("Saved summary to:", summary_csv_path)
    print("Saved fold results to:", fold_csv_path)
    print("Saved plot to:", plot_path)


# ==========================================================
# 2. PARAMETER STUDY CONFIGURATION
# ==========================================================
BASE_CONFIG = {
    "cov_estimator": "lwf",
    "ts_metric": "riemann",
    "c_value": 1.0,
    "solver": "lbfgs",
    "class_weight": "balanced",
}

PARAMETER_STUDIES = [
    {
        "name": "cov_estimator",
        "values": ["scm", "lwf", "oas"],
        "overrides": {},
    },
    {
        "name": "ts_metric",
        "values": ["riemann", "logeuclid", "euclid"],
        "overrides": {"cov_estimator": "lwf"},
    },
    {
        "name": "C",
        "values": [0.1, 1.0, 10.0, 100.0],
        "overrides": {"cov_estimator": "lwf", "ts_metric": "riemann", "solver": "lbfgs"},
    },
    {
        "name": "solver",
        "values": ["lbfgs", "liblinear", "saga"],
        "overrides": {"cov_estimator": "lwf", "ts_metric": "riemann", "c_value": 1.0},
    },
    {
        "name": "class_weight",
        "values": [None, "balanced"],
        "overrides": {"cov_estimator": "lwf", "ts_metric": "riemann", "c_value": 1.0, "solver": "lbfgs"},
    },
]


if __name__ == "__main__":
    for study in PARAMETER_STUDIES:
        study_config = BASE_CONFIG.copy()
        study_config.update(study["overrides"])
        run_parameter_study(study["name"], study["values"], study_config)
