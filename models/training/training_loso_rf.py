import csv
import os

import matplotlib
import numpy as np
from mne.decoding import CSP
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ==========================================================
# 1. LOAD PREPROCESSED DATA
# ==========================================================
BASE_DIR = "models/output"
DATA_DIR = os.path.join(BASE_DIR, "preprocessing_result")
RESULTS_DIR = os.path.join(BASE_DIR, "rf_parameter_study")

X = np.load(os.path.join(DATA_DIR, "X.npy"))
y = np.load(os.path.join(DATA_DIR, "y.npy"))
subjects = np.load(os.path.join(DATA_DIR, "subjects.npy"))

os.makedirs(RESULTS_DIR, exist_ok=True)

print("Dataset shape:", X.shape)
print("Number of subjects:", len(np.unique(subjects)))
print("Results directory:", RESULTS_DIR)


def run_loso_rf(
    X,
    y,
    subjects,
    *,
    n_estimators=200,
    criterion="gini",
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=5,
    max_features="sqrt",
    class_weight="balanced",
    csp_components=6,
    progress_label="",
):
    """Run LOSO with CSP inside each fold and return summary metrics."""
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

        csp = CSP(
            n_components=csp_components,
            reg="ledoit_wolf",
            log=True,
            norm_trace=False,
        )

        X_train_csp = csp.fit_transform(X_train, y_train)
        X_test_csp = csp.transform(X_test)

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            class_weight=class_weight,
            n_jobs=-1,
            random_state=42,
        )

        model.fit(X_train_csp, y_train)

        y_pred = model.predict(X_test_csp)
        y_prob = model.predict_proba(X_test_csp)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_prob)
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
        "n_estimators": n_estimators,
        "criterion": criterion,
        "max_depth": max_depth,
        "min_samples_split": min_samples_split,
        "min_samples_leaf": min_samples_leaf,
        "max_features": max_features,
        "class_weight": class_weight,
        "csp_components": csp_components,
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
        "n_estimators",
        "criterion",
        "max_depth",
        "min_samples_split",
        "min_samples_leaf",
        "max_features",
        "class_weight",
        "csp_components",
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
        "n_estimators",
        "criterion",
        "max_depth",
        "min_samples_split",
        "min_samples_leaf",
        "max_features",
        "class_weight",
        "csp_components",
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
    plt.title(f"RF Parameter Study: {parameter_name}")
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

        if parameter_name == "n_estimators":
            config["n_estimators"] = value
        elif parameter_name == "criterion":
            config["criterion"] = value
        elif parameter_name == "max_depth":
            config["max_depth"] = value
        elif parameter_name == "min_samples_split":
            config["min_samples_split"] = value
        elif parameter_name == "min_samples_leaf":
            config["min_samples_leaf"] = value
        elif parameter_name == "max_features":
            config["max_features"] = value
        elif parameter_name == "class_weight":
            config["class_weight"] = value
        elif parameter_name == "csp_components":
            config["csp_components"] = value
        else:
            raise ValueError(f"Unsupported parameter: {parameter_name}")

        progress_label = f"{parameter_name}={value} [{value_index}/{total_values}]"
        print(f"\nRunning {progress_label}")
        result = run_loso_rf(X, y, subjects, progress_label=progress_label, **config)

        summary_row = {
            "parameter_name": parameter_name,
            "parameter_value": value,
            "n_estimators": result["n_estimators"],
            "criterion": result["criterion"],
            "max_depth": result["max_depth"],
            "min_samples_split": result["min_samples_split"],
            "min_samples_leaf": result["min_samples_leaf"],
            "max_features": result["max_features"],
            "class_weight": result["class_weight"],
            "csp_components": result["csp_components"],
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
                    "n_estimators": result["n_estimators"],
                    "criterion": result["criterion"],
                    "max_depth": result["max_depth"],
                    "min_samples_split": result["min_samples_split"],
                    "min_samples_leaf": result["min_samples_leaf"],
                    "max_features": result["max_features"],
                    "class_weight": result["class_weight"],
                    "csp_components": result["csp_components"],
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
    "n_estimators": 200,
    "criterion": "gini",
    "max_depth": 10,
    "min_samples_split": 10,
    "min_samples_leaf": 5,
    "max_features": "sqrt",
    "class_weight": "balanced",
    "csp_components": 6,
}

PARAMETER_STUDIES = [
    {
        "name": "n_estimators",
        "values": [100, 200, 300, 500],
        "overrides": {},
    },
    {
        "name": "criterion",
        "values": ["gini", "entropy", "log_loss"],
        "overrides": {"n_estimators": 200},
    },
    {
        "name": "max_depth",
        "values": [5, 10, 15, 20, None],
        "overrides": {"criterion": "gini", "n_estimators": 200},
    },
    {
        "name": "min_samples_split",
        "values": [2, 5, 10, 20],
        "overrides": {"criterion": "gini", "n_estimators": 200, "max_depth": 10},
    },
    {
        "name": "min_samples_leaf",
        "values": [1, 2, 5, 10],
        "overrides": {"criterion": "gini", "n_estimators": 200, "max_depth": 10, "min_samples_split": 10},
    },
    {
        "name": "max_features",
        "values": ["sqrt", "log2", None],
        "overrides": {"criterion": "gini", "n_estimators": 200, "max_depth": 10},
    },
    {
        "name": "class_weight",
        "values": [None, "balanced", "balanced_subsample"],
        "overrides": {"criterion": "gini", "n_estimators": 200, "max_depth": 10},
    },
]


if __name__ == "__main__":
    for study in PARAMETER_STUDIES:
        study_config = BASE_CONFIG.copy()
        study_config.update(study["overrides"])
        run_parameter_study(study["name"], study["values"], study_config)
