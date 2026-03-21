import csv
import os

import matplotlib
import numpy as np
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DATA_DIR = os.path.join("models", "preprocessing_result")
RESULTS_DIR = os.path.join("models", "csp", "csp_testing")

X = np.load(os.path.join(DATA_DIR, "X.npy"))
y = np.load(os.path.join(DATA_DIR, "y.npy"))
subjects = np.load(os.path.join(DATA_DIR, "subjects.npy"))

os.makedirs(RESULTS_DIR, exist_ok=True)

print("Dataset shape:", X.shape)
print("Number of subjects:", len(np.unique(subjects)))
print("Results directory:", RESULTS_DIR)


# Fixed model settings for fair CSP comparison across classical CSP-based models.
MODEL_CONFIGS = {
    "svm": {
        "display_name": "SVM",
        "factory": lambda: Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    SVC(
                        kernel="linear",
                        C=1.0,
                        class_weight="balanced",
                        probability=False,
                        random_state=42,
                    ),
                ),
            ]
        ),
    },
    "lr": {
        "display_name": "LR",
        "factory": lambda: Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        penalty="l2",
                        C=1.0,
                        solver="lbfgs",
                        class_weight="balanced",
                        max_iter=1000,
                        random_state=42,
                    ),
                ),
            ]
        ),
    },
    "knn": {
        "display_name": "KNN",
        "factory": lambda: Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    KNeighborsClassifier(
                        n_neighbors=5,
                        weights="distance",
                        metric="minkowski",
                        p=2,
                    ),
                ),
            ]
        ),
    },
    "dt": {
        "display_name": "DT",
        "factory": lambda: DecisionTreeClassifier(
            criterion="gini",
            max_depth=5,
            min_samples_split=2,
            min_samples_leaf=1,
            class_weight="balanced",
            random_state=42,
        ),
    },
    "rf": {
        "display_name": "RF",
        "factory": lambda: RandomForestClassifier(
            n_estimators=200,
            criterion="gini",
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features="sqrt",
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    },
    "lda": {
        "display_name": "LDA",
        "factory": lambda: LinearDiscriminantAnalysis(
            solver="lsqr",
            shrinkage="auto",
        ),
    },
}

CSP_VALUES = [4, 6, 8, 10]


def get_score(model, features):
    if hasattr(model, "decision_function"):
        return model.decision_function(features)
    if hasattr(model, "predict_proba"):
        return model.predict_proba(features)[:, 1]
    return model.predict(features)


def run_loso_with_csp(X, y, subjects, model_name, csp_components):
    logo = LeaveOneGroupOut()

    accuracies = []
    f1_scores = []
    auc_scores = []
    conf_matrices = []
    fold_rows = []
    all_y_true = []
    all_y_pred = []

    total_folds = len(np.unique(subjects))
    model_config = MODEL_CONFIGS[model_name]

    for fold, (train_idx, test_idx) in enumerate(logo.split(X, y, groups=subjects), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        test_subject = int(subjects[test_idx][0])

        print(
            f"[{model_name} | csp={csp_components}] Fold {fold}/{total_folds} "
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

        model = model_config["factory"]()
        model.fit(X_train_csp, y_train)

        y_pred = model.predict(X_test_csp)
        y_score = get_score(model, X_test_csp)

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
                "model_name": model_name,
                "display_name": model_config["display_name"],
                "csp_components": csp_components,
                "fold": fold,
                "subject": test_subject,
                "accuracy": acc,
                "f1_score": f1,
                "roc_auc": auc,
            }
        )

    return {
        "model_name": model_name,
        "display_name": model_config["display_name"],
        "csp_components": csp_components,
        "mean_accuracy": float(np.mean(accuracies)),
        "std_accuracy": float(np.std(accuracies)),
        "mean_f1": float(np.mean(f1_scores)),
        "mean_auc": float(np.mean(auc_scores)),
        "overall_confusion_matrix": confusion_matrix(all_y_true, all_y_pred, labels=[0, 1]),
        "fold_rows": fold_rows,
    }


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_accuracy_by_model(path, summary_rows):
    plt.figure(figsize=(10, 6))

    for model_name, model_config in MODEL_CONFIGS.items():
        model_rows = [row for row in summary_rows if row["model_name"] == model_name]
        model_rows.sort(key=lambda row: row["csp_components"])
        x_values = [row["csp_components"] for row in model_rows]
        y_values = [row["mean_accuracy"] for row in model_rows]
        plt.plot(x_values, y_values, marker="o", linewidth=2, label=model_config["display_name"])

    plt.xlabel("CSP Components")
    plt.ylabel("Mean Accuracy")
    plt.title("CSP Component Comparison Across Classical Models")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_f1_by_model(path, summary_rows):
    plt.figure(figsize=(10, 6))

    for model_name, model_config in MODEL_CONFIGS.items():
        model_rows = [row for row in summary_rows if row["model_name"] == model_name]
        model_rows.sort(key=lambda row: row["csp_components"])
        x_values = [row["csp_components"] for row in model_rows]
        y_values = [row["mean_f1"] for row in model_rows]
        plt.plot(x_values, y_values, marker="o", linewidth=2, label=model_config["display_name"])

    plt.xlabel("CSP Components")
    plt.ylabel("Mean F1-score")
    plt.title("CSP Component Comparison Across Classical Models (F1-score)")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def write_text_summary(path, summary_rows):
    best_overall = max(summary_rows, key=lambda row: (row["mean_accuracy"], row["mean_f1"], row["mean_auc"]))
    lines = [
        "CSP TESTING SUMMARY",
        "=" * 70,
        "",
        "Best overall model/CSP combination",
        f"model: {best_overall['display_name']}",
        f"csp_components: {best_overall['csp_components']}",
        f"mean_accuracy: {best_overall['mean_accuracy']:.4f}",
        f"mean_f1: {best_overall['mean_f1']:.4f}",
        f"mean_auc: {best_overall['mean_auc']:.4f}",
        "",
        "Best CSP component count per model",
    ]

    for model_name, model_config in MODEL_CONFIGS.items():
        model_rows = [row for row in summary_rows if row["model_name"] == model_name]
        best_row = max(model_rows, key=lambda row: (row["mean_accuracy"], row["mean_f1"], row["mean_auc"]))
        lines.append(
            (
                f"{model_config['display_name']}: csp={best_row['csp_components']}, "
                f"accuracy={best_row['mean_accuracy']:.4f}, "
                f"f1={best_row['mean_f1']:.4f}, auc={best_row['mean_auc']:.4f}"
            )
        )

    with open(path, "w", encoding="utf-8") as text_file:
        text_file.write("\n".join(lines))


if __name__ == "__main__":
    summary_rows = []
    fold_rows = []

    for model_name in MODEL_CONFIGS:
        print("\n" + "=" * 70)
        print(f"Testing CSP across model: {MODEL_CONFIGS[model_name]['display_name']}")
        print("=" * 70)

        for csp_components in CSP_VALUES:
            result = run_loso_with_csp(X, y, subjects, model_name, csp_components)
            summary_rows.append(
                {
                    "model_name": result["model_name"],
                    "display_name": result["display_name"],
                    "csp_components": result["csp_components"],
                    "mean_accuracy": result["mean_accuracy"],
                    "std_accuracy": result["std_accuracy"],
                    "mean_f1": result["mean_f1"],
                    "mean_auc": result["mean_auc"],
                }
            )
            fold_rows.extend(result["fold_rows"])

            print(
                f"{result['display_name']} | CSP={csp_components} | "
                f"Accuracy={result['mean_accuracy'] * 100:.2f}% | "
                f"F1={result['mean_f1']:.3f} | AUC={result['mean_auc']:.3f}"
            )

    summary_csv_path = os.path.join(RESULTS_DIR, "csp_summary.csv")
    fold_csv_path = os.path.join(RESULTS_DIR, "csp_fold_results.csv")
    accuracy_plot_path = os.path.join(RESULTS_DIR, "csp_accuracy_comparison.png")
    f1_plot_path = os.path.join(RESULTS_DIR, "csp_f1_comparison.png")
    summary_txt_path = os.path.join(RESULTS_DIR, "csp_summary.txt")

    write_csv(
        summary_csv_path,
        summary_rows,
        [
            "model_name",
            "display_name",
            "csp_components",
            "mean_accuracy",
            "std_accuracy",
            "mean_f1",
            "mean_auc",
        ],
    )
    write_csv(
        fold_csv_path,
        fold_rows,
        [
            "model_name",
            "display_name",
            "csp_components",
            "fold",
            "subject",
            "accuracy",
            "f1_score",
            "roc_auc",
        ],
    )
    plot_accuracy_by_model(accuracy_plot_path, summary_rows)
    plot_f1_by_model(f1_plot_path, summary_rows)
    write_text_summary(summary_txt_path, summary_rows)

    print("Saved summary to:", summary_csv_path)
    print("Saved fold results to:", fold_csv_path)
    print("Saved accuracy plot to:", accuracy_plot_path)
    print("Saved F1 plot to:", f1_plot_path)
    print("Saved text summary to:", summary_txt_path)
