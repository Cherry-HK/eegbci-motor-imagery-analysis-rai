import csv
import itertools
import os

import matplotlib
import numpy as np
from mne.decoding import CSP
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ==========================================================
# 1. LOAD PREPROCESSED DATA
# ==========================================================
DATA_DIR = os.path.join("models", "preprocessing_result")
RESULTS_DIR = os.path.join("models", "svm", "svm_testing")

X = np.load(os.path.join(DATA_DIR, "X.npy"))
y = np.load(os.path.join(DATA_DIR, "y.npy"))
subjects = np.load(os.path.join(DATA_DIR, "subjects.npy"))

os.makedirs(RESULTS_DIR, exist_ok=True)

print("Dataset shape:", X.shape)
print("Number of subjects:", len(np.unique(subjects)))
print("Results directory:", RESULTS_DIR)


def sanitize_param_grid(param_grid):
    """Remove parameters that do not affect a given kernel."""
    kernel = param_grid["kernel"]
    cleaned = dict(param_grid)

    if kernel == "linear":
        cleaned["gamma"] = "unused"
        cleaned["degree"] = "unused"
        cleaned["coef0"] = "unused"
    elif kernel == "rbf":
        cleaned["degree"] = "unused"
        cleaned["coef0"] = "unused"
    elif kernel == "sigmoid":
        cleaned["degree"] = "unused"

    return cleaned


def build_search_space(search_config):
    keys = list(search_config.keys())
    values = [search_config[key] for key in keys]
    combinations = []
    seen = set()

    for combo_values in itertools.product(*values):
        raw_combo = dict(zip(keys, combo_values))
        combo = sanitize_param_grid(raw_combo)
        combo_key = tuple((key, combo[key]) for key in sorted(combo))
        if combo_key in seen:
            continue
        seen.add(combo_key)
        combinations.append(combo)

    return combinations


def run_loso_svm(
    X,
    y,
    subjects,
    *,
    kernel="linear",
    c_value=1.0,
    gamma="scale",
    degree=3,
    coef0=0.0,
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

        svm_kwargs = {
            "kernel": kernel,
            "C": c_value,
            "class_weight": class_weight,
            "probability": False,
            "random_state": 42,
        }
        if kernel in {"rbf", "poly", "sigmoid"}:
            svm_kwargs["gamma"] = gamma
        if kernel == "poly":
            svm_kwargs["degree"] = degree
            svm_kwargs["coef0"] = coef0
        elif kernel == "sigmoid":
            svm_kwargs["coef0"] = coef0

        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("svm", SVC(**svm_kwargs)),
            ]
        )

        model.fit(X_train_csp, y_train)

        y_pred = model.predict(X_test_csp)
        y_score = model.decision_function(X_test_csp)

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
                "support": int(len(y_test)),
            }
        )

    per_subject_rows = sorted(fold_rows, key=lambda row: row["accuracy"], reverse=True)

    return {
        "kernel": kernel,
        "C": c_value,
        "gamma": gamma,
        "degree": degree,
        "coef0": coef0,
        "class_weight": class_weight,
        "csp_components": csp_components,
        "mean_accuracy": float(np.mean(accuracies)),
        "std_accuracy": float(np.std(accuracies)),
        "mean_f1": float(np.mean(f1_scores)),
        "mean_auc": float(np.mean(auc_scores)),
        "avg_confusion_matrix": np.mean(conf_matrices, axis=0),
        "overall_confusion_matrix": confusion_matrix(all_y_true, all_y_pred, labels=[0, 1]),
        "fold_rows": fold_rows,
        "per_subject_rows": per_subject_rows,
        "best_subject": max(fold_rows, key=lambda row: row["accuracy"]),
        "worst_subject": min(fold_rows, key=lambda row: row["accuracy"]),
    }


def make_safe_filename(value):
    safe_value = str(value)
    for old, new in [(" ", "_"), (".", "p"), ("-", "neg"), ("/", "_")]:
        safe_value = safe_value.replace(old, new)
    return safe_value


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


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


def plot_top_combinations(path, rows, top_n=10):
    top_rows = rows[:top_n]
    labels = [f"#{row['rank']} {row['kernel']}" for row in top_rows]
    scores = [row["mean_accuracy"] for row in top_rows]

    plt.figure(figsize=(11, 6))
    bars = plt.bar(labels, scores, color="#3b82f6")
    plt.ylabel("Mean Accuracy")
    plt.xlabel("Combination Rank")
    plt.title(f"Top {len(top_rows)} SVM Hyperparameter Combinations")
    plt.xticks(rotation=30, ha="right")
    plt.ylim(0.0, 1.0)

    for bar, score in zip(bars, scores):
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            score + 0.01,
            f"{score:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_subject_accuracy(path, per_subject_rows):
    sorted_rows = sorted(per_subject_rows, key=lambda row: row["accuracy"], reverse=True)
    subject_ranks = np.arange(1, len(sorted_rows) + 1)
    scores = [row["accuracy"] for row in sorted_rows]

    plt.figure(figsize=(11, 5))
    plt.scatter(subject_ranks, scores, color="#10b981", s=32, alpha=0.9)
    plt.plot(subject_ranks, scores, color="#10b981", linewidth=1.2, alpha=0.45)
    plt.xlabel("Subjects Ranked by Accuracy")
    plt.ylabel("Accuracy")
    plt.title("Best SVM Configuration: LOSO Accuracy per Subject")
    plt.ylim(0.0, 1.0)
    plt.xlim(0, len(sorted_rows) + 1)
    plt.grid(axis="y", linestyle="--", alpha=0.4)

    highlight_rows = sorted_rows[:3] + sorted_rows[-3:]
    highlight_subjects = {row["subject"] for row in highlight_rows}
    for rank, row in enumerate(sorted_rows, start=1):
        if row["subject"] in highlight_subjects:
            plt.annotate(
                f"S{row['subject']}: {row['accuracy']:.2f}",
                (rank, row["accuracy"]),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=8,
            )

    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_subject_metric_comparison(path, per_subject_rows):
    sorted_rows = sorted(per_subject_rows, key=lambda row: row["accuracy"], reverse=True)
    subjects_axis = np.arange(1, len(sorted_rows) + 1)
    accuracy_values = [row["accuracy"] for row in sorted_rows]
    f1_values = [row["f1_score"] for row in sorted_rows]
    auc_values = [row["roc_auc"] for row in sorted_rows]

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    metric_specs = [
        ("Accuracy", accuracy_values, "#2563eb"),
        ("F1-score", f1_values, "#f97316"),
        ("ROC-AUC", auc_values, "#16a34a"),
    ]

    for ax, (label, values, color) in zip(axes, metric_specs):
        ax.scatter(subjects_axis, values, color=color, s=24, alpha=0.85)
        ax.plot(subjects_axis, values, color=color, linewidth=1.0, alpha=0.35)
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel(label)
        ax.grid(True, linestyle="--", alpha=0.35)

    axes[0].set_title("Best SVM Configuration: Per-Subject LOSO Metrics")
    axes[-1].set_xlabel("Subjects Ranked by Accuracy")
    axes[-1].set_xlim(0, len(sorted_rows) + 1)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def write_best_summary(path, best_result, total_combinations, top_rows):
    best_subject = best_result["best_subject"]
    worst_subject = best_result["worst_subject"]
    top_three = top_rows[:3]

    lines = [
        "SVM FINAL TUNING SUMMARY",
        "=" * 70,
        f"Total combinations evaluated: {total_combinations}",
        "",
        "Best overall parameter combination",
        f"kernel: {best_result['kernel']}",
        f"C: {best_result['C']}",
        f"gamma: {best_result['gamma']}",
        f"degree: {best_result['degree']}",
        f"coef0: {best_result['coef0']}",
        f"class_weight: {best_result['class_weight']}",
        f"csp_components: {best_result['csp_components']}",
        "",
        "Best model final metrics",
        f"mean_accuracy: {best_result['mean_accuracy']:.4f}",
        f"std_accuracy: {best_result['std_accuracy']:.4f}",
        f"mean_f1: {best_result['mean_f1']:.4f}",
        f"mean_auc: {best_result['mean_auc']:.4f}",
        "",
        "Per-subject LOSO analysis",
        (
            f"best_subject: {best_subject['subject']} "
            f"(accuracy={best_subject['accuracy']:.4f}, "
            f"f1={best_subject['f1_score']:.4f}, auc={best_subject['roc_auc']:.4f})"
        ),
        (
            f"worst_subject: {worst_subject['subject']} "
            f"(accuracy={worst_subject['accuracy']:.4f}, "
            f"f1={worst_subject['f1_score']:.4f}, auc={worst_subject['roc_auc']:.4f})"
        ),
        "",
        "Interpretation notes",
        (
            "The best combination above is the strongest configuration among all "
            "tested SVM+CSP settings under LOSO evaluation."
        ),
        (
            "Mean accuracy reflects overall subject-independent performance, while "
            "standard deviation indicates how stable the model is across subjects."
        ),
        (
            "A large gap between the best and worst subject suggests subject "
            "variability and should be discussed in the thesis."
        ),
        (
            "Use the confusion matrix to discuss whether the classifier tends to "
            "favor one class over the other."
        ),
        "",
        "Top ranked combinations",
    ]

    for row in top_three:
        lines.append(
            (
                f"rank {row['rank']}: kernel={row['kernel']}, C={row['C']}, "
                f"gamma={row['gamma']}, degree={row['degree']}, coef0={row['coef0']}, "
                f"class_weight={row['class_weight']}, csp_components={row['csp_components']}, "
                f"mean_accuracy={row['mean_accuracy']:.4f}, "
                f"mean_f1={row['mean_f1']:.4f}, mean_auc={row['mean_auc']:.4f}"
            )
        )

    with open(path, "w", encoding="utf-8") as text_file:
        text_file.write("\n".join(lines))


SEARCH_CONFIG = {
    "kernel": ["linear", "rbf", "poly"],
    "c_value": [0.1, 1.0, 10.0],
    "gamma": ["scale", "auto", 0.01, 0.1],
    "degree": [2, 3, 4, 5],
    "coef0": [0.0, 0.5, 1.0, 2.0],
    "class_weight": ["balanced"],
    "csp_components": [6]
}


if __name__ == "__main__":
    combinations = build_search_space(SEARCH_CONFIG)
    summary_rows = []
    fold_rows = []
    best_result = None

    print("\n" + "=" * 70)
    print("Running SVM combined hyperparameter search")
    print("=" * 70)
    print("Total unique combinations:", len(combinations))

    for index, combo in enumerate(combinations, 1):
        progress_label = f"combo {index}/{len(combinations)}"
        print("\n" + "-" * 70)
        print(
            f"Testing {progress_label}: "
            f"kernel={combo['kernel']}, C={combo['c_value']}, gamma={combo['gamma']}, "
            f"degree={combo['degree']}, coef0={combo['coef0']}, "
            f"class_weight={combo['class_weight']}, csp_components={combo['csp_components']}"
        )

        result = run_loso_svm(X, y, subjects, progress_label=progress_label, **combo)

        summary_row = {
            "kernel": result["kernel"],
            "C": result["C"],
            "gamma": result["gamma"],
            "degree": result["degree"],
            "coef0": result["coef0"],
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
                    "kernel": result["kernel"],
                    "C": result["C"],
                    "gamma": result["gamma"],
                    "degree": result["degree"],
                    "coef0": result["coef0"],
                    "class_weight": result["class_weight"],
                    "csp_components": result["csp_components"],
                    **fold_row,
                }
            )

        if best_result is None or result["mean_accuracy"] > best_result["mean_accuracy"]:
            best_result = result

        print(f"Mean Accuracy: {result['mean_accuracy'] * 100:.2f}% +/- {result['std_accuracy'] * 100:.2f}%")
        print(f"Mean F1 Score: {result['mean_f1']:.3f}")
        print(f"Mean ROC-AUC: {result['mean_auc']:.3f}")

    ranked_rows = sorted(
        summary_rows,
        key=lambda row: (row["mean_accuracy"], row["mean_f1"], row["mean_auc"]),
        reverse=True,
    )
    for rank, row in enumerate(ranked_rows, 1):
        row["rank"] = rank

    best_row = ranked_rows[0]
    best_fold_rows = [
        row
        for row in fold_rows
        if row["kernel"] == best_row["kernel"]
        and row["C"] == best_row["C"]
        and row["gamma"] == best_row["gamma"]
        and row["degree"] == best_row["degree"]
        and row["coef0"] == best_row["coef0"]
        and row["class_weight"] == best_row["class_weight"]
        and row["csp_components"] == best_row["csp_components"]
    ]

    summary_csv_path = os.path.join(RESULTS_DIR, "all_combinations_summary.csv")
    fold_csv_path = os.path.join(RESULTS_DIR, "all_combinations_fold_results.csv")
    top_csv_path = os.path.join(RESULTS_DIR, "top_10_combinations.csv")
    best_fold_csv_path = os.path.join(RESULTS_DIR, "best_configuration_per_subject.csv")
    best_summary_path = os.path.join(RESULTS_DIR, "best_configuration_summary.txt")
    confusion_csv_path = os.path.join(RESULTS_DIR, "best_confusion_matrix.csv")
    confusion_plot_path = os.path.join(RESULTS_DIR, "best_confusion_matrix.png")
    top_plot_path = os.path.join(RESULTS_DIR, "top_10_combinations.png")
    subject_accuracy_plot_path = os.path.join(RESULTS_DIR, "best_per_subject_accuracy.png")
    subject_metric_plot_path = os.path.join(RESULTS_DIR, "best_per_subject_metrics.png")

    write_csv(
        summary_csv_path,
        ranked_rows,
        [
            "rank",
            "kernel",
            "C",
            "gamma",
            "degree",
            "coef0",
            "class_weight",
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
            "kernel",
            "C",
            "gamma",
            "degree",
            "coef0",
            "class_weight",
            "csp_components",
            "fold",
            "subject",
            "accuracy",
            "f1_score",
            "roc_auc",
            "support",
        ],
    )
    write_csv(
        top_csv_path,
        ranked_rows[:10],
        [
            "rank",
            "kernel",
            "C",
            "gamma",
            "degree",
            "coef0",
            "class_weight",
            "csp_components",
            "mean_accuracy",
            "std_accuracy",
            "mean_f1",
            "mean_auc",
        ],
    )
    write_csv(
        best_fold_csv_path,
        sorted(best_fold_rows, key=lambda row: row["subject"]),
        [
            "kernel",
            "C",
            "gamma",
            "degree",
            "coef0",
            "class_weight",
            "csp_components",
            "fold",
            "subject",
            "accuracy",
            "f1_score",
            "roc_auc",
            "support",
        ],
    )

    save_confusion_matrix_csv(confusion_csv_path, best_result["overall_confusion_matrix"])
    save_confusion_matrix_plot(
        confusion_plot_path,
        best_result["overall_confusion_matrix"],
        "Best SVM Configuration: Overall LOSO Confusion Matrix",
    )
    plot_top_combinations(top_plot_path, ranked_rows, top_n=10)
    plot_subject_accuracy(subject_accuracy_plot_path, best_result["per_subject_rows"])
    plot_subject_metric_comparison(subject_metric_plot_path, best_result["per_subject_rows"])
    write_best_summary(best_summary_path, best_result, len(combinations), ranked_rows)

    print("\n" + "=" * 70)
    print("Best overall SVM configuration")
    print("=" * 70)
    print(best_row)
    print("Saved all combinations summary to:", summary_csv_path)
    print("Saved all fold results to:", fold_csv_path)
    print("Saved top 10 combinations to:", top_csv_path)
    print("Saved best per-subject results to:", best_fold_csv_path)
    print("Saved text summary to:", best_summary_path)
    print("Saved confusion matrix CSV to:", confusion_csv_path)
    print("Saved confusion matrix plot to:", confusion_plot_path)
    print("Saved top combinations plot to:", top_plot_path)
    print("Saved per-subject accuracy plot to:", subject_accuracy_plot_path)
    print("Saved per-subject metrics plot to:", subject_metric_plot_path)
