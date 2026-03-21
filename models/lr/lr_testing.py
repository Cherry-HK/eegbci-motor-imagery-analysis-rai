import csv
import itertools
import os

import matplotlib
import numpy as np
from mne.decoding import CSP
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DATA_DIR = os.path.join("models", "preprocessing_result")
RESULTS_DIR = os.path.join("models", "lr", "lr_testing")

X = np.load(os.path.join(DATA_DIR, "X.npy"))
y = np.load(os.path.join(DATA_DIR, "y.npy"))
subjects = np.load(os.path.join(DATA_DIR, "subjects.npy"))

os.makedirs(RESULTS_DIR, exist_ok=True)

print("Dataset shape:", X.shape)
print("Number of subjects:", len(np.unique(subjects)))
print("Results directory:", RESULTS_DIR)


def sanitize_param_grid(param_grid):
    """Remove incompatible or irrelevant LR parameter combinations."""
    penalty = param_grid["penalty"]
    solver = param_grid["solver"]
    cleaned = dict(param_grid)

    if penalty == "l1":
        if solver not in {"liblinear", "saga"}:
            return None
        cleaned["l1_ratio"] = "unused"
    elif penalty == "l2":
        if solver not in {"lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"}:
            return None
        cleaned["l1_ratio"] = "unused"
    elif penalty == "elasticnet":
        if solver != "saga":
            return None
    else:
        return None

    return cleaned


def build_search_space(search_config):
    keys = list(search_config.keys())
    values = [search_config[key] for key in keys]
    combinations = []
    seen = set()

    for combo_values in itertools.product(*values):
        raw_combo = dict(zip(keys, combo_values))
        combo = sanitize_param_grid(raw_combo)
        if combo is None:
            continue
        combo_key = tuple((key, combo[key]) for key in sorted(combo))
        if combo_key in seen:
            continue
        seen.add(combo_key)
        combinations.append(combo)

    return combinations


def run_loso_lr(
    X,
    y,
    subjects,
    *,
    penalty="l2",
    c_value=1.0,
    solver="lbfgs",
    class_weight="balanced",
    l1_ratio=None,
    csp_components=6,
    progress_label="",
):
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

        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "lr",
                    LogisticRegression(
                        penalty=penalty,
                        C=c_value,
                        solver=solver,
                        class_weight=class_weight,
                        l1_ratio=l1_ratio if l1_ratio != "unused" else None,
                        max_iter=1000,
                        random_state=42,
                    ),
                ),
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
        "penalty": penalty,
        "C": c_value,
        "solver": solver,
        "class_weight": class_weight,
        "l1_ratio": l1_ratio,
        "csp_components": csp_components,
        "mean_accuracy": float(np.mean(accuracies)),
        "std_accuracy": float(np.std(accuracies)),
        "mean_f1": float(np.mean(f1_scores)),
        "mean_auc": float(np.mean(auc_scores)),
        "overall_confusion_matrix": confusion_matrix(all_y_true, all_y_pred, labels=[0, 1]),
        "fold_rows": fold_rows,
        "per_subject_rows": per_subject_rows,
        "best_subject": max(fold_rows, key=lambda row: row["accuracy"]),
        "worst_subject": min(fold_rows, key=lambda row: row["accuracy"]),
    }


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
    labels = [f"#{row['rank']} {row['solver']}" for row in top_rows]
    scores = [row["mean_accuracy"] for row in top_rows]

    plt.figure(figsize=(11, 6))
    bars = plt.bar(labels, scores, color="#2563eb")
    plt.ylabel("Mean Accuracy")
    plt.xlabel("Combination Rank")
    plt.title(f"Top {len(top_rows)} LR Hyperparameter Combinations")
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
    sorted_rows = sorted(per_subject_rows, key=lambda row: row["subject"])
    labels = [str(row["subject"]) for row in sorted_rows]
    scores = [row["accuracy"] for row in sorted_rows]

    plt.figure(figsize=(10, 5))
    bars = plt.bar(labels, scores, color="#16a34a")
    plt.xlabel("Test Subject")
    plt.ylabel("Accuracy")
    plt.title("Best LR Configuration: LOSO Accuracy per Subject")
    plt.ylim(0.0, 1.0)
    plt.grid(axis="y", linestyle="--", alpha=0.4)

    for bar, score in zip(bars, scores):
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            score + 0.01,
            f"{score:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_subject_metric_comparison(path, per_subject_rows):
    sorted_rows = sorted(per_subject_rows, key=lambda row: row["subject"])
    subjects_axis = np.arange(len(sorted_rows))
    accuracy_values = [row["accuracy"] for row in sorted_rows]
    f1_values = [row["f1_score"] for row in sorted_rows]
    auc_values = [row["roc_auc"] for row in sorted_rows]

    plt.figure(figsize=(11, 5))
    plt.plot(subjects_axis, accuracy_values, marker="o", linewidth=2, label="Accuracy")
    plt.plot(subjects_axis, f1_values, marker="s", linewidth=2, label="F1-score")
    plt.plot(subjects_axis, auc_values, marker="^", linewidth=2, label="ROC-AUC")
    plt.xticks(subjects_axis, [str(row["subject"]) for row in sorted_rows])
    plt.ylim(0.0, 1.0)
    plt.xlabel("Test Subject")
    plt.ylabel("Score")
    plt.title("Best LR Configuration: Per-Subject LOSO Metrics")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def write_best_summary(path, best_result, total_combinations, top_rows):
    best_subject = best_result["best_subject"]
    worst_subject = best_result["worst_subject"]
    top_three = top_rows[:3]

    lines = [
        "LR FINAL TUNING SUMMARY",
        "=" * 70,
        f"Total combinations evaluated: {total_combinations}",
        "",
        "Best overall parameter combination",
        f"penalty: {best_result['penalty']}",
        f"C: {best_result['C']}",
        f"solver: {best_result['solver']}",
        f"class_weight: {best_result['class_weight']}",
        f"l1_ratio: {best_result['l1_ratio']}",
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
        "Top ranked combinations",
    ]

    for row in top_three:
        lines.append(
            (
                f"rank {row['rank']}: penalty={row['penalty']}, C={row['C']}, "
                f"solver={row['solver']}, class_weight={row['class_weight']}, "
                f"l1_ratio={row['l1_ratio']}, csp_components={row['csp_components']}, "
                f"mean_accuracy={row['mean_accuracy']:.4f}, "
                f"mean_f1={row['mean_f1']:.4f}, mean_auc={row['mean_auc']:.4f}"
            )
        )

    with open(path, "w", encoding="utf-8") as text_file:
        text_file.write("\n".join(lines))


SEARCH_CONFIG = {
    "penalty": ["l2", "l1", "elasticnet"],
    "c_value": [0.1, 1.0, 10.0],
    "solver": ["lbfgs", "liblinear", "saga"],
    "class_weight": [None, "balanced"],
    "l1_ratio": [0.2, 0.5, 0.8],
    "csp_components": [6],
}


if __name__ == "__main__":
    combinations = build_search_space(SEARCH_CONFIG)
    summary_rows = []
    fold_rows = []
    best_result = None

    print("\n" + "=" * 70)
    print("Running LR combined hyperparameter search")
    print("=" * 70)
    print("Total unique combinations:", len(combinations))

    for index, combo in enumerate(combinations, 1):
        progress_label = f"combo {index}/{len(combinations)}"
        print("\n" + "-" * 70)
        print(
            f"Testing {progress_label}: penalty={combo['penalty']}, C={combo['c_value']}, "
            f"solver={combo['solver']}, class_weight={combo['class_weight']}, "
            f"l1_ratio={combo['l1_ratio']}, csp_components={combo['csp_components']}"
        )

        result = run_loso_lr(X, y, subjects, progress_label=progress_label, **combo)

        summary_row = {
            "penalty": result["penalty"],
            "C": result["C"],
            "solver": result["solver"],
            "class_weight": result["class_weight"],
            "l1_ratio": result["l1_ratio"],
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
                    "penalty": result["penalty"],
                    "C": result["C"],
                    "solver": result["solver"],
                    "class_weight": result["class_weight"],
                    "l1_ratio": result["l1_ratio"],
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
        if row["penalty"] == best_row["penalty"]
        and row["C"] == best_row["C"]
        and row["solver"] == best_row["solver"]
        and row["class_weight"] == best_row["class_weight"]
        and row["l1_ratio"] == best_row["l1_ratio"]
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
            "penalty",
            "C",
            "solver",
            "class_weight",
            "l1_ratio",
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
            "penalty",
            "C",
            "solver",
            "class_weight",
            "l1_ratio",
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
            "penalty",
            "C",
            "solver",
            "class_weight",
            "l1_ratio",
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
            "penalty",
            "C",
            "solver",
            "class_weight",
            "l1_ratio",
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
        "Best LR Configuration: Overall LOSO Confusion Matrix",
    )
    plot_top_combinations(top_plot_path, ranked_rows, top_n=10)
    plot_subject_accuracy(subject_accuracy_plot_path, best_result["per_subject_rows"])
    plot_subject_metric_comparison(subject_metric_plot_path, best_result["per_subject_rows"])
    write_best_summary(best_summary_path, best_result, len(combinations), ranked_rows)

    print("\n" + "=" * 70)
    print("Best overall LR configuration")
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
