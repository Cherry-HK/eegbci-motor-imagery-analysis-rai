import csv
import itertools
import os

import matplotlib
import numpy as np
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = os.path.join("models", "preprocessing_result")
RESULTS_DIR = os.path.join("models", "lda", "lda_testing")
X = np.load(os.path.join(DATA_DIR, "X.npy"))
y = np.load(os.path.join(DATA_DIR, "y.npy"))
subjects = np.load(os.path.join(DATA_DIR, "subjects.npy"))
os.makedirs(RESULTS_DIR, exist_ok=True)


def sanitize_param_grid(param_grid):
    cleaned = dict(param_grid)
    if cleaned["solver"] == "svd":
        cleaned["shrinkage"] = "unused"
    elif cleaned["solver"] not in {"lsqr", "eigen"}:
        return None
    return cleaned


def build_search_space(search_config):
    keys = list(search_config.keys())
    combos, seen = [], set()
    for vals in itertools.product(*[search_config[k] for k in keys]):
        combo = sanitize_param_grid(dict(zip(keys, vals)))
        if combo is None:
            continue
        key = tuple((k, combo[k]) for k in sorted(combo))
        if key in seen:
            continue
        seen.add(key)
        combos.append(combo)
    return combos


def run_loso_lda(X, y, subjects, *, solver="lsqr", shrinkage="auto", csp_components=6, progress_label=""):
    logo = LeaveOneGroupOut()
    accuracies, f1_scores, auc_scores, conf_matrices = [], [], [], []
    fold_rows, all_y_true, all_y_pred = [], [], []
    total_folds = len(np.unique(subjects))
    for fold, (train_idx, test_idx) in enumerate(logo.split(X, y, groups=subjects), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        test_subject = int(subjects[test_idx][0])
        print(f"[{progress_label}] Fold {fold}/{total_folds} (test subject {test_subject})")
        csp = CSP(n_components=csp_components, reg="ledoit_wolf", log=True, norm_trace=False)
        X_train_csp = csp.fit_transform(X_train, y_train)
        X_test_csp = csp.transform(X_test)
        model = LinearDiscriminantAnalysis(solver=solver, shrinkage=None if shrinkage == "unused" else shrinkage)
        model.fit(X_train_csp, y_train)
        y_pred = model.predict(X_test_csp)
        y_score = model.decision_function(X_test_csp)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_score)
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        accuracies.append(acc); f1_scores.append(f1); auc_scores.append(auc); conf_matrices.append(cm)
        all_y_true.extend(y_test.tolist()); all_y_pred.extend(y_pred.tolist())
        fold_rows.append({"fold": fold, "subject": test_subject, "accuracy": acc, "f1_score": f1, "roc_auc": auc, "support": int(len(y_test))})
    return {
        "solver": solver, "shrinkage": shrinkage, "csp_components": csp_components,
        "mean_accuracy": float(np.mean(accuracies)), "std_accuracy": float(np.std(accuracies)),
        "mean_f1": float(np.mean(f1_scores)), "mean_auc": float(np.mean(auc_scores)),
        "overall_confusion_matrix": confusion_matrix(all_y_true, all_y_pred, labels=[0, 1]),
        "fold_rows": fold_rows, "per_subject_rows": sorted(fold_rows, key=lambda row: row["accuracy"], reverse=True),
        "best_subject": max(fold_rows, key=lambda row: row["accuracy"]), "worst_subject": min(fold_rows, key=lambda row: row["accuracy"]),
    }


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_confusion_matrix_csv(path, matrix):
    np.savetxt(path, matrix, delimiter=",", fmt="%d")


def save_confusion_matrix_plot(path, matrix, title):
    plt.figure(figsize=(5, 4))
    plt.imshow(matrix, interpolation="nearest", cmap="Blues")
    plt.title(title)
    plt.colorbar()
    ticks = np.arange(2)
    plt.xticks(ticks, ["Left", "Right"])
    plt.yticks(ticks, ["Left", "Right"])
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    threshold = matrix.max() / 2.0 if matrix.size else 0.0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            plt.text(j, i, f"{int(matrix[i, j])}", ha="center", va="center", color="white" if matrix[i, j] > threshold else "black")
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
    plt.title(f"Top {len(top_rows)} LDA Hyperparameter Combinations")
    plt.xticks(rotation=30, ha="right")
    plt.ylim(0.0, 1.0)
    for bar, score in zip(bars, scores):
        plt.text(bar.get_x() + bar.get_width() / 2.0, score + 0.01, f"{score:.3f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_subject_accuracy(path, per_subject_rows, title):
    rows = sorted(per_subject_rows, key=lambda row: row["subject"])
    labels = [str(row["subject"]) for row in rows]
    scores = [row["accuracy"] for row in rows]
    plt.figure(figsize=(10, 5))
    bars = plt.bar(labels, scores, color="#16a34a")
    plt.xlabel("Test Subject")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.ylim(0.0, 1.0)
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    for bar, score in zip(bars, scores):
        plt.text(bar.get_x() + bar.get_width() / 2.0, score + 0.01, f"{score:.2f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_subject_metric_comparison(path, per_subject_rows, title):
    rows = sorted(per_subject_rows, key=lambda row: row["subject"])
    axis = np.arange(len(rows))
    plt.figure(figsize=(11, 5))
    plt.plot(axis, [row["accuracy"] for row in rows], marker="o", linewidth=2, label="Accuracy")
    plt.plot(axis, [row["f1_score"] for row in rows], marker="s", linewidth=2, label="F1-score")
    plt.plot(axis, [row["roc_auc"] for row in rows], marker="^", linewidth=2, label="ROC-AUC")
    plt.xticks(axis, [str(row["subject"]) for row in rows])
    plt.ylim(0.0, 1.0)
    plt.xlabel("Test Subject")
    plt.ylabel("Score")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


SEARCH_CONFIG = {"solver": ["svd", "lsqr", "eigen"], "shrinkage": [None, "auto", 0.1, 0.5], "csp_components": [6]}


if __name__ == "__main__":
    combinations = build_search_space(SEARCH_CONFIG)
    summary_rows, fold_rows = [], []
    best_result = None
    for index, combo in enumerate(combinations, 1):
        result = run_loso_lda(X, y, subjects, progress_label=f"combo {index}/{len(combinations)}", **combo)
        summary_rows.append({"solver": result["solver"], "shrinkage": result["shrinkage"], "csp_components": result["csp_components"], "mean_accuracy": result["mean_accuracy"], "std_accuracy": result["std_accuracy"], "mean_f1": result["mean_f1"], "mean_auc": result["mean_auc"]})
        for fold_row in result["fold_rows"]:
            fold_rows.append({"solver": result["solver"], "shrinkage": result["shrinkage"], "csp_components": result["csp_components"], **fold_row})
        if best_result is None or result["mean_accuracy"] > best_result["mean_accuracy"]:
            best_result = result
    ranked_rows = sorted(summary_rows, key=lambda row: (row["mean_accuracy"], row["mean_f1"], row["mean_auc"]), reverse=True)
    for rank, row in enumerate(ranked_rows, 1):
        row["rank"] = rank
    best_row = ranked_rows[0]
    best_fold_rows = [row for row in fold_rows if row["solver"] == best_row["solver"] and row["shrinkage"] == best_row["shrinkage"] and row["csp_components"] == best_row["csp_components"]]
    write_csv(os.path.join(RESULTS_DIR, "all_combinations_summary.csv"), ranked_rows, ["rank", "solver", "shrinkage", "csp_components", "mean_accuracy", "std_accuracy", "mean_f1", "mean_auc"])
    write_csv(os.path.join(RESULTS_DIR, "all_combinations_fold_results.csv"), fold_rows, ["solver", "shrinkage", "csp_components", "fold", "subject", "accuracy", "f1_score", "roc_auc", "support"])
    write_csv(os.path.join(RESULTS_DIR, "top_10_combinations.csv"), ranked_rows[:10], ["rank", "solver", "shrinkage", "csp_components", "mean_accuracy", "std_accuracy", "mean_f1", "mean_auc"])
    write_csv(os.path.join(RESULTS_DIR, "best_configuration_per_subject.csv"), sorted(best_fold_rows, key=lambda row: row["subject"]), ["solver", "shrinkage", "csp_components", "fold", "subject", "accuracy", "f1_score", "roc_auc", "support"])
    save_confusion_matrix_csv(os.path.join(RESULTS_DIR, "best_confusion_matrix.csv"), best_result["overall_confusion_matrix"])
    save_confusion_matrix_plot(os.path.join(RESULTS_DIR, "best_confusion_matrix.png"), best_result["overall_confusion_matrix"], "Best LDA Configuration: Overall LOSO Confusion Matrix")
    plot_top_combinations(os.path.join(RESULTS_DIR, "top_10_combinations.png"), ranked_rows, top_n=10)
    plot_subject_accuracy(os.path.join(RESULTS_DIR, "best_per_subject_accuracy.png"), best_result["per_subject_rows"], "Best LDA Configuration: LOSO Accuracy per Subject")
    plot_subject_metric_comparison(os.path.join(RESULTS_DIR, "best_per_subject_metrics.png"), best_result["per_subject_rows"], "Best LDA Configuration: Per-Subject LOSO Metrics")
    with open(os.path.join(RESULTS_DIR, "best_configuration_summary.txt"), "w", encoding="utf-8") as handle:
        handle.write(
            "\n".join(
                [
                    "LDA combined-search summary",
                    "=" * 40,
                    f"Total combinations tested: {len(combinations)}",
                    "",
                    "Best configuration:",
                    f"- solver: {best_result['solver']}",
                    f"- shrinkage: {best_result['shrinkage']}",
                    f"- csp_components: {best_result['csp_components']}",
                    f"- mean_accuracy: {best_result['mean_accuracy']:.4f}",
                    f"- std_accuracy: {best_result['std_accuracy']:.4f}",
                    f"- mean_f1: {best_result['mean_f1']:.4f}",
                    f"- mean_auc: {best_result['mean_auc']:.4f}",
                ]
            )
        )
