import csv
import json
import os

import matplotlib
import numpy as np
import optuna
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = os.path.join("models", "preprocessing_result")
RESULTS_DIR = os.path.join("models", "lda", "lda_optuna")
X = np.load(os.path.join(DATA_DIR, "X.npy"))
y = np.load(os.path.join(DATA_DIR, "y.npy"))
subjects = np.load(os.path.join(DATA_DIR, "subjects.npy"))
os.makedirs(RESULTS_DIR, exist_ok=True)

print("Dataset shape:", X.shape)
print("Number of subjects:", len(np.unique(subjects)))
print("Results directory:", RESULTS_DIR)

N_TRIALS = 30
OPTUNA_SEARCH_SPACE = {
    "solver": ["svd", "lsqr", "eigen"],
    "csp_components": [6],
}


def run_loso_lda(X, y, subjects, *, solver="lsqr", shrinkage="auto", csp_components=6, progress_label=""):
    logo = LeaveOneGroupOut()
    accuracies, f1_scores, auc_scores = [], [], []
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
        accuracies.append(acc)
        f1_scores.append(f1)
        auc_scores.append(auc)
        all_y_true.extend(y_test.tolist())
        all_y_pred.extend(y_pred.tolist())
        fold_rows.append({"fold": fold, "subject": test_subject, "accuracy": acc, "f1_score": f1, "roc_auc": auc, "support": int(len(y_test))})
    return {
        "solver": solver,
        "shrinkage": shrinkage,
        "csp_components": csp_components,
        "mean_accuracy": float(np.mean(accuracies)),
        "std_accuracy": float(np.std(accuracies)),
        "mean_f1": float(np.mean(f1_scores)),
        "mean_auc": float(np.mean(auc_scores)),
        "overall_confusion_matrix": confusion_matrix(all_y_true, all_y_pred, labels=[0, 1]),
        "fold_rows": fold_rows,
        "per_subject_rows": sorted(fold_rows, key=lambda row: row["accuracy"], reverse=True),
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


def plot_top_trials(path, rows, top_n=10):
    top_rows = rows[:top_n]
    labels = [f"#{row['rank']} {row['solver']}" for row in top_rows]
    scores = [row["mean_accuracy"] for row in top_rows]
    plt.figure(figsize=(11, 6))
    bars = plt.bar(labels, scores, color="#2563eb")
    plt.ylabel("Mean Accuracy")
    plt.xlabel("Trial Rank")
    plt.title(f"Top {len(top_rows)} LDA Optuna Trials")
    plt.xticks(rotation=30, ha="right")
    plt.ylim(0.0, 1.0)
    for bar, score in zip(bars, scores):
        plt.text(bar.get_x() + bar.get_width() / 2.0, score + 0.01, f"{score:.3f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_subject_accuracy(path, per_subject_rows, title):
    rows = sorted(per_subject_rows, key=lambda row: row["accuracy"], reverse=True)
    axis = np.arange(1, len(rows) + 1)
    scores = [row["accuracy"] for row in rows]
    plt.figure(figsize=(11, 5))
    plt.scatter(axis, scores, color="#16a34a", s=32, alpha=0.9)
    plt.plot(axis, scores, color="#16a34a", linewidth=1.2, alpha=0.45)
    plt.xlabel("Subjects Ranked by Accuracy")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.ylim(0.0, 1.0)
    plt.xlim(0, len(rows) + 1)
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    highlight_subjects = {row["subject"] for row in rows[:3] + rows[-3:]}
    for rank, row in enumerate(rows, start=1):
        if row["subject"] in highlight_subjects:
            plt.annotate(f"S{row['subject']}: {row['accuracy']:.2f}", (rank, row["accuracy"]), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_subject_metric_comparison(path, per_subject_rows, title):
    rows = sorted(per_subject_rows, key=lambda row: row["accuracy"], reverse=True)
    axis = np.arange(1, len(rows) + 1)
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    metric_specs = [
        ("Accuracy", [row["accuracy"] for row in rows], "#2563eb"),
        ("F1-score", [row["f1_score"] for row in rows], "#f97316"),
        ("ROC-AUC", [row["roc_auc"] for row in rows], "#16a34a"),
    ]
    for ax, (label, values, color) in zip(axes, metric_specs):
        ax.scatter(axis, values, color=color, s=24, alpha=0.85)
        ax.plot(axis, values, color=color, linewidth=1.0, alpha=0.35)
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel(label)
        ax.grid(True, linestyle="--", alpha=0.35)
    axes[0].set_title(title)
    axes[-1].set_xlabel("Subjects Ranked by Accuracy")
    axes[-1].set_xlim(0, len(rows) + 1)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def write_best_summary(path, best_result, total_trials, top_rows):
    lines = [
        "LDA OPTUNA TUNING SUMMARY",
        "=" * 70,
        f"Total Optuna trials evaluated: {total_trials}",
        "",
        "Best overall parameter combination",
        f"solver: {best_result['solver']}",
        f"shrinkage: {best_result['shrinkage']}",
        f"csp_components: {best_result['csp_components']}",
        "",
        "Best model final metrics",
        f"mean_accuracy: {best_result['mean_accuracy']:.4f}",
        f"std_accuracy: {best_result['std_accuracy']:.4f}",
        f"mean_f1: {best_result['mean_f1']:.4f}",
        f"mean_auc: {best_result['mean_auc']:.4f}",
        "",
        "Top ranked trials",
    ]
    for row in top_rows[:3]:
        lines.append(f"rank {row['rank']}: solver={row['solver']}, shrinkage={row['shrinkage']}, csp_components={row['csp_components']}, mean_accuracy={row['mean_accuracy']:.4f}, mean_f1={row['mean_f1']:.4f}, mean_auc={row['mean_auc']:.4f}")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def suggest_lda_params(trial):
    solver = trial.suggest_categorical("solver", OPTUNA_SEARCH_SPACE["solver"])
    shrinkage = "unused"
    if solver in {"lsqr", "eigen"}:
        shrinkage = trial.suggest_categorical("shrinkage_supported", ["auto", 0.1, 0.3, 0.5])
    return {
        "solver": solver,
        "shrinkage": shrinkage,
        "csp_components": trial.suggest_categorical("csp_components", OPTUNA_SEARCH_SPACE["csp_components"]),
    }


if __name__ == "__main__":
    trial_summaries, fold_rows = [], []
    best_result_holder = {"result": None}

    def objective(trial):
        config = suggest_lda_params(trial)
        result = run_loso_lda(X, y, subjects, progress_label=f"trial {trial.number + 1}/{N_TRIALS}", **config)
        trial_summaries.append({"rank": 0, "trial_number": trial.number, "solver": result["solver"], "shrinkage": result["shrinkage"], "csp_components": result["csp_components"], "mean_accuracy": result["mean_accuracy"], "std_accuracy": result["std_accuracy"], "mean_f1": result["mean_f1"], "mean_auc": result["mean_auc"]})
        for fold_row in result["fold_rows"]:
            fold_rows.append({"trial_number": trial.number, "solver": result["solver"], "shrinkage": result["shrinkage"], "csp_components": result["csp_components"], **fold_row})
        if best_result_holder["result"] is None or result["mean_accuracy"] > best_result_holder["result"]["mean_accuracy"]:
            best_result_holder["result"] = dict(result)
            best_result_holder["result"]["trial_number"] = trial.number
        trial.set_user_attr("mean_f1", result["mean_f1"])
        trial.set_user_attr("mean_auc", result["mean_auc"])
        return result["mean_accuracy"]

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=N_TRIALS)
    best_result = best_result_holder["result"]

    ranked_rows = sorted(trial_summaries, key=lambda row: (row["mean_accuracy"], row["mean_f1"], row["mean_auc"]), reverse=True)
    for rank, row in enumerate(ranked_rows, 1):
        row["rank"] = rank
    best_row = ranked_rows[0]
    best_fold_rows = [row for row in fold_rows if row["trial_number"] == best_row["trial_number"]]

    write_csv(os.path.join(RESULTS_DIR, "all_trials_summary.csv"), ranked_rows, ["rank", "trial_number", "solver", "shrinkage", "csp_components", "mean_accuracy", "std_accuracy", "mean_f1", "mean_auc"])
    write_csv(os.path.join(RESULTS_DIR, "all_trials_fold_results.csv"), fold_rows, ["trial_number", "solver", "shrinkage", "csp_components", "fold", "subject", "accuracy", "f1_score", "roc_auc", "support"])
    write_csv(os.path.join(RESULTS_DIR, "top_10_trials.csv"), ranked_rows[:10], ["rank", "trial_number", "solver", "shrinkage", "csp_components", "mean_accuracy", "std_accuracy", "mean_f1", "mean_auc"])
    write_csv(os.path.join(RESULTS_DIR, "best_configuration_per_subject.csv"), sorted(best_fold_rows, key=lambda row: row["subject"]), ["trial_number", "solver", "shrinkage", "csp_components", "fold", "subject", "accuracy", "f1_score", "roc_auc", "support"])
    save_confusion_matrix_csv(os.path.join(RESULTS_DIR, "best_confusion_matrix.csv"), best_result["overall_confusion_matrix"])
    save_confusion_matrix_plot(os.path.join(RESULTS_DIR, "best_confusion_matrix.png"), best_result["overall_confusion_matrix"], "Best LDA Optuna Configuration: Overall LOSO Confusion Matrix")
    plot_top_trials(os.path.join(RESULTS_DIR, "top_10_trials.png"), ranked_rows, top_n=10)
    plot_subject_accuracy(os.path.join(RESULTS_DIR, "best_per_subject_accuracy.png"), best_result["per_subject_rows"], "Best LDA Optuna Configuration: LOSO Accuracy per Subject")
    plot_subject_metric_comparison(os.path.join(RESULTS_DIR, "best_per_subject_metrics.png"), best_result["per_subject_rows"], "Best LDA Optuna Configuration: Per-Subject LOSO Metrics")
    write_best_summary(os.path.join(RESULTS_DIR, "best_configuration_summary.txt"), best_result, len(ranked_rows), ranked_rows)
    with open(os.path.join(RESULTS_DIR, "study_metadata.json"), "w", encoding="utf-8") as jsonfile:
        json.dump({"optimizer": "optuna", "n_trials": N_TRIALS, "best_trial_number": study.best_trial.number, "best_value": study.best_value, "best_params": study.best_trial.params}, jsonfile, indent=2)

    print("Saved Optuna trial summary to:", os.path.join(RESULTS_DIR, "all_trials_summary.csv"))
    print("Saved Optuna fold results to:", os.path.join(RESULTS_DIR, "all_trials_fold_results.csv"))
    print("Saved Optuna metadata to:", os.path.join(RESULTS_DIR, "study_metadata.json"))
