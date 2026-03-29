import json
import os

import numpy as np
import optuna

from models.deep_learning_utils import (
    DEVICE,
    plot_subject_accuracy,
    plot_subject_metrics,
    plot_top_combinations,
    plot_train_loss_curves,
    run_loso_deep,
    save_confusion_matrix_csv,
    save_confusion_matrix_plot,
    set_seed,
    write_csv,
)
from models.deep_model_architectures import EEGTransformer


DATA_DIR = os.path.join("models", "preprocessing_result")
RESULTS_DIR = os.path.join("models", "transformer", "transformer_optuna")

X = np.load(os.path.join(DATA_DIR, "X.npy")).astype(np.float32)
y = np.load(os.path.join(DATA_DIR, "y.npy")).astype(np.int64)
subjects = np.load(os.path.join(DATA_DIR, "subjects.npy"))

os.makedirs(RESULTS_DIR, exist_ok=True)

print("Dataset shape:", X.shape)
print("Number of subjects:", len(np.unique(subjects)))
print("Results directory:", RESULTS_DIR)
print("Device:", DEVICE)


N_TRIALS = 3
FIXED_CONFIG = {
    "batch_size": 64,
    "epochs": 75,
    "weight_decay": 1e-3,
    "validation_fraction": 0.2,
    "early_stopping_patience": 15,
    "lr_scheduler_patience": 5,
    "lr_scheduler_factor": 0.7,
    "use_class_weight": True,
    "seed": 42,
}


def build_model(config, n_channels, n_samples):
    return EEGTransformer(
        n_channels=n_channels,
        d_model=config["d_model"],
        nhead=config["nhead"],
        num_layers=config["num_layers"],
        dim_feedforward=config["dim_feedforward"],
        dropout_rate=config["dropout_rate"],
    )


def suggest_params(trial):
    # d_model = trial.suggest_categorical("d_model", [16, 32, 64])
    d_model = trial.suggest_categorical("d_model", [16, 32])
    # nhead = trial.suggest_categorical("nhead", [2, 8])
    nhead = trial.suggest_categorical("nhead", [8])

    if d_model % nhead != 0:
        raise optuna.TrialPruned()
    
    return {
        "d_model": d_model,
        "nhead": nhead,
        "num_layers": trial.suggest_categorical("num_layers", [1, 3]),
        "dim_feedforward": trial.suggest_categorical("dim_feedforward", [256]),
        # "dim_feedforward": trial.suggest_categorical("dim_feedforward", [128, 256]),
        "dropout_rate": trial.suggest_categorical("dropout_rate", [0.5]),
        # "dropout_rate": trial.suggest_categorical("dropout_rate", [0.3, 0.5]),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-3, log=True),
        **FIXED_CONFIG,
    }


def write_best_summary(path, best_result, total_trials, top_rows):
    lines = [
        "TRANSFORMER OPTUNA TUNING SUMMARY",
        "=" * 70,
        f"Total Optuna trials evaluated: {total_trials}",
        "",
        "Best overall parameter combination",
        f"d_model: {best_result['d_model']}",
        f"nhead: {best_result['nhead']}",
        f"num_layers: {best_result['num_layers']}",
        f"dim_feedforward: {best_result['dim_feedforward']}",
        f"dropout_rate: {best_result['dropout_rate']}",
        f"learning_rate: {best_result['learning_rate']}",
        "",
        "Best model final metrics",
        f"mean_accuracy: {best_result['mean_accuracy']:.4f}",
        f"std_accuracy: {best_result['std_accuracy']:.4f}",
        f"mean_f1: {best_result['mean_f1']:.4f}",
        f"mean_auc: {best_result['mean_auc']:.4f}",
        f"mean_final_train_loss: {best_result['mean_final_train_loss']:.4f}",
        f"mean_best_val_loss: {best_result['mean_best_val_loss']:.4f}",
        f"mean_best_epoch: {best_result['mean_best_epoch']:.2f}",
        f"mean_stopped_epoch: {best_result['mean_stopped_epoch']:.2f}",
        "",
        "Top ranked trials",
    ]
    for row in top_rows[:3]:
        lines.append(
            (
                f"rank {row['rank']}: d_model={row['d_model']}, nhead={row['nhead']}, "
                f"num_layers={row['num_layers']}, dim_feedforward={row['dim_feedforward']}, "
                f"dropout_rate={row['dropout_rate']}, learning_rate={row['learning_rate']}, "
                f"mean_accuracy={row['mean_accuracy']:.4f}, mean_f1={row['mean_f1']:.4f}, "
                f"mean_auc={row['mean_auc']:.4f}"
            )
        )
    with open(path, "w", encoding="utf-8") as text_file:
        text_file.write("\n".join(lines))


if __name__ == "__main__":
    set_seed(42)

    trial_summaries = []
    fold_rows = []
    loss_rows = []
    best_result_holder = {"result": None}

    def objective(trial):
        config = suggest_params(trial)
        progress_label = f"trial {trial.number + 1}/{N_TRIALS}"
        result = run_loso_deep(
            X,
            y,
            subjects,
            config=config,
            build_model=build_model,
            add_channel_dim=True,
            progress_label=progress_label,
        )

        summary_row = {
            "rank": 0,
            "trial_number": trial.number,
            "d_model": config["d_model"],
            "nhead": config["nhead"],
            "num_layers": config["num_layers"],
            "dim_feedforward": config["dim_feedforward"],
            "dropout_rate": config["dropout_rate"],
            "learning_rate": config["learning_rate"],
            "batch_size": config["batch_size"],
            "epochs": config["epochs"],
            "weight_decay": config["weight_decay"],
            "validation_fraction": config["validation_fraction"],
            "early_stopping_patience": config["early_stopping_patience"],
            "lr_scheduler_patience": config["lr_scheduler_patience"],
            "lr_scheduler_factor": config["lr_scheduler_factor"],
            "use_class_weight": config["use_class_weight"],
            "seed": config["seed"],
            "mean_accuracy": result["mean_accuracy"],
            "std_accuracy": result["std_accuracy"],
            "mean_f1": result["mean_f1"],
            "mean_auc": result["mean_auc"],
            "mean_train_time_sec": result["mean_train_time_sec"],
            "mean_inference_time_sec": result["mean_inference_time_sec"],
            "mean_inference_ms_per_sample": result["mean_inference_ms_per_sample"],
            "mean_model_size_mb": result["mean_model_size_mb"],
            "mean_num_parameters": result["mean_num_parameters"],
            "mean_train_memory_delta_mb": result["mean_train_memory_delta_mb"],
            "mean_peak_gpu_memory_mb": result["mean_peak_gpu_memory_mb"],
            "mean_final_train_loss": result["mean_final_train_loss"],
            "mean_best_val_loss": result["mean_best_val_loss"],
            "mean_best_epoch": result["mean_best_epoch"],
            "mean_stopped_epoch": result["mean_stopped_epoch"],
        }
        trial_summaries.append(summary_row)

        for fold_row in result["fold_rows"]:
            fold_rows.append({"trial_number": trial.number, **config, **fold_row})
        for loss_row in result["loss_rows"]:
            loss_rows.append({"trial_number": trial.number, **loss_row})

        if (
            best_result_holder["result"] is None
            or result["mean_accuracy"] > best_result_holder["result"]["mean_accuracy"]
        ):
            best_result_holder["result"] = dict(result)
            best_result_holder["result"].update(config)
            best_result_holder["result"]["trial_number"] = trial.number

        trial.set_user_attr("mean_f1", result["mean_f1"])
        trial.set_user_attr("mean_auc", result["mean_auc"])
        return result["mean_accuracy"]

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=N_TRIALS)
    best_result = best_result_holder["result"]

    ranked_rows = sorted(
        trial_summaries,
        key=lambda row: (row["mean_accuracy"], row["mean_f1"], row["mean_auc"]),
        reverse=True,
    )
    for rank, row in enumerate(ranked_rows, 1):
        row["rank"] = rank

    best_row = ranked_rows[0]
    best_fold_rows = [row for row in fold_rows if row["trial_number"] == best_row["trial_number"]]

    write_csv(os.path.join(RESULTS_DIR, "all_trials_summary.csv"), ranked_rows, list(ranked_rows[0].keys()))
    write_csv(os.path.join(RESULTS_DIR, "all_trials_fold_results.csv"), fold_rows, list(fold_rows[0].keys()))
    write_csv(os.path.join(RESULTS_DIR, "all_trials_train_loss.csv"), loss_rows, list(loss_rows[0].keys()))
    write_csv(os.path.join(RESULTS_DIR, "top_10_trials.csv"), ranked_rows[:10], list(ranked_rows[0].keys()))
    write_csv(
        os.path.join(RESULTS_DIR, "best_configuration_per_subject.csv"),
        sorted(best_fold_rows, key=lambda row: row["subject"]),
        list(best_fold_rows[0].keys()),
    )

    save_confusion_matrix_csv(
        os.path.join(RESULTS_DIR, "best_confusion_matrix.csv"),
        best_result["overall_confusion_matrix"],
    )
    save_confusion_matrix_plot(
        os.path.join(RESULTS_DIR, "best_confusion_matrix.png"),
        best_result["overall_confusion_matrix"],
        "Best Transformer Optuna Configuration: Overall LOSO Confusion Matrix",
    )
    plot_top_combinations(
        os.path.join(RESULTS_DIR, "top_10_trials.png"),
        ranked_rows,
        top_n=10,
        model_label="Transformer Optuna Trials",
        label_key="d_model",
    )
    plot_subject_accuracy(
        os.path.join(RESULTS_DIR, "best_per_subject_accuracy.png"),
        best_result["per_subject_rows"],
        model_label="Transformer Optuna",
    )
    plot_subject_metrics(
        os.path.join(RESULTS_DIR, "best_per_subject_metrics.png"),
        best_result["per_subject_rows"],
        model_label="Transformer Optuna",
    )
    plot_train_loss_curves(
        os.path.join(RESULTS_DIR, "all_trials_train_loss.png"),
        loss_rows,
        model_label="Transformer Optuna",
        group_key="d_model",
    )
    write_best_summary(
        os.path.join(RESULTS_DIR, "best_configuration_summary.txt"),
        best_result,
        len(ranked_rows),
        ranked_rows,
    )

    with open(os.path.join(RESULTS_DIR, "study_metadata.json"), "w", encoding="utf-8") as json_file:
        json.dump(
            {
                "optimizer": "optuna",
                "n_trials": N_TRIALS,
                "best_trial_number": study.best_trial.number,
                "best_value": study.best_value,
                "best_params": study.best_trial.params,
            },
            json_file,
            indent=2,
        )

    print("Saved Optuna trial summary to:", os.path.join(RESULTS_DIR, "all_trials_summary.csv"))
    print("Saved Optuna fold results to:", os.path.join(RESULTS_DIR, "all_trials_fold_results.csv"))
    print("Saved Optuna metadata to:", os.path.join(RESULTS_DIR, "study_metadata.json"))
