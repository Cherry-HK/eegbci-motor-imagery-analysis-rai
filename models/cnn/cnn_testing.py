import os

import numpy as np

from models.deep_model_architectures import EEGNetLite
from models.deep_learning_utils import (
    DEVICE,
    build_search_space,
    plot_subject_accuracy,
    plot_subject_metrics,
    plot_top_combinations,
    plot_train_loss_curves,
    run_loso_deep,
    save_confusion_matrix_csv,
    save_confusion_matrix_plot,
    set_seed,
    write_csv,
    write_summary_text,
)


DATA_DIR = os.path.join("models", "preprocessing_result")
RESULTS_DIR = os.path.join("models", "cnn", "cnn_testing")

X = np.load(os.path.join(DATA_DIR, "X.npy")).astype(np.float32)
y = np.load(os.path.join(DATA_DIR, "y.npy")).astype(np.int64)
subjects = np.load(os.path.join(DATA_DIR, "subjects.npy"))

os.makedirs(RESULTS_DIR, exist_ok=True)

print("Dataset shape:", X.shape)
print("Number of subjects:", len(np.unique(subjects)))
print("Results directory:", RESULTS_DIR)
print("Device:", DEVICE)


def build_model(config, n_channels, n_samples):
    return EEGNetLite(
        n_channels=n_channels,
        n_samples=n_samples,
        temporal_filters=config["temporal_filters"],
        depth_multiplier=config["depth_multiplier"],
        kernel_length=config["kernel_length"],
        dropout_rate=config["dropout_rate"],
    )


SEARCH_CONFIG = {
    "temporal_filters": [4, 8, 16],
    "depth_multiplier": [2],
    "kernel_length": [64],
    "dropout_rate": [0.25, 0.5],
    "learning_rate": [1e-4, 1e-3],
    "batch_size": [32],
    "epochs": [50],
    "weight_decay": [0.0],
    "validation_fraction": [0.1],
    "early_stopping_patience": [10],
    "lr_scheduler_patience": [5],
    "lr_scheduler_factor": [0.5],
    "use_class_weight": [True],
    "seed": [42],
}


if __name__ == "__main__":
    set_seed(42)
    combinations = build_search_space(SEARCH_CONFIG)
    print("Total combinations:", len(combinations))

    summary_rows = []
    fold_rows = []
    loss_rows = []
    best_result = None

    for combo_index, config in enumerate(combinations, 1):
        progress_label = f"combo {combo_index}/{len(combinations)}"
        result = run_loso_deep(
            X,
            y,
            subjects,
            config=config,
            build_model=build_model,
            add_channel_dim=True,
            progress_label=progress_label,
        )

        row = {
            "rank": 0,
            "temporal_filters": config["temporal_filters"],
            "depth_multiplier": config["depth_multiplier"],
            "kernel_length": config["kernel_length"],
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
        summary_rows.append(row)

        for fold_row in result["fold_rows"]:
            fold_with_config = dict(fold_row)
            fold_with_config.update(config)
            fold_rows.append(fold_with_config)

        loss_rows.extend(result["loss_rows"])

        if best_result is None or result["mean_accuracy"] > best_result["mean_accuracy"]:
            best_result = dict(result)
            best_result.update(config)

    summary_rows.sort(key=lambda row: (row["mean_accuracy"], row["mean_f1"], row["mean_auc"]), reverse=True)
    for rank, row in enumerate(summary_rows, 1):
        row["rank"] = rank

    top_10_rows = summary_rows[:10]

    write_csv(
        os.path.join(RESULTS_DIR, "all_combinations_summary.csv"),
        summary_rows,
        list(summary_rows[0].keys()),
    )
    write_csv(
        os.path.join(RESULTS_DIR, "all_combinations_fold_results.csv"),
        fold_rows,
        list(fold_rows[0].keys()),
    )
    write_csv(
        os.path.join(RESULTS_DIR, "all_combinations_train_loss.csv"),
        loss_rows,
        list(loss_rows[0].keys()),
    )
    write_csv(
        os.path.join(RESULTS_DIR, "top_10_combinations.csv"),
        top_10_rows,
        list(top_10_rows[0].keys()),
    )
    write_csv(
        os.path.join(RESULTS_DIR, "best_configuration_per_subject.csv"),
        best_result["per_subject_rows"],
        list(best_result["per_subject_rows"][0].keys()),
    )

    save_confusion_matrix_csv(
        os.path.join(RESULTS_DIR, "best_confusion_matrix.csv"),
        best_result["overall_confusion_matrix"],
    )
    save_confusion_matrix_plot(
        os.path.join(RESULTS_DIR, "best_confusion_matrix.png"),
        best_result["overall_confusion_matrix"],
        "CNN Best Configuration Confusion Matrix",
    )
    plot_top_combinations(
        os.path.join(RESULTS_DIR, "top_10_combinations.png"),
        summary_rows,
        top_n=10,
        model_label="CNN",
        label_key="temporal_filters",
    )
    plot_subject_accuracy(
        os.path.join(RESULTS_DIR, "best_per_subject_accuracy.png"),
        best_result["per_subject_rows"],
        model_label="CNN",
    )
    plot_subject_metrics(
        os.path.join(RESULTS_DIR, "best_per_subject_metrics.png"),
        best_result["per_subject_rows"],
        model_label="CNN",
    )
    write_summary_text(
        os.path.join(RESULTS_DIR, "best_configuration_summary.txt"),
        "CNN",
        best_result,
        len(combinations),
    )
    plot_train_loss_curves(
        os.path.join(RESULTS_DIR, "all_combinations_train_loss.png"),
        loss_rows,
        model_label="CNN",
        group_key="temporal_filters",
    )

    print("Saved CNN testing outputs to:", RESULTS_DIR)
