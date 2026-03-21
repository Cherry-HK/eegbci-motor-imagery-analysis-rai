import os

import numpy as np
from braindecode.models import EEGNetv4

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
RESULTS_DIR = os.path.join("models", "eegnet", "eegnet_testing")

X = np.load(os.path.join(DATA_DIR, "X.npy")).astype(np.float32)
y = np.load(os.path.join(DATA_DIR, "y.npy")).astype(np.int64)
subjects = np.load(os.path.join(DATA_DIR, "subjects.npy"))

os.makedirs(RESULTS_DIR, exist_ok=True)

print("Dataset shape:", X.shape)
print("Number of subjects:", len(np.unique(subjects)))
print("Results directory:", RESULTS_DIR)
print("Device:", DEVICE)


def build_model(config, n_channels, n_samples):
    return EEGNetv4(
        n_chans=n_channels,
        n_outputs=2,
        n_times=n_samples,
        final_conv_length="auto",
        pool_mode="mean",
        F1=config["f1"],
        D=config["depth_multiplier"],
        F2=config["f2"],
        kernel_length=config["kernel_length"],
        drop_prob=config["dropout_rate"],
    )


def sanitize_config(config):
    cleaned = dict(config)
    cleaned["f2"] = cleaned["f1"] * cleaned["depth_multiplier"]
    return cleaned


SEARCH_CONFIG = {
    "f1": [4, 8, 16],
    "depth_multiplier": [1, 2],
    "kernel_length": [64],
    "dropout_rate": [0.25],
    "learning_rate": [1e-4, 1e-3],
    "batch_size": [32],
    "epochs": [20],
    "weight_decay": [0.0],
}


if __name__ == "__main__":
    set_seed(42)
    combinations = build_search_space(SEARCH_CONFIG, sanitize_fn=sanitize_config)
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
            add_channel_dim=False,
            progress_label=progress_label,
        )

        row = {
            "rank": 0,
            "f1": config["f1"],
            "depth_multiplier": config["depth_multiplier"],
            "f2": config["f2"],
            "kernel_length": config["kernel_length"],
            "dropout_rate": config["dropout_rate"],
            "learning_rate": config["learning_rate"],
            "batch_size": config["batch_size"],
            "epochs": config["epochs"],
            "weight_decay": config["weight_decay"],
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
        "EEGNet Best Configuration Confusion Matrix",
    )
    plot_top_combinations(
        os.path.join(RESULTS_DIR, "top_10_combinations.png"),
        summary_rows,
        top_n=10,
        model_label="EEGNet",
        label_key="f1",
    )
    plot_subject_accuracy(
        os.path.join(RESULTS_DIR, "best_per_subject_accuracy.png"),
        best_result["per_subject_rows"],
        model_label="EEGNet",
    )
    plot_subject_metrics(
        os.path.join(RESULTS_DIR, "best_per_subject_metrics.png"),
        best_result["per_subject_rows"],
        model_label="EEGNet",
    )
    write_summary_text(
        os.path.join(RESULTS_DIR, "best_configuration_summary.txt"),
        "EEGNet",
        best_result,
        len(combinations),
    )
    plot_train_loss_curves(
        os.path.join(RESULTS_DIR, "all_combinations_train_loss.png"),
        loss_rows,
        model_label="EEGNet",
        group_key="f1",
    )

    print("Saved EEGNet testing outputs to:", RESULTS_DIR)
