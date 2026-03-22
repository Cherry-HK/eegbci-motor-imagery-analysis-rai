import csv
import os

import matplotlib
import numpy as np

from models.deep_learning_utils import DEVICE, run_loso_deep, set_seed
from models.deep_model_architectures import EEGShallowConvNet

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DATA_DIR = os.path.join("models", "preprocessing_result")
RESULTS_DIR = os.path.join("models", "shallowconvnet", "shallowconvnet_parameter_study")

X = np.load(os.path.join(DATA_DIR, "X.npy")).astype(np.float32)
y = np.load(os.path.join(DATA_DIR, "y.npy")).astype(np.int64)
subjects = np.load(os.path.join(DATA_DIR, "subjects.npy"))

os.makedirs(RESULTS_DIR, exist_ok=True)

print("Dataset shape:", X.shape)
print("Number of subjects:", len(np.unique(subjects)))
print("Results directory:", RESULTS_DIR)
print("Device:", DEVICE)


def build_model(config, n_channels, n_samples):
    return EEGShallowConvNet(
        n_channels=n_channels,
        n_samples=n_samples,
        temporal_filters=config["temporal_filters"],
        filter_length=config["filter_length"],
        pool_length=config["pool_length"],
        pool_stride=config["pool_stride"],
        dropout_rate=config["dropout_rate"],
    )


def write_csv(path, rows, fieldnames):
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
    plt.title(f"ShallowConvNet Parameter Study: {parameter_name}")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_train_loss(path, parameter_name, loss_rows):
    plt.figure(figsize=(8, 5))
    for value in dict.fromkeys(row["parameter_value"] for row in loss_rows):
        value_rows = [row for row in loss_rows if row["parameter_value"] == value]
        epoch_values = sorted(set(row["epoch"] for row in value_rows))
        mean_epoch_losses = []
        for epoch in epoch_values:
            epoch_rows = [
                row["train_loss"]
                for row in value_rows
                if row["epoch"] == epoch and row["train_loss"] is not None
            ]
            mean_epoch_losses.append(float(np.mean(epoch_rows)) if epoch_rows else np.nan)
        plt.plot(epoch_values, mean_epoch_losses, marker="o", linewidth=2, label=str(value))

    plt.xlabel("Epoch")
    plt.ylabel("Mean Train Loss")
    plt.title(f"ShallowConvNet Train Loss: {parameter_name}")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(title=parameter_name)
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
    loss_rows = []
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
        config[parameter_name] = value

        progress_label = f"{parameter_name}={value} [{value_index}/{total_values}]"
        print(f"\nRunning {progress_label}")
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
            "parameter_name": parameter_name,
            "parameter_value": value,
            "temporal_filters": config["temporal_filters"],
            "filter_length": config["filter_length"],
            "pool_length": config["pool_length"],
            "pool_stride": config["pool_stride"],
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
        summary_rows.append(summary_row)

        for fold_row in result["fold_rows"]:
            fold_rows.append(
                {
                    "parameter_name": parameter_name,
                    "parameter_value": value,
                    "temporal_filters": config["temporal_filters"],
                    "filter_length": config["filter_length"],
                    "pool_length": config["pool_length"],
                    "pool_stride": config["pool_stride"],
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
                    **fold_row,
                }
            )
        for loss_row in result["loss_rows"]:
            loss_rows.append(
                {
                    "parameter_name": parameter_name,
                    "parameter_value": value,
                    "temporal_filters": config["temporal_filters"],
                    "filter_length": config["filter_length"],
                    "pool_length": config["pool_length"],
                    "pool_stride": config["pool_stride"],
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
                    **loss_row,
                }
            )

        plot_labels.append(str(value))
        plot_scores.append(result["mean_accuracy"])

        print(f"Mean Accuracy: {result['mean_accuracy'] * 100:.2f}% +/- {result['std_accuracy'] * 100:.2f}%")
        print(f"Mean F1 Score: {result['mean_f1']:.3f}")
        print(f"Mean ROC-AUC: {result['mean_auc']:.3f}")
        print(f"Mean Train Time (s): {result['mean_train_time_sec']:.2f}")
        print(f"Mean Inference ms/sample: {result['mean_inference_ms_per_sample']:.4f}")
        print(f"Mean Model Size (MB): {result['mean_model_size_mb']:.3f}")

        safe_value = make_safe_filename(value)
        confusion_csv_path = os.path.join(parameter_dir, f"confusion_{parameter_name}_{safe_value}.csv")
        confusion_plot_path = os.path.join(parameter_dir, f"confusion_{parameter_name}_{safe_value}.png")
        save_confusion_matrix_csv(confusion_csv_path, result["overall_confusion_matrix"])
        save_confusion_matrix_plot(
            confusion_plot_path,
            result["overall_confusion_matrix"],
            f"Overall LOSO Confusion Matrix: {parameter_name}={value}",
        )

    summary_csv_path = os.path.join(parameter_dir, f"summary_{parameter_name}.csv")
    fold_csv_path = os.path.join(parameter_dir, f"fold_results_{parameter_name}.csv")
    loss_csv_path = os.path.join(parameter_dir, f"train_loss_{parameter_name}.csv")
    plot_path = os.path.join(parameter_dir, f"plot_{parameter_name}.png")
    loss_plot_path = os.path.join(parameter_dir, f"train_loss_{parameter_name}.png")

    write_csv(summary_csv_path, summary_rows, list(summary_rows[0].keys()))
    write_csv(fold_csv_path, fold_rows, list(fold_rows[0].keys()))
    write_csv(loss_csv_path, loss_rows, list(loss_rows[0].keys()))
    plot_parameter_results(
        plot_path,
        parameter_name,
        plot_labels,
        plot_scores,
    )
    plot_train_loss(
        loss_plot_path,
        parameter_name,
        loss_rows,
    )

    best_row = max(summary_rows, key=lambda row: row["mean_accuracy"])
    print("\nBest result for", parameter_name)
    print(best_row)
    print("Saved summary to:", summary_csv_path)
    print("Saved fold results to:", fold_csv_path)
    print("Saved train loss to:", loss_csv_path)
    print("Saved train loss plot to:", loss_plot_path)
    print("Saved plot to:", plot_path)


BASE_CONFIG = {
    "temporal_filters": 40,
    "filter_length": 25,
    "pool_length": 75,
    "pool_stride": 15,
    "dropout_rate": 0.5,
    "learning_rate": 1e-3,
    "batch_size": 32,
    "epochs": 50,
    "weight_decay": 0.0,
    "validation_fraction": 0.1,
    "early_stopping_patience": 10,
    "lr_scheduler_patience": 5,
    "lr_scheduler_factor": 0.5,
    "use_class_weight": True,
    "seed": 42,
}

PARAMETER_STUDIES = [
    {"name": "temporal_filters", "values": [20, 40, 60], "overrides": {}},
    {"name": "filter_length", "values": [15, 25, 35], "overrides": {"temporal_filters": 40}},
    {"name": "pool_length", "values": [50, 75, 100], "overrides": {"temporal_filters": 40}},
    {"name": "dropout_rate", "values": [0.25, 0.5, 0.75], "overrides": {"temporal_filters": 40}},
    {"name": "learning_rate", "values": [1e-4, 1e-3, 1e-2], "overrides": {"temporal_filters": 40}},
]


if __name__ == "__main__":
    set_seed(42)
    for study in PARAMETER_STUDIES:
        study_config = BASE_CONFIG.copy()
        study_config.update(study["overrides"])
        run_parameter_study(study["name"], study["values"], study_config)
