import csv
import os

import matplotlib
import numpy as np
from braindecode.models import EEGNetv4

from models.deep_learning_utils import (
    DEVICE,
    plot_subject_accuracy,
    plot_subject_metrics,
    plot_train_loss_curves,
    run_loso_deep,
    save_confusion_matrix_csv,
    save_confusion_matrix_plot,
    set_seed,
)

matplotlib.use("Agg")


DATA_DIR = os.path.join("models", "preprocessing_result")
RESULTS_DIR = os.path.join("models", "eegnet", "eegnet_training_hparams")

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


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_parameter_results(path, parameter_name, parameter_values, metric_values):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))
    plt.plot(parameter_values, metric_values, marker="o", linewidth=2)
    plt.xlabel(parameter_name)
    plt.ylabel("Mean Accuracy")
    plt.title(f"EEGNet Training Hyperparameter Study: {parameter_name}")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def make_safe_filename(value):
    safe_value = str(value)
    for old, new in [(" ", "_"), (".", "p"), ("-", "neg"), ("/", "_")]:
        safe_value = safe_value.replace(old, new)
    return safe_value


BASE_CONFIG = {
    "f1": 8,
    "depth_multiplier": 2,
    "f2": 16,
    "kernel_length": 64,
    "dropout_rate": 0.25,
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
    # {"name": "batch_size", "values": [16, 32, 64], "overrides": {}},
    # {"name": "epochs", "values": [50, 75, 100], "overrides": {}},
    # {"name": "weight_decay", "values": [0.0, 1e-4, 1e-3], "overrides": {}},
    # {"name": "validation_fraction", "values": [0.1, 0.15, 0.2], "overrides": {}},
    # {"name": "early_stopping_patience", "values": [5, 10, 15], "overrides": {}},
    # {"name": "lr_scheduler_patience", "values": [3, 5, 8], "overrides": {}},
    {"name": "lr_scheduler_factor", "values": [0.3, 0.5, 0.7], "overrides": {}},
    {"name": "use_class_weight", "values": [True, False], "overrides": {}},
]


def apply_parameter_value(config, parameter_name, value):
    if parameter_name not in config:
        raise ValueError(f"Unsupported parameter: {parameter_name}")
    config[parameter_name] = value
    config["f2"] = config["f1"] * config["depth_multiplier"]


def run_parameter_study(parameter_name, values, base_config):
    study_dir = os.path.join(RESULTS_DIR, parameter_name)
    os.makedirs(study_dir, exist_ok=True)

    summary_rows = []
    fold_rows = []
    loss_rows = []
    best_result = None
    plot_labels = []
    plot_scores = []

    print("\n" + "=" * 70)
    print(f"Testing training hyperparameter: {parameter_name}")
    print("=" * 70)

    for value_index, value in enumerate(values, 1):
        config = dict(base_config)
        apply_parameter_value(config, parameter_name, value)

        progress_label = f"{parameter_name}={value} [{value_index}/{len(values)}]"
        print(f"\nRunning {progress_label}")

        result = run_loso_deep(
            X,
            y,
            subjects,
            config=config,
            build_model=build_model,
            add_channel_dim=False,
            progress_label=progress_label,
        )

        summary_row = {
            "rank": 0,
            "parameter_name": parameter_name,
            "parameter_value": value,
            "f1": config["f1"],
            "depth_multiplier": config["depth_multiplier"],
            "f2": config["f2"],
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
        summary_rows.append(summary_row)

        for fold_row in result["fold_rows"]:
            fold_rows.append(
                {
                    "parameter_name": parameter_name,
                    "parameter_value": value,
                    **config,
                    **fold_row,
                }
            )

        for loss_row in result["loss_rows"]:
            loss_rows.append(
                {
                    "parameter_name": parameter_name,
                    "parameter_value": value,
                    **loss_row,
                }
            )

        if best_result is None or result["mean_accuracy"] > best_result["mean_accuracy"]:
            best_result = dict(result)
            best_result.update(config)

        plot_labels.append(str(value))
        plot_scores.append(result["mean_accuracy"])

        print(f"Mean Accuracy: {result['mean_accuracy'] * 100:.2f}% +/- {result['std_accuracy'] * 100:.2f}%")
        print(f"Mean F1 Score: {result['mean_f1']:.3f}")
        print(f"Mean ROC-AUC: {result['mean_auc']:.3f}")
        print(f"Mean Train Time (s): {result['mean_train_time_sec']:.2f}")

        safe_value = make_safe_filename(value)
        save_confusion_matrix_csv(
            os.path.join(study_dir, f"confusion_{parameter_name}_{safe_value}.csv"),
            result["overall_confusion_matrix"],
        )
        save_confusion_matrix_plot(
            os.path.join(study_dir, f"confusion_{parameter_name}_{safe_value}.png"),
            result["overall_confusion_matrix"],
            f"EEGNet Training Hyperparameter Study: {parameter_name}={value}",
        )

    summary_rows.sort(key=lambda row: (row["mean_accuracy"], row["mean_f1"], row["mean_auc"]), reverse=True)
    for rank, row in enumerate(summary_rows, 1):
        row["rank"] = rank

    write_csv(os.path.join(study_dir, f"summary_{parameter_name}.csv"), summary_rows, list(summary_rows[0].keys()))
    write_csv(os.path.join(study_dir, f"fold_results_{parameter_name}.csv"), fold_rows, list(fold_rows[0].keys()))
    write_csv(os.path.join(study_dir, f"train_loss_{parameter_name}.csv"), loss_rows, list(loss_rows[0].keys()))
    plot_parameter_results(
        os.path.join(study_dir, f"plot_{parameter_name}.png"),
        parameter_name,
        plot_labels,
        plot_scores,
    )

    plot_train_loss_curves(
        os.path.join(study_dir, f"train_loss_{parameter_name}.png"),
        loss_rows,
        model_label="EEGNet Training Hyperparameters",
        group_key="parameter_value",
    )
    plot_subject_accuracy(
        os.path.join(study_dir, f"best_per_subject_accuracy_{parameter_name}.png"),
        best_result["per_subject_rows"],
        model_label="EEGNet Training Hyperparameters",
    )
    plot_subject_metrics(
        os.path.join(study_dir, f"best_per_subject_metrics_{parameter_name}.png"),
        best_result["per_subject_rows"],
        model_label="EEGNet Training Hyperparameters",
    )


if __name__ == "__main__":
    set_seed(BASE_CONFIG["seed"])
    for study in PARAMETER_STUDIES:
        study_config = dict(BASE_CONFIG)
        study_config.update(study["overrides"])
        run_parameter_study(study["name"], study["values"], study_config)
