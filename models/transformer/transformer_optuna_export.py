import os

import numpy as np
import torch

from models.deep_learning_utils import (
    DEVICE,
    load_best_row,
    parse_numeric,
    save_metadata,
    set_seed,
    train_full_deep_model,
)
from models.deep_model_architectures import EEGTransformer


DATA_DIR = os.path.join("models", "preprocessing_result")
MODEL_DIR = os.path.join("models", "transformer")
OPTUNA_DIR = os.path.join(MODEL_DIR, "transformer_optuna")
BEST_CONFIG_CSV = os.path.join(OPTUNA_DIR, "all_trials_summary.csv")
MODEL_PATH = os.path.join(MODEL_DIR, "transformer_optuna_model.pth")
METADATA_PATH = os.path.join(MODEL_DIR, "transformer_optuna_metadata.json")


def build_model(config, n_channels, n_samples):
    return EEGTransformer(
        n_channels=n_channels,
        d_model=config["d_model"],
        nhead=config["nhead"],
        num_layers=config["num_layers"],
        dim_feedforward=config["dim_feedforward"],
        dropout_rate=config["dropout_rate"],
    )


def load_best_configuration(path):
    best_row = load_best_row(path, "models/transformer/transformer_optuna.py")
    return {
        "trial_number": int(best_row["trial_number"]),
        "d_model": parse_numeric(best_row["d_model"]),
        "nhead": parse_numeric(best_row["nhead"]),
        "num_layers": parse_numeric(best_row["num_layers"]),
        "dim_feedforward": parse_numeric(best_row["dim_feedforward"]),
        "dropout_rate": parse_numeric(best_row["dropout_rate"]),
        "learning_rate": parse_numeric(best_row["learning_rate"]),
        "batch_size": parse_numeric(best_row["batch_size"]),
        "epochs": parse_numeric(best_row["epochs"]),
        "weight_decay": parse_numeric(best_row["weight_decay"]),
        "validation_fraction": parse_numeric(best_row["validation_fraction"]),
        "early_stopping_patience": parse_numeric(best_row["early_stopping_patience"]),
        "lr_scheduler_patience": parse_numeric(best_row["lr_scheduler_patience"]),
        "lr_scheduler_factor": parse_numeric(best_row["lr_scheduler_factor"]),
        "use_class_weight": parse_numeric(best_row["use_class_weight"]),
        "seed": parse_numeric(best_row["seed"]),
        "rank": int(best_row["rank"]),
        "mean_accuracy": float(best_row["mean_accuracy"]),
        "std_accuracy": float(best_row["std_accuracy"]),
        "mean_f1": float(best_row["mean_f1"]),
        "mean_auc": float(best_row["mean_auc"]),
        "mean_final_train_loss": parse_numeric(best_row.get("mean_final_train_loss")),
        "mean_best_val_loss": parse_numeric(best_row.get("mean_best_val_loss")),
        "mean_best_epoch": parse_numeric(best_row.get("mean_best_epoch")),
        "mean_stopped_epoch": parse_numeric(best_row.get("mean_stopped_epoch")),
    }


if __name__ == "__main__":
    set_seed(42)
    os.makedirs(MODEL_DIR, exist_ok=True)

    X = np.load(os.path.join(DATA_DIR, "X.npy")).astype(np.float32)
    y = np.load(os.path.join(DATA_DIR, "y.npy")).astype(np.int64)
    subjects = np.load(os.path.join(DATA_DIR, "subjects.npy"))

    best_config = load_best_configuration(BEST_CONFIG_CSV)
    print("Loaded best Transformer Optuna configuration:", best_config)
    print("Device:", DEVICE)

    model, epoch_losses, normalization_stats = train_full_deep_model(
        X,
        y,
        config=best_config,
        build_model=build_model,
        add_channel_dim=True,
    )
    torch.save(model.state_dict(), MODEL_PATH)

    metadata = {
        "model_type": "transformer_optuna",
        "saved_model_path": MODEL_PATH,
        "source_results_csv": BEST_CONFIG_CSV,
        "device_used_for_export": str(DEVICE),
        "class_labels": [0, 1],
        "subjects_seen": sorted(np.unique(subjects).astype(int).tolist()),
        "input_shape": {
            "n_trials": int(X.shape[0]),
            "n_channels": int(X.shape[1]),
            "n_samples": int(X.shape[2]),
        },
        "best_config": {
            key: best_config[key]
            for key in [
                "trial_number",
                "d_model",
                "nhead",
                "num_layers",
                "dim_feedforward",
                "dropout_rate",
                "learning_rate",
                "batch_size",
                "epochs",
                "weight_decay",
                "validation_fraction",
                "early_stopping_patience",
                "lr_scheduler_patience",
                "lr_scheduler_factor",
                "use_class_weight",
                "seed",
            ]
        },
        "evaluation_summary": {
            "rank": best_config["rank"],
            "trial_number": best_config["trial_number"],
            "mean_accuracy": best_config["mean_accuracy"],
            "std_accuracy": best_config["std_accuracy"],
            "mean_f1": best_config["mean_f1"],
            "mean_auc": best_config["mean_auc"],
            "mean_final_train_loss": best_config["mean_final_train_loss"],
            "mean_best_val_loss": best_config["mean_best_val_loss"],
            "mean_best_epoch": best_config["mean_best_epoch"],
            "mean_stopped_epoch": best_config["mean_stopped_epoch"],
        },
        "normalization_stats": normalization_stats,
        "export_train_loss_history": epoch_losses,
    }
    save_metadata(METADATA_PATH, metadata)

    print("Saved Transformer Optuna weights to:", MODEL_PATH)
    print("Saved metadata to:", METADATA_PATH)
