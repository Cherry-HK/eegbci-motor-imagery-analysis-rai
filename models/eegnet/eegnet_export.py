import os

import numpy as np
import torch
from braindecode.models import EEGNetv4

from models.deep_learning_utils import (
    DEVICE,
    load_best_row,
    parse_numeric,
    save_metadata,
    set_seed,
    train_full_deep_model,
)


DATA_DIR = os.path.join("models", "preprocessing_result")
MODEL_DIR = os.path.join("models", "eegnet")
TESTING_DIR = os.path.join(MODEL_DIR, "eegnet_testing")
BEST_CONFIG_CSV = os.path.join(TESTING_DIR, "all_combinations_summary.csv")
MODEL_PATH = os.path.join(MODEL_DIR, "eegnet_model.pth")
METADATA_PATH = os.path.join(MODEL_DIR, "eegnet_metadata.json")


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


def load_best_configuration(path):
    best_row = load_best_row(path, "models/eegnet/eegnet_testing.py")
    return {
        "f1": parse_numeric(best_row["f1"]),
        "depth_multiplier": parse_numeric(best_row["depth_multiplier"]),
        "f2": parse_numeric(best_row["f2"]),
        "kernel_length": parse_numeric(best_row["kernel_length"]),
        "dropout_rate": parse_numeric(best_row["dropout_rate"]),
        "learning_rate": parse_numeric(best_row["learning_rate"]),
        "batch_size": parse_numeric(best_row["batch_size"]),
        "epochs": parse_numeric(best_row["epochs"]),
        "weight_decay": parse_numeric(best_row["weight_decay"]),
        "rank": int(best_row["rank"]),
        "mean_accuracy": float(best_row["mean_accuracy"]),
        "std_accuracy": float(best_row["std_accuracy"]),
        "mean_f1": float(best_row["mean_f1"]),
        "mean_auc": float(best_row["mean_auc"]),
        "mean_final_train_loss": parse_numeric(best_row["mean_final_train_loss"]),
    }


if __name__ == "__main__":
    set_seed(42)
    os.makedirs(MODEL_DIR, exist_ok=True)

    X = np.load(os.path.join(DATA_DIR, "X.npy")).astype(np.float32)
    y = np.load(os.path.join(DATA_DIR, "y.npy")).astype(np.int64)
    subjects = np.load(os.path.join(DATA_DIR, "subjects.npy"))

    best_config = load_best_configuration(BEST_CONFIG_CSV)
    print("Loaded best EEGNet configuration:", best_config)
    print("Device:", DEVICE)

    model, epoch_losses = train_full_deep_model(
        X,
        y,
        config=best_config,
        build_model=build_model,
        add_channel_dim=False,
    )
    torch.save(model.state_dict(), MODEL_PATH)

    metadata = {
        "model_type": "eegnet",
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
                "f1",
                "depth_multiplier",
                "f2",
                "kernel_length",
                "dropout_rate",
                "learning_rate",
                "batch_size",
                "epochs",
                "weight_decay",
            ]
        },
        "evaluation_summary": {
            "rank": best_config["rank"],
            "mean_accuracy": best_config["mean_accuracy"],
            "std_accuracy": best_config["std_accuracy"],
            "mean_f1": best_config["mean_f1"],
            "mean_auc": best_config["mean_auc"],
            "mean_final_train_loss": best_config["mean_final_train_loss"],
        },
        "export_train_loss_history": epoch_losses,
    }
    save_metadata(METADATA_PATH, metadata)

    print("Saved EEGNet weights to:", MODEL_PATH)
    print("Saved metadata to:", METADATA_PATH)
