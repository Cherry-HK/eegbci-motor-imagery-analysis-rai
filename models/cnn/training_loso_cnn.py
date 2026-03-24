import csv
import io
import os
import random
import time
from copy import deepcopy

import matplotlib
import numpy as np
import torch
import torch.nn as nn
from models.deep_learning_utils import run_loso_deep
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
from torch.utils.data import DataLoader, TensorDataset

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import psutil
except ImportError:
    psutil = None

# ==========================================================
# 1. LOAD PREPROCESSED DATA
# ==========================================================
DATA_DIR = os.path.join("models", "preprocessing_result")
RESULTS_DIR = os.path.join("models", "cnn", "cnn_parameter_study")

X = np.load(os.path.join(DATA_DIR, "X.npy")).astype(np.float32)
y = np.load(os.path.join(DATA_DIR, "y.npy")).astype(np.int64)
subjects = np.load(os.path.join(DATA_DIR, "subjects.npy"))

os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Dataset shape:", X.shape)
print("Number of subjects:", len(np.unique(subjects)))
print("Results directory:", RESULTS_DIR)
print("Device:", DEVICE)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class EEGNetLite(nn.Module):
    def __init__(
        self,
        n_channels,
        n_samples,
        temporal_filters=8,
        depth_multiplier=2,
        kernel_length=64,
        dropout_rate=0.5,
    ):
        super().__init__()

        spatial_filters = temporal_filters * depth_multiplier
        kernel_length = min(kernel_length, n_samples)
        padding = kernel_length // 2

        self.features = nn.Sequential(
            nn.Conv2d(
                1,
                temporal_filters,
                kernel_size=(1, kernel_length),
                padding=(0, padding),
                bias=False,
            ),
            nn.BatchNorm2d(temporal_filters),
            nn.Conv2d(
                temporal_filters,
                spatial_filters,
                kernel_size=(n_channels, 1),
                groups=temporal_filters,
                bias=False,
            ),
            nn.BatchNorm2d(spatial_filters),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(dropout_rate),
            nn.Conv2d(
                spatial_filters,
                spatial_filters,
                kernel_size=(1, 16),
                padding=(0, 8),
                groups=spatial_filters,
                bias=False,
            ),
            nn.Conv2d(spatial_filters, spatial_filters, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(spatial_filters),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(dropout_rate),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_samples)
            flattened_dim = self.features(dummy).reshape(1, -1).shape[1]

        self.classifier = nn.Linear(flattened_dim, 2)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        return self.classifier(x)


def build_model(config, n_channels, n_samples):
    return EEGNetLite(
        n_channels=n_channels,
        n_samples=n_samples,
        temporal_filters=config["temporal_filters"],
        depth_multiplier=config["depth_multiplier"],
        kernel_length=config["kernel_length"],
        dropout_rate=config["dropout_rate"],
    ).to(DEVICE)


def get_process_memory_mb():
    if psutil is None:
        return None
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def create_dataloader(features, labels, batch_size, shuffle):
    x_tensor = torch.from_numpy(features).unsqueeze(1)
    y_tensor = torch.from_numpy(labels)
    dataset = TensorDataset(x_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def split_train_validation(X_data, y_data, validation_fraction, seed):
    if validation_fraction <= 0.0 or len(np.unique(y_data)) < 2:
        return X_data, y_data, None, None

    rng = np.random.default_rng(seed)
    train_indices = []
    val_indices = []

    for class_label in np.unique(y_data):
        class_indices = np.where(y_data == class_label)[0]
        shuffled = rng.permutation(class_indices)
        n_val = int(round(len(shuffled) * validation_fraction))
        n_val = max(1, n_val) if len(shuffled) > 1 else 0
        n_val = min(n_val, max(0, len(shuffled) - 1))
        val_indices.extend(shuffled[:n_val].tolist())
        train_indices.extend(shuffled[n_val:].tolist())

    if not val_indices or not train_indices:
        return X_data, y_data, None, None

    train_indices = np.array(train_indices, dtype=np.int64)
    val_indices = np.array(val_indices, dtype=np.int64)
    return X_data[train_indices], y_data[train_indices], X_data[val_indices], y_data[val_indices]


def compute_fold_normalization(X_train_reference):
    mean = X_train_reference.mean(axis=(0, 2), keepdims=True)
    std = X_train_reference.std(axis=(0, 2), keepdims=True) + 1e-8
    return mean.astype(np.float32), std.astype(np.float32)


def apply_fold_normalization(X_data, mean, std):
    return ((X_data - mean) / std).astype(np.float32)


def build_class_weight_tensor(y_train):
    class_counts = np.bincount(y_train.astype(np.int64), minlength=2).astype(np.float32)
    class_counts[class_counts == 0] = 1.0
    total = float(class_counts.sum())
    weights = total / (len(class_counts) * class_counts)
    return torch.tensor(weights, dtype=torch.float32, device=DEVICE)


def train_one_fold(
    X_train,
    y_train,
    X_test,
    y_test,
    config,
):
    X_subtrain, y_subtrain, X_val, y_val = split_train_validation(
        X_train,
        y_train,
        config.get("validation_fraction", 0.1),
        config.get("seed", 42),
    )
    if X_val is None:
        X_subtrain, y_subtrain = X_train, y_train

    normalization_mean, normalization_std = compute_fold_normalization(X_subtrain)
    X_subtrain = apply_fold_normalization(X_subtrain, normalization_mean, normalization_std)
    X_test = apply_fold_normalization(X_test, normalization_mean, normalization_std)
    if X_val is not None:
        X_val = apply_fold_normalization(X_val, normalization_mean, normalization_std)

    n_channels = X_subtrain.shape[1]
    n_samples = X_subtrain.shape[2]

    model = EEGNetLite(
        n_channels=n_channels,
        n_samples=n_samples,
        temporal_filters=config["temporal_filters"],
        depth_multiplier=config["depth_multiplier"],
        kernel_length=config["kernel_length"],
        dropout_rate=config["dropout_rate"],
    ).to(DEVICE)

    class_weight_tensor = None
    if config.get("use_class_weight", True):
        class_weight_tensor = build_class_weight_tensor(y_subtrain)
    criterion = nn.CrossEntropyLoss(weight=class_weight_tensor)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.get("lr_scheduler_factor", 0.5),
        patience=config.get("lr_scheduler_patience", 5),
    )

    train_loader = create_dataloader(X_subtrain, y_subtrain, config["batch_size"], shuffle=True)
    val_loader = None
    if X_val is not None:
        val_loader = create_dataloader(X_val, y_val, config["batch_size"], shuffle=False)
    test_loader = create_dataloader(X_test, y_test, config["batch_size"], shuffle=False)

    model.train()
    memory_before_mb = get_process_memory_mb()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    train_start = time.perf_counter()
    epoch_losses = []

    best_state_dict = None
    best_val_loss = float("inf")
    best_epoch = 0
    best_train_loss = None
    epochs_without_improvement = 0
    stopped_epoch = config["epochs"]

    for epoch in range(1, config["epochs"] + 1):
        batch_losses = []
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.item()))
        train_loss = float(np.mean(batch_losses)) if batch_losses else None
        model.eval()
        val_loss = train_loss
        if val_loader is not None:
            val_batch_losses = []
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(DEVICE)
                    batch_y = batch_y.to(DEVICE)
                    logits = model(batch_x)
                    loss = criterion(logits, batch_y)
                    val_batch_losses.append(float(loss.item()))
            val_loss = float(np.mean(val_batch_losses)) if val_batch_losses else train_loss
        model.train()
        scheduler.step(val_loss)
        epoch_losses.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
        if val_loss is not None and val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_train_loss = train_loss
            best_state_dict = deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.get("early_stopping_patience", 10):
                stopped_epoch = epoch
                break

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    train_time_sec = time.perf_counter() - train_start
    memory_after_mb = get_process_memory_mb()

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    model.eval()
    all_probs = []
    all_preds = []

    inference_start = time.perf_counter()
    with torch.no_grad():
        for batch_x, _ in test_loader:
            batch_x = batch_x.to(DEVICE)
            logits = model(batch_x)
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)
            all_probs.extend(probs.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    inference_time_sec = time.perf_counter() - inference_start

    y_pred = np.array(all_preds, dtype=np.int64)
    y_prob = np.array(all_probs, dtype=np.float32)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    model_size_mb = len(buffer.getvalue()) / (1024 * 1024)

    num_parameters = sum(parameter.numel() for parameter in model.parameters())
    train_memory_delta_mb = None
    if memory_before_mb is not None and memory_after_mb is not None:
        train_memory_delta_mb = max(0.0, memory_after_mb - memory_before_mb)

    peak_gpu_memory_mb = None
    if torch.cuda.is_available():
        peak_gpu_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    inference_ms_per_sample = (inference_time_sec / max(1, len(y_test))) * 1000.0

    return {
        "accuracy": accuracy,
        "f1_score": f1,
        "roc_auc": auc,
        "confusion_matrix": cm,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "train_time_sec": train_time_sec,
        "inference_time_sec": inference_time_sec,
        "inference_ms_per_sample": inference_ms_per_sample,
        "model_size_mb": model_size_mb,
        "num_parameters": num_parameters,
        "train_memory_delta_mb": train_memory_delta_mb,
        "peak_gpu_memory_mb": peak_gpu_memory_mb,
        "epoch_losses": epoch_losses,
        "final_train_loss": epoch_losses[-1]["train_loss"] if epoch_losses else None,
        "best_train_loss": best_train_loss,
        "best_val_loss": best_val_loss if best_val_loss != float("inf") else None,
        "best_epoch": best_epoch,
        "stopped_epoch": stopped_epoch,
    }


def run_loso_cnn(
    X,
    y,
    subjects,
    *,
    temporal_filters=8,
    depth_multiplier=2,
    kernel_length=64,
    dropout_rate=0.5,
    learning_rate=1e-3,
    batch_size=32,
    epochs=20,
    weight_decay=0.0,
    validation_fraction=0.1,
    early_stopping_patience=10,
    lr_scheduler_patience=5,
    lr_scheduler_factor=0.5,
    use_class_weight=True,
    seed=42,
    progress_label="",
):
    """Run LOSO with a PyTorch CNN and return summary metrics."""
    logo = LeaveOneGroupOut()

    accuracies = []
    f1_scores = []
    auc_scores = []
    conf_matrices = []
    fold_rows = []
    all_y_true = []
    all_y_pred = []
    train_times = []
    inference_times = []
    inference_ms_per_sample_values = []
    model_sizes = []
    parameter_counts = []
    train_memory_deltas = []
    peak_gpu_memories = []
    final_train_losses = []
    best_val_losses = []
    best_epochs = []
    stopped_epochs = []
    loss_rows = []

    total_folds = len(np.unique(subjects))

    for fold, (train_idx, test_idx) in enumerate(logo.split(X, y, groups=subjects), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        test_subject = int(subjects[test_idx][0])

        print(
            f"[{progress_label}] Fold {fold}/{total_folds} "
            f"(test subject {test_subject})"
        )

        result = train_one_fold(
            X_train,
            y_train,
            X_test,
            y_test,
            {
                "temporal_filters": temporal_filters,
                "depth_multiplier": depth_multiplier,
                "kernel_length": kernel_length,
                "dropout_rate": dropout_rate,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "epochs": epochs,
                "weight_decay": weight_decay,
                "validation_fraction": validation_fraction,
                "early_stopping_patience": early_stopping_patience,
                "lr_scheduler_patience": lr_scheduler_patience,
                "lr_scheduler_factor": lr_scheduler_factor,
                "use_class_weight": use_class_weight,
                "seed": seed,
            },
        )

        accuracies.append(result["accuracy"])
        f1_scores.append(result["f1_score"])
        auc_scores.append(result["roc_auc"])
        conf_matrices.append(result["confusion_matrix"])
        all_y_true.extend(y_test.tolist())
        all_y_pred.extend(result["y_pred"].tolist())
        train_times.append(result["train_time_sec"])
        inference_times.append(result["inference_time_sec"])
        inference_ms_per_sample_values.append(result["inference_ms_per_sample"])
        model_sizes.append(result["model_size_mb"])
        parameter_counts.append(result["num_parameters"])

        if result["train_memory_delta_mb"] is not None:
            train_memory_deltas.append(result["train_memory_delta_mb"])
        if result["peak_gpu_memory_mb"] is not None:
            peak_gpu_memories.append(result["peak_gpu_memory_mb"])
        if result["final_train_loss"] is not None:
            final_train_losses.append(result["final_train_loss"])
        if result["best_val_loss"] is not None:
            best_val_losses.append(result["best_val_loss"])
        if result["best_epoch"] is not None:
            best_epochs.append(result["best_epoch"])
        if result["stopped_epoch"] is not None:
            stopped_epochs.append(result["stopped_epoch"])

        fold_rows.append(
            {
                "fold": fold,
                "subject": test_subject,
                "accuracy": result["accuracy"],
                "f1_score": result["f1_score"],
                "roc_auc": result["roc_auc"],
                "train_time_sec": result["train_time_sec"],
                "inference_time_sec": result["inference_time_sec"],
                "inference_ms_per_sample": result["inference_ms_per_sample"],
                "model_size_mb": result["model_size_mb"],
                "num_parameters": result["num_parameters"],
                "train_memory_delta_mb": result["train_memory_delta_mb"],
                "peak_gpu_memory_mb": result["peak_gpu_memory_mb"],
                "final_train_loss": result["final_train_loss"],
                "best_train_loss": result["best_train_loss"],
                "best_val_loss": result["best_val_loss"],
                "best_epoch": result["best_epoch"],
                "stopped_epoch": result["stopped_epoch"],
            }
        )
        for epoch_row in result["epoch_losses"]:
            loss_rows.append(
                {
                    "fold": fold,
                    "subject": test_subject,
                    "epoch": epoch_row["epoch"],
                    "train_loss": epoch_row["train_loss"],
                    "val_loss": epoch_row.get("val_loss"),
                    "learning_rate": epoch_row.get("learning_rate"),
                }
            )

    return {
        "temporal_filters": temporal_filters,
        "depth_multiplier": depth_multiplier,
        "kernel_length": kernel_length,
        "dropout_rate": dropout_rate,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs": epochs,
        "weight_decay": weight_decay,
        "validation_fraction": validation_fraction,
        "early_stopping_patience": early_stopping_patience,
        "lr_scheduler_patience": lr_scheduler_patience,
        "lr_scheduler_factor": lr_scheduler_factor,
        "use_class_weight": use_class_weight,
        "seed": seed,
        "mean_accuracy": float(np.mean(accuracies)),
        "std_accuracy": float(np.std(accuracies)),
        "mean_f1": float(np.mean(f1_scores)),
        "mean_auc": float(np.mean(auc_scores)),
        "avg_confusion_matrix": np.mean(conf_matrices, axis=0),
        "overall_confusion_matrix": confusion_matrix(all_y_true, all_y_pred, labels=[0, 1]),
        "mean_train_time_sec": float(np.mean(train_times)),
        "mean_inference_time_sec": float(np.mean(inference_times)),
        "mean_inference_ms_per_sample": float(np.mean(inference_ms_per_sample_values)),
        "mean_model_size_mb": float(np.mean(model_sizes)),
        "mean_num_parameters": float(np.mean(parameter_counts)),
        "mean_train_memory_delta_mb": float(np.mean(train_memory_deltas)) if train_memory_deltas else None,
        "mean_peak_gpu_memory_mb": float(np.mean(peak_gpu_memories)) if peak_gpu_memories else None,
        "mean_final_train_loss": float(np.mean(final_train_losses)) if final_train_losses else None,
        "mean_best_val_loss": float(np.mean(best_val_losses)) if best_val_losses else None,
        "mean_best_epoch": float(np.mean(best_epochs)) if best_epochs else None,
        "mean_stopped_epoch": float(np.mean(stopped_epochs)) if stopped_epochs else None,
        "fold_rows": fold_rows,
        "loss_rows": loss_rows,
    }


def write_summary_csv(path, rows):
    fieldnames = [
        "parameter_name",
        "parameter_value",
        "temporal_filters",
        "depth_multiplier",
        "kernel_length",
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
        "mean_accuracy",
        "std_accuracy",
        "mean_f1",
        "mean_auc",
        "mean_train_time_sec",
        "mean_inference_time_sec",
        "mean_inference_ms_per_sample",
        "mean_model_size_mb",
        "mean_num_parameters",
        "mean_train_memory_delta_mb",
        "mean_peak_gpu_memory_mb",
        "mean_final_train_loss",
        "mean_best_val_loss",
        "mean_best_epoch",
        "mean_stopped_epoch",
    ]

    with open(path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_fold_csv(path, rows):
    fieldnames = [
        "parameter_name",
        "parameter_value",
        "temporal_filters",
        "depth_multiplier",
        "kernel_length",
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
        "fold",
        "subject",
        "accuracy",
        "f1_score",
        "roc_auc",
        "support",
        "train_time_sec",
        "inference_time_sec",
        "inference_ms_per_sample",
        "model_size_mb",
        "num_parameters",
        "train_memory_delta_mb",
        "peak_gpu_memory_mb",
        "final_train_loss",
        "best_train_loss",
        "best_val_loss",
        "best_epoch",
        "stopped_epoch",
    ]

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
    plt.title(f"CNN Parameter Study: {parameter_name}")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def write_loss_csv(path, rows):
    fieldnames = [
        "parameter_name",
        "parameter_value",
        "temporal_filters",
        "depth_multiplier",
        "kernel_length",
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
        "fold",
        "subject",
        "epoch",
        "train_loss",
        "val_loss",
        "learning_rate",
    ]

    with open(path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_train_loss(path, parameter_name, loss_rows):
    plt.figure(figsize=(8, 5))

    for value in dict.fromkeys(row["parameter_value"] for row in loss_rows):
        value_rows = [row for row in loss_rows if row["parameter_value"] == value]
        epoch_values = sorted(set(row["epoch"] for row in value_rows))
        mean_epoch_losses = []
        for epoch in epoch_values:
            epoch_rows = [row["train_loss"] for row in value_rows if row["epoch"] == epoch and row["train_loss"] is not None]
            mean_epoch_losses.append(float(np.mean(epoch_rows)) if epoch_rows else np.nan)
        plt.plot(epoch_values, mean_epoch_losses, marker="o", linewidth=2, label=str(value))

    plt.xlabel("Epoch")
    plt.ylabel("Mean Train Loss")
    plt.title(f"CNN Train Loss: {parameter_name}")
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

        if parameter_name == "temporal_filters":
            config["temporal_filters"] = value
        elif parameter_name == "depth_multiplier":
            config["depth_multiplier"] = value
        elif parameter_name == "kernel_length":
            config["kernel_length"] = value
        elif parameter_name == "dropout_rate":
            config["dropout_rate"] = value
        elif parameter_name == "learning_rate":
            config["learning_rate"] = value
        elif parameter_name == "batch_size":
            config["batch_size"] = value
        elif parameter_name == "epochs":
            config["epochs"] = value
        elif parameter_name == "weight_decay":
            config["weight_decay"] = value
        else:
            raise ValueError(f"Unsupported parameter: {parameter_name}")

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
        }
        summary_rows.append(summary_row)

        for fold_row in result["fold_rows"]:
            fold_rows.append(
                {
                    "parameter_name": parameter_name,
                    "parameter_value": value,
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
                    **fold_row,
                }
            )
        for loss_row in result["loss_rows"]:
            loss_rows.append(
                {
                    "parameter_name": parameter_name,
                    "parameter_value": value,
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
        confusion_csv_path = os.path.join(
            parameter_dir, f"confusion_{parameter_name}_{safe_value}.csv"
        )
        confusion_plot_path = os.path.join(
            parameter_dir, f"confusion_{parameter_name}_{safe_value}.png"
        )
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

    write_summary_csv(summary_csv_path, summary_rows)
    write_fold_csv(fold_csv_path, fold_rows)
    write_loss_csv(loss_csv_path, loss_rows)
    plot_parameter_results(plot_path, parameter_name, plot_labels, plot_scores)
    plot_train_loss(loss_plot_path, parameter_name, loss_rows)

    best_row = max(summary_rows, key=lambda row: row["mean_accuracy"])
    print("\nBest result for", parameter_name)
    print(best_row)
    print("Saved summary to:", summary_csv_path)
    print("Saved fold results to:", fold_csv_path)
    print("Saved train loss to:", loss_csv_path)
    print("Saved train loss plot to:", loss_plot_path)
    print("Saved plot to:", plot_path)


# ==========================================================
# 2. PARAMETER STUDY CONFIGURATION
# ==========================================================
BASE_CONFIG = {
    "temporal_filters": 8,
    "depth_multiplier": 2,
    "kernel_length": 64,
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
    {
        "name": "temporal_filters",
        "values": [4, 8, 16],
        "overrides": {},
    },
    {
        "name": "depth_multiplier",
        "values": [1, 2, 4],
        "overrides": {"temporal_filters": 8},
    },
    {
        "name": "dropout_rate",
        "values": [0.25, 0.5, 0.75],
        "overrides": {"temporal_filters": 8},
    },
    {
        "name": "learning_rate",
        "values": [1e-4, 1e-3, 1e-2],
        "overrides": {"temporal_filters": 8},
    },
    {
        "name": "kernel_length",
        "values": [32, 64, 128],
        "overrides": {"temporal_filters": 8},
    },
]


if __name__ == "__main__":
    set_seed(42)
    for study in PARAMETER_STUDIES:
        study_config = BASE_CONFIG.copy()
        study_config.update(study["overrides"])
        run_parameter_study(study["name"], study["values"], study_config)
