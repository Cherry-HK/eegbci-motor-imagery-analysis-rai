import csv
import io
import math
import os
import random
import time

import matplotlib
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
from torch.utils.data import DataLoader, TensorDataset
from models.deep_learning_utils import run_loso_deep

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
RESULTS_DIR = os.path.join("models", "transformer", "transformer_parameter_study")

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


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2048):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class EEGTransformer(nn.Module):
    def __init__(
        self,
        n_channels,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout_rate=0.1,
    ):
        super().__init__()
        self.input_projection = nn.Linear(n_channels, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout_rate,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(d_model, 2)

    def forward(self, x):
        # Input: [batch, 1, channels, samples]
        x = x.squeeze(1).transpose(1, 2)
        x = self.input_projection(x)
        x = self.positional_encoding(x)
        x = self.encoder(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.classifier(x)


def build_model(config, n_channels, n_samples):
    return EEGTransformer(
        n_channels=n_channels,
        d_model=config["d_model"],
        nhead=config["nhead"],
        num_layers=config["num_layers"],
        dim_feedforward=config["dim_feedforward"],
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


def train_one_fold(X_train, y_train, X_test, y_test, config):
    n_channels = X_train.shape[1]

    model = EEGTransformer(
        n_channels=n_channels,
        d_model=config["d_model"],
        nhead=config["nhead"],
        num_layers=config["num_layers"],
        dim_feedforward=config["dim_feedforward"],
        dropout_rate=config["dropout_rate"],
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    train_loader = create_dataloader(X_train, y_train, config["batch_size"], shuffle=True)
    test_loader = create_dataloader(X_test, y_test, config["batch_size"], shuffle=False)

    model.train()
    memory_before_mb = get_process_memory_mb()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    train_start = time.perf_counter()
    epoch_losses = []
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
        epoch_losses.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(batch_losses)) if batch_losses else None,
            }
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    train_time_sec = time.perf_counter() - train_start
    memory_after_mb = get_process_memory_mb()

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
        "train_time_sec": train_time_sec,
        "inference_time_sec": inference_time_sec,
        "inference_ms_per_sample": inference_ms_per_sample,
        "model_size_mb": model_size_mb,
        "num_parameters": num_parameters,
        "train_memory_delta_mb": train_memory_delta_mb,
        "peak_gpu_memory_mb": peak_gpu_memory_mb,
        "epoch_losses": epoch_losses,
        "final_train_loss": epoch_losses[-1]["train_loss"] if epoch_losses else None,
    }


def run_loso_transformer(
    X,
    y,
    subjects,
    *,
    d_model=64,
    nhead=4,
    num_layers=2,
    dim_feedforward=128,
    dropout_rate=0.1,
    learning_rate=1e-3,
    batch_size=32,
    epochs=20,
    weight_decay=0.0,
    progress_label="",
):
    """Run LOSO with a PyTorch Transformer and return summary metrics."""
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
                "d_model": d_model,
                "nhead": nhead,
                "num_layers": num_layers,
                "dim_feedforward": dim_feedforward,
                "dropout_rate": dropout_rate,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "epochs": epochs,
                "weight_decay": weight_decay,
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
            }
        )
        for epoch_row in result["epoch_losses"]:
            loss_rows.append(
                {
                    "fold": fold,
                    "subject": test_subject,
                    "epoch": epoch_row["epoch"],
                    "train_loss": epoch_row["train_loss"],
                }
            )

    return {
        "d_model": d_model,
        "nhead": nhead,
        "num_layers": num_layers,
        "dim_feedforward": dim_feedforward,
        "dropout_rate": dropout_rate,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs": epochs,
        "weight_decay": weight_decay,
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
        "fold_rows": fold_rows,
        "loss_rows": loss_rows,
    }


def write_summary_csv(path, rows):
    fieldnames = [
        "parameter_name",
        "parameter_value",
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
        "fold",
        "subject",
        "accuracy",
        "f1_score",
        "roc_auc",
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
    plt.title(f"Transformer Parameter Study: {parameter_name}")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def write_loss_csv(path, rows):
    fieldnames = [
        "parameter_name",
        "parameter_value",
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
    parameter_values = []
    mean_losses = []

    for value in dict.fromkeys(row["parameter_value"] for row in loss_rows):
        value_rows = [row for row in loss_rows if row["parameter_value"] == value]
        epoch_values = sorted(set(row["epoch"] for row in value_rows))
        mean_epoch_losses = []
        for epoch in epoch_values:
            epoch_rows = [row["train_loss"] for row in value_rows if row["epoch"] == epoch and row["train_loss"] is not None]
            mean_epoch_losses.append(float(np.mean(epoch_rows)) if epoch_rows else np.nan)
        plt.plot(epoch_values, mean_epoch_losses, marker="o", linewidth=2, label=str(value))
        parameter_values.append(value)
        mean_losses.append(mean_epoch_losses[-1] if mean_epoch_losses else np.nan)

    plt.xlabel("Epoch")
    plt.ylabel("Mean Train Loss")
    plt.title(f"Transformer Train Loss: {parameter_name}")
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

        if parameter_name == "d_model":
            config["d_model"] = value
        elif parameter_name == "nhead":
            config["nhead"] = value
        elif parameter_name == "num_layers":
            config["num_layers"] = value
        elif parameter_name == "dim_feedforward":
            config["dim_feedforward"] = value
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
        summary_rows.append(summary_row)

        for fold_row in result["fold_rows"]:
            fold_rows.append(
                {
                    "parameter_name": parameter_name,
                    "parameter_value": value,
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
                    **fold_row,
                }
            )
        for loss_row in result["loss_rows"]:
            loss_rows.append(
                {
                    "parameter_name": parameter_name,
                    "parameter_value": value,
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
    "d_model": 64,
    "nhead": 4,
    "num_layers": 2,
    "dim_feedforward": 128,
    "dropout_rate": 0.1,
    "learning_rate": 1e-3,
    "batch_size": 32,
    "epochs": 75,
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
        "name": "d_model",
        "values": [32, 64, 128],
        "overrides": {},
    },
    {
        "name": "nhead",
        "values": [2, 4, 8],
        "overrides": {"d_model": 64},
    },
    {
        "name": "num_layers",
        "values": [1, 2, 3],
        "overrides": {"d_model": 64, "nhead": 4},
    },
    {
        "name": "dropout_rate",
        "values": [0.1, 0.3, 0.5],
        "overrides": {"d_model": 64, "nhead": 4, "num_layers": 2},
    },
    {
        "name": "learning_rate",
        "values": [1e-4, 1e-3, 1e-2],
        "overrides": {"d_model": 64, "nhead": 4, "num_layers": 2},
    },
]


if __name__ == "__main__":
    set_seed(42)
    for study in PARAMETER_STUDIES:
        study_config = BASE_CONFIG.copy()
        study_config.update(study["overrides"])
        run_parameter_study(study["name"], study["values"], study_config)
