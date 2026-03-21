import csv
import io
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

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import psutil
except ImportError:
    psutil = None

# ==========================================================
# 1. LOAD PREPROCESSED DATA
# ==========================================================
BASE_DIR = "models/output"
DATA_DIR = os.path.join(BASE_DIR, "preprocessing_result")
RESULTS_DIR = os.path.join(BASE_DIR, "lstm_parameter_study")

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


class EEGLSTM(nn.Module):
    def __init__(
        self,
        n_channels,
        hidden_size=64,
        num_layers=1,
        dropout_rate=0.3,
        bidirectional=False,
    ):
        super().__init__()

        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        effective_dropout = dropout_rate if num_layers > 1 else 0.0

        self.lstm = nn.LSTM(
            input_size=n_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=effective_dropout,
            bidirectional=bidirectional,
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(hidden_size * self.num_directions, 2)

    def forward(self, x):
        # Input comes in as [batch, 1, channels, samples].
        x = x.squeeze(1).transpose(1, 2)
        output, _ = self.lstm(x)
        last_step = output[:, -1, :]
        last_step = self.dropout(last_step)
        return self.classifier(last_step)


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


def train_one_fold(
    X_train,
    y_train,
    X_test,
    y_test,
    config,
):
    n_channels = X_train.shape[1]

    model = EEGLSTM(
        n_channels=n_channels,
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        dropout_rate=config["dropout_rate"],
        bidirectional=config["bidirectional"],
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

    for _ in range(config["epochs"]):
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

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
    }


def run_loso_lstm(
    X,
    y,
    subjects,
    *,
    hidden_size=64,
    num_layers=1,
    dropout_rate=0.3,
    bidirectional=False,
    learning_rate=1e-3,
    batch_size=32,
    epochs=20,
    weight_decay=0.0,
    progress_label="",
):
    """Run LOSO with a PyTorch LSTM and return summary metrics."""
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
                "hidden_size": hidden_size,
                "num_layers": num_layers,
                "dropout_rate": dropout_rate,
                "bidirectional": bidirectional,
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
            }
        )

    return {
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "dropout_rate": dropout_rate,
        "bidirectional": bidirectional,
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
        "fold_rows": fold_rows,
    }


def write_summary_csv(path, rows):
    fieldnames = [
        "parameter_name",
        "parameter_value",
        "hidden_size",
        "num_layers",
        "dropout_rate",
        "bidirectional",
        "learning_rate",
        "batch_size",
        "epochs",
        "weight_decay",
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
        "hidden_size",
        "num_layers",
        "dropout_rate",
        "bidirectional",
        "learning_rate",
        "batch_size",
        "epochs",
        "weight_decay",
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
    plt.title(f"LSTM Parameter Study: {parameter_name}")
    plt.grid(True, linestyle="--", alpha=0.4)
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

        if parameter_name == "hidden_size":
            config["hidden_size"] = value
        elif parameter_name == "num_layers":
            config["num_layers"] = value
        elif parameter_name == "dropout_rate":
            config["dropout_rate"] = value
        elif parameter_name == "bidirectional":
            config["bidirectional"] = value
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
        result = run_loso_lstm(X, y, subjects, progress_label=progress_label, **config)

        summary_row = {
            "parameter_name": parameter_name,
            "parameter_value": value,
            "hidden_size": result["hidden_size"],
            "num_layers": result["num_layers"],
            "dropout_rate": result["dropout_rate"],
            "bidirectional": result["bidirectional"],
            "learning_rate": result["learning_rate"],
            "batch_size": result["batch_size"],
            "epochs": result["epochs"],
            "weight_decay": result["weight_decay"],
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
        }
        summary_rows.append(summary_row)

        for fold_row in result["fold_rows"]:
            fold_rows.append(
                {
                    "parameter_name": parameter_name,
                    "parameter_value": value,
                    "hidden_size": result["hidden_size"],
                    "num_layers": result["num_layers"],
                    "dropout_rate": result["dropout_rate"],
                    "bidirectional": result["bidirectional"],
                    "learning_rate": result["learning_rate"],
                    "batch_size": result["batch_size"],
                    "epochs": result["epochs"],
                    "weight_decay": result["weight_decay"],
                    **fold_row,
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
    plot_path = os.path.join(parameter_dir, f"plot_{parameter_name}.png")

    write_summary_csv(summary_csv_path, summary_rows)
    write_fold_csv(fold_csv_path, fold_rows)
    plot_parameter_results(plot_path, parameter_name, plot_labels, plot_scores)

    best_row = max(summary_rows, key=lambda row: row["mean_accuracy"])
    print("\nBest result for", parameter_name)
    print(best_row)
    print("Saved summary to:", summary_csv_path)
    print("Saved fold results to:", fold_csv_path)
    print("Saved plot to:", plot_path)


# ==========================================================
# 2. PARAMETER STUDY CONFIGURATION
# ==========================================================
BASE_CONFIG = {
    "hidden_size": 64,
    "num_layers": 1,
    "dropout_rate": 0.3,
    "bidirectional": False,
    "learning_rate": 1e-3,
    "batch_size": 32,
    "epochs": 20,
    "weight_decay": 0.0,
}

PARAMETER_STUDIES = [
    {
        "name": "hidden_size",
        "values": [32, 64, 128],
        "overrides": {},
    },
    {
        "name": "num_layers",
        "values": [1, 2, 3],
        "overrides": {"hidden_size": 64},
    },
    {
        "name": "dropout_rate",
        "values": [0.1, 0.3, 0.5],
        "overrides": {"hidden_size": 64, "num_layers": 2},
    },
    {
        "name": "bidirectional",
        "values": [False, True],
        "overrides": {"hidden_size": 64, "num_layers": 2},
    },
    {
        "name": "learning_rate",
        "values": [1e-4, 1e-3, 1e-2],
        "overrides": {"hidden_size": 64, "num_layers": 2},
    },
]


if __name__ == "__main__":
    set_seed(42)
    for study in PARAMETER_STUDIES:
        study_config = BASE_CONFIG.copy()
        study_config.update(study["overrides"])
        run_parameter_study(study["name"], study["values"], study_config)
