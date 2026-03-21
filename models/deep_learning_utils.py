import csv
import itertools
import json
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


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_process_memory_mb():
    if psutil is None:
        return None
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def create_dataloader(features, labels, batch_size, shuffle, add_channel_dim):
    x_tensor = torch.from_numpy(features)
    if add_channel_dim:
        x_tensor = x_tensor.unsqueeze(1)
    y_tensor = torch.from_numpy(labels)
    dataset = TensorDataset(x_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_one_fold_deep(
    X_train,
    y_train,
    X_test,
    y_test,
    config,
    build_model,
    add_channel_dim,
):
    n_channels = X_train.shape[1]
    n_samples = X_train.shape[2]

    model = build_model(config, n_channels, n_samples).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    train_loader = create_dataloader(
        X_train, y_train, config["batch_size"], shuffle=True, add_channel_dim=add_channel_dim
    )
    test_loader = create_dataloader(
        X_test, y_test, config["batch_size"], shuffle=False, add_channel_dim=add_channel_dim
    )

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

    train_memory_delta_mb = None
    if memory_before_mb is not None and memory_after_mb is not None:
        train_memory_delta_mb = max(0.0, memory_after_mb - memory_before_mb)

    peak_gpu_memory_mb = None
    if torch.cuda.is_available():
        peak_gpu_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    inference_ms_per_sample = (inference_time_sec / max(1, len(y_test))) * 1000.0
    num_parameters = sum(parameter.numel() for parameter in model.parameters())
    model_size_mb = sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())
    model_size_mb /= 1024 * 1024

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


def run_loso_deep(
    X,
    y,
    subjects,
    config,
    build_model,
    add_channel_dim,
    progress_label="",
):
    logo = LeaveOneGroupOut()

    accuracies = []
    f1_scores = []
    auc_scores = []
    conf_matrices = []
    fold_rows = []
    loss_rows = []
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

    total_folds = len(np.unique(subjects))

    for fold, (train_idx, test_idx) in enumerate(logo.split(X, y, groups=subjects), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        test_subject = int(subjects[test_idx][0])

        print(
            f"[{progress_label}] Fold {fold}/{total_folds} "
            f"(test subject {test_subject})"
        )

        result = train_one_fold_deep(
            X_train,
            y_train,
            X_test,
            y_test,
            config,
            build_model,
            add_channel_dim,
        )

        accuracies.append(result["accuracy"])
        f1_scores.append(result["f1_score"])
        auc_scores.append(result["roc_auc"])
        conf_matrices.append(result["confusion_matrix"])
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

        all_y_true.extend(y_test.tolist())
        all_y_pred.extend(result["y_pred"].tolist())
        fold_rows.append(
            {
                "fold": fold,
                "subject": test_subject,
                "accuracy": result["accuracy"],
                "f1_score": result["f1_score"],
                "roc_auc": result["roc_auc"],
                "support": int(len(y_test)),
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
        for epoch_entry in result["epoch_losses"]:
            loss_row = {
                "fold": fold,
                "subject": test_subject,
                "epoch": epoch_entry["epoch"],
                "train_loss": epoch_entry["train_loss"],
            }
            for key, value in config.items():
                loss_row[key] = value
            loss_rows.append(loss_row)

    per_subject_rows = sorted(fold_rows, key=lambda row: row["accuracy"], reverse=True)

    return {
        "mean_accuracy": float(np.mean(accuracies)),
        "std_accuracy": float(np.std(accuracies)),
        "mean_f1": float(np.mean(f1_scores)),
        "mean_auc": float(np.mean(auc_scores)),
        "overall_confusion_matrix": confusion_matrix(all_y_true, all_y_pred, labels=[0, 1]),
        "fold_rows": fold_rows,
        "loss_rows": loss_rows,
        "per_subject_rows": per_subject_rows,
        "best_subject": max(fold_rows, key=lambda row: row["accuracy"]),
        "worst_subject": min(fold_rows, key=lambda row: row["accuracy"]),
        "mean_train_time_sec": float(np.mean(train_times)),
        "mean_inference_time_sec": float(np.mean(inference_times)),
        "mean_inference_ms_per_sample": float(np.mean(inference_ms_per_sample_values)),
        "mean_model_size_mb": float(np.mean(model_sizes)),
        "mean_num_parameters": float(np.mean(parameter_counts)),
        "mean_train_memory_delta_mb": float(np.mean(train_memory_deltas)) if train_memory_deltas else None,
        "mean_peak_gpu_memory_mb": float(np.mean(peak_gpu_memories)) if peak_gpu_memories else None,
        "mean_final_train_loss": float(np.mean(final_train_losses)) if final_train_losses else None,
    }


def build_search_space(search_config, sanitize_fn=None):
    keys = list(search_config.keys())
    values = [search_config[key] for key in keys]
    combinations = []
    seen = set()

    for combo_values in itertools.product(*values):
        combo = dict(zip(keys, combo_values))
        if sanitize_fn is not None:
            combo = sanitize_fn(combo)
            if combo is None:
                continue
        combo_key = tuple((key, combo[key]) for key in sorted(combo))
        if combo_key in seen:
            continue
        seen.add(combo_key)
        combinations.append(combo)

    return combinations


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


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


def plot_top_combinations(path, rows, top_n, model_label, label_key):
    top_rows = rows[:top_n]
    labels = [f"#{row['rank']} {row[label_key]}" for row in top_rows]
    scores = [row["mean_accuracy"] for row in top_rows]

    plt.figure(figsize=(11, 6))
    bars = plt.bar(labels, scores, color="#2563eb")
    plt.ylabel("Mean Accuracy")
    plt.xlabel("Combination Rank")
    plt.title(f"Top {len(top_rows)} {model_label} Hyperparameter Combinations")
    plt.xticks(rotation=30, ha="right")
    plt.ylim(0.0, 1.0)

    for bar, score in zip(bars, scores):
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            score + 0.01,
            f"{score:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_subject_accuracy(path, per_subject_rows, model_label):
    sorted_rows = sorted(per_subject_rows, key=lambda row: row["subject"])
    labels = [str(row["subject"]) for row in sorted_rows]
    scores = [row["accuracy"] for row in sorted_rows]

    plt.figure(figsize=(10, 5))
    bars = plt.bar(labels, scores, color="#0f766e")
    plt.ylabel("Accuracy")
    plt.xlabel("Held-out Subject")
    plt.title(f"{model_label} Best Configuration: Per-Subject LOSO Accuracy")
    plt.ylim(0.0, 1.0)
    plt.xticks(rotation=45, ha="right")

    for bar, score in zip(bars, scores):
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            score + 0.01,
            f"{score:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_subject_metrics(path, per_subject_rows, model_label):
    sorted_rows = sorted(per_subject_rows, key=lambda row: row["subject"])
    labels = [str(row["subject"]) for row in sorted_rows]
    accuracies = [row["accuracy"] for row in sorted_rows]
    f1_scores = [row["f1_score"] for row in sorted_rows]
    auc_scores = [row["roc_auc"] for row in sorted_rows]

    plt.figure(figsize=(11, 5))
    plt.plot(labels, accuracies, marker="o", label="Accuracy")
    plt.plot(labels, f1_scores, marker="s", label="F1-score")
    plt.plot(labels, auc_scores, marker="^", label="ROC-AUC")
    plt.ylabel("Score")
    plt.xlabel("Held-out Subject")
    plt.title(f"{model_label} Best Configuration: Per-Subject Metrics")
    plt.ylim(0.0, 1.0)
    plt.xticks(rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_train_loss_curves(path, loss_rows, model_label, group_key):
    grouped = {}
    for row in loss_rows:
        group_value = str(row[group_key])
        epoch = int(row["epoch"])
        train_loss = row["train_loss"]
        if train_loss is None:
            continue
        grouped.setdefault(group_value, {}).setdefault(epoch, []).append(float(train_loss))

    plt.figure(figsize=(10, 5))
    for group_value, epoch_map in grouped.items():
        epochs = sorted(epoch_map.keys())
        mean_losses = [float(np.mean(epoch_map[epoch])) for epoch in epochs]
        plt.plot(epochs, mean_losses, marker="o", label=group_value)

    plt.xlabel("Epoch")
    plt.ylabel("Mean Train Loss")
    plt.title(f"{model_label}: Mean Training Loss Across LOSO Folds")
    plt.legend(title=group_key)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def write_summary_text(path, model_label, best_result, total_combinations):
    lines = [
        f"{model_label} combined-search summary",
        "=" * 40,
        f"Total combinations tested: {total_combinations}",
        "",
        "Best configuration:",
    ]
    for key, value in best_result.items():
        if key in {"fold_rows", "loss_rows", "per_subject_rows", "best_subject", "worst_subject", "overall_confusion_matrix"}:
            continue
        lines.append(f"- {key}: {value}")

    lines.extend(
        [
            "",
            f"Best subject: {best_result['best_subject']['subject']} ({best_result['best_subject']['accuracy']:.4f})",
            f"Worst subject: {best_result['worst_subject']['subject']} ({best_result['worst_subject']['accuracy']:.4f})",
        ]
    )

    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def parse_optional(value):
    if value in {"None", "", None}:
        return None
    return value


def parse_bool(value):
    if value in {True, False}:
        return value
    if value == "True":
        return True
    if value == "False":
        return False
    return value


def parse_numeric(value):
    value = parse_optional(value)
    value = parse_bool(value)
    if value is None or isinstance(value, bool):
        return value
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return value
    if numeric.is_integer():
        return int(numeric)
    return numeric


def load_best_row(path, script_hint):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Best-configuration CSV not found: {path}. Run {script_hint} first."
        )

    with open(path, "r", newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        best_row = next(reader, None)

    if best_row is None:
        raise ValueError(f"No rows found in best-configuration CSV: {path}")
    return best_row


def train_full_deep_model(X, y, config, build_model, add_channel_dim):
    n_channels = X.shape[1]
    n_samples = X.shape[2]
    model = build_model(config, n_channels, n_samples).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    loader = create_dataloader(
        X, y, config["batch_size"], shuffle=True, add_channel_dim=add_channel_dim
    )

    model.train()
    epoch_losses = []
    for epoch in range(1, config["epochs"] + 1):
        batch_losses = []
        for batch_x, batch_y in loader:
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
    return model, epoch_losses


def save_metadata(path, metadata):
    with open(path, "w", encoding="utf-8") as jsonfile:
        json.dump(metadata, jsonfile, indent=2)
