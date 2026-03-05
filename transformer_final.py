"""
train_transformer_eeg.py
========================
EEG Motor Imagery Classification using a Transformer-based model.

- No files are written to disk (no .pth / .csv / .png saves)
- All metrics printed to terminal as percentages
- Plots displayed via plt.show()
- CLS token used for temporal pooling

Usage : python train_transformer_eeg.py
Deps  : pip install torch numpy scikit-learn matplotlib seaborn tqdm
"""

# ─────────────────────────────────────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────────────────────────────────────
import os
import time
import math
import random
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, auc
)
from sklearn.preprocessing import label_binarize

# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────
SEED = 42

def set_seed(seed: int = SEED):
    """Fix all random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
class Config:
    # Data
    data_dir        : str   = "."       # directory containing the .npy files
    # Model
    d_model         : int   = 128       # transformer embedding dimension
    nhead           : int   = 8         # number of attention heads
    num_layers      : int   = 4         # number of transformer encoder layers
    dim_feedforward : int   = 256       # inner dim of the FFN
    dropout         : float = 0.3       # dropout rate
    # Training
    batch_size      : int   = 64
    epochs          : int   = 100
    lr              : float = 1e-3
    weight_decay    : float = 1e-4
    grad_clip       : float = 1.0       # max gradient norm for clipping
    patience        : int   = 15        # early-stopping patience
    # Misc
    num_workers     : int   = 0
    device          : str   = "cuda" if torch.cuda.is_available() else "cpu"

cfg = Config()
print(f"[Config] Device : {cfg.device}")

# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────
class EEGDataset(Dataset):
    """
    Wraps preprocessed EEG arrays.

    Expected input shape  : (N, C, T)
    __getitem__ returns   : (T, C)  — time as the sequence dimension —
    plus the integer class label.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)   # (N, C, T)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        # (C, T) → (T, C)  so the Transformer sees time as its sequence axis
        x = self.X[idx].permute(1, 0)
        return x, self.y[idx]

# ─────────────────────────────────────────────────────────────────────────────
# Positional Encoding
# ─────────────────────────────────────────────────────────────────────────────
class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding (Vaswani et al., 2017).
    Injects position information into token embeddings without extra parameters.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe       = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)                   # (1, max_len, d_model)
        self.register_buffer("pe", pe)         # not a learnable parameter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_model)"""
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)

# ─────────────────────────────────────────────────────────────────────────────
# EEG Transformer Model  (CLS token pooling)
# ─────────────────────────────────────────────────────────────────────────────
class EEGTransformer(nn.Module):
    """
    Transformer encoder for EEG time-series classification using a CLS token.

    Architecture:
        Input projection  (C → d_model)
        → Prepend learnable CLS token
        → Positional Encoding
        → N × TransformerEncoderLayer  (MHSA + FFN + residuals + LayerNorm)
        → Extract CLS token output  [position 0]
        → Classification head  (Linear → GELU → Dropout → Linear)

    The CLS token (BERT-style) concentrates global sequence information into
    a single vector. The model learns which time steps to attend to rather than
    treating every step equally (as mean pooling would).
    """

    def __init__(
        self,
        in_channels     : int,
        n_classes       : int,
        d_model         : int   = 128,
        nhead           : int   = 8,
        num_layers      : int   = 4,
        dim_feedforward : int   = 256,
        dropout         : float = 0.3,
    ):
        super().__init__()

        # ── 1. Input projection: map C EEG channels → d_model ──────────────
        self.input_proj = nn.Linear(in_channels, d_model)

        # ── 2. Learnable CLS token  (1, 1, d_model) ────────────────────────
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # ── 3. Positional encoding ──────────────────────────────────────────
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)

        # ── 4. Transformer encoder stack ────────────────────────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model         = d_model,
            nhead           = nhead,
            dim_feedforward = dim_feedforward,
            dropout         = dropout,
            activation      = "gelu",   # GELU throughout
            batch_first     = True,     # input shape: (B, T, d_model)
            norm_first      = True,     # Pre-LN for improved stability
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers = num_layers,
            norm       = nn.LayerNorm(d_model),
        )

        # ── 5. Classification head ──────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_classes),
        )

        self._init_weights()

    def _init_weights(self):
        """Xavier uniform initialisation for all linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x      : (B, T, C)  — batch of EEG sequences
        Returns:
            logits : (B, n_classes)
        """
        B = x.size(0)

        # Project EEG channels to the model dimension
        x = self.input_proj(x)                          # (B, T, d_model)

        # Prepend the CLS token to every sequence in the batch
        cls = self.cls_token.expand(B, -1, -1)          # (B, 1, d_model)
        x   = torch.cat([cls, x], dim=1)                # (B, T+1, d_model)

        # Add positional encoding
        x = self.pos_enc(x)                             # (B, T+1, d_model)

        # Transformer encoder: MHSA + FFN with residuals & LayerNorm
        x = self.transformer_encoder(x)                 # (B, T+1, d_model)

        # Extract only the CLS token (position 0) for classification
        cls_out = x[:, 0, :]                            # (B, d_model)

        # Classification head
        logits = self.classifier(cls_out)               # (B, n_classes)
        return logits

# ─────────────────────────────────────────────────────────────────────────────
# Metrics helper
# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred, y_prob, n_classes) -> dict:
    """
    Compute all classification metrics.
    Returns decimal values; callers multiply by 100 for display.
    """
    avg     = "macro"
    acc     = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    prec    = precision_score(y_true, y_pred, average=avg, zero_division=0)
    rec     = recall_score   (y_true, y_pred, average=avg, zero_division=0)
    f1      = f1_score       (y_true, y_pred, average=avg, zero_division=0)

    if n_classes == 2:
        roc = roc_auc_score(y_true, y_prob[:, 1])
    else:
        y_bin = label_binarize(y_true, classes=list(range(n_classes)))
        roc   = roc_auc_score(y_bin, y_prob, multi_class="ovr", average=avg)

    return dict(
        accuracy          = acc,
        balanced_accuracy = bal_acc,
        precision         = prec,
        recall            = rec,
        f1                = f1,
        roc_auc           = roc,
    )

# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, device, grad_clip) -> float:
    """Run one training epoch and return the mean loss."""
    model.train()
    total_loss = 0.0

    for X_batch, y_batch in tqdm(loader, desc="  Train", leave=False):
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        logits = model(X_batch)
        loss   = criterion(logits, y_batch)
        loss.backward()

        # Gradient clipping prevents exploding gradients
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()
        total_loss += loss.item() * X_batch.size(0)

    return total_loss / len(loader.dataset)

# ─────────────────────────────────────────────────────────────────────────────
# Validation / evaluation loop
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, criterion, device, n_classes):
    """Evaluate on a DataLoader. Returns (mean_loss, metrics, y_true, y_pred, y_prob)."""
    model.eval()
    total_loss = 0.0
    all_labels, all_preds, all_probs = [], [], []

    for X_batch, y_batch in tqdm(loader, desc="  Valid", leave=False):
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        logits = model(X_batch)
        loss   = criterion(logits, y_batch)
        total_loss += loss.item() * X_batch.size(0)

        probs = F.softmax(logits, dim=-1)
        preds = probs.argmax(dim=-1)

        all_labels.extend(y_batch.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    mean_loss = total_loss / len(loader.dataset)
    y_true    = np.array(all_labels)
    y_pred    = np.array(all_preds)
    y_prob    = np.array(all_probs)
    metrics   = compute_metrics(y_true, y_pred, y_prob, n_classes)

    return mean_loss, metrics, y_true, y_pred, y_prob

# ─────────────────────────────────────────────────────────────────────────────
# Visualisation  — plt.show() only, nothing written to disk
# ─────────────────────────────────────────────────────────────────────────────
def show_training_curves(history: dict):
    """Display loss, accuracy, and per-epoch timing in a single figure."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Training Summary", fontsize=15, fontweight="bold")

    # Loss curves
    ax = axes[0]
    ax.plot(epochs, history["train_loss"], label="Train Loss", lw=2)
    ax.plot(epochs, history["val_loss"],   label="Val Loss",   lw=2, linestyle="--")
    ax.set_title("Loss Curves"); ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.legend(); ax.grid(alpha=0.3)

    # Validation accuracy (as %)
    ax = axes[1]
    ax.plot(epochs, [v * 100 for v in history["val_acc"]],
            label="Val Accuracy", lw=2, color="green")
    ax.set_title("Validation Accuracy"); ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy (%)")
    ax.legend(); ax.grid(alpha=0.3)

    # Training time per epoch
    ax = axes[2]
    ax.bar(epochs, history["epoch_time"], color="steelblue", alpha=0.7)
    ax.set_title("Training Time per Epoch"); ax.set_xlabel("Epoch"); ax.set_ylabel("Seconds")
    ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    plt.show()


def show_roc_curves(y_true, y_prob, n_classes):
    """Display one-vs-rest ROC curves for every class."""
    fig, ax = plt.subplots(figsize=(8, 6))
    colors  = plt.cm.tab10(np.linspace(0, 1, n_classes))

    if n_classes == 2:
        fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
        roc_auc     = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc * 100:.2f}%")
    else:
        y_bin = label_binarize(y_true, classes=list(range(n_classes)))
        for i, color in zip(range(n_classes), colors):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
            roc_auc     = auc(fpr, tpr)
            ax.plot(fpr, tpr, lw=2, color=color,
                    label=f"Class {i}  AUC = {roc_auc * 100:.2f}%")

    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves (One-vs-Rest)", fontsize=13)
    ax.legend(loc="lower right"); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def show_confusion_matrix(y_true, y_pred, n_classes):
    """Display the confusion matrix as a seaborn heatmap."""
    cm  = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(max(6, n_classes), max(5, n_classes - 1)))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=[f"C{i}" for i in range(n_classes)],
        yticklabels=[f"C{i}" for i in range(n_classes)],
        ax=ax
    )
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion Matrix", fontsize=13)
    plt.tight_layout()
    plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────
def load_data(data_dir: str = "."):
    """Load preprocessed NumPy arrays and remap labels to 0-indexed integers."""
    X_train = np.load(os.path.join(data_dir, "X_train.npy"))
    X_test  = np.load(os.path.join(data_dir, "X_test.npy"))
    y_train = np.load(os.path.join(data_dir, "y_train.npy"))
    y_test  = np.load(os.path.join(data_dir, "y_test.npy"))

    # Remap labels to 0-indexed regardless of original encoding
    classes   = np.unique(y_train)
    label_map = {c: i for i, c in enumerate(classes)}
    y_train   = np.array([label_map[l] for l in y_train])
    y_test    = np.array([label_map[l] for l in y_test])

    print(f"[Data] X_train : {X_train.shape}   y_train : {y_train.shape}")
    print(f"[Data] X_test  : {X_test.shape}    y_test  : {y_test.shape}")
    print(f"[Data] Classes : {classes}  →  remapped to 0..{len(classes) - 1}")
    return X_train, X_test, y_train, y_test, len(classes)

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    set_seed(SEED)

    # ── Load data ─────────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test, n_classes = load_data(cfg.data_dir)
    _, n_channels, _ = X_train.shape           # (N, C, T)

    # ── DataLoaders ───────────────────────────────────────────────────────────
    train_loader = DataLoader(
        EEGDataset(X_train, y_train),
        batch_size  = cfg.batch_size,
        shuffle     = True,
        num_workers = cfg.num_workers,
        pin_memory  = (cfg.device == "cuda"),
    )
    test_loader = DataLoader(
        EEGDataset(X_test, y_test),
        batch_size  = cfg.batch_size,
        shuffle     = False,
        num_workers = cfg.num_workers,
        pin_memory  = (cfg.device == "cuda"),
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = EEGTransformer(
        in_channels     = n_channels,
        n_classes       = n_classes,
        d_model         = cfg.d_model,
        nhead           = cfg.nhead,
        num_layers      = cfg.num_layers,
        dim_feedforward = cfg.dim_feedforward,
        dropout         = cfg.dropout,
    ).to(cfg.device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[Model] EEGTransformer  |  trainable parameters : {n_params:,}")

    # ── Optimizer, scheduler, loss ────────────────────────────────────────────
    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    # Cosine annealing decays lr smoothly to lr * 1e-2 over all epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=cfg.lr * 1e-2)
    criterion = nn.CrossEntropyLoss()

    # ── Training state ────────────────────────────────────────────────────────
    history = dict(train_loss=[], val_loss=[], val_acc=[], epoch_time=[])
    best_val_acc   = -1.0
    best_metrics   = {}
    best_y_true    = best_y_pred = best_y_prob = None
    patience_count = 0

    sep = "─" * 70
    print(f"\n{sep}")
    print(f"  Starting training  |  up to {cfg.epochs} epochs  |  device: {cfg.device}")
    print(sep)

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()

        # ── Train ─────────────────────────────────────────────────────────
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, cfg.device, cfg.grad_clip
        )

        # ── Validate ──────────────────────────────────────────────────────
        val_loss, metrics, y_true, y_pred, y_prob = evaluate(
            model, test_loader, criterion, cfg.device, n_classes
        )

        scheduler.step()
        elapsed = time.time() - t0

        # Record history for plots
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(metrics["accuracy"])
        history["epoch_time"].append(elapsed)

        # ── Per-epoch terminal print (all metrics as %) ────────────────────
        print(
            f"\nEpoch {epoch:03d}/{cfg.epochs}\n"
            f"  train_loss        = {train_loss:.4f}\n"
            f"  val_loss          = {val_loss:.4f}\n"
            f"  accuracy          = {metrics['accuracy']          * 100:.2f}%\n"
            f"  balanced_accuracy = {metrics['balanced_accuracy'] * 100:.2f}%\n"
            f"  f1_score          = {metrics['f1']                * 100:.2f}%\n"
            f"  roc_auc           = {metrics['roc_auc']           * 100:.2f}%\n"
            f"  training_time     = {elapsed:.2f}s"
        )

        # ── Keep best results in memory (no checkpoint file written) ───────
        if metrics["accuracy"] > best_val_acc:
            best_val_acc   = metrics["accuracy"]
            best_metrics   = metrics.copy()
            best_y_true    = y_true.copy()
            best_y_pred    = y_pred.copy()
            best_y_prob    = y_prob.copy()
            patience_count = 0
            print(f"  ✔ New best accuracy : {best_val_acc * 100:.2f}%")
        else:
            patience_count += 1
            if patience_count >= cfg.patience:
                print(f"\n[Train] Early stopping triggered at epoch {epoch}.")
                break

    # ── Final results printed to terminal ────────────────────────────────────
    print(f"\n{sep}")
    print("  Final Results")
    print(sep)
    print(f"  Accuracy          : {best_metrics['accuracy']          * 100:.2f}%")
    print(f"  Balanced Accuracy : {best_metrics['balanced_accuracy'] * 100:.2f}%")
    print(f"  Precision         : {best_metrics['precision']         * 100:.2f}%")
    print(f"  Recall            : {best_metrics['recall']            * 100:.2f}%")
    print(f"  F1 Score          : {best_metrics['f1']                * 100:.2f}%")
    print(f"  ROC-AUC           : {best_metrics['roc_auc']           * 100:.2f}%")
    print(sep)

    # ── Plots  (displayed only — nothing saved to disk) ───────────────────────
    show_training_curves(history)
    show_roc_curves(best_y_true, best_y_prob, n_classes)
    show_confusion_matrix(best_y_true, best_y_pred, n_classes)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()