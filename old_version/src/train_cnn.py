"""
EEGNet-Motor CNN Model for Motor Imagery EEG Classification

This script trains an EEGNet-inspired convolutional neural network
on raw EEG signals for binary motor imagery classification (left vs right hand).

Architecture: Temporal Conv -> Depthwise Spatial Conv -> Separable Conv -> Classifier
  - ~3,500 parameters (compact design for small datasets)
  - Depthwise spatial convolutions learn CSP-like spatial filters
  - Temporal kernels tuned to mu/beta rhythm frequencies at 128 Hz

Dataset: EEG Motor Movement/Imagery Dataset (PhysioNet)
Features: Raw EEG (9 channels x 513 timesteps)
Task: Left hand vs Right hand imagery
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

print("=" * 70)
print("Motor Imagery BCI - EEGNet-Motor CNN Classifier")
print("=" * 70)

# ============================================================================
# 1. Device Setup
# ============================================================================
print("\n[Step 1/8] Setting up device...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"   Using device: {device}")
if device.type == "cuda":
    print(f"   GPU: {torch.cuda.get_device_name(0)}")

# ============================================================================
# 2. Load Data
# ============================================================================
print("\n[Step 2/8] Loading preprocessed data...")

try:
    X_train_raw = np.load('X_train.npy')
    X_test_raw  = np.load('X_test.npy')
    y_train     = np.load('y_train.npy')
    y_test      = np.load('y_test.npy')

    print(f"   X_train: {X_train_raw.shape}")
    print(f"   X_test:  {X_test_raw.shape}")
    print(f"   Classes: {np.unique(y_train, return_counts=True)}")

except FileNotFoundError:
    print("\n   ERROR: Preprocessed data files not found!")
    print("   Please run preprocessing first:")
    print("     python src/preprocessing.py")
    exit(1)

# ============================================================================
# 3. Reshape for CNN: (N, 1, channels, timesteps)
# ============================================================================
print("\n[Step 3/8] Reshaping data for CNN...")

n_channels = 9

if X_train_raw.ndim == 2:
    n_times    = X_train_raw.shape[1] // n_channels
    X_train_3d = X_train_raw[:, :n_channels * n_times].reshape(-1, n_channels, n_times)
    X_test_3d  = X_test_raw[:,  :n_channels * n_times].reshape(-1, n_channels, n_times)
elif X_train_raw.ndim == 3:
    if X_train_raw.shape[1] > X_train_raw.shape[2]:
        # (N, timesteps, channels) -> (N, channels, timesteps)
        X_train_3d = X_train_raw.transpose(0, 2, 1)
        X_test_3d  = X_test_raw.transpose(0, 2, 1)
    else:
        # Already (N, channels, timesteps)
        X_train_3d = X_train_raw
        X_test_3d  = X_test_raw

n_times = X_train_3d.shape[2]
print(f"   Shape: {X_train_3d.shape} (samples, channels, timesteps)")

# ============================================================================
# 4. Normalize per-channel
# ============================================================================
print("\n[Step 4/8] Normalizing data per-channel...")

scaler = StandardScaler()
# Reshape to (N*timesteps, channels) for per-channel normalization
X_train_flat = X_train_3d.transpose(0, 2, 1).reshape(-1, n_channels)
X_test_flat  = X_test_3d.transpose(0, 2, 1).reshape(-1, n_channels)

X_train_flat = scaler.fit_transform(X_train_flat)
X_test_flat  = scaler.transform(X_test_flat)

X_train_3d = X_train_flat.reshape(-1, n_times, n_channels).transpose(0, 2, 1)
X_test_3d  = X_test_flat.reshape(-1, n_times, n_channels).transpose(0, 2, 1)

# Add batch channel dim: (N, channels, timesteps) -> (N, 1, channels, timesteps)
X_train_4d = X_train_3d[:, np.newaxis, :, :]
X_test_4d  = X_test_3d[:, np.newaxis, :, :]

print(f"   Normalized shape: {X_train_4d.shape} (samples, 1, channels, timesteps)")

# ============================================================================
# 5. DataLoaders with Augmentation
# ============================================================================
print("\n[Step 5/8] Creating DataLoaders...")


class AugmentedEEGDataset(Dataset):
    """EEG dataset with optional data augmentation (Gaussian noise + time shift)."""

    def __init__(self, X, y, augment=False):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        self.augment = augment

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x, y = self.X[idx].clone(), self.y[idx]

        if self.augment:
            # Gaussian noise (50% probability)
            if np.random.rand() > 0.5:
                noise = torch.randn_like(x) * 0.01
                x = x + noise

            # Time shift (50% probability, +/- 20 steps)
            if np.random.rand() > 0.5:
                shift = np.random.randint(-20, 21)
                if shift != 0:
                    x = torch.roll(x, shifts=shift, dims=-1)

        return x, y


BATCH_SIZE = 16
train_dataset = AugmentedEEGDataset(X_train_4d, y_train, augment=True)
test_dataset  = AugmentedEEGDataset(X_test_4d,  y_test,  augment=False)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

print(f"   DataLoaders ready. Batch size: {BATCH_SIZE}")

# ============================================================================
# 6. Model Definition - EEGNet-Motor
# ============================================================================
print("\n[Step 6/8] Building EEGNet-Motor CNN model...")


class EEGNet_Motor(nn.Module):
    """
    EEGNet-inspired CNN for motor imagery classification.

    Architecture:
      Block 1: Temporal convolution  (captures mu/beta rhythms)
      Block 2: Depthwise spatial conv (learns CSP-like spatial filters)
      Block 3: Separable convolution  (temporal refinement + channel mixing)
      Classifier: Flatten -> Linear -> 2 classes

    Parameters: ~3,500 (compact for small datasets)
    """

    def __init__(self, n_channels, n_times, num_classes=2,
                 F1=16, D=2, F2=32, kern_len=64, drop_rate=0.25):
        super(EEGNet_Motor, self).__init__()

        # Block 1: Temporal convolution
        self.conv1 = nn.Conv2d(1, F1, (1, kern_len), padding=(0, kern_len // 2),
                               bias=False)
        self.bn1 = nn.BatchNorm2d(F1)

        # Block 2: Depthwise spatial convolution
        self.conv2 = nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(drop_rate)

        # Block 3: Separable convolution
        self.conv3_dw = nn.Conv2d(F2, F2, (1, 16), padding=(0, 8), groups=F2,
                                  bias=False)
        self.conv3_pw = nn.Conv2d(F2, F2, (1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.pool2 = nn.AvgPool2d((1, 8))
        self.drop2 = nn.Dropout(drop_rate)

        # Dynamically compute flat size via dummy forward pass
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_times)
            dummy = self._forward_features(dummy)
            flat_size = dummy.shape[1] * dummy.shape[2] * dummy.shape[3]

        # Classifier
        self.classifier = nn.Linear(flat_size, num_classes)

    def _forward_features(self, x):
        # Block 1: Temporal
        x = self.bn1(self.conv1(x))

        # Block 2: Depthwise spatial
        x = F.elu(self.bn2(self.conv2(x)))
        x = self.drop1(self.pool1(x))

        # Block 3: Separable
        x = self.conv3_dw(x)
        x = F.elu(self.bn3(self.conv3_pw(x)))
        x = self.drop2(self.pool2(x))

        return x

    def forward(self, x):
        x = self._forward_features(x)
        x = x.flatten(start_dim=1)
        return self.classifier(x)


model = EEGNet_Motor(
    n_channels=n_channels,
    n_times=n_times,
    num_classes=2
).to(device)

print(f"   Model architecture:")
print(model)
total_params = sum(p.numel() for p in model.parameters())
print(f"\n   Total parameters: {total_params:,}")

# ============================================================================
# 7. Training
# ============================================================================
print("\n[Step 7/8] Training...")

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', patience=7, factor=0.5
)


class EarlyStopping:
    def __init__(self, patience=15):
        self.patience   = patience
        self.counter    = 0
        self.best_score = None
        self.stop       = False
        self.best_state = None

    def __call__(self, score, model):
        if self.best_score is None or score > self.best_score:
            self.best_score = score
            self.best_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}
            self.counter    = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True


early_stopping = EarlyStopping(patience=15)

EPOCHS = 150
train_losses, train_accs = [], []
val_losses,   val_accs   = [], []

print(f"\n{'=' * 60}")
print(f"  TRAINING EEGNet-Motor CNN ({EPOCHS} epochs max)")
print(f"{'=' * 60}")

for epoch in range(EPOCHS):
    # --- Training ---
    model.train()
    t_loss, t_correct, t_total = 0, 0, 0

    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss    = criterion(outputs, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        t_loss    += loss.item()
        preds      = outputs.argmax(dim=1)
        t_correct += (preds == y_batch).sum().item()
        t_total   += y_batch.size(0)

    # --- Validation ---
    model.eval()
    v_loss, v_correct, v_total = 0, 0, 0

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs  = model(X_batch)
            loss     = criterion(outputs, y_batch)
            v_loss  += loss.item()
            preds    = outputs.argmax(dim=1)
            v_correct += (preds == y_batch).sum().item()
            v_total   += y_batch.size(0)

    t_acc = t_correct / t_total
    v_acc = v_correct / v_total
    t_l   = t_loss / len(train_loader)
    v_l   = v_loss / len(test_loader)

    train_losses.append(t_l)
    train_accs.append(t_acc)
    val_losses.append(v_l)
    val_accs.append(v_acc)

    scheduler.step(v_acc)
    early_stopping(v_acc, model)

    if (epoch + 1) % 10 == 0:
        best_marker = " *" if v_acc == early_stopping.best_score else ""
        print(f"  Epoch [{epoch+1:3d}/{EPOCHS}] "
              f"Train Loss: {t_l:.4f} | Train Acc: {t_acc:.4f} | "
              f"Val Acc: {v_acc:.4f}{best_marker}")

    if early_stopping.stop:
        print(f"\n  Early stopping at epoch {epoch+1}")
        break

# Load best model weights
model.load_state_dict(early_stopping.best_state)
model.to(device)
print(f"\n   Best Validation Accuracy: {early_stopping.best_score:.4f} "
      f"({early_stopping.best_score:.2%})")

# ============================================================================
# 8. Final Evaluation & Save
# ============================================================================
print("\n[Step 8/8] Final evaluation and saving...")

model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        outputs = model(X_batch)
        preds   = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y_batch.numpy())

all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)

test_acc = accuracy_score(all_labels, all_preds)
cm = confusion_matrix(all_labels, all_preds)

print(f"\n  Test Accuracy: {test_acc * 100:.2f}%")
print(f"\n  Classification Report:")
print(classification_report(all_labels, all_preds,
                            target_names=['Left Hand', 'Right Hand']))

# Save model and results
os.makedirs('models', exist_ok=True)

model_path = 'models/cnn_model.pt'
torch.save(model.state_dict(), model_path)
print(f"   Model saved: {model_path}")

results = {
    'test_accuracy': test_acc,
    'best_val_accuracy': early_stopping.best_score,
    'confusion_matrix': cm,
    'train_losses': train_losses,
    'train_accs': train_accs,
    'val_losses': val_losses,
    'val_accs': val_accs,
    'epochs_trained': len(train_losses),
}
results_path = 'models/cnn_results.pkl'
with open(results_path, 'wb') as f:
    pickle.dump(results, f)
print(f"   Results saved: {results_path}")

# Visualizations
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Training curves - Loss
axes[0, 0].plot(train_losses, label='Train Loss')
axes[0, 0].plot(val_losses, label='Val Loss')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].set_title('Training & Validation Loss')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Training curves - Accuracy
axes[0, 1].plot(train_accs, label='Train Acc')
axes[0, 1].plot(val_accs, label='Val Acc')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Accuracy')
axes[0, 1].set_title('Training & Validation Accuracy')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Confusion Matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Left', 'Right'],
            yticklabels=['Left', 'Right'], ax=axes[1, 0])
axes[1, 0].set_xlabel('Predicted', fontsize=12)
axes[1, 0].set_ylabel('True', fontsize=12)
axes[1, 0].set_title(f'EEGNet-Motor CNN Confusion Matrix\nAccuracy: {test_acc:.2%}',
                     fontsize=13, fontweight='bold')

# Per-class accuracy
class_acc = cm.diagonal() / cm.sum(axis=1)
bars = axes[1, 1].bar(['Left Hand', 'Right Hand'], class_acc,
                       color=['#3498db', '#e67e22'])
axes[1, 1].set_ylabel('Accuracy', fontsize=12)
axes[1, 1].set_title('Per-Class Accuracy', fontsize=13, fontweight='bold')
axes[1, 1].set_ylim(0, 1.0)
for bar, val in zip(bars, class_acc):
    axes[1, 1].text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                    f'{val:.2%}', ha='center', fontsize=11, fontweight='bold')
axes[1, 1].grid(axis='y', alpha=0.3)

plt.tight_layout()
fig_path = 'models/cnn_visualization.png'
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
print(f"   Visualization saved: {fig_path}")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 70)
print("TRAINING SUMMARY")
print("=" * 70)
print(f"Model:            EEGNet-Motor CNN")
print(f"Parameters:       {total_params:,}")
print(f"Training samples: {len(y_train)}")
print(f"Test samples:     {len(y_test)}")
print(f"Epochs trained:   {len(train_losses)}")
print(f"Best Val Acc:     {early_stopping.best_score * 100:.2f}%")
print(f"Test Accuracy:    {test_acc * 100:.2f}%")
print("=" * 70)

print("\nTraining complete!")

plt.show()
