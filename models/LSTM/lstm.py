import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


DATA_PATH = r"C:\Users\herme\Desktop\capstone\code"

X_train_raw = np.load(f"{DATA_PATH}\\X_train.npy")
X_test_raw  = np.load(f"{DATA_PATH}\\X_test.npy")
y_train     = np.load(f"{DATA_PATH}\\y_train.npy")
y_test      = np.load(f"{DATA_PATH}\\y_test.npy")

print(f"\n✅ Data loaded!")
print(f"   X_train: {X_train_raw.shape}")
print(f"   X_test:  {X_test_raw.shape}")
print(f"   Classes: {np.unique(y_train, return_counts=True)}")


if X_train_raw.ndim == 2:
    n_channels = 9
    n_times    = X_train_raw.shape[1] // n_channels
    X_train_3d = X_train_raw[:, :n_channels*n_times].reshape(-1, n_channels, n_times)
    X_test_3d  = X_test_raw[:,  :n_channels*n_times].reshape(-1, n_channels, n_times)
    X_train_3d = X_train_3d.transpose(0, 2, 1)
    X_test_3d  = X_test_3d.transpose(0, 2, 1)
elif X_train_raw.ndim == 3:
    if X_train_raw.shape[1] < X_train_raw.shape[2]:
        X_train_3d = X_train_raw.transpose(0, 2, 1)
        X_test_3d  = X_test_raw.transpose(0, 2, 1)
    else:
        X_train_3d = X_train_raw
        X_test_3d  = X_test_raw

print(f"\n✅ Shape: {X_train_3d.shape} (samples, timesteps, channels)")

n_timesteps = X_train_3d.shape[1]
n_channels  = X_train_3d.shape[2]



scaler = StandardScaler()
X_train_flat = X_train_3d.reshape(-1, n_channels)
X_test_flat  = X_test_3d.reshape(-1, n_channels)

X_train_flat = scaler.fit_transform(X_train_flat)
X_test_flat  = scaler.transform(X_test_flat)

X_train_3d = X_train_flat.reshape(-1, n_timesteps, n_channels)
X_test_3d  = X_test_flat.reshape(-1, n_timesteps, n_channels)

print("✅ Data normalized!")


class AugmentedEEGDataset(Dataset):
    def __init__(self, X, y, augment=False):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        self.augment = augment
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        x, y = self.X[idx], self.y[idx]
        
        if self.augment and np.random.rand() > 0.5:
            # Add small random noise
            noise = torch.randn_like(x) * 0.01
            x = x + noise
        
        return x, y

train_dataset = AugmentedEEGDataset(X_train_3d, y_train, augment=True)
test_dataset  = AugmentedEEGDataset(X_test_3d,  y_test,  augment=False)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False)

print(f"✅ DataLoaders ready! Batch size: 32")


class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Linear(hidden_size, 1)
    
    def forward(self, lstm_output):
        # lstm_output: (batch, seq_len, hidden_size)
        attn_weights = torch.softmax(self.attention(lstm_output), dim=1)
        context = torch.sum(attn_weights * lstm_output, dim=1)
        return context, attn_weights

class EEG_CNN_LSTM(nn.Module):
    def __init__(self, n_timesteps, n_channels, num_classes=2):
        super(EEG_CNN_LSTM, self).__init__()
        
        # Temporal CNN to extract features across time
        self.conv1 = nn.Conv1d(n_channels, 32, kernel_size=5, padding=2)
        self.bn1   = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2)
        
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2   = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2)
        
        # Calculate size after convolutions
        conv_out_size = n_timesteps // 4  # Due to 2 pooling layers
        
        # LSTM to capture temporal dependencies
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )
        
        # Attention mechanism
        self.attention = AttentionLayer(256)  # 128*2 for bidirectional
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        # x: (batch, timesteps, channels)
        x = x.permute(0, 2, 1)  # -> (batch, channels, timesteps) for CNN
        
        # CNN layers
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        # Back to (batch, timesteps, features) for LSTM
        x = x.permute(0, 2, 1)
        
        # LSTM
        lstm_out, _ = self.lstm(x)
        
        # Attention
        context, attn_weights = self.attention(lstm_out)
        
        # Classify
        return self.classifier(context)

model = EEG_CNN_LSTM(
    n_timesteps=n_timesteps,
    n_channels=n_channels,
    num_classes=2
).to(device)

print(f"\n✅ CNN-LSTM Model Architecture:")
print(model)
total_params = sum(p.numel() for p in model.parameters())
print(f"\n   Total parameters: {total_params:,}")


riterion = nn.CrossEntropyLoss()
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

print(f"\n{'='*60}")
print(f"  TRAINING CNN-LSTM ({EPOCHS} epochs max)")
print(f"{'='*60}")

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
        print(f"  Epoch [{epoch+1:3d}/{EPOCHS}] "
              f"Train Loss: {t_l:.4f} | Train Acc: {t_acc:.4f} | "
              f"Val Acc: {v_acc:.4f} {'⭐' if v_acc == early_stopping.best_score else ''}")
    
    if early_stopping.stop:
        print(f"\n  ⏹ Early stopping at epoch {epoch+1}")
        break

# Load best model weights
model.load_state_dict(early_stopping.best_state)
model.to(device)
print(f"\n✅ Best Validation Accuracy: {early_stopping.best_score:.4f} ({early_stopping.best_score:.2%})")