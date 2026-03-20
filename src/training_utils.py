import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


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


class AugmentedEEGDataset(Dataset):
    """EEG dataset with optional data augmentation."""

    def __init__(self, X, y, augment=False, time_shift=0):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        self.augment = augment
        self.time_shift = time_shift

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x, y = self.X[idx].clone(), self.y[idx]

        if self.augment:
            # Gaussian noise (50% probability)
            if np.random.rand() > 0.5:
                noise = torch.randn_like(x) * 0.01
                x = x + noise

            # Time shift (50% probability)
            if self.time_shift > 0 and np.random.rand() > 0.5:
                shift = np.random.randint(-self.time_shift, self.time_shift + 1)
                if shift != 0:
                    x = torch.roll(x, shifts=shift, dims=-1)

        return x, y
