import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats   import randint, skew, kurtosis
from scipy.signal  import welch


# ─────────────────────────────────────────────────────────────────
#  DEVICE DETECTION
# ─────────────────────────────────────────────────────────────────

def detect_device():
    try:
        import cupy as cp  # type: ignore[import-untyped]
        if cp.cuda.runtime.getDeviceCount() == 0:
            raise RuntimeError("No CUDA device found.")
        import cuml  # type: ignore[import-untyped]  # noqa: F401
        props    = cp.cuda.runtime.getDeviceProperties(0)
        gpu_name = props["name"].decode() if isinstance(props["name"], bytes) else props["name"]
        print(f"  ✔  CUDA detected — using cuML  [{gpu_name}]")
        return True, f"GPU ({gpu_name})"
    except Exception as exc:
        print(f"  ℹ  cuML / CUDA not available  ({exc.__class__.__name__}: {exc})")
        print("     Falling back to sklearn (CPU).")
        return False, "CPU (sklearn)"


# ─────────────────────────────────────────────────────────────────
#  1. LOAD DATA
# ─────────────────────────────────────────────────────────────────

def load_data():
    print("=" * 62)
    print("  EEG Decision Tree Classifier  —  Feature Engineering")
    print("=" * 62)
    print("\n[1/6] Loading dataset...")

    X_train = np.load("X_train.npy")
    X_test  = np.load("X_test.npy")
    y_train = np.load("y_train.npy")
    y_test  = np.load("y_test.npy")

    print(f"  X_train : {X_train.shape}  |  y_train : {y_train.shape}")
    print(f"  X_test  : {X_test.shape}  |  y_test  : {y_test.shape}")

    n_classes = len(np.unique(y_train))
    print(f"  Classes : {np.unique(y_train)}  (n = {n_classes})")
    return X_train, X_test, y_train, y_test, n_classes


# ─────────────────────────────────────────────────────────────────
#  2. DEVICE SETUP
# ─────────────────────────────────────────────────────────────────

def setup_device():
    print("\n[2/6] Detecting compute device...")
    use_gpu, device_label = detect_device()
    print(f"  Device : {device_label}")
    return use_gpu, device_label


# ─────────────────────────────────────────────────────────────────
#  3. FEATURE EXTRACTION  (N, C, T) → (N, C × 14)
# ─────────────────────────────────────────────────────────────────

def _band_power(psd, freqs, fmin, fmax):
    """Integrate PSD within a frequency band (trapezoidal rule)."""
    idx = np.logical_and(freqs >= fmin, freqs <= fmax)
    return np.trapezoid(psd[idx], freqs[idx]) if idx.any() else 0.0


def _spectral_entropy(psd):
    """Normalised spectral entropy of a PSD vector."""
    p = psd / (psd.sum() + 1e-12)
    return -np.sum(p * np.log2(p + 1e-12))


def _hjorth(signal):
    """Hjorth mobility and complexity."""
    diff1 = np.diff(signal)
    diff2 = np.diff(diff1)
    var0  = np.var(signal)  + 1e-12
    var1  = np.var(diff1)   + 1e-12
    var2  = np.var(diff2)   + 1e-12
    mobility   = np.sqrt(var1 / var0)
    complexity = np.sqrt(var2 / var1) / mobility
    return mobility, complexity


def extract_features(X, fs=128, label=""):
    """
    Extract 14 features per channel from EEG data (N, C, T).

    Features per channel (14 total):
      [0]  mean
      [1]  std
      [2]  variance
      [3]  skewness
      [4]  kurtosis
      [5]  peak-to-peak amplitude
      [6]  zero-crossing rate
      [7]  Hjorth mobility
      [8]  Hjorth complexity
      [9]  delta  band power  (0.5–4 Hz)
      [10] theta  band power  (4–8 Hz)
      [11] alpha  band power  (8–13 Hz)
      [12] beta   band power  (13–30 Hz)
      [13] gamma  band power  (30–50 Hz)  [+ spectral entropy below]

    Returns array of shape (N, C * 14).
    """
    N, C, T = X.shape
    N_FEAT  = 14
    out     = np.zeros((N, C * N_FEAT), dtype=np.float32)

    nperseg = min(T, 256)   # Welch segment length

    for i in range(N):
        for c in range(C):
            sig  = X[i, c].astype(np.float64)
            base = c * N_FEAT

            # ── Time-domain ─────────────────────────────────────
            out[i, base + 0] = np.mean(sig)
            out[i, base + 1] = np.std(sig)
            out[i, base + 2] = np.var(sig)
            out[i, base + 3] = float(skew(sig))
            out[i, base + 4] = float(kurtosis(sig))
            out[i, base + 5] = np.ptp(sig)                              # peak-to-peak
            out[i, base + 6] = ((np.diff(np.sign(sig)) != 0).sum()      # zero-crossing rate
                                 / (T - 1))
            mob, cmp         = _hjorth(sig)
            out[i, base + 7] = mob
            out[i, base + 8] = cmp

            # ── Frequency-domain (Welch PSD) ─────────────────────
            freqs, psd       = welch(sig, fs=fs, nperseg=nperseg)
            out[i, base +  9] = _band_power(psd, freqs,  0.5,  4.0)    # delta
            out[i, base + 10] = _band_power(psd, freqs,  4.0,  8.0)    # theta
            out[i, base + 11] = _band_power(psd, freqs,  8.0, 13.0)    # alpha
            out[i, base + 12] = _band_power(psd, freqs, 13.0, 30.0)    # beta
            out[i, base + 13] = _band_power(psd, freqs, 30.0, 50.0)    # gamma

        if (i + 1) % max(1, N // 5) == 0:
            print(f"    {label}  {i + 1:>5}/{N} samples processed...")

    return out


def build_features(X_train, X_test, fs=128):
    print("\n[3/6] Extracting EEG features (N, C, T) → (N, C × 14)...")
    print("  Features per channel : mean, std, var, skewness, kurtosis,")
    print("                         peak-to-peak, zero-crossing rate,")
    print("                         Hjorth mobility & complexity,")
    print("                         delta / theta / alpha / beta / gamma power")

    t0 = time.time()
    X_train_feat = extract_features(X_train, fs=fs, label="train")
    X_test_feat  = extract_features(X_test,  fs=fs, label="test ")
    elapsed      = time.time() - t0

    print(f"  Feature matrix X_train : {X_train_feat.shape}")
    print(f"  Feature matrix X_test  : {X_test_feat.shape}")
    print(f"  Extraction completed in {elapsed:.1f}s")
    return X_train_feat, X_test_feat


# ─────────────────────────────────────────────────────────────────
#  4. OPTIONAL GPU TRANSFER
# ─────────────────────────────────────────────────────────────────

def to_device(X_train_feat, X_test_feat, y_train, use_gpu):
    if use_gpu:
        import cupy as cp  # type: ignore[import-untyped]
        print("\n  Transferring feature arrays to GPU (CuPy)...")
        X_train_feat = cp.asarray(X_train_feat)
        X_test_feat  = cp.asarray(X_test_feat)
        y_train      = cp.asarray(y_train.astype(np.int32))
        print("  ✔  Data resident on GPU.")
    return X_train_feat, X_test_feat, y_train


# ─────────────────────────────────────────────────────────────────
#  5. HYPERPARAMETER SEARCH + TRAINING
# ─────────────────────────────────────────────────────────────────

def train_with_search(X_train_feat, y_train, use_gpu):
    print("\n[4/6] Hyperparameter optimisation (RandomizedSearchCV, n_iter=50)...")

    criteria = ["gini", "entropy"] if use_gpu else ["gini", "entropy", "log_loss"]

    param_dist = {
        "criterion"        : criteria,
        "max_depth"        : [None] + list(range(5, 61, 5)),
        "min_samples_split": randint(2, 20),
        "min_samples_leaf" : randint(1, 10),
        "max_features"     : [None, "sqrt", "log2", 0.5, 0.75],
        "splitter"         : ["best", "random"],
        "class_weight"     : [None, "balanced"],
    }

    from sklearn.model_selection import RandomizedSearchCV

    if use_gpu:
        from cuml.tree import DecisionTreeClassifier as DTC  # type: ignore[import-untyped]
        base_clf = DTC()
        n_jobs   = 1
        print("  Backend : cuML  (GPU-accelerated)")
    else:
        from sklearn.tree import DecisionTreeClassifier as DTC
        base_clf = DTC(random_state=42)
        n_jobs   = -1
        print("  Backend : sklearn  (CPU)")

    search = RandomizedSearchCV(
        estimator           = base_clf,
        param_distributions = param_dist,
        n_iter              = 50,
        scoring             = "accuracy",
        cv                  = 5,
        n_jobs              = n_jobs,
        verbose             = 1,
        random_state        = 42,
    )

    t0 = time.time()
    search.fit(X_train_feat, y_train)
    elapsed = time.time() - t0

    print(f"\n  Search completed in {elapsed:.1f}s")
    print("  ── Best hyperparameters ────────────────────────────")
    for k, v in search.best_params_.items():
        print(f"     {k:<22}: {v}")
    print(f"     {'CV accuracy':<22}: {search.best_score_ * 100:.2f}%")
    print("  ────────────────────────────────────────────────────")

    return search.best_estimator_


# ─────────────────────────────────────────────────────────────────
#  6. EVALUATION
# ─────────────────────────────────────────────────────────────────

def evaluate(model, X_test_feat, y_test, n_classes, use_gpu, device_label):
    print("\n[5/6] Evaluating on test set...")

    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score,
        precision_score, recall_score, f1_score, roc_auc_score,
    )

    y_pred_raw = model.predict(X_test_feat)
    y_prob_raw = model.predict_proba(X_test_feat)

    if use_gpu:
        import cupy as cp  # type: ignore[import-untyped]
        y_pred = cp.asnumpy(y_pred_raw).astype(int)
        y_prob = cp.asnumpy(y_prob_raw).astype(float)
    else:
        y_pred = np.array(y_pred_raw, dtype=int)
        y_prob = np.array(y_prob_raw, dtype=float)

    avg = "macro" if n_classes > 2 else "binary"

    roc_auc = (
        roc_auc_score(y_test, y_prob[:, 1])
        if n_classes == 2
        else roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")
    )

    acc     = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    prec    = precision_score(y_test, y_pred, average=avg, zero_division=0)
    rec     = recall_score(y_test, y_pred, average=avg, zero_division=0)
    f1      = f1_score(y_test, y_pred, average=avg, zero_division=0)

    print("\n" + "=" * 62)
    print(f"  Evaluation Results  [{device_label}]")
    print("=" * 62)
    print(f"  Accuracy          : {acc     * 100:.2f}%")
    print(f"  Balanced Accuracy : {bal_acc * 100:.2f}%")
    print(f"  Precision         : {prec    * 100:.2f}%")
    print(f"  Recall            : {rec     * 100:.2f}%")
    print(f"  F1 Score          : {f1      * 100:.2f}%")
    print(f"  ROC-AUC           : {roc_auc * 100:.2f}%")
    print("=" * 62)

    return y_pred, y_prob


# ─────────────────────────────────────────────────────────────────
#  7. VISUALISATION
# ─────────────────────────────────────────────────────────────────

def plot_results(y_test, y_pred, y_prob, n_classes, device_label):
    from sklearn.metrics import (
        confusion_matrix, roc_auc_score, roc_curve, ConfusionMatrixDisplay,
    )

    classes = np.unique(y_test)
    fig = plt.figure(figsize=(15, 6))
    fig.suptitle(
        f"EEG Decision Tree — Evaluation  [{device_label}]",
        fontsize=14, fontweight="bold",
    )
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

    # ── Confusion Matrix ──────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix(y_test, y_pred),
        display_labels=classes,
    ).plot(ax=ax1, colorbar=True, cmap="Blues")
    ax1.set_title("Confusion Matrix", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Predicted Label")
    ax1.set_ylabel("True Label")

    # ── ROC Curve ─────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])

    if n_classes == 2:
        fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
        auc_val = roc_auc_score(y_test, y_prob[:, 1])
        ax2.plot(fpr, tpr, lw=2, color="#2563EB",
                 label=f"ROC (AUC = {auc_val:.2f})")
    else:
        from sklearn.preprocessing import label_binarize
        y_bin  = label_binarize(y_test, classes=classes)
        colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
        for i, (cls, col) in enumerate(zip(classes, colors)):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
            auc_val = roc_auc_score(y_bin[:, i], y_prob[:, i])
            ax2.plot(fpr, tpr, lw=1.8, color=col,
                     label=f"Class {cls}  (AUC = {auc_val:.2f})")

    ax2.plot([0, 1], [0, 1], "k--", lw=1, label="Chance")
    ax2.set_xlim([0, 1]);  ax2.set_ylim([0, 1.02])
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.set_title("ROC Curve", fontsize=12, fontweight="bold")
    ax2.legend(loc="lower right", fontsize=8)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────

def main():
    X_train, X_test, y_train, y_test, n_classes = load_data()
    use_gpu, device_label                        = setup_device()
    X_train_feat, X_test_feat                    = build_features(X_train, X_test, fs=128)
    X_train_feat, X_test_feat, y_train_dev       = to_device(X_train_feat, X_test_feat,
                                                              y_train, use_gpu)
    best_model                                   = train_with_search(X_train_feat, y_train_dev, use_gpu)
    y_pred, y_prob                               = evaluate(best_model, X_test_feat, y_test,
                                                            n_classes, use_gpu, device_label)
    plot_results(y_test, y_pred, y_prob, n_classes, device_label)


if __name__ == "__main__":
    main()