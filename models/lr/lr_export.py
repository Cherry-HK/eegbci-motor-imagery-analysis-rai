import csv
import json
import os

import joblib
import numpy as np
from mne.decoding import CSP
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DATA_DIR = os.path.join("models", "preprocessing_result")
MODEL_DIR = os.path.join("models", "lr")
TESTING_DIR = os.path.join(MODEL_DIR, "lr_testing")
BEST_CONFIG_CSV = os.path.join(TESTING_DIR, "all_combinations_summary.csv")
MODEL_PATH = os.path.join(MODEL_DIR, "lr_model.joblib")
METADATA_PATH = os.path.join(MODEL_DIR, "lr_metadata.json")


def parse_optional(value):
    if value in {"None", "", None}:
        return None
    if value == "unused":
        return value
    return value


def parse_numeric(value):
    value = parse_optional(value)
    if value in {None, "unused"}:
        return value
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return value
    if numeric.is_integer():
        return int(numeric)
    return numeric


def load_best_configuration(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Best-configuration CSV not found: {path}. Run models/lr/lr_testing.py first."
        )

    with open(path, "r", newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        best_row = next(reader, None)

    if best_row is None:
        raise ValueError(f"No rows found in best-configuration CSV: {path}")

    return {
        "penalty": best_row["penalty"],
        "C": parse_numeric(best_row["C"]),
        "solver": best_row["solver"],
        "class_weight": parse_optional(best_row["class_weight"]),
        "l1_ratio": parse_numeric(best_row["l1_ratio"]),
        "csp_components": parse_numeric(best_row["csp_components"]),
        "mean_accuracy": float(best_row["mean_accuracy"]),
        "std_accuracy": float(best_row["std_accuracy"]),
        "mean_f1": float(best_row["mean_f1"]),
        "mean_auc": float(best_row["mean_auc"]),
        "rank": int(best_row["rank"]),
    }


def build_lr_pipeline(best_config):
    csp = CSP(
        n_components=best_config["csp_components"],
        reg="ledoit_wolf",
        log=True,
        norm_trace=False,
    )
    classifier = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "lr",
                LogisticRegression(
                    penalty=best_config["penalty"],
                    C=best_config["C"],
                    solver=best_config["solver"],
                    class_weight=best_config["class_weight"],
                    l1_ratio=best_config["l1_ratio"] if best_config["l1_ratio"] != "unused" else None,
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )
    return csp, classifier


if __name__ == "__main__":
    os.makedirs(MODEL_DIR, exist_ok=True)

    X = np.load(os.path.join(DATA_DIR, "X.npy"))
    y = np.load(os.path.join(DATA_DIR, "y.npy"))
    subjects = np.load(os.path.join(DATA_DIR, "subjects.npy"))

    best_config = load_best_configuration(BEST_CONFIG_CSV)
    print("Loaded best LR configuration:", best_config)

    csp, classifier = build_lr_pipeline(best_config)
    X_csp = csp.fit_transform(X, y)
    classifier.fit(X_csp, y)

    export_bundle = {
        "csp": csp,
        "classifier": classifier,
        "class_labels": [0, 1],
        "subjects_seen": sorted(np.unique(subjects).astype(int).tolist()),
        "best_config": {
            "penalty": best_config["penalty"],
            "C": best_config["C"],
            "solver": best_config["solver"],
            "class_weight": best_config["class_weight"],
            "l1_ratio": best_config["l1_ratio"],
            "csp_components": best_config["csp_components"],
        },
    }
    joblib.dump(export_bundle, MODEL_PATH)

    metadata = {
        "model_type": "lr",
        "saved_model_path": MODEL_PATH,
        "source_results_csv": BEST_CONFIG_CSV,
        "class_labels": [0, 1],
        "subjects_seen": sorted(np.unique(subjects).astype(int).tolist()),
        "best_config": export_bundle["best_config"],
        "evaluation_summary": {
            "rank": best_config["rank"],
            "mean_accuracy": best_config["mean_accuracy"],
            "std_accuracy": best_config["std_accuracy"],
            "mean_f1": best_config["mean_f1"],
            "mean_auc": best_config["mean_auc"],
        },
    }
    with open(METADATA_PATH, "w", encoding="utf-8") as jsonfile:
        json.dump(metadata, jsonfile, indent=2)

    print("Saved deployable model to:", MODEL_PATH)
    print("Saved metadata to:", METADATA_PATH)
