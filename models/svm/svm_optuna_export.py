import csv
import json
import os

import joblib
import numpy as np
from mne.decoding import CSP
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


DATA_DIR = os.path.join("models", "preprocessing_result")
MODEL_DIR = os.path.join("models", "svm")
OPTUNA_DIR = os.path.join(MODEL_DIR, "svm_optuna")
BEST_CONFIG_CSV = os.path.join(OPTUNA_DIR, "all_trials_summary.csv")
MODEL_PATH = os.path.join(MODEL_DIR, "svm_optuna_model.joblib")
METADATA_PATH = os.path.join(MODEL_DIR, "svm_optuna_metadata.json")


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
            f"Optuna summary CSV not found: {path}. Run models/svm/svm_optuna.py first."
        )

    with open(path, "r", newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        best_row = next(reader, None)

    if best_row is None:
        raise ValueError(f"No rows found in Optuna summary CSV: {path}")

    return {
        "trial_number": int(best_row["trial_number"]),
        "kernel": best_row["kernel"],
        "C": parse_numeric(best_row["C"]),
        "gamma": parse_numeric(best_row["gamma"]),
        "degree": parse_numeric(best_row["degree"]),
        "coef0": parse_numeric(best_row["coef0"]),
        "class_weight": parse_optional(best_row["class_weight"]),
        "csp_components": parse_numeric(best_row["csp_components"]),
        "mean_accuracy": float(best_row["mean_accuracy"]),
        "std_accuracy": float(best_row["std_accuracy"]),
        "mean_f1": float(best_row["mean_f1"]),
        "mean_auc": float(best_row["mean_auc"]),
        "rank": int(best_row["rank"]),
    }


def build_svm_pipeline(best_config):
    svm_kwargs = {
        "kernel": best_config["kernel"],
        "C": best_config["C"],
        "class_weight": best_config["class_weight"],
        "probability": False,
        "random_state": 42,
    }

    if best_config["kernel"] in {"rbf", "poly", "sigmoid"} and best_config["gamma"] != "unused":
        svm_kwargs["gamma"] = best_config["gamma"]
    if best_config["kernel"] == "poly":
        if best_config["degree"] != "unused":
            svm_kwargs["degree"] = best_config["degree"]
        if best_config["coef0"] != "unused":
            svm_kwargs["coef0"] = best_config["coef0"]
    elif best_config["kernel"] == "sigmoid" and best_config["coef0"] != "unused":
        svm_kwargs["coef0"] = best_config["coef0"]

    csp = CSP(
        n_components=best_config["csp_components"],
        reg="ledoit_wolf",
        log=True,
        norm_trace=False,
    )
    classifier = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svm", SVC(**svm_kwargs)),
        ]
    )
    return csp, classifier


if __name__ == "__main__":
    os.makedirs(MODEL_DIR, exist_ok=True)

    X = np.load(os.path.join(DATA_DIR, "X.npy"))
    y = np.load(os.path.join(DATA_DIR, "y.npy"))
    subjects = np.load(os.path.join(DATA_DIR, "subjects.npy"))

    best_config = load_best_configuration(BEST_CONFIG_CSV)
    print("Loaded best SVM Optuna configuration:", best_config)

    csp, classifier = build_svm_pipeline(best_config)
    X_csp = csp.fit_transform(X, y)
    classifier.fit(X_csp, y)

    export_bundle = {
        "csp": csp,
        "classifier": classifier,
        "class_labels": [0, 1],
        "subjects_seen": sorted(np.unique(subjects).astype(int).tolist()),
        "best_config": {
            "trial_number": best_config["trial_number"],
            "kernel": best_config["kernel"],
            "C": best_config["C"],
            "gamma": best_config["gamma"],
            "degree": best_config["degree"],
            "coef0": best_config["coef0"],
            "class_weight": best_config["class_weight"],
            "csp_components": best_config["csp_components"],
        },
    }
    joblib.dump(export_bundle, MODEL_PATH)

    metadata = {
        "model_type": "svm_optuna",
        "saved_model_path": MODEL_PATH,
        "source_results_csv": BEST_CONFIG_CSV,
        "class_labels": [0, 1],
        "subjects_seen": sorted(np.unique(subjects).astype(int).tolist()),
        "best_config": export_bundle["best_config"],
        "evaluation_summary": {
            "rank": best_config["rank"],
            "trial_number": best_config["trial_number"],
            "mean_accuracy": best_config["mean_accuracy"],
            "std_accuracy": best_config["std_accuracy"],
            "mean_f1": best_config["mean_f1"],
            "mean_auc": best_config["mean_auc"],
        },
    }
    with open(METADATA_PATH, "w", encoding="utf-8") as jsonfile:
        json.dump(metadata, jsonfile, indent=2)

    print("Saved deployable Optuna model to:", MODEL_PATH)
    print("Saved metadata to:", METADATA_PATH)
