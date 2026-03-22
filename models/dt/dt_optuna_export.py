import csv
import json
import os

import joblib
import numpy as np
from mne.decoding import CSP
from sklearn.tree import DecisionTreeClassifier

DATA_DIR = os.path.join("models", "preprocessing_result")
MODEL_DIR = os.path.join("models", "dt")
OPTUNA_DIR = os.path.join(MODEL_DIR, "dt_optuna")
BEST_CONFIG_CSV = os.path.join(OPTUNA_DIR, "all_trials_summary.csv")
MODEL_PATH = os.path.join(MODEL_DIR, "dt_optuna_model.joblib")
METADATA_PATH = os.path.join(MODEL_DIR, "dt_optuna_metadata.json")


def parse_value(value):
    if value in {"None", "", None}:
        return None
    try:
        numeric = float(value)
        return int(numeric) if numeric.is_integer() else numeric
    except ValueError:
        return value


if __name__ == "__main__":
    with open(BEST_CONFIG_CSV, "r", newline="", encoding="utf-8") as csvfile:
        best = next(csv.DictReader(csvfile))

    best_config = {
        "trial_number": int(best["trial_number"]),
        "criterion": best["criterion"],
        "max_depth": parse_value(best["max_depth"]),
        "min_samples_split": parse_value(best["min_samples_split"]),
        "min_samples_leaf": parse_value(best["min_samples_leaf"]),
        "class_weight": parse_value(best["class_weight"]),
        "csp_components": parse_value(best["csp_components"]),
    }

    X = np.load(os.path.join(DATA_DIR, "X.npy"))
    y = np.load(os.path.join(DATA_DIR, "y.npy"))
    subjects = np.load(os.path.join(DATA_DIR, "subjects.npy"))

    csp = CSP(n_components=best_config["csp_components"], reg="ledoit_wolf", log=True, norm_trace=False)
    X_csp = csp.fit_transform(X, y)
    classifier = DecisionTreeClassifier(
        criterion=best_config["criterion"],
        max_depth=best_config["max_depth"],
        min_samples_split=best_config["min_samples_split"],
        min_samples_leaf=best_config["min_samples_leaf"],
        class_weight=best_config["class_weight"],
        random_state=42,
    )
    classifier.fit(X_csp, y)

    bundle = {
        "csp": csp,
        "classifier": classifier,
        "class_labels": [0, 1],
        "subjects_seen": sorted(np.unique(subjects).astype(int).tolist()),
        "best_config": best_config,
    }
    joblib.dump(bundle, MODEL_PATH)

    with open(METADATA_PATH, "w", encoding="utf-8") as jsonfile:
        json.dump(
            {
                "model_type": "dt_optuna",
                "saved_model_path": MODEL_PATH,
                "source_results_csv": BEST_CONFIG_CSV,
                "class_labels": [0, 1],
                "subjects_seen": sorted(np.unique(subjects).astype(int).tolist()),
                "best_config": best_config,
                "evaluation_summary": {
                    "rank": int(best["rank"]),
                    "trial_number": int(best["trial_number"]),
                    "mean_accuracy": float(best["mean_accuracy"]),
                    "std_accuracy": float(best["std_accuracy"]),
                    "mean_f1": float(best["mean_f1"]),
                    "mean_auc": float(best["mean_auc"]),
                },
            },
            jsonfile,
            indent=2,
        )
