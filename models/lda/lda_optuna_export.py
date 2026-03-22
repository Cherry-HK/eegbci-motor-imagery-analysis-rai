import csv
import json
import os

import joblib
import numpy as np
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

DATA_DIR = os.path.join("models", "preprocessing_result")
MODEL_DIR = os.path.join("models", "lda")
OPTUNA_DIR = os.path.join(MODEL_DIR, "lda_optuna")
BEST_CONFIG_CSV = os.path.join(OPTUNA_DIR, "all_trials_summary.csv")
MODEL_PATH = os.path.join(MODEL_DIR, "lda_optuna_model.joblib")
METADATA_PATH = os.path.join(MODEL_DIR, "lda_optuna_metadata.json")


def parse_value(value):
    if value in {"None", "", None, "unused"}:
        return value
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
        "solver": best["solver"],
        "shrinkage": parse_value(best["shrinkage"]),
        "csp_components": parse_value(best["csp_components"]),
    }
    X = np.load(os.path.join(DATA_DIR, "X.npy"))
    y = np.load(os.path.join(DATA_DIR, "y.npy"))
    subjects = np.load(os.path.join(DATA_DIR, "subjects.npy"))
    csp = CSP(n_components=best_config["csp_components"], reg="ledoit_wolf", log=True, norm_trace=False)
    X_csp = csp.fit_transform(X, y)
    classifier = LinearDiscriminantAnalysis(solver=best_config["solver"], shrinkage=None if best_config["shrinkage"] == "unused" else best_config["shrinkage"])
    classifier.fit(X_csp, y)
    bundle = {"csp": csp, "classifier": classifier, "class_labels": [0, 1], "subjects_seen": sorted(np.unique(subjects).astype(int).tolist()), "best_config": best_config}
    joblib.dump(bundle, MODEL_PATH)
    with open(METADATA_PATH, "w", encoding="utf-8") as jsonfile:
        json.dump({
            "model_type": "lda_optuna",
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
        }, jsonfile, indent=2)
