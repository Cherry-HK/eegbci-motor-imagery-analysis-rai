import csv
import json
import os

import joblib
import numpy as np
from mne.decoding import CSP
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA_DIR = os.path.join("models", "preprocessing_result")
MODEL_DIR = os.path.join("models", "knn")
OPTUNA_DIR = os.path.join(MODEL_DIR, "knn_optuna")
BEST_CONFIG_CSV = os.path.join(OPTUNA_DIR, "all_trials_summary.csv")
MODEL_PATH = os.path.join(MODEL_DIR, "knn_optuna_model.joblib")
METADATA_PATH = os.path.join(MODEL_DIR, "knn_optuna_metadata.json")


def parse_numeric(value):
    if value in {"None", "", None, "unused"}:
        return value
    numeric = float(value)
    return int(numeric) if numeric.is_integer() else numeric


if __name__ == "__main__":
    with open(BEST_CONFIG_CSV, "r", newline="", encoding="utf-8") as csvfile:
        best = next(csv.DictReader(csvfile))

    best_config = {
        "trial_number": int(best["trial_number"]),
        "n_neighbors": parse_numeric(best["n_neighbors"]),
        "weights": best["weights"],
        "metric": best["metric"],
        "p": parse_numeric(best["p"]),
        "csp_components": parse_numeric(best["csp_components"]),
    }

    X = np.load(os.path.join(DATA_DIR, "X.npy"))
    y = np.load(os.path.join(DATA_DIR, "y.npy"))
    subjects = np.load(os.path.join(DATA_DIR, "subjects.npy"))

    csp = CSP(n_components=best_config["csp_components"], reg="ledoit_wolf", log=True, norm_trace=False)
    X_csp = csp.fit_transform(X, y)
    kwargs = {"n_neighbors": best_config["n_neighbors"], "weights": best_config["weights"], "metric": best_config["metric"]}
    if best_config["metric"] == "minkowski":
        kwargs["p"] = best_config["p"]
    classifier = Pipeline([("scaler", StandardScaler()), ("knn", KNeighborsClassifier(**kwargs))])
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
                "model_type": "knn_optuna",
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
