import csv
import json
import os

import joblib
import numpy as np
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

DATA_DIR = os.path.join("models", "preprocessing_result")
MODEL_DIR = os.path.join("models", "riemann")
OPTUNA_DIR = os.path.join(MODEL_DIR, "riemann_optuna")
BEST_CONFIG_CSV = os.path.join(OPTUNA_DIR, "all_trials_summary.csv")
MODEL_PATH = os.path.join(MODEL_DIR, "riemann_optuna_model.joblib")
METADATA_PATH = os.path.join(MODEL_DIR, "riemann_optuna_metadata.json")


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
        "cov_estimator": best["cov_estimator"],
        "ts_metric": best["ts_metric"],
        "C": parse_value(best["C"]),
        "solver": best["solver"],
        "class_weight": parse_value(best["class_weight"]),
    }
    X = np.load(os.path.join(DATA_DIR, "X.npy"))
    y = np.load(os.path.join(DATA_DIR, "y.npy"))
    subjects = np.load(os.path.join(DATA_DIR, "subjects.npy"))
    pipeline = Pipeline([("cov", Covariances(estimator=best_config["cov_estimator"])), ("ts", TangentSpace(metric=best_config["ts_metric"])), ("lr", LogisticRegression(C=best_config["C"], solver=best_config["solver"], class_weight=best_config["class_weight"], max_iter=1000, random_state=42))])
    pipeline.fit(X, y)
    bundle = {"pipeline": pipeline, "class_labels": [0, 1], "subjects_seen": sorted(np.unique(subjects).astype(int).tolist()), "best_config": best_config}
    joblib.dump(bundle, MODEL_PATH)
    with open(METADATA_PATH, "w", encoding="utf-8") as jsonfile:
        json.dump({
            "model_type": "riemann_optuna",
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
