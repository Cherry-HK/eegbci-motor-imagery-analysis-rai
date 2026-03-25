import csv
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_ROOT / "models"
OUTPUT_ROOT = REPO_ROOT / "summary"

# 🔥 CHANGE THIS ONLY
MODEL_NAME = "transformer"


def read_csv_rows(path: Path):
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def write_csv(path: Path, rows):
    if not rows:
        print(f"No rows to write for {path.name}")
        return

    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def collect_summary_files():
    """
    Example:
    models/svm/svm_parameter_study/C/summary_C.csv
    """
    model_path = MODELS_DIR / MODEL_NAME / f"{MODEL_NAME}_training_hparams"
    return sorted(model_path.glob("*/summary_*.csv"))


def main():
    OUTPUT_ROOT.mkdir(exist_ok=True)

    summary_files = collect_summary_files()
    if not summary_files:
        print(f"No summary files found for model: {MODEL_NAME}")
        return

    combined_rows = []

    for summary_file in summary_files:
        parameter_name = summary_file.parent.name  # C, kernel, gamma, etc.

        rows = read_csv_rows(summary_file)

        for row in rows:
            row_with_info = {
                "model": MODEL_NAME,
                "parameter": parameter_name,
                "source_file": str(summary_file.relative_to(REPO_ROOT)),
                **row,
            }
            combined_rows.append(row_with_info)

    output_path = OUTPUT_ROOT / f"{MODEL_NAME}_dl_hparams.csv"
    write_csv(output_path, combined_rows)

    print(f"{MODEL_NAME} done ✅")
    print(f"Collected {len(summary_files)} files")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()