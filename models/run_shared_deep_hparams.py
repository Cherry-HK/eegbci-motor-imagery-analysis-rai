import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent

SCRIPTS = [
    ROOT / "models" / "eegnet" / "eegnet_training_hparams.py",
    ROOT / "models" / "lstm" / "lstm_training_hparams.py",
    ROOT / "models" / "transformer" / "transformer_training_hparams.py",
]


def main():
    for script in SCRIPTS:
        print("\n" + "=" * 80)
        print(f"Running shared training-hyperparameter study: {script}")
        print("=" * 80)
        result = subprocess.run([sys.executable, str(script)], cwd=ROOT)
        if result.returncode != 0:
            print(f"\nStopping because {script.name} failed with exit code {result.returncode}")
            sys.exit(result.returncode)

    print("\nAll shared deep-learning hyperparameter studies completed successfully.")


if __name__ == "__main__":
    main()
