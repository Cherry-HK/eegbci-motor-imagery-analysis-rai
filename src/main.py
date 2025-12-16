"""
Main Example Script for EEG Preprocessing

This script demonstrates the complete preprocessing pipeline for EEG data
from the PhysioNet EEG Motor Movement/Imagery Dataset.

The preprocessing pipeline includes:
1. Data loading from PhysioNet
2. Re-referencing (optional)
3. Filtering (bandpass and notch)
4. ICA artifact removal (optional)
5. Epoching
6. Bad epoch rejection (optional)
"""

import argparse
import logging
from pathlib import Path
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Import preprocessing functions
from preprocessing import (
    preprocess_pipeline,
    get_motor_imagery_event_dict,
    get_recommended_runs
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_preprocessing_example(
    subject: int = 1,
    task_type: str = 'imagery_left_right',
    apply_ica: bool = False,
    use_enhanced_ica: bool = False,
    use_autoreject: bool = False,
    ref_type: str = None,
    output_dir: str = '../outputs',
    data_path: str = None,
    save_epochs: bool = False,
    verbose: bool = True
):
    """
    Run preprocessing example for a single subject.

    Parameters
    ----------
    subject : int
        Subject number (1-109)
    task_type : str
        Type of motor imagery task:
        - 'imagery_left_right': Imagined left/right fist movement
        - 'imagery_fists_feet': Imagined both fists/feet movement
        - 'real_left_right': Real left/right fist movement
        - 'real_fists_feet': Real both fists/feet movement
    apply_ica : bool
        Whether to apply basic ICA for artifact removal
    use_enhanced_ica : bool
        Whether to use enhanced ICA with EOG/ECG/muscle detection
    use_autoreject : bool
        Whether to use automatic bad epoch rejection
    ref_type : str, optional
        Type of re-referencing: 'average', 'mastoid', 'custom'
        If None, no re-referencing is applied
    output_dir : str
        Directory to save outputs
    data_path : str, optional
        Path to data directory
    save_epochs : bool
        Whether to save the preprocessed epochs to disk
    verbose : bool
        Whether to print progress information
    """
    if verbose:
        print("=" * 70)
        print(f"EEG PREPROCESSING EXAMPLE - SUBJECT {subject}")
        print("=" * 70)

    # Setup output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get recommended runs for task type
    runs = get_recommended_runs(task_type)

    # Get event dictionary
    if 'left_right' in task_type:
        event_id = get_motor_imagery_event_dict('left_right_fist')
    else:
        event_id = get_motor_imagery_event_dict('fists_feet')

    if verbose:
        print(f"\nConfiguration:")
        print(f"  Subject: {subject}")
        print(f"  Task Type: {task_type}")
        print(f"  Runs: {runs}")
        print(f"  Events: {event_id}")
        print(f"  Re-referencing: {ref_type if ref_type else 'None'}")
        print(f"  ICA: {'Enhanced' if use_enhanced_ica else ('Basic' if apply_ica else 'None')}")
        print(f"  Auto-reject: {use_autoreject}")

    # Run preprocessing pipeline
    if verbose:
        print("\n" + "-" * 70)
        print("RUNNING PREPROCESSING PIPELINE")
        print("-" * 70)

    epochs, raw = preprocess_pipeline(
        subject=subject,
        runs=runs,
        event_id=event_id,
        data_path=data_path,
        apply_ica=apply_ica,
        use_enhanced_ica=use_enhanced_ica,
        use_autoreject=use_autoreject,
        ref_type=ref_type,
        l_freq=7.0,
        h_freq=30.0,
        notch_freq=60.0,
        tmin=-1.0,
        tmax=4.0,
        baseline=(-1.0, 0.0),
        verbose=verbose
    )

    # Print summary
    if verbose:
        print("\n" + "-" * 70)
        print("PREPROCESSING SUMMARY")
        print("-" * 70)
        print(f"Number of epochs: {len(epochs)}")
        print(f"Number of channels: {len(epochs.ch_names)}")
        print(f"Sampling frequency: {epochs.info['sfreq']} Hz")
        print(f"Epoch duration: {epochs.tmax - epochs.tmin} seconds")
        print(f"Data shape: {epochs.get_data().shape}")
        print(f"  (n_epochs, n_channels, n_timepoints)")

    # Save epochs if requested
    if save_epochs:
        epochs_file = output_dir / f"subject_{subject}_epochs_preprocessed.fif"
        epochs.save(epochs_file, overwrite=True)
        if verbose:
            print(f"\nPreprocessed epochs saved to: {epochs_file}")

    if verbose:
        print("\n" + "=" * 70)
        print("PREPROCESSING COMPLETE!")
        print("=" * 70)
        print(f"\nNow can use these preprocessed epochs for:")
        print("  - Feature extraction")
        print("  - Machine learning classification")
        print("  - Statistical analysis")
        print("  - Further analysis as needed")

    return epochs, raw


def main():
    """Main entry point for the preprocessing example."""
    parser = argparse.ArgumentParser(
        description='EEG Preprocessing Pipeline Example',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic preprocessing (no ICA, no re-referencing)
  python main.py --subject 1 --task imagery_left_right

  # With average reference and basic ICA
  python main.py --subject 1 --task imagery_left_right --ref average --ica

  # With enhanced ICA (EOG/ECG detection) and auto-reject
  python main.py --subject 1 --task imagery_left_right --enhanced-ica --autoreject

  # Save preprocessed epochs to disk
  python main.py --subject 1 --task imagery_left_right --save-epochs
        """
    )

    parser.add_argument(
        '--subject',
        type=int,
        default=1,
        help='Subject number (1-109). Default: 1'
    )

    parser.add_argument(
        '--task',
        type=str,
        default='imagery_left_right',
        choices=['imagery_left_right', 'imagery_fists_feet',
                 'real_left_right', 'real_fists_feet'],
        help='Type of motor imagery task. Default: imagery_left_right'
    )

    parser.add_argument(
        '--ica',
        action='store_true',
        help='Apply basic ICA for artifact removal (EOG detection only)'
    )

    parser.add_argument(
        '--enhanced-ica',
        action='store_true',
        help='Apply enhanced ICA with EOG/ECG/muscle artifact detection'
    )

    parser.add_argument(
        '--autoreject',
        action='store_true',
        help='Use automatic bad epoch rejection based on amplitude'
    )

    parser.add_argument(
        '--ref',
        type=str,
        default=None,
        choices=['average', 'mastoid', 'custom'],
        help='Type of re-referencing to apply. Default: None'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='../outputs',
        help='Output directory for results. Default: ../outputs'
    )

    parser.add_argument(
        '--data-path',
        type=str,
        default=None,
        help='Path to data directory. Default: ~/mne_data'
    )

    parser.add_argument(
        '--save-epochs',
        action='store_true',
        help='Save preprocessed epochs to disk (.fif format)'
    )

    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress verbose output'
    )

    args = parser.parse_args()

    verbose = not args.quiet

    # Run preprocessing
    run_preprocessing_example(
        subject=args.subject,
        task_type=args.task,
        apply_ica=args.ica,
        use_enhanced_ica=args.enhanced_ica,
        use_autoreject=args.autoreject,
        ref_type=args.ref,
        output_dir=args.output,
        data_path=args.data_path,
        save_epochs=args.save_epochs,
        verbose=verbose
    )


if __name__ == '__main__':
    main()
