"""
EEG Preprocessing Module for Motor Imagery Analysis

This module provides functions for loading, filtering, and cleaning EEG data
from the PhysioNet EEG Motor Movement/Imagery Dataset.
"""

import mne
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import logging

# Setup logger
logger = logging.getLogger(__name__)


def load_eeg_data(
    subject: int,
    runs: List[int],
    data_path: Optional[str] = None
) -> mne.io.Raw:
    """
    Load EEG data for a specific subject and runs.

    Parameters
    ----------
    subject : int
        Subject number (1-109)
    runs : list of int
        List of run numbers to load (1-14)
    data_path : str, optional
        Path to store/load data. If None, uses default MNE data directory

    Returns
    -------
    raw : mne.io.Raw
        Concatenated raw EEG data

    Raises
    ------
    ValueError
        If subject number is not in range 1-109
        If any run number is not in range 1-14
    """
    # Input validation
    if not (1 <= subject <= 109):
        raise ValueError(
            f"Subject must be between 1 and 109 (inclusive), got {subject}"
        )

    if not all(1 <= run <= 14 for run in runs):
        invalid_runs = [r for r in runs if not (1 <= r <= 14)]
        raise ValueError(
            f"All run numbers must be between 1 and 14 (inclusive). "
            f"Invalid runs: {invalid_runs}"
        )

    if data_path is None:
        data_path = str(Path.home() / 'mne_data')

    # Load raw data files
    raw_fnames = mne.datasets.eegbci.load_data(
        subjects=subject,
        runs=runs,
        path=data_path,
        update_path=True
    )

    # Read and concatenate raw files
    raw_list = [mne.io.read_raw_edf(fname, preload=True, verbose=False)
                for fname in raw_fnames]
    raw = mne.concatenate_raws(raw_list)

    # Standardize channel names for EEGBCI dataset
    mne.datasets.eegbci.standardize(raw)

    # Set standard montage
    montage = mne.channels.make_standard_montage('standard_1005')
    raw.set_montage(montage, on_missing='warn')

    return raw


def apply_filtering(
    raw: mne.io.Raw,
    l_freq: float = 7.0,
    h_freq: float = 30.0,
    notch_freq: Optional[float] = 60.0
) -> mne.io.Raw:
    """
    Apply bandpass and notch filtering to raw EEG data.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data
    l_freq : float
        Low cutoff frequency in Hz (default: 7 Hz for Mu band)
    h_freq : float
        High cutoff frequency in Hz (default: 30 Hz for Beta band)
    notch_freq : float, optional
        Frequency for notch filter in Hz (default: 60 Hz for US power line)
        Set to None to skip notch filtering

    Returns
    -------
    raw : mne.io.Raw
        Filtered raw EEG data
    """
    # Apply bandpass filter
    raw.filter(
        l_freq=l_freq,
        h_freq=h_freq,
        fir_design='firwin',
        picks='eeg',
        verbose=False
    )

    # Apply notch filter if specified
    if notch_freq is not None:
        raw.notch_filter(
            freqs=notch_freq,
            picks='eeg',
            verbose=False
        )

    return raw


def apply_reference(
    raw: mne.io.Raw,
    ref_type: str = 'average',
    ref_channels: Optional[List[str]] = None
) -> mne.io.Raw:
    """
    Apply re-referencing to raw EEG data.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data
    ref_type : str
        Type of reference to apply:
        - 'average': Average reference (CAR - Common Average Reference)
        - 'mastoid': Linked-ears reference using Tp7/Tp8 (EEGBCI) or A1/A2
        - 'custom': Custom reference using specified channels
        Default: 'average'
    ref_channels : list of str, optional
        Channel names for custom reference (used when ref_type='custom')
        For mastoid reference, will auto-detect available channels

    Returns
    -------
    raw : mne.io.Raw
        Re-referenced raw EEG data

    Notes
    -----
    For EEGBCI dataset, 'mastoid' reference uses Tp7/Tp8 (linked-ears)
    as true mastoid channels (A1/A2) are not recorded. This is the closest
    anatomical alternative to traditional mastoid referencing.

    Re-referencing is applied BEFORE filtering to avoid interpolation artifacts
    in the filtered signal.
    """
    if ref_type == 'average':
        # Apply common average reference (CAR)
        raw.set_eeg_reference(ref_channels='average', projection=False, verbose=False)
        logger.info("Applied average reference (CAR)")

    elif ref_type == 'mastoid':
        # Try to use mastoid or linked-ears channels
        if ref_channels is None:
            available_channels = raw.ch_names

            # Priority 1: True mastoid channels (A1, A2)
            if 'A1' in available_channels and 'A2' in available_channels:
                ref_channels = ['A1', 'A2']
                logger.info("Using true mastoid channels (A1, A2)")

            # Priority 2: Alternative mastoid (TP9, TP10)
            elif 'TP9' in available_channels and 'TP10' in available_channels:
                ref_channels = ['TP9', 'TP10']
                logger.info("Using alternative mastoid channels (TP9, TP10)")

            # Priority 3: EEGBCI linked-ears (Tp7, Tp8)
            elif 'Tp7' in available_channels and 'Tp8' in available_channels:
                ref_channels = ['Tp7', 'Tp8']
                logger.info(
                    "Using linked-ears reference (Tp7, Tp8) - "
                    "EEGBCI dataset does not have dedicated mastoid channels. "
                    "Tp7/Tp8 provide closest anatomical alternative."
                )

            else:
                logger.warning(
                    "No suitable reference channels found (A1/A2, TP9/TP10, or Tp7/Tp8). "
                    "Falling back to average reference (CAR)."
                )
                raw.set_eeg_reference(ref_channels='average', projection=False, verbose=False)
                return raw

        raw.set_eeg_reference(ref_channels=ref_channels, projection=False, verbose=False)
        logger.info(f"Applied reference using {ref_channels}")

    elif ref_type == 'custom':
        if ref_channels is None:
            raise ValueError("ref_channels must be provided when ref_type='custom'")

        raw.set_eeg_reference(ref_channels=ref_channels, projection=False, verbose=False)
        logger.info(f"Applied custom reference using {ref_channels}")

    else:
        raise ValueError(f"Unknown ref_type: {ref_type}. Use 'average', 'mastoid', or 'custom'")

    return raw


def remove_artifacts_ica(
    raw: mne.io.Raw,
    n_components: int = 15,
    method: str = 'infomax',
    detect_eog: bool = True,
    detect_ecg: bool = True,
    detect_muscle: bool = False,
    eog_threshold: float = 3.0,
    ecg_threshold: float = 3.0,
    random_state: int = 42
) -> Tuple[mne.io.Raw, mne.preprocessing.ICA, Dict]:
    """
    Remove artifacts using Independent Component Analysis (ICA).

    This comprehensive function detects and removes multiple artifact types
    including EOG (eye movements), ECG (heartbeat), and muscle artifacts.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data
    n_components : int
        Number of ICA components (default: 15)
    method : str
        ICA decomposition method: 'fastica', 'infomax', 'picard'
        Default: 'infomax' (better for artifact separation)
    detect_eog : bool
        Detect and remove EOG (eye movement) artifacts (default: True)
    detect_ecg : bool
        Detect and remove ECG (heartbeat) artifacts (default: True)
    detect_muscle : bool
        Detect and remove muscle artifacts (default: False)
    eog_threshold : float
        Threshold for EOG component detection (default: 3.0)
    ecg_threshold : float
        Threshold for ECG component detection (default: 3.0)
    random_state : int
        Random seed for reproducibility

    Returns
    -------
    raw : mne.io.Raw
        Cleaned raw EEG data
    ica : mne.preprocessing.ICA
        Fitted ICA object
    artifact_report : dict
        Dictionary containing artifact detection report
        Contains: eog_indices, ecg_indices, muscle_indices, total_excluded
    """
    artifact_report = {
        'eog_indices': [],
        'ecg_indices': [],
        'muscle_indices': [],
        'total_excluded': 0
    }

    # Create and fit ICA
    ica = mne.preprocessing.ICA(
        n_components=n_components,
        method=method,
        random_state=random_state,
        max_iter='auto'
    )

    ica.fit(raw, verbose=False)
    logger.info(f"ICA fitted with {n_components} components using {method} method")

    # Detect EOG artifacts
    if detect_eog:
        try:
            eog_indices, eog_scores = ica.find_bads_eog(
                raw,
                ch_name=['Fp1', 'Fp2'],
                threshold=eog_threshold,
                verbose=False
            )
            artifact_report['eog_indices'] = eog_indices
            ica.exclude.extend(eog_indices)
            logger.info(f"Detected {len(eog_indices)} EOG components: {eog_indices}")
        except Exception as e:
            logger.warning(f"EOG detection failed: {e}")

    # Detect ECG artifacts
    if detect_ecg:
        try:
            # Create virtual ECG channel from EEG data
            ecg_indices, ecg_scores = ica.find_bads_ecg(
                raw,
                method='correlation',
                threshold=ecg_threshold,
                verbose=False
            )
            artifact_report['ecg_indices'] = ecg_indices
            # Only add if not already in exclude list
            new_ecg = [idx for idx in ecg_indices if idx not in ica.exclude]
            ica.exclude.extend(new_ecg)
            logger.info(f"Detected {len(ecg_indices)} ECG components: {ecg_indices}")
        except Exception as e:
            logger.warning(f"ECG detection failed: {e}")

    # Detect muscle artifacts (experimental)
    if detect_muscle:
        try:
            muscle_indices, muscle_scores = ica.find_bads_muscle(
                raw,
                threshold=0.5,
                verbose=False
            )
            artifact_report['muscle_indices'] = muscle_indices
            # Only add if not already in exclude list
            new_muscle = [idx for idx in muscle_indices if idx not in ica.exclude]
            ica.exclude.extend(new_muscle)
            logger.info(f"Detected {len(muscle_indices)} muscle components: {muscle_indices}")
        except Exception as e:
            logger.warning(f"Muscle detection failed: {e}")

    artifact_report['total_excluded'] = len(ica.exclude)
    logger.info(f"Total ICA components excluded: {artifact_report['total_excluded']}")

    # Apply ICA to remove artifacts
    raw = ica.apply(raw, verbose=False)

    return raw, ica, artifact_report


def create_epochs(
    raw: mne.io.Raw,
    event_id: dict,
    tmin: float = -1.0,
    tmax: float = 4.0,
    baseline: Tuple[float, float] = (-1.0, 0.0),
    reject: Optional[dict] = None
) -> mne.Epochs:
    """
    Create epochs from raw EEG data based on events.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data
    event_id : dict
        Dictionary mapping event names to event codes
        Example: {'left_fist': 1, 'right_fist': 2}
    tmin : float
        Start time before event in seconds (default: -1.0)
    tmax : float
        End time after event in seconds (default: 4.0)
    baseline : tuple of float
        Baseline correction interval (default: (-1.0, 0.0))
    reject : dict, optional
        Rejection parameters for bad epochs
        Example: {'eeg': 100e-6}

    Returns
    -------
    epochs : mne.Epochs
        Epoched EEG data
    """
    # Extract events from annotations
    events, _ = mne.events_from_annotations(raw, verbose=False)

    # Create epochs
    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_id,
        tmin=tmin,
        tmax=tmax,
        baseline=baseline,
        reject=reject,
        preload=True,
        verbose=False
    )

    return epochs


def reject_bad_epochs_auto(
    epochs: mne.Epochs,
    method: str = 'peak_to_peak',
    threshold: Optional[float] = None,
    n_interpolate: Optional[int] = None,
    consensus: Optional[float] = None,
    random_state: int = 42
) -> Tuple[mne.Epochs, Dict]:
    """
    Automatically reject or repair bad epochs using autoreject or peak-to-peak method.

    Parameters
    ----------
    epochs : mne.Epochs
        Epoched EEG data
    method : str
        Method for bad epoch detection:
        - 'peak_to_peak': Simple amplitude-based rejection (default)
        - 'autoreject': Use autoreject library (requires installation)
        Default: 'peak_to_peak'
    threshold : float, optional
        Peak-to-peak threshold in volts (e.g., 100e-6 for 100 µV)
        If None, uses adaptive threshold based on data statistics
    n_interpolate : int, optional
        Number of channels to interpolate per epoch (autoreject only)
        If None, autoreject will determine automatically
    consensus : float, optional
        Fraction of channels that must agree (autoreject only)
        If None, autoreject will determine automatically
    random_state : int
        Random seed for reproducibility

    Returns
    -------
    epochs_clean : mne.Epochs
        Cleaned epochs with bad epochs removed/repaired
    rejection_report : dict
        Dictionary containing rejection statistics
    """
    rejection_report = {
        'n_epochs_original': len(epochs),
        'n_epochs_rejected': 0,
        'n_epochs_interpolated': 0,
        'n_epochs_final': 0,
        'rejection_rate': 0.0,
        'method': method
    }

    if method == 'peak_to_peak':
        # Simple peak-to-peak amplitude rejection
        data = epochs.get_data()

        if threshold is None:
            # Adaptive threshold: median + 3 * MAD (Median Absolute Deviation)
            peak_to_peak = np.ptp(data, axis=2)  # Peak-to-peak per epoch per channel
            median = np.median(peak_to_peak)
            mad = np.median(np.abs(peak_to_peak - median))
            threshold = median + 3 * mad
            logger.info(f"Using adaptive threshold: {threshold * 1e6:.2f} µV")

        # Calculate peak-to-peak amplitude for each epoch
        peak_to_peak = np.ptp(data, axis=2)
        max_peak_to_peak = np.max(peak_to_peak, axis=1)

        # Find bad epochs
        bad_epochs = max_peak_to_peak > threshold
        n_bad = np.sum(bad_epochs)

        rejection_report['n_epochs_rejected'] = n_bad
        rejection_report['threshold_uv'] = threshold * 1e6

        # Drop bad epochs
        epochs_clean = epochs.copy()
        if n_bad > 0:
            good_indices = np.where(~bad_epochs)[0]
            epochs_clean = epochs_clean[good_indices]
            logger.info(f"Rejected {n_bad} epochs using peak-to-peak threshold")

    elif method == 'autoreject':
        # Try to use autoreject library
        try:
            from autoreject import AutoReject

            # Create AutoReject object
            ar = AutoReject(
                n_interpolate=n_interpolate,
                consensus=consensus,
                random_state=random_state,
                n_jobs=1,
                verbose=False
            )

            # Fit and transform
            epochs_clean, reject_log = ar.fit_transform(epochs, return_log=True)

            # Get statistics
            rejection_report['n_epochs_rejected'] = np.sum(reject_log.bad_epochs)
            rejection_report['n_epochs_interpolated'] = len(reject_log.bad_epochs) - rejection_report['n_epochs_rejected']
            rejection_report['bad_channels_per_epoch'] = reject_log.labels

            logger.info(f"AutoReject: {rejection_report['n_epochs_rejected']} epochs rejected, "
                       f"{rejection_report['n_epochs_interpolated']} epochs interpolated")

        except ImportError:
            logger.warning("autoreject library not installed. Falling back to peak-to-peak method.")
            # Recursive call with peak_to_peak method
            return reject_bad_epochs_auto(epochs, method='peak_to_peak', threshold=threshold)

    else:
        raise ValueError(f"Unknown method: {method}. Use 'peak_to_peak' or 'autoreject'")

    rejection_report['n_epochs_final'] = len(epochs_clean)
    rejection_report['rejection_rate'] = rejection_report['n_epochs_rejected'] / rejection_report['n_epochs_original']

    logger.info(f"Epoch rejection complete: {rejection_report['n_epochs_final']}/{rejection_report['n_epochs_original']} epochs retained "
               f"({100 * (1 - rejection_report['rejection_rate']):.1f}%)")

    return epochs_clean, rejection_report


def preprocess_pipeline(
    subject: int,
    runs: List[int],
    event_id: dict,
    data_path: Optional[str] = None,
    apply_ica: bool = False,
    use_enhanced_ica: bool = False,
    use_autoreject: bool = False,
    ref_type: Optional[str] = None,
    l_freq: float = 7.0,
    h_freq: float = 30.0,
    notch_freq: Optional[float] = 60.0,
    tmin: float = -1.0,
    tmax: float = 4.0,
    baseline: Tuple[float, float] = (-1.0, 0.0),
    reject: Optional[dict] = None,
    verbose: bool = True
) -> Tuple[mne.Epochs, mne.io.Raw]:
    """
    Complete preprocessing pipeline for EEG motor imagery data.

    Pipeline order (following MNE best practices):
    1. Load data
    2. Re-reference (optional) - BEFORE filtering to avoid interpolation artifacts
    3. Filter (bandpass + notch)
    4. ICA artifact removal (optional) - on filtered continuous data
    5. Epoch creation
    6. Bad epoch rejection (optional)

    Parameters
    ----------
    subject : int
        Subject number (1-109)
    runs : list of int
        List of run numbers to load (1-14)
    event_id : dict
        Dictionary mapping event names to event codes
        Example: {'left_fist': 1, 'right_fist': 2}
    data_path : str, optional
        Path to store/load data
    apply_ica : bool
        Whether to apply basic ICA for artifact removal (default: False)
    use_enhanced_ica : bool
        Whether to use enhanced ICA with EOG/ECG/muscle detection (default: False)
        If True, overrides apply_ica
    use_autoreject : bool
        Whether to use automatic bad epoch rejection (default: False)
    ref_type : str, optional
        Type of re-referencing: 'average', 'mastoid', 'custom'
        If None, no re-referencing is applied (default: None)
        Note: Re-referencing is applied BEFORE filtering
    l_freq : float
        Low cutoff frequency in Hz (default: 7 Hz for Mu band)
    h_freq : float
        High cutoff frequency in Hz (default: 30 Hz for Beta band)
    notch_freq : float, optional
        Frequency for notch filter in Hz (default: 60 Hz for US power line)
        Set to None to skip notch filtering
    tmin : float
        Start time before event in seconds (default: -1.0)
    tmax : float
        End time after event in seconds (default: 4.0)
    baseline : tuple of float
        Baseline correction interval (default: (-1.0, 0.0))
    reject : dict, optional
        Rejection parameters for bad epochs
        Example: {'eeg': 100e-6}
    verbose : bool
        Whether to print progress information

    Returns
    -------
    epochs : mne.Epochs
        Preprocessed epoched data
    raw : mne.io.Raw
        Preprocessed raw data (for visualization)

    Notes
    -----
    Pipeline Order Rationale:
    - Re-referencing before filtering prevents interpolation artifacts in filtered signal
    - Filtering before ICA removes slow drifts that waste ICA components
    - ICA on continuous data provides better decomposition than on epochs
    - Epoch rejection is final quality control after all preprocessing

    Examples
    --------
    >>> # Basic preprocessing without ICA
    >>> epochs, raw = preprocess_pipeline(
    ...     subject=1,
    ...     runs=[4, 8, 12],
    ...     event_id={'left_fist': 1, 'right_fist': 2}
    ... )
    >>> # With ICA and average reference
    >>> epochs, raw = preprocess_pipeline(
    ...     subject=1,
    ...     runs=[4, 8, 12],
    ...     event_id={'left_fist': 1, 'right_fist': 2},
    ...     use_enhanced_ica=True,
    ...     ref_type='average'
    ... )
    """
    if verbose:
        print(f"Loading data for subject {subject}, runs {runs}...")

    # Step 1: Load data
    raw = load_eeg_data(subject, runs, data_path)

    # Step 2: Re-reference BEFORE filtering (to avoid interpolation artifacts)
    if ref_type is not None:
        if verbose:
            print(f"Applying {ref_type} reference...")
        raw = apply_reference(raw, ref_type=ref_type)

    # Step 3: Apply filtering
    if verbose:
        print(f"Applying filtering (bandpass: {l_freq}-{h_freq} Hz)...")
    raw = apply_filtering(raw, l_freq, h_freq, notch_freq)

    # Step 4: Optional ICA artifact removal (on filtered continuous data)
    if use_enhanced_ica or apply_ica:
        if verbose:
            if use_enhanced_ica:
                print("Applying ICA for artifact removal (EOG/ECG detection)...")
            else:
                print("Applying ICA for artifact removal (EOG detection only)...")

        # Use comprehensive ICA with configurable artifact detection
        raw, _, artifact_report = remove_artifacts_ica(
            raw,
            detect_eog=True,
            detect_ecg=use_enhanced_ica,  # Only detect ECG if enhanced mode
            detect_muscle=False
        )

        if verbose:
            print(f"  Excluded {artifact_report['total_excluded']} ICA components")

    # Step 5: Create epochs
    if verbose:
        print("Creating epochs...")
    epochs = create_epochs(raw, event_id, tmin, tmax, baseline, reject)

    # Step 6: Optional automatic bad epoch rejection
    if use_autoreject:
        if verbose:
            print("Applying automatic bad epoch rejection...")
        epochs, rejection_report = reject_bad_epochs_auto(epochs, method='peak_to_peak')
        if verbose:
            print(f"  Rejected {rejection_report['n_epochs_rejected']} bad epochs")
            print(f"  Retained {rejection_report['n_epochs_final']} epochs ({100 * (1 - rejection_report['rejection_rate']):.1f}%)")

    if verbose:
        print(f"Preprocessing complete. {len(epochs)} epochs created.")

    return epochs, raw


def get_motor_imagery_event_dict(task_type: str = 'left_right_fist') -> dict:
    """
    Get event dictionary for common motor imagery tasks.

    Parameters
    ----------
    task_type : str
        Type of motor imagery task:
        - 'left_right_fist': Left vs Right fist (imagined or real)
        - 'fists_feet': Both fists vs Both feet (imagined or real)

    Returns
    -------
    event_id : dict
        Event dictionary mapping task names to event codes
    """
    # Standard event codes from EEGBCI dataset
    # T0: Rest
    # T1: Left fist (or both fists)
    # T2: Right fist (or both feet)

    if task_type == 'left_right_fist':
        event_id = {
            'left_fist': 1,
            'right_fist': 2
        }
    elif task_type == 'fists_feet':
        event_id = {
            'both_fists': 1,
            'both_feet': 2
        }
    else:
        raise ValueError(f"Unknown task type: {task_type}")

    return event_id


def get_recommended_runs(task_type: str = 'imagery_left_right') -> List[int]:
    """
    Get recommended run numbers for different task types.

    Parameters
    ----------
    task_type : str
        Type of task:
        - 'imagery_left_right': Imagined left/right fist movement
        - 'imagery_fists_feet': Imagined both fists/feet movement
        - 'real_left_right': Real left/right fist movement
        - 'real_fists_feet': Real both fists/feet movement

    Returns
    -------
    runs : list of int
        Recommended run numbers
    """
    run_mapping = {
        'imagery_left_right': [4, 8, 12],
        'imagery_fists_feet': [6, 10, 14],
        'real_left_right': [3, 7, 11],
        'real_fists_feet': [5, 9, 13]
    }

    if task_type not in run_mapping:
        raise ValueError(f"Unknown task type: {task_type}")

    return run_mapping[task_type]
