import os
import numpy as np
import mne
from sklearn.model_selection import GroupShuffleSplit
from mne.decoding import CSP
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

class EEGPreprocessor:
    
    def __init__(self, data_path, subjects='all', freq_band=(8, 30), 
                 target_sfreq=128, notch_freq=60):
        
        self.data_path = data_path
        self.subjects = list(range(1, 110)) if subjects == 'all' else subjects
        self.freq_band = freq_band
        self.target_sfreq = target_sfreq
        self.notch_freq = notch_freq
        
        self.selected_channels = [
            'FC3.', 'FC1.', 'FCz.', 'FC2.', 'FC4.',
            'C5..', 'C3..', 'C1..', 'Cz..', 'C2..', 'C4..', 'C6..',
            'CP3.', 'CP1.', 'CPz.', 'CP2.', 'CP4.',
            'Fz..', 'Pz..'
        ]
        
        # Motor imagery runs (left vs right hand)
        self.mi_runs = [4, 8, 12]
        
        
    def load_subject_data(self, subject_id):
        
        raw_list = []
        subject_str = f'S{subject_id:03d}' # example: S001
        
        for run in self.mi_runs:
            run_str = f'{subject_str}R{run:02d}.edf' # example: S001R04.edf
            file_path = os.path.join(self.data_path, subject_str, run_str)
            
            if os.path.exists(file_path):
                raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
                raw_list.append(raw)
            else:
                print(f"Warning: File not found - {file_path}")
                
        return raw_list
    
    def preprocess_raw(self, raw):

        # copy to avoid modifying original data
        raw = raw.copy()
        
        # Select only EEG channels (remove EOG, etc.)
        # raw.pick_types(eeg=True, exclude='bads')
        raw.pick(picks='eeg', exclude='bads')
        
        # Select specific motor cortex channels
        available_channels = [ch for ch in self.selected_channels if ch in raw.ch_names]
        if len(available_channels) == 0:
            raise RuntimeError("No selected motor channels found in raw.ch_names")
        
        # raw.pick_channels(available_channels)
        raw.pick(picks=available_channels)
        
        # 1. Notch filter
        raw.notch_filter(freqs=self.notch_freq, verbose=False)
        
        # 2. Bandpass filter
        raw.filter(l_freq=self.freq_band[0], h_freq=self.freq_band[1], 
                   verbose=False, method='iir')  # iir/fir but iir is more common for EEG
        
        # 3. Resampling
        if raw.info['sfreq'] != self.target_sfreq:
            raw.resample(self.target_sfreq, verbose=False)
        
        # 4. Re-reference 
        # Surface Laplacian (CSD) referencing could not be applied because the EEGMMIDB EDF files do not contain reliable electrode digitization points required by MNE’s CSD implementation; therefore, Common Average Referencing combined with CSP was used for spatial filtering.
        # raw.set_montage('standard_1020', on_missing='ignore')
        # try:
        #     raw = compute_current_source_density(raw)
        # except Exception as exc:
        #     raise RuntimeError(f"CSD failed: {exc}")
        raw.set_eeg_reference('average', projection=False, verbose=False) # CAR

        return raw
    
    # Not necessary needed anymore since we are using CSD
    """
    def apply_ica(self, raw, n_components=15, random_state=42):
        
        # Fit ICA
        ica = ICA(n_components=n_components, random_state=random_state, 
                  max_iter=500, verbose=False)
        ica.fit(raw, verbose=False)
        
        # Detect EOG artifacts automatically
        eog_indices, eog_scores = ica.find_bads_eog(raw, verbose=False)
        
        # Mark components as bad
        ica.exclude = eog_indices
        
        # Apply ICA to remove artifacts
        raw_clean = ica.apply(raw.copy(), verbose=False)
        
        return raw_clean
    """
    
    def create_epochs(self, raw, tmin=0.0, tmax=4.0, baseline=None):

        # Convert EDF annotations (T0, T1, T2) to MNE events
        events, event_id = mne.events_from_annotations(raw, verbose=False)

        # Check that T1/T2 exist in the annotations
        if 'T1' not in event_id or 'T2' not in event_id:
            # Return an empty Epochs object to allow the caller to skip this run
            n_ch = len(raw.ch_names)
            n_times = int((tmax - tmin) * raw.info['sfreq'])
            info = raw.info.copy()
            empty_data = np.empty((0, n_ch, n_times))
            return mne.EpochsArray(empty_data, info=info, tmin=tmin)

        # Select only motor imagery events (T1: left hand, T2: right hand)
        mi_event_id = {
            'left_hand': event_id['T1'],
            'right_hand': event_id['T2']
        }

        # Epoch data from 0 to 4 seconds after cue onset
        epochs = mne.Epochs(
            raw,
            events,
            event_id=mi_event_id,
            tmin=tmin,
            tmax=tmax,
            baseline=baseline,
            preload=True,
            verbose=False
        )

        return epochs

    
    def process_all_subjects(self):
        
        all_epochs = []
        all_labels = []
        all_subject_ids = []
        
        for subject_id in self.subjects:
            print(f"Processing subject {subject_id}/109...")
            try:
                # Load subject data
                raw_list = self.load_subject_data(subject_id) 
                if not raw_list:
                    print(f"  No data found for subject {subject_id}")
                    continue
                
                subject_epochs = []
                for raw in raw_list:
                    try:
                        # Preprocess
                        raw_processed = self.preprocess_raw(raw)
                    except Exception as e:
                        print(f"  Preprocessing failed for subject {subject_id}: {str(e)}")
                        continue

                    # Create epochs
                    epochs = self.create_epochs(raw_processed)
                    if len(epochs) > 0:
                        subject_epochs.append(epochs)
                

                # Concatenate epochs from all runs
                if subject_epochs:
                    subject_epochs_combined = mne.concatenate_epochs(subject_epochs)
                    
                    # Get data and labels
                    X_subject = subject_epochs_combined.get_data()

                    event_id = subject_epochs_combined.event_id
                    y_subject = np.array([
                        0 if e == event_id['left_hand'] else 1
                        for e in subject_epochs_combined.events[:, 2]
                        ])
                    
                    all_epochs.append(X_subject)
                    all_labels.append(y_subject)
                    all_subject_ids.extend([subject_id] * len(y_subject))
                    
                    print(f"  Collected {len(y_subject)} epochs")
                    
            except Exception as e:
                print(f"  Error processing subject {subject_id}: {str(e)}")
                continue

        if len(all_epochs) == 0:
            # return empty arrays instead of crashing
            return np.empty((0, 0, 0)), np.array([]), np.array([])
        
        # Combine all subjects
        X = np.vstack(all_epochs)
        y = np.hstack(all_labels)
        subjects_info = np.array(all_subject_ids)
        
        print(f"\nTotal dataset: {len(X)} epochs")
        print(f"Data shape: {X.shape}")
        print(f"Class distribution - Left hand (0): {np.sum(y == 0)}, Right hand (1): {np.sum(y == 1)}")
        
        return X, y, subjects_info
    
    def extract_csp_features(self, X_train, y_train, X_test, n_components=6):
  
        # Initialize CSP
        csp = CSP(n_components=n_components, reg=None, log=True, norm_trace=False)
        
        # Fit on training data
        csp.fit(X_train, y_train)
        
        # Transform both train and test
        X_train_csp = csp.transform(X_train)
        X_test_csp = csp.transform(X_test)
        
        print(f"CSP features extracted: {X_train_csp.shape}")
        
        return X_train_csp, X_test_csp, csp

# we are using GroupShuffleSplit for subject-independent splitting
# for future reference we can also try LOSO
if __name__ == "__main__":

    data_path = "data/raw"

    preprocessor = EEGPreprocessor(
        data_path=data_path,
        subjects='all',
        freq_band=(8, 30),
        target_sfreq=128,
        notch_freq=60
    )

    # Process all subjects
    X, y, subjects = preprocessor.process_all_subjects()

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=subjects))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Extract CSP features
    X_train_csp, X_test_csp, csp = preprocessor.extract_csp_features(
        X_train, y_train, X_test, n_components=6
    )

    # Save outputs
    np.save('X_train.npy', X_train)
    np.save('X_test.npy', X_test)
    np.save('y_train.npy', y_train)
    np.save('y_test.npy', y_test)
    np.save('X_train_csp.npy', X_train_csp)
    np.save('X_test_csp.npy', X_test_csp)
    np.save('subjects.npy', subjects)

    print("\nPreprocessing complete!")
    print(f"Raw data shapes - Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"CSP features - Train: {X_train_csp.shape}, Test: {X_test_csp.shape}")
