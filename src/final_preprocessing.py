import os
import numpy as np
import mne
# import warnings
import pandas as pd
# warnings.filterwarnings('ignore', category=RuntimeWarning)

EXPERIMENT_NAME = "preprocessing_result"  # change
OUTPUT_DIR = os.path.join("models", EXPERIMENT_NAME)

class EEGPreprocessor:
    
    def __init__(self, data_path, subjects='all', freq_band=(8, 30), notch_freq=60, tmin=0.0, n_samples=512):
        
        self.data_path = data_path
        self.subjects = list(range(1, 110)) if subjects == 'all' else subjects
        self.freq_band = freq_band
        self.notch_freq = notch_freq
        self.tmin = tmin
        self.n_samples = n_samples
        self.audit = []
        
        # 9 channels
        # self.selected_channels = [
        #     'Fc3.', 'Fcz.', 'Fc4.',
        #     'C3..', 'Cz..', 'C4..', 
        #     'Cp3.', 'Cpz.', 'Cp4.',
        # ]
        
        self.selected_channels = [
            'Fc3.', 'Fc1.', 'Fcz.', 'Fc2.', 'Fc4.',
            'C3..', 'C1..', 'Cz..', 'C2..', 'C4..',
            'Cp3.', 'Cp1.', 'Cpz.', 'Cp2.', 'Cp4.'
        ]
        
        # Fc5.,Fc3.,Fc1.,Fcz.,Fc2.,Fc4.,Fc6.,C5..,C3..,C1..,Cz..,C2..,C4..,C6..,Cp5.,Cp3.,Cp1.,Cpz.,Cp2.,Cp4.,Cp6.,Fp1.,Fpz.,Fp2.,Af7.,Af3.,Afz.,Af4.,Af8.,F7..,F5..,F3..,F1..,Fz..,F2..,F4..,F6..,F8..,Ft7.,Ft8.,T7..,T8..,T9..,T10.,Tp7.,Tp8.,P7..,P5..,P3..,P1..,Pz..,P2..,P4..,P6..,P8..,Po7.,Po3.,Poz.,Po4.,Po8.,O1..,Oz..,O2..,Iz..

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
                raw_list.append((run, raw))
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
        # if raw.info['sfreq'] != self.target_sfreq:
        #     raw.resample(self.target_sfreq, verbose=False)
        
        # 4. Re-reference 
        # Surface Laplacian (CSD) referencing could not be applied because the EEGMMIDB EDF files do not contain reliable electrode digitization points required by MNE’s CSD implementation; therefore, Common Average Referencing combined with CSP was used for spatial filtering.
        # raw.set_montage('standard_1020', on_missing='ignore')
        # try:
        #     raw = compute_current_source_density(raw)
        # except Exception as exc:
        #     raise RuntimeError(f"CSD failed: {exc}")
        raw.set_eeg_reference('average', projection=False, verbose=False) # CAR

        return raw
    
    # ICA was considered but is not used in the final pipeline.
    # Final preprocessing uses CAR + bandpass + CSP.
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
    
    def create_epochs(self, raw, baseline=None):

        sfreq = raw.info['sfreq']
        print(f"Original sampling frequency: {sfreq} Hz")
        tmin = self.tmin
        tmax = tmin + (self.n_samples - 1) / sfreq

        # Convert EDF annotations (T0, T1, T2) to MNE events
        events, event_id = mne.events_from_annotations(raw, verbose=False)

        # Check that T1/T2 exist in the annotations
        if 'T1' not in event_id or 'T2' not in event_id:
            self.audit.append({
                "subject": self.current_subject,
                "run": self.current_run,
                "total_events": len(events),
                "epochs_kept": 0,
                "ignored_non_target": 0,
                "artifact_rejections": 0,
                "target_events_estimated": 0,
                "bad_channels": "None",
                "status": "missing_T1_or_T2"
            })
            # Return an empty Epochs object to allow the caller to skip this run
            n_ch = len(raw.ch_names)
            n_times = self.n_samples
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
            reject=dict(eeg=150e-6),
            preload=True,
            verbose=False
        )

        print(epochs)
        # print("Dropped epochs:", sum(len(x) > 0 for x in epochs.drop_log))
        # print(epochs.drop_log[:20])

        total = len(events)
        kept = len(epochs)

        ignored = 0
        rejected = 0
        bad_channels = []

        for r in epochs.drop_log:
            if len(r) == 0:
                continue

            if 'IGNORED' in r:
                ignored += 1
            else:
                rejected += 1
                bad_channels.extend(r)

        # remove duplicates and join nicely
        bad_channels = ",".join(sorted(set(bad_channels))) if bad_channels else "None"

        self.audit.append({
            "subject": self.current_subject,
            "run": self.current_run,
            "total_events": total,
            "epochs_kept": kept,
            "ignored_non_target": ignored,
            "artifact_rejections": rejected,
            "target_events_estimated": kept + rejected,
            "bad_channels": bad_channels,
            "status": "ok"
        })

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
                for run, raw in raw_list:
                    self.current_subject = subject_id
                    self.current_run = run

                    try:
                        # Preprocess
                        raw_processed = self.preprocess_raw(raw)
                    except Exception as e:
                        print(f"  Preprocessing failed for subject {subject_id}, run {run}: {str(e)}")
                        continue

                    # Create epochs
                    epochs = self.create_epochs(raw_processed)
                    if len(epochs) > 0:
                        subject_epochs.append(epochs)
                

                # Concatenate epochs from all runs
                if subject_epochs:
                    subject_epochs_combined = mne.concatenate_epochs(subject_epochs)

                    print("Combined epochs shape:", subject_epochs_combined.get_data().shape)
                    print("Combined events shape:", subject_epochs_combined.events.shape)
                    print("Event ID:", subject_epochs_combined.event_id)
                    print("Unique event codes:", np.unique(subject_epochs_combined.events[:, 2], return_counts=True))

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
        
        print("\n" + "="*50)
        print(f"Total dataset: {len(X)} epochs")
        print(f"Data shape: {X.shape}")
        print(f"y shape: {y.shape}")
        print(f"Subjects: {len(np.unique(subjects_info))}")
        print(f"Class distribution - Left hand (0): {np.sum(y == 0)}, Right hand (1): {np.sum(y == 1)}")
        print("="*50)
        
        return X, y, subjects_info
    

if __name__ == "__main__":

    data_path = "data/MNE-eegbci-data/files/eegmmidb/1.0.0"

    preprocessor = EEGPreprocessor(
        data_path=data_path,
        subjects='all',
        freq_band=(8, 30),
        notch_freq=60, 
        tmin=0.0,
        n_samples=512
    )

    print(f"Running preprocessing with tmin={preprocessor.tmin}, n_samples={preprocessor.n_samples}, "
      f"band={preprocessor.freq_band}, notch={preprocessor.notch_freq}")

    # Process all subjects
    X, y, subjects = preprocessor.process_all_subjects()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save outputs
    np.save(os.path.join(OUTPUT_DIR, "X.npy"), X)
    np.save(os.path.join(OUTPUT_DIR, "y.npy"), y)
    np.save(os.path.join(OUTPUT_DIR, "subjects.npy"), subjects)

    print("\n Preprocessing finished.")
    print("Saved:")
    print("X.npy, y.npy, subjects.npy")

    df = pd.DataFrame(preprocessor.audit)
    df.to_csv(os.path.join(OUTPUT_DIR, "epoch_audit.csv"), index=False)
    print("Saved epoch_audit.csv")

