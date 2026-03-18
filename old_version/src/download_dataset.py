import mne
import os
 
download_path = 'data/MNE-eegbci-data/files/eegmmidb/1.0.0'

# Download specific subjects (start with a few for testing)
subjects_to_download = list(range(1, 4))  # Download subjects 1-3

for subject in subjects_to_download:
    print(f"\nDownloading subject {subject}...")
    
    # Download runs 4, 8, 12 (motor imagery tasks)
    mne.datasets.eegbci.load_data(
        subject,
        runs=[4, 8, 12],  # Only motor imagery runs
        path=download_path,
        update_path=False,
        verbose=True
    )

print(f"Data saved to: {os.path.abspath(download_path)}")
