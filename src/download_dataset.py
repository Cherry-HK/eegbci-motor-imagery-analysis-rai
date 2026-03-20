import mne
import os

download_path = 'data/MNE-eegbci-data/files/eegmmidb/1.0.0'

# Download all 109 subjects
subjects_to_download = list(range(1, 110))

for subject in subjects_to_download:
    print(f"\nDownloading subject {subject}/109...")

    # Download runs 4, 8, 12 (motor imagery: left fist vs right fist)
    try:
        mne.datasets.eegbci.load_data(
            subject,
            runs=[4, 8, 12],
            path=download_path,
            update_path=False,
            verbose=True
        )
    except Exception as e:
        print(f"  Failed to download subject {subject}: {e}")

print(f"\nData saved to: {os.path.abspath(download_path)}")
