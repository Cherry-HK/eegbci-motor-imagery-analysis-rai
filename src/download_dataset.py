import mne
import os
import shutil

download_path = 'data/raw'

# MNE creates a nested structure: {path}/MNE-eegbci-data/files/eegmmidb/1.0.0/S001/...
# We download there first, then move subject folders directly into data/raw/
mne_nested_dir = os.path.join(download_path, 'MNE-eegbci-data', 'files', 'eegmmidb', '1.0.0')

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

# Move subject folders from MNE's nested structure to data/raw/
if os.path.isdir(mne_nested_dir):
    for item in sorted(os.listdir(mne_nested_dir)):
        src = os.path.join(mne_nested_dir, item)
        dst = os.path.join(download_path, item)
        if os.path.isdir(src):
            shutil.move(src, dst)
    shutil.rmtree(os.path.join(download_path, 'MNE-eegbci-data'))
    print("\nReorganized files into flat structure.")

print(f"\nData saved to: {os.path.abspath(download_path)}")
