## Comparative Analysis of Classical and Deep Learning Models for Motor Imagery EEG Classification with a Training Application for Neuro-Rehabilitation

### Team Members:
1. 65011255 Apisara Luengnaruemitchai
2. 65011289 Cherry Hlaing Kyaw
3. 65011302 Hermes Maryprasith
4. 65011619 Varis Saligupta

### Dataset
This project uses the **EEG Motor Movement/Imagery Dataset (EEGMMIDB)** provided by **PhysioNet**.
- **109 subjects**, 64-channel EEG
- **Motor imagery runs**: R04, R08, R12 (left fist vs right fist)
- **15 selected motor-cortex channels**: FC3, FC1, FCz, FC2, FC4, C3, C1, Cz, C2, C4, CP3, CP1, CPz, CP2, CP4

**Dataset source:**
https://physionet.org/content/eegmmidb/1.0.0/

### Download Instructions
1. Run the download script to fetch all 109 subjects via MNE:
   ```bash
   python src/download_dataset.py
   ```
2. Or visit the PhysioNet dataset page and download manually.

### Expected Directory Structure

```
data/
└── raw/
    ├── S001/
    │   ├── S001R04.edf
    │   ├── S001R08.edf
    │   └── S001R12.edf
    ├── S002/
    ├── ...
    └── S109/
```

### Usage
1. Download dataset (all 109 subjects)
```bash
python src/download_dataset.py
```
2. Preprocess the data: bandpass filter, epoch, CSP features
```bash
python src/preprocessing.py
```
3. Run the parameter study
   Example:
```bash
python models/lr/training_loso_lr.py
```
4. Run the best-combination search using optuna
   Example:
```bash
python models/lr/lr_optuna.py
```
5. Export the final model
   Example:
```bash
python models/lr/lr_optuna_export.py
```

### Models
| Model | Type | Features |
|-------|------|----------|
| Logistic Regression | Classical ML | CSP |
| KNN | Classical ML | CSP + Raw |
| EEGNet-Motor CNN | Deep Learning | Raw EEG |
| CNN-LSTM with Attention | Deep Learning | Raw EEG |

### Requirements
```
pip install -r requirements.txt
```
