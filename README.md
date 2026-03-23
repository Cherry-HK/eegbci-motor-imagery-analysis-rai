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
```bash
# for example:
python models/lr/training_loso_lr.py
```
4. Run the best-combination search using optuna
```bash
# for example:
python models/lr/lr_optuna.py
```
5. Export the final model
```bash
# for example:
python models/lr/lr_optuna_export.py
```

### Models
| Model | Type | Features |
|-------|------|----------|
| SVM | Classical ML | CSP features |
| Logistic Regression (LR) | Classical ML | CSP features |
| KNN | Classical ML | CSP features |
| LDA | Classical ML | CSP features |
| Decision Tree (DT) | Classical ML | CSP features |
| Random Forest (RF) | Classical ML | CSP features |
| Riemannian Classifier | Classical ML | Riemannian covariance / tangent-space features |
| CNN | Deep Learning | Raw EEG epochs |
| EEGNet | Deep Learning | Raw EEG epochs |
| LSTM | Deep Learning | Raw EEG epochs |
| Transformer | Deep Learning | Raw EEG epochs |
| ShallowConvNet | Deep Learning | Raw EEG epochs |

### Requirements
```
pip install -r requirements.txt
```
