# Repo for "Benchmarking imputation methods on real-world clinical time series with simulated spatio-temporal missingness"

## Model Card (Short)

- **Task:** Missing value imputation in multivariate clinical time series  
- **Data:** ICU data from PhysioNet (MIMIC‑III, HiRID)  
- **Models:** Baseline statistical imputers and deep learning autoencoder‑based methods  
- **Use case:** Controlled evaluation of imputation methods under realistic missingness patterns  

---

## Table of Contents

1. Introduction  
2. Installation  
3. Data Loading  
4. Preprocessing Clinical Data  
5. Creating Complete Case Data  
6. Inducing Spatio‑Temporal Missingness  
7. Imputation Methods  
8. Using Stored Models and Applying Imputation  
9. Example End‑to‑End Workflow  
10. License & Citation  

---

## Introduction

`mi_icu` is a benchmarking suite for missing data imputation on real‑world ICU time‑series data. It simulates clinically realistic **spatio‑temporal missingness patterns** by fitting a **Markov chain model** to fully observed ICU time series and then evaluates several imputation strategies ranging from statistical baselines to deep learning models.

---

## Installation

```bash
git clone https://github.com/ngiesa/mi_icu.git
cd mi_icu
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Data Loading

```python
from data_loader import ICUDataLoader

loader = ICUDataLoader(
    mimic_path="data/mimiciii",
    hirid_path="data/hirid"
)

ts_data, meta = loader.load_all()
```

---

## Preprocessing Clinical Data

```python
from preprocess import resample_and_clean

clean_data = resample_and_clean(
    ts_data,
    resample_freq='5min',
    outlier_strategy='clip'
)
```

---

## Creating Complete Case Data

```python
from preprocessing.complete_case import create_complete_cases

complete_data = create_complete_cases(
    ts_data,
    resample_freq="5min",
    outlier_strategy="clip"
)
```

---

## Inducing Spatio‑Temporal Missingness

```python
from induction import MissingnessInducer

inducer = MissingnessInducer()
mask = inducer.fit_sample(complete_data, pattern="mar_amplified")
missing_data = apply_mask(complete_data, mask)
```

---

## Imputation Methods

### Baseline

```python
from sklearn.impute import SimpleImputer

imp = SimpleImputer(strategy='mean')
filled = imp.fit_transform(masked_data)
```

### Deep Learning

```python
from imputation.stae import STAEImputer

model = STAEImputer(input_dim=masked_data.shape[-1])
model.fit(masked_data, complete_data)
imputed = model.impute(masked_data)
```

---

## Using Stored Models

```python
import torch

model = STAEImputer(input_dim=num_features)
model.load_state_dict(torch.load("models/stae_mimic.pth"))
model.eval()

imputed = model.impute(new_masked_data)
```

---

## Example End‑to‑End Workflow

```bash
python preprocess.py
python make_complete_cases.py
python create_missingness.py
python run_imputation.py
python evaluate.py
```

---

## License & Citation

Please cite the associated publication by Giesa et al. when using this repository.
