# mi_icu

**Multiple imputation methods for ICU time series**

This repository contains code to benchmark imputation methods on real-world ICU data using simulated spatio-temporal missingness patterns. It supports data from **MIMIC-III** and **HiRID**, both available via PhysioNet.

---

## 📌 Model Card (Short)

- **Task:** Missing value imputation in multivariate clinical time series  
- **Data:** ICU data from PhysioNet (MIMIC-III, HiRID)  
- **Models:** Baseline statistical imputers and deep learning autoencoder-based methods  
- **Use case:** Controlled evaluation of imputation methods under realistic missingness patterns  

---

## 📁 1. Data Loading

Raw ICU data must be downloaded manually from PhysioNet and stored locally. The data loader scripts expect a structured directory and convert raw events into aligned time-series per ICU stay.

**Expected directory structure:**
```
data/
├── mimiciii/
│   ├── raw/
│   └── processed/
└── hirid/
    ├── raw/
    └── processed/
```

Loader scripts read from these directories and return unified pandas/NumPy time-series.

**Example:**
```python
from data_loader import ICUDataLoader

loader = ICUDataLoader(
    mimic_path="data/mimiciii",
    hirid_path="data/hirid"
)

ts_data, meta = loader.load_all()
print(ts_data.shape)
```

---

## 🧹 2. Creation of Complete Case Data

To establish ground-truth data, *complete cases* are created by:

1. Removing ICU stays with missing values in required variables  
2. Resampling time-series to a fixed temporal grid (e.g. 5 minutes)  
3. Handling outliers using one of two strategies:
   - **Discard:** remove samples containing outliers  
   - **Clip:** cap values to predefined physiological bounds  

**Example:**
```python
from preprocessing.complete_case import create_complete_cases

complete_data = create_complete_cases(
    ts_data,
    resample_freq="5min",
    outlier_strategy="clip"  # or "discard"
)

print(complete_data.shape)
```

---

## 🌪️ 3. Induction of Missingness Patterns

Artificial missingness is introduced to simulate realistic ICU data corruption. The repository supports different **spatio-temporal patterns**, including:

- **MCAR:** random missingness  
- **Temporal blocks:** contiguous time segments removed  
- **Feature-wise patterns:** correlated feature dropouts  

These patterns are applied only after complete case generation.

**Example:**
```python
from missingness import MissingnessGenerator

generator = MissingnessGenerator(
    pattern="block",
    miss_rate=0.3,
    temporal_span=12
)

masked_data = generator.apply(complete_data)
```

---

## 🤖 4. Application of Imputation Methods

### 4.1 Baseline Imputation Methods

Standard statistical imputers serve as baselines.

**Example (mean imputation):**
```python
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="mean")
imputed = imputer.fit_transform(masked_data)
```

---

### 4.2 Deep Learning Imputation Models

Deep learning models (e.g., GRU/LSTM autoencoders) learn temporal dependencies and feature correlations to recover missing values.

**Example (STAE model):**
```python
from imputation.stae import STAEImputer

model = STAEImputer(input_dim=masked_data.shape[-1])
model.fit(masked_data, complete_data)

imputed_dl = model.impute(masked_data)
```

---

## 🔁 Example End-to-End Workflow

```bash
# 1. Download ICU data from PhysioNet
# 2. Store data under data/mimiciii or data/hirid

# 3. Create complete cases
python scripts/make_complete_cases.py

# 4. Induce missingness
python scripts/create_missingness.py --pattern block --rate 0.3

# 5. Run imputation
python scripts/run_imputation.py --method stae
```

---

## 🛠️ Dependencies

- Python ≥ 3.8  
- numpy, pandas  
- scikit-learn  
- torch  

Additional utilities are provided in `utils.py` and `stats.py`.

---

## 📄 License & Citation

This repository is intended for research and benchmarking.  
Please cite the associated work when using this codebase.
