# Repository for "Benchmarking imputation methods on real-world clinical time series with simulated spatio-temporal missingness"

We provide code and instructions how to use our models. The MIMIC and HIRID datasets can be downloaded directly via physionet (https://physionet.org/content/hirid/1.1.1/, https://physionet.org/content/mimiciii/1.4/). ICDEP cannot be shared due to German privacy regulations but we aim to make an anonymous version available soon. Besides, complete case data and imputed datasets related to MIMIC are currently submitted to Physionet. 


# Model Card for Autoencoder Imputers

<!-- Provide a quick summary of what the model is/does. -->

A spatio-temporal autoencoder (STAE) is a type of deep learning model designed to learn compact representations of data that vary across both space and time. Its purpose is to capture spatial structures (e.g., relationships between features, pixels, or locations) together with temporal dynamics (e.g., evolution, trends, dependencies over time). Here, this model is used for imputations

## Model Details

### Model Description

<!-- Provide a longer summary of what this model is. -->

- **Developed by:** {Niklas Giesa}
- **Funded by [optional]:** {Institute of Medical Informatics}
- **Model type:** {LSTM / BILSTM STAE}
- **License:** {cc}

### Model Sources [optional]

<!-- Provide the basic links for the model. -->

- **Repository:** {(https://github.com/ngiesa/mi_icu)}

## Uses

Users who want to test a vaiation of imputation methods can use the models

### Direct Use

Imputations, apriori to downstream prediction tasks. 

### Downstream Use [optional]

<!-- This section is for the model use when fine-tuned for a task, or when plugged into a larger ecosystem/app -->

Clinical prediction taks for death, or delirium 

### Out-of-Scope Use

Direct clinical decision making. 

## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

Sample size, selection bias, healthsystem properties. 

### Recommendations

Using benchmark on center-specific datasets and comparing results with benchmarks against our results. 

## How to Get Started with the Model

<code>
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load("model.pth", map_location=device))
model.to(device)
</code>


## Training Details

### Training Data

Cross validation (CV) sets dataset specific MIMIC, HIRID, ICDEP

### Training Procedure

Hyperband optimization via CV.

#### Preprocessing [optional]

Methods in main manuscript. 


#### Training Hyperparameters

Numer of hidden layers, accumulate graident batches (Batch of 1 due to unevenly sized sequences), number of stacked modules, preprocessing (GT, LOCF preimpute for training) etc. 

## Evaluation


### Testing Data, Factors & Metrics

#### Testing Data

Hold-out CV fold. 

#### Factors

Subsets of CV. 

#### Metrics

Cross-correlation and auto-correlation errors, RSME, MSE, Wasserstein distance 

### Results

Results in main manuscript. 

#### Summary

Summary in main manuscript. 

## Model Examination [optional]


## Environmental Impact

<!-- Total emissions (in grams of CO2eq) and additional considerations, such as electricity usage, go here. Edit the suggested text below accordingly -->

Carbon emissions can be estimated using the [Machine Learning Impact calculator](https://mlco2.github.io/impact#compute) 

Multiple trainings for approxamitely 167 hours on 5 GPUs as NVIDIA A100 Tensor Core, approx. 23.38 CO2 emitted. 

## Technical Specifications

### Model Architecture and Objective

STAE, additional transformer model specified in https://www.nature.com/articles/s43856-024-00681-x. 

### Compute Infrastructure / Hardware

High Performance Cluster https://www.hpc.bihealth.org/


#### Software

VSCode, Pytorch environment


## Model Card Contact

niklas.giesa@charite.de

