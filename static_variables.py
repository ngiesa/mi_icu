
import pandas as pd

# clip values to corr bound
VALID_RANGES = {
    "hr": [5, 400],
    "rr": [2, 300],
    "spo2": [10, 100],
    "bp_sys": [20, 400],
    "bp_dia": [10, 300]
}

FEATURES = ['hr', 'bp_sys', 'bp_dia', 'spo2', 'rr']
DATASETS = ["MIMIC", "HIRID", "ICDEP"]


FM_ORIGINAL = {
    "MIMIC": {
        'hr':0.104, 'bp_sys':0.132, 'bp_dia':0.132, 'spo2':0.143, 'rr':0.118
    },
    "HIRID": {
        'hr':0.006, 'bp_sys':0.039, 'bp_dia':0.039, 'spo2':0.025, 'rr':0.266
    },
    "ICDEP": {
        'hr':0.047, 'bp_sys':0.235, 'bp_dia':0.235, 'spo2':0.049, 'rr':0.113
    }
}

# mapping from 
FEATURE_MAPPING = {
        "hirid": ["HR", "ABPs", 'ABPd  ', "SpO2", "RR"],
        "mimic": ['heart rate', 'systolic blood pressure','diastolic blood pressure', 'oxygen saturation', 'respiratory rate'],
        "icdep": ["heart_rate",  "bp_sys", "bp_dia", "spo2", "rr_total"] #TODO
    }

# setting resampling config x T times
AGGREGATION_STEP = {
    "MIMIC": 1,
    "HIRID": 2,
    "ICDEP": 2
}

def load_complete_sequences(clean):
    
    """ loading complete data """
    
    list_complete_sequences=[]
    for link in ["sequence_mimic_{}.csv", "sequence_hirid_{}.csv", "sequence_icdep_{}.csv"]:
        complete_data = pd.read_csv("./complete_sequences/" + link.format(clean), index_col = 0)
        complete_data = complete_data.assign(time_index = complete_data.groupby("sequence_id")["hr"].cumcount()).sort_values(by=["sequence_id", "time_index"])
        list_complete_sequences.append(complete_data)
    return list_complete_sequences

def load_resampled_sequences(clean):
    
    """ loading resampled data """
    
    list_resampled_sequences=[]
    for link in ["sequence_1_mimic_{}.csv", "sequence_2_hirid_{}.csv", "sequence_2_icdep_{}.csv"]:
        list_resampled_sequences.append(pd.read_csv("./resampled_sequences/" + link.format(clean), index_col = 0))
    return list_resampled_sequences