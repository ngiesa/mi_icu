

from data_loader import DataLoader
from plotting import plot_frequency_lens, plot_missing_rates_heatmap, plot_scatter_feat_combo, plot_scatter_resample_feats
from stats import combine_features

# example usage
conf_items = {
    "2h set": {
        "sampling_interval": 2,
        "features": ['heart rate', 'systolic blood pressure', 'diastolic blood pressure', 'respiratory rate', 'oxygen saturation'], 
        "patient_id": "subject_id"
    },
    "4h set": {
        "sampling_interval": 4,
        "features": ['heart rate', 'systolic blood pressure', 'diastolic blood pressure', 'respiratory rate', 'oxygen saturation'], 
        "patient_id": "subject_id"
    },
    "6h set": {
        "sampling_interval": 6,
        "features": ['heart rate', 'systolic blood pressure', 'diastolic blood pressure', 'respiratory rate', 'oxygen saturation'],
        "patient_id": "subject_id"
    },
}

DATA_PATH = "./data/all_hourly_data.h5"
data_loader = DataLoader(data_path=DATA_PATH)

#plot_missing_rates_heatmap(data_loader=data_loader, sampling_intervals=[1, 2, 4, 6])

#combine_features(data_loader=data_loader, number_of_features=8)
#plot_scatter_feat_combo()

plot_scatter_resample_feats(data_loader=data_loader)

data_loader.conf_items= conf_items
data_loader.apply_conf_items()

plot_frequency_lens(data_loader=data_loader)
