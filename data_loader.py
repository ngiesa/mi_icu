import pandas as pd
from utils import categorize_age

class DataLoader():

    def __init__(self, data_path):
        self.data_path = data_path
        self.data_meas, self.data_demo = self.load_mimic_hourly_data()

    def load_mimic_hourly_data(self):
        ''' load hourly measurements and static demogr data'''

        data_meas = pd.read_hdf(DATA_PATH, 'vitals_labs_mean')
        data_meas.columns = data_meas.columns.droplevel('Aggregation Function')

        data_demo = pd.read_hdf(DATA_PATH, 'patients')
        data_demo.loc[:,'age'] = data_demo['age'].apply(categorize_age)

        return data_meas.reset_index(), data_demo.reset_index()
    
    def resample_data(self, data: pd.DataFrame=None, interval_h: int = 4, features: list = [], patient_id:str = "subject_id"):
        '''
        Resamples hourly dataset by sampling interval in h, selects features, and drops all patients with any NaN

                Parameters:
                        data (DataFrame): Dataset with NaN, features and patient identifier 
                        features (list): List of features 
                        patient_id (str): Patient identifier column 

                Returns:
                        df (DataFrame): Complete Case Set with resampled mean values
        '''
        # selection of features with patient id
        data = data.loc[:, features + [patient_id], ]
        # setting interval index for groups 
        data['{}h_index'.format(str(interval_h))] = data.groupby([patient_id]).cumcount()//interval_h
        # resample values with mean and keeping outliers 
        data = data.groupby([patient_id, '{}h_index'.format(str(interval_h))]).mean().reset_index()
        # dropping all patients with any NaN in any feature per time index group
        return data.groupby(patient_id).filter(lambda x: x.notna().all().all())
    

    def apply_conf_items(self, conf_items: dict = None):
        ''' iterates through item dict and resamples data '''
        for item in conf_items.keys():
            print("resampling dataset ", item)
            config = conf_items[item]
            conf_items[item]["data"] = self.resample_data(
                                        data=self.data_meas, 
                                        interval_h= config["sampling_interval"], 
                                        features=config["features"], 
                                        patient_id=config["patient_id"]
                                    )
            conf_items[item]["data"].to_csv("./dataset_{}.csv".format(item.replace(" ", "_")))
            conf_items[item]["n_subjects"] = len(conf_items[item]["data"][config["patient_id"]].unique())
        return conf_items



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
        "features": ['heart rate', 'systolic blood pressure', 'diastolic blood pressure', 'respiratory rate', 'oxygen saturation', 'temperature'],
        "patient_id": "subject_id"
    },
}

DATA_PATH = "./data/all_hourly_data.h5"
data_loader = DataLoader(data_path=DATA_PATH)
data_loader.apply_conf_items(conf_items=conf_items)