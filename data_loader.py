import pandas as pd
from utils import categorize_age
import os

class DataLoader():

    ''' class for handling mimic preprocessed file and derived complete case sets '''

    def __init__(self, data_path:str = "", conf_items: dict = {}, resample_feats: list = []):
        self.data_path = data_path
        self.data_meas, self.data_demo = self.load_mimic_hourly_data()
        self.conf_items = conf_items
        self.all_features = list(self.data_meas\
                             .drop(columns=["hadm_id", "subject_id", "hours_in", "icustay_id"]).columns)
        self.complete_datasets = self.load_complete_datasets()
        
    
    def load_complete_datasets(self):
        ''' load resampled complete datasets if available'''
        complete_ds_names, ds = [f for f in os.listdir("./data") if not ".h5" in f], {}
        for ds_name in complete_ds_names:
            ds[ds_name.replace(".csv", "")] = pd.read_csv("./data/{}".format(ds_name), index_col=0)
        return ds
        

    def load_mimic_hourly_data(self):
        ''' load hourly measurements and static demogr data'''
        data_meas = pd.read_hdf(self.data_path, 'vitals_labs_mean')
        data_meas.columns = data_meas.columns.droplevel('Aggregation Function')

        data_demo = pd.read_hdf(self.data_path, 'patients')
        data_demo.loc[:,'age'] = data_demo['age'].apply(categorize_age)

        return data_meas.reset_index(), data_demo.reset_index()
    
    def resample_data(self, interval_h: int = 4, features: list = [], patient_id:str = "subject_id", drop_na:bool=True):
        '''
        resamples hourly dataset by sampling interval in h, selects features, and drops all patients with any NaN

                Parameters:
                        data (DataFrame): Dataset with NaN, features and patient identifier 
                        features (list): List of features 
                        patient_id (str): Patient identifier column 

                Returns:
                        df (DataFrame): Complete Case Set with resampled mean values
        '''
        data=self.data_meas
        # selection of features with patient id
        data = data.loc[:, features + [patient_id] ]
        # setting interval index for groups 
        data['{}h_index'.format(str(interval_h))] = data.groupby([patient_id]).cumcount()//interval_h
        # resample values with mean and keeping outliers 
        data = data.groupby([patient_id, '{}h_index'.format(str(interval_h))]).mean().reset_index()
        if not drop_na:
            return data
        # dropping all patients with any NaN in any feature per time index group
        data = data.groupby(patient_id).filter(lambda x: x.notna().all().all())
        return data
    

    def apply_conf_items(self):
        ''' iterates through item dict and resamples data '''
        for item in self.conf_items.keys():
            print("resampling dataset ", item)
            config = self.conf_items[item]
            self.conf_items[item]["data"] = self.resample_data(
                                        interval_h= config["sampling_interval"], 
                                        features=config["features"], 
                                        patient_id=config["patient_id"]
                                    )
            self.conf_items[item]["data"].to_csv("./data/dataset_{}.csv".format(item.replace(" ", "_")))
            self.conf_items[item]["n_subjects"] = len(self.conf_items[item]["data"][config["patient_id"]].unique())
        return self.conf_items