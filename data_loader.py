import random
import os
import pandas as pd
import json 
import numpy as np

class DataLoader():

    ''' class for loading different datasets and resampling '''

    def __init__(self):
        self.data_path_mimic = "../deploy_scripts/data/source_files/all_hourly_data.h5"
        self.data_path_hirid = "../deploy_scripts/data/source_files/hirid_processed_sampled.h5"#TODO change for respiratory rate
        self.data_path_icdep = "../deploy_scripts/data/source_files/icdep_sequences.csv"
        self.dataset_sequences = {}
        self.dataset_static = {}
        self.complete_datasets = {}
        self.sequence_features = []
        self.identifier = 0
        self.dataset_name = ""
    
    def load_dataset(self, name="MIMIC", chunk=False):
        
        """ loading dataset 
        
            Parameters: database name (name): str either MIMIC or HIRID
            Returns: none
            
        """

        # load feature names for time dynamic variables
        with open("../deploy_scripts/data/features_dataset.json", "r+") as f:
            features_map = json.load(f)
            
        self.sequence_features = features_map[name.upper()]["time_dynamic_features"]
        self.dataset_name = name
        
        print("reading: ",  name)
        
        if name == "MIMIC":
            self.identifier = "subject_id"
            self.dataset_sequences = pd.read_hdf(self.data_path_mimic, 'vitals_labs_mean', mode="r").reset_index()
            # drop identifiers that are redundant
            self.dataset_sequences.columns = self.dataset_sequences.columns.droplevel('Aggregation Function')
            self.dataset_sequences = self.dataset_sequences.drop(columns=["hadm_id", "icustay_id", "hours_in"])
            self.dataset_static = pd.read_hdf(self.data_path_mimic, 'patients')
            # random set ages over 300 due to HIPPA deidentification between 90 and 100
            rdm_limit= len(self.dataset_static.loc[(self.dataset_static["age"] > 300)])
            self.dataset_static.loc[(self.dataset_static["age"] > 300), "age"] = \
                [np.random.choice(np.arange(90, 100), p=list(np.linspace(20,0,10)/100)) for _ in range(0, rdm_limit)]
            
        if name == "ICDEP":
            self.identifier = "sequence_id"
            self.dataset_sequences = pd.read_csv(self.data_path_icdep, index_col=0)
            self.dataset_static = pd.read_csv("../deploy_scripts/data/source_files/icdep_gender_age.csv", index_col = 0)
            self.dataset_static["gender"] = ["F" if x == "W" else "M" for x in self.dataset_static["gender"]]
        
        if name == "HIRID":
            print("loading HIRID")
            self.identifier = "patientid"
            #if chunk:
            #    self.dataset_sequences = pd.read_hdf(self.data_path_hirid, 'hirid', mode='r', chunksize=10000)
            #else:
            #    self.dataset_sequences = pd.read_hdf(self.data_path_hirid, 'hirid', mode='r', columns=["patientid", "timestamp", "ABPs", 'ABPd  ', "SpO2", "HR", "RR"])\
            #    .drop(columns=["timestamp"]) #TODO col restriction
            #self.dataset_sequences = self.dataset_sequences.set_index("timestamp")
            self.dataset_sequences = pd.read_csv("/home/giesan/mi_icu/deploy_scripts/data/source_files/hirid_preprocessed_sampled.csv")
            print("HIRID static loaded")
            self.dataset_static = pd.read_csv("../deploy_scripts/data/source_files/hirid_demographics.csv", index_col = 0).rename(columns={"sex": "gender"})
    
    def load_complete_case(self):   
        
        """ loading created complete case sets """
        
        for file in os.listdir("../deploy_scripts/data/complete/"):
            if not self.dataset_name.lower() in file:
                continue
            self.complete_datasets[file.split(".csv")[0].replace("_complete", "")] = pd.read_csv("../deploy_scripts/data/complete/" + file, index_col=0)


