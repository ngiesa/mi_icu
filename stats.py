import pandas as pd

from data_loader import DataLoader
from itertools import chain, combinations
from functools import reduce


def combine_features(data_loader:DataLoader=None, number_of_features:int=0, patient_id:str = "subject_id"):
    
    df_miss = pd.read_csv("./missing_rates.csv", index_col=0)

    # draw first n columns and combine iteratievly to datasets
    print("combine dataset for ", df_miss.iloc[:number_of_features])

    # define features
    features = list(df_miss.iloc[:number_of_features]["LEVEL2"])

    res_outer_dfs = []

    # try these wit different sampling intervals
    for sampling in df_miss.columns:
        if "sampling" in sampling:
            print("sampling ", sampling)

            comb_feats, interval_h = [[]], int(sampling.replace("sampling", "")\
                                             .replace("_", "").replace("h", ""))
            
            # combine n features
            for i, f in enumerate(features):
                comb_feats.append([f] + comb_feats[i])

            _ = comb_feats.pop(0)

            res_inner_dfs = []

            # resample the data for interval accordingly
            if interval_h == 1:
                data = data_loader.data_meas
            else:
                data = data_loader.resample_data(features=data_loader.all_features, interval_h=interval_h, drop_na=False)

            for j, combo in enumerate(comb_feats):

                # go through feature sets and get number of patients without any NaN value
                data_combo = data[combo + [patient_id]].groupby(patient_id).filter(lambda x: x.notna().all().all())

                data_combo = data_combo.groupby(patient_id).filter(lambda x: x.notna().all().all())
                    
                if data_combo.empty:
                    continue
                
                n = len(data_combo[patient_id].unique())

                # append results 
                res_inner_dfs.append(pd.DataFrame({
                                            "combo_set": j+1, "features": str(combo), 
                                            "n_patients": n,
                                            "sampling": interval_h}, 
                                            index=[j]))

                print(pd.concat(res_inner_dfs))

            res_outer_dfs.append(pd.concat(res_inner_dfs))

    if len(res_outer_dfs) > 0:
        df_write = pd.concat(res_outer_dfs)
    else:
        df_write=res_outer_dfs[0]
    
    df_write.to_csv("./combo_miss_rate_feats.csv")