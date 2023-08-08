import pandas as pd

from data_loader import DataLoader
from itertools import chain, combinations
from functools import reduce

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

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

            comb_feats, interval_h = [], int(sampling.replace("sampling", "")\
                                             .replace("_", "").replace("h", ""))

            # create all possible combinations of features
            for i, combo in enumerate(powerset(features), 1):
                if (not set(combo) in [set(c) for c in comb_feats]) & (combo!=()):
                    comb_feats.append(list(combo))

            res_inner_dfs = []

            # resample the data for interval accordingly
            if interval_h == 1:
                data = data_loader.data_meas
            else:
                data = data_loader.resample_data(features=data_loader.features, interval_h=interval_h)

            for j, combo in enumerate(comb_feats):

                # go through feature sets and get number of patients without any NaN value
                data_combo = data[combo + [patient_id]].groupby(patient_id).filter(lambda x: x.notna().all().all())
                
                if data_combo.empty:
                    continue
                
                n = len(data_combo[patient_id].unique())

                # append results 
                res_inner_dfs.append(pd.DataFrame({"combo_set": j, "features": str(combo), 
                                            "n_patients_{}_sampling".format(interval_h): n}, 
                                            index=[j]))

                print(pd.concat(res_inner_dfs))

            res_outer_dfs.append(pd.concat(res_inner_dfs))

    if len(res_outer_dfs) > 0:
        df_write = reduce(lambda left,right: pd.merge(left,right,on=['features'],how='inner'), res_outer_dfs)
    else:
        df_write=res_outer_dfs[0]
    
    df_write.to_csv("./combo_miss_rate_feats.csv")