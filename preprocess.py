import pandas as pd
from data_loader import DataLoader
from statsmodels.tsa.stattools import acf
import numpy as np
from plotting import plot_baseline_infos, plot_demographics
from static_variables import DATASETS, FEATURE_MAPPING, FEATURES, VALID_RANGES, AGGREGATION_STEP

# getting identifier mapping
df_identifier = pd.read_excel("../deploy_scripts/stats/common_feat_miss.xlsx")


def prepare_cv_splits():
    # splits of three fold cross validation 

    df_cv = []
    
    for c, clean in enumerate(["pres", "drop"]):
        completes = load_complete_sequences("drop")
        
        for i, ds in enumerate(completes):
            seq_id = ds.sequence_id.unique()
            step = int(len(seq_id)/3)
            
            for r in (0, 3):
                frac_test = seq_id[int(r* step): int((r* step)+step)]
                frac_train = [s for s in seq_id if s not in frac_test]
                
                df_cv.append(pd.DataFrame({"DATASET": DATASETS[i], "CLEAN":clean, "CV": r, 
                                        "SET": len(frac_test) * ["TEST"] + len(frac_train) * ["TRAIN"],
                                        "sequence_id": list(frac_test) + list(frac_train)}))
                
    pd.concat(df_cv).to_csv("/home/giesan/mi_icu/spatio_temporal_pattern/complete_sequences/cv_splits_all.csv")



# function for resampling sequences
def resampling_sequences(sequences, aggregation_step, dataset, identifier, features):
    return sequences.\
            assign(time_index = (sequences.groupby(identifier).cumcount()/aggregation_step[dataset.upper()]).astype(int))\
                .groupby([identifier, "time_index"])[features].mean().reset_index()
                
# cleansing as clipping      
def cleansing(df_seq, remove = False):
    rows_affected = []
    for f in FEATURES:
        # either remove or clip ranges
        if remove:
            rows_affected.append(round(1 - len([v for v in df_seq[f] if (v >= VALID_RANGES[f][0]) & (v <= VALID_RANGES[f][1])]) / len(df_seq), 3))
            df_seq[f] = [v if (v >= VALID_RANGES[f][0])\
                & (v <= VALID_RANGES[f][1]) \
                    else float("NaN") for v in df_seq[f]]
        else:
            df_seq[f] = df_seq[f].clip(lower=VALID_RANGES[f][0], 
                                        upper=VALID_RANGES[f][1])
    return df_seq, rows_affected

STATIC_DATA = []
    
def resample_data():
    
    """ sampling data, cleansing and plotting differences """
    
    list_resampled_sequences, static_data, df_seq_len_org= [], [], []

    plotting_data = {}

    df_rows_cleansing = []
    for clean in ["pres", "drop"]:
        
        for d, dataset in enumerate(DATASETS):
            
            dl = DataLoader()
            dl.load_dataset(name=dataset.upper())
            
            if dataset.upper() != "ICDEP":
                continue
            
            # store single missingness
            df_single_missing = []
            
            # get sequential data 
            sequences = dl.dataset_sequences[FEATURE_MAPPING[dataset.lower()] + [dl.identifier]]
            static =  dl.dataset_static.reset_index()
            
            # merge data with all adults
            sequences = sequences.merge(static[static.age >= 18][dl.identifier], on=dl.identifier)
            
            
            # append static data
            static_append = static[static.age >= 18].assign(sequence_id = dl.dataset_static.reset_index()[dl.identifier])
            static_data.append(static_append)
            
            static_append.to_csv("./desc/static_data_{}.csv".format(dataset.lower()))
            
            l = list(sequences.columns)
            l.remove(dl.identifier)
            
            # resample sequences according to sampling intervals
            resampled_sequences = resampling_sequences(
                sequences=sequences, 
                aggregation_step=AGGREGATION_STEP,
                dataset=dataset,
                identifier=dl.identifier, features=l)
            
            # rename sequences
            for feature in FEATURE_MAPPING[dataset.lower()]:
                abbr_feat = df_identifier[df_identifier["{}_org".format(dataset.lower())] == feature]\
                    ["{}_abbr".format(dataset.lower())].iloc[0]
                resampled_sequences = resampled_sequences.rename(columns={feature: abbr_feat})
            
            # apply cleansing processes
            if clean == "preserve":
                resampled_sequences, rows = cleansing(df_seq=resampled_sequences, remove=False)
            if clean == "drop":
                resampled_sequences, rows = cleansing(df_seq=resampled_sequences, remove=True)
                df_rows_cleansing.append(pd.DataFrame({"dataset": dataset, "feature": FEATURES, "rows": rows, "len": len(resampled_sequences)}))
            
            # store the sequence lengths
            df_seq_len_org.append(resampled_sequences.groupby(dl.identifier)["time_index"].count().reset_index())
            
            acf_univariate = []
            # univariate autocorrelation
            for feature in FEATURES:
                acf_res = acf(nlags=1, x=resampled_sequences.drop(columns=["time_index"]).isna().astype(int)[feature], fft=True)
                acf_univariate.append(pd.DataFrame({"dataset": dataset, "feature": feature, "acf_coef": acf_res[1]}, index=[feature]))
                
                df_single_missing.append(pd.DataFrame({"feature": feature,
                                                        "single_miss": resampled_sequences[feature].isna()\
                                                        .sum()/len(resampled_sequences)},index=[feature]))
            # cross_correlation
            df_acf_analyse = resampled_sequences.drop(columns=["time_index", dl.identifier])
            df_acf_analyse.columns = FEATURES
            list_resampled_sequences.append(df_acf_analyse.assign(sequence_id = resampled_sequences[dl.identifier]))
            
            df_acf_analyse.assign(sequence_id = resampled_sequences[dl.identifier])\
                .to_csv("./resampled_sequences/sequence_{}_{}_{}.csv"\
                    .format(AGGREGATION_STEP[dataset.upper()], dataset.lower(), clean))
                
            print("resampled stored")
            
            corr = df_acf_analyse.isna().astype(int).corr()
            
            plotting_data[dataset.upper()] = {
                "acf": pd.concat(acf_univariate),
                "miss": pd.concat(df_single_missing),
                "corr": corr
            }
            
            return
            
        plot_baseline_infos(plotting_data, DATASETS, clean)
        

    pd.concat(df_rows_cleansing).to_csv("./desc/clean_rows.csv")
    
    return static_data
    

def cut_na_sequences(sequences, features, identifier = "sequence_id"):
    
    """ shortening sequence data to complete """
    
    # get NAs for included features
    df_miss = sequences[features].isna().astype(int)
    df_miss = df_miss.assign(sequence_id = sequences[identifier])
    # get first Na occurrence in sequence 
    df_miss["first_NA"] = df_miss[features].max(axis=1)
    df_miss["time_index"] = df_miss.groupby(identifier).cumcount()
    # filter all sequences that are less than 3 time steps 
    df_miss["drop_row"] = df_miss.groupby(["sequence_id"])["first_NA"].cumsum()
    # unify results and drop all NAs
    sequences["drop_row"] = df_miss["drop_row"]
    sequences["time_index"] = df_miss["time_index"]
    return sequences[(sequences.drop_row == 0)].drop(columns=["drop_row", "time_index"])


def load_resampled_sequences(clean):
    
    """ loading resampled data """
    
    list_resampled_sequences=[]
    for link in ["sequence_1_mimic_{}.csv", "sequence_2_hirid_{}.csv", "sequence_2_icdep_{}.csv"]:
        list_resampled_sequences.append(pd.read_csv("./resampled_sequences/" + link.format(clean), index_col = 0))
    return list_resampled_sequences


def load_complete_sequences(clean):
    
    """ loading complete data """
    
    list_complete_sequences=[]
    for link in ["sequence_mimic_{}.csv", "sequence_hirid_{}.csv", "sequence_icdep_{}.csv"]:
        complete_data = pd.read_csv("./complete_sequences/" + link.format(clean), index_col = 0)
        complete_data = complete_data.assign(time_index = complete_data.groupby("sequence_id")["hr"].cumcount()).sort_values(by=["sequence_id", "time_index"])
        list_complete_sequences.append(complete_data)
    return list_complete_sequences


def build_complete(plot = False):
    
    """ shortening sequences NA """
    
    df_desc_all = []
    for clean in ["pres", "drop"]:
        list_complete_sequences = []
        list_resampled_sequences = load_resampled_sequences(clean=clean)
            
        for s, res_sequence in enumerate(list_resampled_sequences):
            
            # drop all sequences with any NaN and get the number of remaining sequences 
            not_na_sequences = cut_na_sequences(sequences=res_sequence, features=FEATURES, identifier="sequence_id")
            # omit all seq that are less than 3 time steps and complete 
            not_na_sequences = not_na_sequences.merge(not_na_sequences.groupby("sequence_id")["hr"].count(), on="sequence_id", suffixes=["", "_len"])
            list_complete_sequences.append(not_na_sequences[not_na_sequences.hr_len > 2].drop(columns=["hr_len"]))
            not_na_sequences[not_na_sequences.hr_len > 5].drop(columns=["hr_len"]).to_csv("./complete_sequences/sequence_{}_{}.csv".format(DATASETS[s].lower(), clean))
            
            # store data how much remained after dropping and creating complete set
            df_desc_all.append(
                    pd.DataFrame(
                        {
                            "resampled_records": len(res_sequence),
                            "resampled_sequences": len(res_sequence["sequence_id"].unique()),
                            "complete_records": len(not_na_sequences),
                            "complete_sequences": len(not_na_sequences["sequence_id"].unique()),
                            "frac_records": round(len(not_na_sequences) / len(res_sequence), 3),
                            "frac_sequences": round(len(not_na_sequences["sequence_id"].unique()) / len(res_sequence["sequence_id"].unique()), 3),
                            "clean": clean,
                            "dataset": DATASETS[s]
                        }, index=[s]
                    )
                )
        
        if plot:
            plot_demographics(list_complete_sequences, list_resampled_sequences, DATASETS, clean)    
        
    pd.concat(df_desc_all).to_csv("./desc/compl_records.csv")


#plot_demographics(load_complete_sequences("pres"), load_resampled_sequences("pres"), DATASETS, "pres")