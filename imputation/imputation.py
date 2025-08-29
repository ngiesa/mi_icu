import pandas as pd
import numpy as np
import random
from methods import apply_interpolation, apply_averages, apply_mice_mf, apply_mice_lr_stats, apply_expanding, apply_averages, apply_locf, apply_mice_knn
from static_variables import FEATURES, DATASETS, load_complete_sequences, FM_ORIGINAL

def prepare_bootstrapps(pattern = "mcar", clean = "pres", max_boot = 1, direction = "rev"):
    
    "prepares bootstrapped simulated missingness"
    
    list_complete_sequences = load_complete_sequences(clean=clean)
    
    bootstrapps = {}
    
    for d, dataset in enumerate(DATASETS):
        
        print("preparing boot for ", dataset)
        
        bootstrapps[dataset] = {}
        complete = list_complete_sequences[d]
        bootstrapps[dataset]["complete"] = complete
        induced_boots = []
        for b in range(0, max_boot):
            boot_occurence = pd.read_csv("/home/giesan/mi_icu/spatio_temporal_pattern/induced_sequences/{}/missing_matrix_{}_boot_{}_{}_{}.csv".format(
                                    dataset.lower(), dataset.lower(), str(b), pattern, clean if ((pattern != "mnar") and (direction != "rev")) else clean), index_col = 0)
            induced_df  = pd.DataFrame(np.array(complete[FEATURES]) * np.array(boot_occurence[FEATURES]\
            .replace(1, float('NaN')).replace(0.0, 1)))
            induced_df.columns = FEATURES
            induced_df = induced_df.assign(sequence_id = complete.reset_index().sequence_id)
            induced_df = induced_df.assign(time_index = induced_df.groupby("sequence_id")["hr"].cumcount())
            induced_boots.append(induced_df)
        bootstrapps[dataset]["boots"] = induced_boots
        
    return bootstrapps

def prepare_static_data(dataset = "MIMIC"):
        static_data = pd.read_csv("/home/giesan/mi_icu/spatio_temporal_pattern/desc/static_data_{}.csv".format(dataset.lower()))
        static_data = static_data[["age", "gender", "sequence_id"]].sort_values(by="sequence_id")
        static_data = static_data.assign(gender = [0 if x == "M" else 1 for x in static_data.gender])
        return static_data.assign(age = [(x-static_data.age.min())/(static_data.age.max()-static_data.age.min()) for x in static_data.age])

def apply_all_imputation_methods(clean = "pres", pattern="mcar", max_boot=1, direction = "", mode="dl"):
    
    # mode: "baseline" or "dl"
    # direction: "rev" or ""
    
    """ apply more traditional imputation methods omitting DL like AE"""
    
    bootstrapps = prepare_bootstrapps(max_boot=max_boot, pattern=pattern, direction = direction)
    
    for dataset in DATASETS:
        
        print(dataset)
        
        complete = bootstrapps[dataset]['complete'].reset_index(drop=True)
        
        print(len(complete))
        
        for b, boot_df in enumerate(bootstrapps[dataset]["boots"]):

            print("bootstrapp: ", b)
            cv_splits = pd.read_csv("/home/giesan/mi_icu/spatio_temporal_pattern/complete_sequences/cv_splits_all.csv", index_col=0)
            cv_splits = cv_splits[(cv_splits["DATASET"] == dataset.upper()) & ((cv_splits["CLEAN"] == clean.lower()))]
            
            # apply 3 fold cross validation 
            for cv in range(0, 3):
                
                print("cv round ", cv)
                
                cv_seq_ids = cv_splits[cv_splits["CV"] == cv]

                df_induced = boot_df[boot_df.sequence_id.isin(list(cv_seq_ids[cv_seq_ids["SET"] == "TEST"].sequence_id))]
                df_train = boot_df[boot_df.sequence_id.isin(list(cv_seq_ids[cv_seq_ids["SET"] == "TRAIN"].sequence_id))]
                
                if mode == "baseline":
                    
                    # induction of MCAR as baseline comparison
                    
                    if not "baseline" in clean:
                        clean = "{}_{}".format(clean, mode) 
                    
                    random.seed(10)
                    
                    for f in FM_ORIGINAL[dataset].keys():
                        df_drop = complete[f].copy()
                        drop_indices = np.random.choice(df_drop.index, int(len(df_drop) *  FM_ORIGINAL[dataset][f]), replace=False)
                        df_drop.iloc[drop_indices] = float('NaN')
                        complete[f] = df_drop
                    
                    df_induced = complete[complete.sequence_id.isin(list(cv_seq_ids[cv_seq_ids["SET"] == "TEST"].sequence_id))]
                    df_train = complete[complete.sequence_id.isin(list(cv_seq_ids[cv_seq_ids["SET"] == "TRAIN"].sequence_id))]
                
                print("len cv test ", len(df_induced))
                print("len cv train ", len(df_train))
                
                print("RUNNING METHODS")
                apply_expanding(induced_df=df_induced, training_df=df_train, aggr="mean").to_csv("./imputed_sequences/{}/rmean_{}_{}_{}_cv_{}.csv".format(dataset.lower(), b, pattern, clean + direction, cv))
                apply_expanding(induced_df=df_induced, training_df=df_train,aggr="median").to_csv("./imputed_sequences/{}/rmedi_{}_{}_{}_cv_{}.csv".format(dataset.lower(), b, pattern, clean  + direction, cv))
                print("AVERAGINGS")
                apply_averages(induced_df=df_induced, training_df=df_train, aggr="mean", fill="pop").to_csv("./imputed_sequences/{}/pmean_{}_{}_{}_cv_{}.csv".format(dataset.lower(), b, pattern, clean + direction, cv))
                apply_averages(induced_df=df_induced, training_df=df_train, aggr="median", fill="pop").to_csv("./imputed_sequences/{}/pmedi_{}_{}_{}_cv_{}.csv".format(dataset.lower(), b, pattern, clean + direction, cv))
                apply_averages(induced_df=df_induced, training_df=df_train, aggr="mean", fill="1st").to_csv("./imputed_sequences/{}/1mean_{}_{}_{}_cv_{}.csv".format(dataset.lower(), b, pattern, clean + direction, cv))
                apply_averages(induced_df=df_induced, training_df=df_train, aggr="median", fill="1st").to_csv("./imputed_sequences/{}/1median_{}_{}_{}_cv_{}.csv".format(dataset.lower(), b, pattern, clean + direction, cv))
                print("INTERPOLATION METHODS")
                apply_interpolation(induced_df=df_induced, training_df=df_train, regress="linear").to_csv("./imputed_sequences/{}/linpol_{}_{}_{}_cv_{}.csv".format(dataset.lower(), b, pattern, clean + direction, cv))
                apply_interpolation(induced_df=df_induced, training_df=df_train, regress="nearest").to_csv("./imputed_sequences/{}/neapol_{}_{}_{}_cv_{}.csv".format(dataset.lower(), b, pattern, clean + direction, cv))
                apply_interpolation(induced_df=df_induced, training_df=df_train, regress="quadratic").to_csv("./imputed_sequences/{}/quapol_{}_{}_{}_cv_{}.csv".format(dataset.lower(), b, pattern, clean + direction, cv))
                print("EXTRAPOLATION METHODS")
                apply_interpolation(induced_df=df_induced, training_df=df_train, regress="linear", type_polation="extra").to_csv("./imputed_sequences/{}/linexp_{}_{}_{}_cv_{}.csv".format(dataset.lower(), b, pattern, clean + direction, cv))
                apply_interpolation(induced_df=df_induced, training_df=df_train, regress="quadratic", type_polation="extra").to_csv("./imputed_sequences/{}/quaexp_{}_{}_{}_cv_{}.csv".format(dataset.lower(), b, pattern, clean + direction, cv))
                apply_locf(induced_df=df_induced,  training_df=df_train, fill="1st").to_csv("./imputed_sequences/{}/1locf_{}_{}_{}_cv_{}.csv".format(dataset.lower(), b, pattern, clean + direction, cv))
                apply_locf(induced_df=df_induced, training_df=df_train, fill="pop").to_csv("./imputed_sequences/{}/plocf_{}_{}_{}_cv_{}.csv".format(dataset.lower(), b, pattern, clean + direction, cv))
                print("MICE")
                apply_mice_mf(induced_df=df_induced).to_csv("./imputed_sequences/{}/mictre_{}_{}_{}_cv_{}.csv".format(dataset.lower(), b, pattern, clean + direction, cv))
                apply_mice_lr_stats(induced_df=df_induced).to_csv("./imputed_sequences/{}/miclr_{}_{}_{}_cv_{}.csv".format(dataset.lower(), b, pattern, clean + direction, cv))
                apply_mice_knn(induced_df=df_induced).to_csv("./imputed_sequences/{}/micknn_{}_{}_{}_cv_{}.csv".format(dataset.lower(), b, pattern, clean + direction, cv))
            
                if (mode != "baseline") and (clean == "pres"):
                    # merge with static data gender and age
                    static_data = prepare_static_data(dataset=dataset)
                    boot_df_static = df_induced.merge(static_data, on=["sequence_id"])
                    apply_mice_mf(induced_df=boot_df_static).to_csv("./imputed_sequences/{}/mictrestat_{}_{}_{}_cv_{}.csv".format(dataset.lower(), b, pattern, clean + direction, cv))
                    apply_mice_lr_stats(induced_df=boot_df_static, with_static=True).to_csv("./imputed_sequences/{}/miclrstat_{}_{}_{}_cv_{}.csv".format(dataset.lower(), b, pattern, clean + direction, cv))
                    apply_mice_knn(induced_df=boot_df_static, with_static=True).to_csv("./imputed_sequences/{}/micknnstat_{}_{}_{}_cv_{}.csv".format(dataset.lower(), b, pattern, clean + direction, cv))