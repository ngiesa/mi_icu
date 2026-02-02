import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf, ccf
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import wasserstein_distance
import os

import statsmodels.stats.api as sms
from static_variables import FEATURES

# config parameters
clean="pres"
pattern="mcar"
dataset = "icdep"
method_select = None
boot=0
mode = "" # "", baseline

direction = "" # or "", "rev"
    
""" validate the imputation methods per single patient / sequence and over the whole population """


def root_mean_squared_error(a, b):
    """ personalized RMSE """
    return np.sqrt(mean_squared_error(a, b))

def wsd_norm(s):
    """ normalization of WSD """
    return (max(list(s))- min(list(s)))

def calc_cross_corr(ts):
    
    """ calculating cross correlation for multidim. time series """
    
    ts = np.array(ts[FEATURES])
    F = len(FEATURES)
    result = np.zeros((F, F, ts.shape[0]))
    # cross-correlation  
    for i in range(F):
        for j in range(F):
            t1, t2 = ts[:, i], ts[:, j]
            tcff = ccf(t1, t2)
            result[i, j, :] = tcff
    return result


def get_autocorrelation_and_wsd(series_complete, series_imputed):
    
    """ calculating autocorrelation and corresponding ACor divergence """
    
    acf_real = [] if np.std(np.array(series_complete)) == 0 else acf(np.array(series_complete))
    acf_imp = [] if np.std(np.array(series_imputed)) == 0 else acf(np.array(series_imputed))
    
    AUTCD, AUTWSD = float("NaN"), float("NaN")
    if (len(acf_real) != 0) & (len(acf_imp) != 0):
        AUTCD = mean_absolute_error(acf_real, acf_imp)
        # also get the WSD of autocorrelation coefficients 
        norm_autwsd = (max(list(acf_real))- min(list(acf_real)))
        if norm_autwsd != 0:
            AUTWSD = wasserstein_distance(acf_real, acf_imp)/ norm_autwsd
    return (AUTCD, AUTWSD)


def get_cross_corr_divergence(series_complete, series_imputed):
    
    """ calculation of cross correlation divergence per sequence """
    
    # calculation of cross- and autocorrelation divergence
    ccd_compl = calc_cross_corr(series_complete)
    ccd_imp = calc_cross_corr(series_imputed)
    
    # reshaping coefficients to flat dimension
    ccd_compl = list(ccd_compl[:,:,0].reshape(-1))
    ccd_imp = list(ccd_imp[:,:,0].reshape(-1))
    # getting inf or NaN mask
    mask_compl = np.isnan(np.array(ccd_compl)).astype(int)
    mask_imp = np.isnan(np.array(ccd_imp)).astype(int)
    mask = [abs(1-np.max([x,y])) for x,y in zip(mask_compl, mask_imp)]
    
    # calculating divergence
    return mean_absolute_error(
            np.multiply(mask,pd.Series(ccd_compl).fillna(0)),
            np.multiply(mask,pd.Series(ccd_imp).fillna(0))
        )
    
def apply_metric_locally(metric, df_cmpl_not_na, df_imp_not_na, df_ind_not_na, norm, f, normalize):
    """ apply any metric with norm to all missing values per sequence """
    return [metric(
                    #pd.Series(np.array(c[1][f]) * np.array(i[1][f].replace(0, float("NaN")))).dropna(), 
                    #pd.Series(np.array(p[1][f]) * np.array(i[1][f].replace(0, float("NaN")))).dropna(), 
                    pd.Series(np.array(c[1][f])), 
                    pd.Series(np.array(p[1][f])),
                    ) / (norm(c[1][f]) if normalize else 1)
                    for c, p, i \
                    in zip(
                        df_cmpl_not_na.groupby("sequence_id"), 
                        df_imp_not_na.groupby("sequence_id"),
                        df_ind_not_na.groupby("sequence_id"),
                        )]
    
def benchmark(
    clean="pres",
    pattern="mcar",
    dataset = "icdep",
    is_stae_baselines = False,
    mode = "",
    normalize = True
    ):
    
    # go through datasets and read complete data and methods 
    complete_data = pd.read_csv("./complete_sequences/sequence_{}_{}.csv".format(dataset.lower(), clean), index_col = 0) 
    complete_data = complete_data.assign(time_index = complete_data.groupby("sequence_id")["hr"].cumcount()).sort_values(by=["sequence_id", "time_index"])
    FILES_DATASET = os.listdir("./imputed_sequences/{}/".format(dataset.lower()))
    if is_stae_baselines:
        FILES_DATASET = os.listdir("../stae_baselines/{}/".format(dataset.lower()))
    
    methods = [x.split("_")[0] for x in FILES_DATASET]

    # opening missing pattern for errors on imputed only or all
    induced_data = pd.read_csv("./induced_sequences/{}/missing_matrix_{}_boot_{}_{}_{}.csv"\
        .format(dataset.lower(),dataset.lower(), boot, pattern, clean), index_col = 0)

    print("validating results for ", dataset)
    
    """ applying define constant variables for retrieving benchmarking statistcs """  

    cv_splits = pd.read_csv("/home/giesan/mi_icu/spatio_temporal_pattern/complete_sequences/cv_splits_all.csv", index_col=0)
    cv_splits = cv_splits[(cv_splits["DATASET"] == dataset.upper()) & ((cv_splits["CLEAN"] == clean.lower()))]

    main_res_local = []
    main_res_global = []
    
    files_dataset = [f for f in FILES_DATASET if ("_{}_".format(boot) in f) and (clean in f) and (mode in f) and (pattern in f)]
    
    print(files_dataset)

    for CV in range(0, 3): 
        
        print("cross validation fold: ", CV)
        
        cv_seq_ids = cv_splits[cv_splits["CV"] == CV]
        
        df_test_induced = induced_data[induced_data.sequence_id.isin(list(cv_seq_ids[cv_seq_ids["SET"] == "TEST"].sequence_id))]
        df_test_complete = complete_data[complete_data.sequence_id.isin(list(cv_seq_ids[cv_seq_ids["SET"] == "TEST"].sequence_id))]
        
        df_na_seq = df_test_induced.groupby(["sequence_id"])[FEATURES].sum().reset_index()
        df_na_seq.set_index("sequence_id")
            
        for method in list(set(methods)):
            
            print(method)
            
            # TODO
            if method not in ["lstae", "1locf", "linpol", "bilstaestat"]: #["BRITS", "SAITS", "stgae"]: #"CTA"
                continue
            
            method_path = [f for f in files_dataset if (("cv_{}".format(CV) in f) and (f.split("_")[0] == method))]
            if mode == "":
                method_path = [m for m in method_path if not "baseline" in m]
            
            if len(method_path) == 0:
                print("NO METHODS FOUND")
                continue
            
            print("open : ", method_path[0])
            
            if is_stae_baselines:
                print("/home/giesan/mi_icu/stae_baselines/{}/{}".format(dataset, method_path[0]))
                df_imputed = pd.read_csv("/home/giesan/mi_icu/stae_baselines/{}/{}".format(dataset, method_path[0]), index_col=0)
            else:
                df_imputed = pd.read_csv("/home/giesan/mi_icu/spatio_temporal_pattern/imputed_sequences/{}/{}".format(dataset, method_path[0]), index_col=0)
            df_imputed = df_imputed.assign(time_index = df_imputed.groupby("sequence_id")["hr"].cumcount())

            #print(df_imputed)
            
            #print(df_test_complete)
            
            print(len(df_imputed))
            print(len(df_test_complete))

            assert len(df_imputed) == len(df_test_complete)
            
            #ensure same sizes
            if len(df_imputed) != len(df_test_complete):
                print("CV length adaption")
                df_imputed = df_imputed.merge(df_test_complete[["sequence_id", "time_index"]], on=["sequence_id", "time_index"])
                df_test_complete = df_test_complete.merge(df_imputed[["sequence_id", "time_index"]], on=["sequence_id", "time_index"])
                    
            # calculating cross correlation divergence
            cc_div = [get_cross_corr_divergence(c[1], p[1]) for c, p in zip(df_test_complete.groupby("sequence_id"), df_imputed.groupby("sequence_id"))]
            
            main_res_local.append(pd.DataFrame({
                    "feature": "all",
                    "CV": CV,
                    "method": method,
                    "metric": "CCorD_loc",
                    "value": cc_div,
                    "sequence_id": df_test_complete.sequence_id.unique()
                }))
                
            for f in FEATURES:
                
                # filter sequences with NaS for local na metrics
                seq_na_feat = df_na_seq[df_na_seq[f] > 0]
                
                print("remaining missings ", seq_na_feat)
                
                # get feature series 
                series_imp = df_imputed[f].reset_index(drop=True)
                series_cmpl = df_test_complete[f].reset_index(drop=True)
                series_ind = df_test_induced[f].reset_index(drop=True)
                
                assert len(series_imp) == len(series_cmpl)
                assert len(series_cmpl) == len(series_ind)
                
                # drop all available CCD values
                series_imp_na = pd.Series(np.array(series_imp) * np.array(series_ind.replace(0, float("NaN")))).dropna()
                series_cmpl_na = pd.Series(np.array(series_cmpl) * np.array(series_ind.replace(0, float("NaN")))).dropna()
                
                # calculation of global metrics the error terms are calculated across missing values only
                #nrmse_all = root_mean_squared_error(series_cmpl, series_ind) / np.mean(series_cmpl)
                if normalize:
                    nrmse_na = root_mean_squared_error(series_cmpl_na, series_imp_na) / np.mean(series_cmpl_na)
                    nmae_na = mean_absolute_error(series_cmpl_na, series_imp_na) / np.mean(series_cmpl_na)
                    nwsd_all = wasserstein_distance(series_cmpl, series_imp) / wsd_norm(series_cmpl)
                else:
                    nrmse_na = root_mean_squared_error(series_cmpl_na, series_imp_na)
                    nmae_na = mean_absolute_error(series_cmpl_na, series_imp_na)
                    nwsd_all = wasserstein_distance(series_cmpl, series_imp)
                    
                
                # appending global metrics
                main_res_global.append(pd.DataFrame({"feature": f,
                                            "CV": CV,
                                            "method": method,
                                            "metric": ["NRMSE_glob", "NMAE_glob", "NWSD_glob"], 
                                            "value": [nrmse_na, nmae_na, nwsd_all],
                                            }))
                
                # calculate individual errors per sequence and feature for all missings only
                df_cmpl_not_na = df_test_complete.merge(seq_na_feat[["sequence_id"]], on="sequence_id")
                df_ind_not_na = df_test_induced.merge(seq_na_feat[["sequence_id"]], on="sequence_id")
                df_imp_not_na = df_imputed.merge(seq_na_feat[["sequence_id"]], on="sequence_id") 
                
                # apply all metrics locally and get confidence intervals to 
                nrmse_loc = apply_metric_locally(root_mean_squared_error, df_cmpl_not_na, df_imp_not_na, df_ind_not_na, np.mean, f, normalize) 
                nmae_loc = apply_metric_locally(mean_absolute_error, df_cmpl_not_na, df_imp_not_na, df_ind_not_na, np.mean, f, normalize)
                nwsd_loc = apply_metric_locally(wasserstein_distance, df_cmpl_not_na, df_imp_not_na, df_ind_not_na, wsd_norm, f, normalize)
                
                assert len(nrmse_loc) == len(nmae_loc)
                assert len(nmae_loc) == len(nwsd_loc)
                
                # get autocorrelation divergence and autocorrelation WSD
                auto_corr = [get_autocorrelation_and_wsd(
                        pd.Series(np.array(c[1][f])), 
                        pd.Series(np.array(p[1][f])), 
                        )
                        for c, p \
                        in zip(
                            df_cmpl_not_na.groupby("sequence_id"), 
                            df_imp_not_na.groupby("sequence_id"))
                        ]
                
                assert len(auto_corr) == len(nwsd_loc)
                
                # append results to res df
                main_res_local.append(pd.DataFrame({
                    "feature": f,
                    "CV": CV,
                    "method": method,
                    "metric": ["NRMSE_loc"] * len(nrmse_loc) + ["NMAE_loc"] * len(nmae_loc) \
                        + ["NWSD_loc"] * len(nwsd_loc) + ["ACorD_loc"] * len(auto_corr) + ["ACorWSD_loc"] * len(auto_corr),
                    "value": nrmse_loc + nmae_loc + nwsd_loc + [a[0] for a in auto_corr] + [a[1] for a in auto_corr],
                    "sequence_id": list(df_cmpl_not_na.sequence_id.unique()) * 5
                })
        )

    # getting local and global metrics and discard where not defined
    seq_errors = pd.concat(main_res_local).replace(np.inf, np.nan).dropna()
    pop_errors = pd.concat(main_res_global)
    

    def aggr_ci_func(a):
        return "{:.3f}±{:.3f}".format(
            round(np.mean(a) * 100, 3),
            round((sms.DescrStatsW(a).tconfint_mean()[1] - np.mean(a)) * 100, 3)
        )
        
    df_display_seq = seq_errors.groupby(['method','metric'])["value"].apply(lambda a: aggr_ci_func(a)).unstack()
    df_display_seq["rank"] = seq_errors.groupby(['method','metric'])["value"].mean().unstack().rank().mean(axis=1).rank() 
    
    df_display_pop = pop_errors.groupby(['method','metric'])["value"].apply(lambda a: aggr_ci_func(a)).unstack()
    df_display_pop["rank"] = pop_errors.groupby(['method','metric'])["value"].mean().unstack().rank().mean(axis=1).rank()
    
    df_display_pop.to_csv("/home/giesan/mi_icu/spatio_temporal_pattern/errors/patterns/pop_errors_{}_{}_{}_boot_{}_display_patterns_tmp.csv".format(dataset, pattern, clean + direction, boot))
    df_display_seq.to_csv("/home/giesan/mi_icu/spatio_temporal_pattern/errors/patterns/seq_errors_{}_{}_{}_boot_{}_display_patterns_tmp.csv".format(dataset, pattern, clean + direction, boot))
    
    # write out the benchmarking results
    pop_errors.to_csv("/home/giesan/mi_icu/spatio_temporal_pattern/errors/patterns/pop_errors_{}_{}_{}_boot_{}_{}_tmp_28_01_2026.csv".format(dataset, pattern, clean + direction, boot, mode))
    seq_errors.to_csv("/home/giesan/mi_icu/spatio_temporal_pattern/errors/patterns/seq_errors_{}_{}_{}_boot_{}_{}_tmp_28_01_2026.csv".format(dataset, pattern, clean + direction, boot, mode))
    
    df_store = df_display_pop.merge(df_display_seq, suffixes=["_loc", "_glob"], on="method")
    print(df_store.head())
    df_store[["ACorD_loc","ACorWSD_loc","CCorD_loc","NMAE_loc","NRMSE_loc","NWSD_loc","NMAE_glob","NRMSE_glob","NWSD_glob"]]\
        .to_csv("/home/giesan/mi_icu/spatio_temporal_pattern/errors/patterns/errors_{}_{}_{}_boot_{}_display_patterns_{}_28_01_2026.csv".format(dataset, pattern, clean + direction, boot, mode))