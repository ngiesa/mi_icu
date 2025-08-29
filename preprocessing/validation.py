import pandas as pd
import numpy as np
import os
import scipy.stats as st
from scipy.stats import wasserstein_distance
from sklearn.metrics import mean_squared_error, mean_absolute_error
from static_variables import FEATURES, DATASETS
from functools import reduce
from statsmodels.tsa.stattools import acf, ccf

import warnings
warnings.filterwarnings("ignore")

def calc_cross_corr_div(ts):
    
    """ calculating cross correlation divergence """
    
    ts = np.array(ts[FEATURES])
    F = len(FEATURES)
    result = np.zeros((F, F, ts.shape[0]))
    # cross-correlation divergence 
    for i in range(F):
        for j in range(F):
            t1, t2 = ts[:, i], ts[:, j]
            tcff = ccf(t1, t2)
            result[i, j, :] = tcff
    return result

def return_errors(U_real, W_imputed, filter):
    
    #import warnings
    #warnings.filterwarnings("error")    
    
    """ calculate error metrics """
    
    #norm_factor for scaling the error terms except the WSD
    norm_factor_errors = np.mean(U_real)
    norm_factor_dist = (max(list(U_real))- min(list(U_real)))
    
    U_real_filter = (filter * U_real).dropna().reset_index(drop=True)
    W_imputed_filter = (filter * W_imputed).dropna().reset_index(drop=True)

    #if len(U_real_filter) != len(W_imputed_filter):
    #    print(U_real_filter)
    #    print(W_imputed_filter)
    

    norm_factor_filter_errors = 0
    
    if not U_real_filter.empty | W_imputed_filter.empty :
        norm_factor_filter_errors = np.mean(U_real_filter)
        norm_factor_filter_dist = (max(list(U_real_filter))- min(list(U_real_filter)))
    
    NRMSE_filter, NMAE_filter, NWSD_filter = float("NaN"), float("NaN"), float("NaN")

    # if both series are equal, metrics are 0
    if (norm_factor_dist == 0) | (norm_factor_errors == 0):
        
        #print(W_imputed)
        #print(U_real)
        
        NMAE, NWSD, NRMSE = 0, 0, 0
    else:
        NMAE = mean_absolute_error(U_real, W_imputed) / norm_factor_errors
        NWSD = wasserstein_distance(U_real, W_imputed) / norm_factor_dist
        NRMSE = np.sqrt(mean_squared_error(U_real, W_imputed)) / norm_factor_errors
        
        if norm_factor_filter_errors != 0:
            
            if len(U_real_filter) == len(W_imputed_filter): #TODO failed at hirid??
        
                NRMSE_filter = np.sqrt(mean_squared_error(U_real_filter, W_imputed_filter)) / norm_factor_filter_errors
                NMAE_filter = mean_absolute_error(U_real_filter, W_imputed_filter) / norm_factor_filter_errors
                NWSD_filter = wasserstein_distance(U_real_filter, W_imputed_filter) / norm_factor_filter_dist
    
    #if NWSD == 0:
     #   if set(U_real) != set(W_imputed):
            
           # print("") # TODO think about the problem if just the real is 0
            #print(U_real)
            #print("imputed")
            #print(W_imputed)
            
            #print(norm_factor_dist)
            #print(norm_factor_errors)
    
        #assert set(U_real) == set(W_imputed) #TODO check MAYBE RETURN NaN??? if dist are equal return null 
        
    #print(U_real)
    #print(W_imputed)
    # calculate autocorrelation divergence
    
    #try: TODO think of auto and cross correlation??? 
    #    
    ## autocorrelation divergence just for one lag? mean across all lags? for lag = 1 [1]
    acf_real = [] if np.std(np.array(U_real)) == 0 else acf(np.array(U_real))
    acf_imp = [] if np.std(np.array(W_imputed)) == 0 else acf(np.array(W_imputed))
    # getting the autocorrelation divergence
    AUTCD, AUTWSD = float("NaN"), float("NaN")
    if (acf_real != []) & (acf_imp != []):
        AUTCD = mean_absolute_error(acf_real, acf_imp)
        # also get the WSD of autocorrelation coefficients 
        norm_autwsd = (max(list(acf_real))- min(list(acf_imp)))
        if norm_autwsd != 0:
            AUTWSD = wasserstein_distance(acf_real, acf_imp)/ norm_autwsd
    
    #except RuntimeWarning as e:
    #    print(e)
    #    #print(U_real)
    #    print(W_imputed)
    #    print(np.std(np.array(W_imputed)))
        
    metrics_dict = {
        "NWSD": NWSD,
        "NRMSE": NRMSE,
        "NMAE": NMAE,
        "NRMSE_filter": NRMSE_filter,
        "NMAE_filter": NMAE_filter,
        "NWSD_filter": NWSD_filter,
        "RMSE": np.sqrt(mean_squared_error(U_real, W_imputed)),
        "ACORD" : AUTCD,
        "ACORWSD": AUTWSD
        }

    return metrics_dict

def validate_all_imputation_methods(clean="pres", pattern="mcar", dataset = "mimic", method_select = None, boot=0):
    
    """ validate the imputation methods per single patient / sequence anf over the whole population """
    
    main_res_seq = []
    main_res_pop = []
    
    # go through datasets and read complete data and methods 
    complete_data = pd.read_csv("./complete_sequences/sequence_{}_{}.csv".format(dataset.lower(), clean), index_col = 0) #TODO LOAD COMPLETE WITH METHOD!!!
    complete_data = complete_data.assign(time_index = complete_data.groupby("sequence_id")["hr"].cumcount()).sort_values(by=["sequence_id", "time_index"])
    files_dataset = os.listdir("./imputed_sequences/{}/".format(dataset.lower()))
    files_dataset = [f for f in files_dataset if ("_{}_".format(boot) in f) and (clean in f) and (pattern in f)]
    methods = [x.split("_")[0] for x in files_dataset]
    # restrict on the bootstrap only
    
    print(methods)
    
    # opening missing pattern for errors on imputed only or all
    induced_data = pd.read_csv("./induced_sequences/{}/missing_matrix_{}_boot_{}_{}_{}.csv"\
        .format(dataset.lower(),dataset.lower(), boot, pattern, clean), index_col = 0)
    
    print("validating results for ", dataset)
    
    for m, method in enumerate(methods): #TODO maybe implement iterarive loading
        
        print("calc errors for ", method)
        
        if method_select:
            # select the according method
            if method != method_select:
                continue
            
        files_method = [f for f in files_dataset if (method == f.split("_")[0]) and (pattern in f)]
        
        for i, file in enumerate(files_method):
            
            print("{}/{}".format(i, len(methods)))
            
            print("./imputed_sequences/{}/".format(dataset.lower()) + file)
            
            imputed_data = pd.read_csv("./imputed_sequences/{}/".format(dataset.lower()) + file, index_col = 0, on_bad_lines="skip")
            if not "time_index" in list(imputed_data.columns):
                imputed_data = imputed_data.assign(time_index = imputed_data.groupby("sequence_id")["hr"].cumcount())
            imputed_data=imputed_data.sort_values(by=["sequence_id", "time_index"])
            
            print(len(imputed_data))
            print(len(complete_data))
            
            #assert len(imputed_data) == len(complete_data) TODO STGAE MIMIC!!
            #if len(imputed_data) != len(complete_data)
            
            print("calc population error ")
            
            # TODO apply induced data only
            
            # calculate across the whole population
            for f, feat in enumerate(FEATURES):
                U_real, W_imputed = complete_data[feat].reset_index(drop=True), imputed_data[feat].reset_index(drop=True)
                
                # get filter for sequences
                filter = induced_data[feat].replace(0, float("NaN")).reset_index(drop=True)
                df_res = pd.DataFrame(return_errors(U_real=U_real, W_imputed=W_imputed, filter = filter), index=[method])
                main_res_pop.append(df_res.assign(feature = feat).assign(boot=i).assign(method=method).assign(dataset = dataset))
            
            seqs = []
            # also iterate through sequences
            for s, seq in enumerate(list(imputed_data.sequence_id.unique())):
                if s % 1000 == 0:
                    print("sequence ", s)
                    
                    #TODO debugging for first 1000 sequences
                    if s == 4000: #TODO
                        break
                
                complete_data_seq = complete_data[complete_data.sequence_id == seq]
                imputed_data_seq = imputed_data[imputed_data.sequence_id == seq]
                induced_data_seq = induced_data[induced_data.sequence_id == seq]
                
                # TODO get cross-corr divergence
                ccd_compl = calc_cross_corr_div(complete_data_seq)
                ccd_imp = calc_cross_corr_div(imputed_data_seq)
                
                ccd_compl = list(ccd_compl[:,:,0].reshape(-1))
                ccd_imp = list(ccd_imp[:,:,0].reshape(-1))
                
                mask_compl = np.isnan(np.array(ccd_compl)).astype(int)
                mask_imp = np.isnan(np.array(ccd_imp)).astype(int)
                mask = [abs(1-np.max([x,y])) for x,y in zip(mask_compl, mask_imp)]
                
                #print(np.multiply(mask,ccd_compl))
                
                ccd_div = mean_absolute_error(
                    np.multiply(mask,pd.Series(ccd_compl).fillna(0)),
                    np.multiply(mask,pd.Series(ccd_imp).fillna(0))
                )
                
                for f, feat in enumerate(FEATURES):
                                        
                    filter = induced_data_seq[feat].replace(0, float("NaN")).reset_index(drop=True)
                    
                    assert len(complete_data_seq[feat]) == len(imputed_data_seq[feat])
                        
                    if len(complete_data_seq[feat]) != len(imputed_data_seq[feat]):
                        print("No same length!!")
                    
                    errors = return_errors(U_real=complete_data_seq[feat], 
                                        W_imputed=imputed_data_seq[feat],
                                        filter = filter
                                        )
                    errors["CCORD"] = ccd_div
                    df_res = pd.DataFrame(errors, index=[method]).assign(feature = feat).assign(sequence_id = seq)
                    seqs.append(df_res)
            
            main_res_seq.append(pd.concat(seqs)\
                    .assign(boot=i)\
                        .assign(method=method)\
                                .assign(dataset = dataset))
        
        df_res_pop = pd.concat(main_res_pop)
        df_res_seq = pd.concat(main_res_seq)
        
        if method_select: #TODO iteration per dataset, pattern, boot, clean 
            # incremental loading
            df_res_seq_exist = pd.read_csv("./errors/seq_errors_{}_{}_{}_boot_{}.csv".format(dataset.lower(), pattern, clean, boot), index_col = 0)
            df_res_pop_exist = pd.read_csv("./errors/pop_errors_{}_{}_{}_boot_{}.csv".format(dataset.lower(), pattern, clean, boot), index_col = 0)
            df_res_seq = pd.concat([df_res_seq, df_res_seq_exist[df_res_seq_exist.method != method_select]])
            df_res_pop = pd.concat([df_res_pop, df_res_pop_exist[df_res_pop_exist.method != method_select]])
            df_res_seq.to_csv("./errors/seq_errors_{}_{}_{}_boot_{}.csv".format(dataset.lower(), pattern, clean, boot)) 
            df_res_pop.to_csv("./errors/pop_errors_{}_{}_{}_boot_{}.csv".format(dataset.lower(), pattern, clean, boot))  
        else:
            df_res_seq.to_csv("./errors/seq_errors_{}_{}_{}_boot_{}.csv".format(dataset.lower(), pattern, clean, boot)) 
            df_res_pop.to_csv("./errors/pop_errors_{}_{}_{}_boot_{}.csv".format(dataset.lower(), pattern, clean, boot))   
    
    ci_type = ["seq", "pop"]
    
    for d, df_res_ci in enumerate([df_res_seq, df_res_pop]):
        selected_metrics = ["NWSD", "NRMSE", "NMAE", "RMSE", "NRMSE_filter", "NMAE_filter"]
        # aggregating CIs for subjects and whole population across bootstrapps
        if ci_type[d] == "seq":
            selected_metrics += ["ACORD", "CCORD", "ACORWSD"]
        ci_aggrs = []
        for metric in selected_metrics:
            ci_mean = df_res_ci.groupby(["method", "feature", "dataset"])[metric].mean().reset_index().rename(columns={metric: metric+ "_ci_mean"})
            ci_low = df_res_ci.groupby(["method", "feature", "dataset"])[metric]\
                .apply(lambda a: (st.t.interval(0.95, len(a)-1, loc=np.mean(a), scale=st.sem(a)))[0]).reset_index().rename(columns={metric: metric + "_ci_low"})
            ci_high = df_res_ci.groupby(["method", "feature", "dataset"])[metric]\
                .apply(lambda a: (st.t.interval(0.95, len(a)-1, loc=np.mean(a), scale=st.sem(a)))[1]).reset_index().rename(columns={metric: metric + "_ci_high"})
                
            ci_aggrs.append(ci_mean)
            ci_aggrs.append(ci_low)
            ci_aggrs.append(ci_high)
        
        df_ci_res = reduce(lambda x, y: x.merge(y, on=["method", "feature", "dataset"]), ci_aggrs)
        df_ci_res.to_csv("./errors/ci_errors_{}_{}_{}_{}_boot_{}.csv".format(dataset.lower(), ci_type[d], pattern, clean, boot)) 
        print("results stored") #TODO iteration per dataset, pattern, boot, clean etc.