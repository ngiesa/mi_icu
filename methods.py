import pandas as pd
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from HOBIT import RegressionForTrigonometric
from statsmodels.imputation.mice import MICEData, MICE
from statsmodels.regression.linear_model import OLS
from static_variables import FEATURES
from sklearn.neighbors import KNeighborsRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

#enable_iterative_imputer()

def apply_locf(induced_df,  training_df, fill="1st", identifier="sequence_id"):
    # perform ffill per sequence group LOCF
    imputed_data_ffill = induced_df.groupby(identifier).ffill()
    if fill=="1st": # still nan in data fill with mean of first index of population 
        imputed_data_ffill[FEATURES] = imputed_data_ffill[FEATURES]\
            .fillna(training_df[training_df.time_index == 0][FEATURES].mean())
    if fill=="pop": # still nan in data fill with mean of whole population 
        imputed_data_ffill[FEATURES] = imputed_data_ffill[FEATURES]\
            .fillna(training_df[FEATURES].mean())
    return imputed_data_ffill.reset_index().assign(sequence_id = induced_df.reset_index().sequence_id)

# applying expanding median / running median
def expanding_median_impute(s: pd.Series):
    """ imputes median from expanding time series """
    return s.fillna(s.expanding().median())

def expanding_mean_impute(s: pd.Series):
    """ imputes median from expanding time series """
    return s.fillna(s.expanding().mean())

def apply_expanding(induced_df, training_df, aggr="mean", identifier="sequence_id"):
    if aggr == "mean":
        df_res =  induced_df.groupby(identifier)[FEATURES]\
                    .apply(lambda s: expanding_mean_impute(s))\
                        .reset_index().assign(sequence_id = induced_df[identifier])
    if aggr == "median":
        df_res =  induced_df.groupby(identifier)[FEATURES]\
                    .apply(lambda s: expanding_median_impute(s))\
                        .reset_index().assign(sequence_id = induced_df[identifier])
                        
    return apply_averages(induced_df=df_res, training_df=training_df, aggr="mean", fill="1st").reset_index()\
        .assign(time_index = induced_df.reset_index().time_index).assign(sequence_id = induced_df.reset_index().sequence_id)    
                        
def apply_averages(induced_df, training_df, aggr="mean", fill="1st"):
    if aggr == "mean":
        if fill == "1st":
            df_res =  induced_df[FEATURES].fillna(training_df[training_df.time_index == 0].mean())
        else:
            df_res = induced_df[FEATURES].fillna(training_df[FEATURES].mean())
    if aggr == "median":
        if fill == "1st":
            df_res =  induced_df[FEATURES].fillna(training_df[training_df.time_index == 0].median())
        else:
            df_res =  induced_df[FEATURES].fillna(training_df[FEATURES].median())
    return df_res.reset_index().assign(sequence_id = induced_df.reset_index().sequence_id)


# Performing MICE tree forrest 
import miceforest as mf

def apply_mice_mf(induced_df):

    kds = mf.KernelDataSet(
            induced_df.drop(columns=["sequence_id", "time_index"]),
            random_state=0)

    df_imp = kds.complete_data()

    df_imp["sequence_id"] = induced_df["sequence_id"]
    df_imp["time_index"] = induced_df["time_index"]
    
    # retrieving data
    return df_imp


def fit_cosine_function(s: pd.Series):
    """ apply cosine fitting """
    trig_reg = RegressionForTrigonometric()
    trig_reg.fit_cos((np.array(list(s.index))), (np.array(list(s))), max_evals=500)
    next_index = max(list(s.index)) + 1
    return trig_reg.predict([next_index])[0]

# define extrapolation methods 

def apply_mice_knn(induced_df, with_static = False):
    """ apply knn mice imputation """
    
    features = FEATURES
    if with_static:
        features = FEATURES + ["age", "gender"]
    
    knn = KNeighborsRegressor()
    imp = IterativeImputer(estimator = knn, max_iter = 1, initial_strategy = 'median', imputation_order='ascending',random_state=42)
    imp.fit(induced_df[features])
    MICE_imputed = imp.transform(induced_df[features])
    MICE_imputed = pd.DataFrame(MICE_imputed)
    MICE_imputed.columns = features
    MICE_imputed["sequence_id"] = induced_df.reset_index()["sequence_id"]
    MICE_imputed["time_index"] = induced_df.reset_index()["time_index"]
    return MICE_imputed
        

def extrapolate_temporal(s, method):
    """ apply extrapolation with past data points """
    s = s.dropna()
    if method in ['linear', 'quadratic']:
        k = ['linear', 'quadratic'].index(method) + 1
        if len(s) < k + 1:
            res = s if type(s) != type(pd.Series([0.001])) else s.iloc[-1]
        else:
            spline = InterpolatedUnivariateSpline((list(s.index)), s, k=k)
            res = spline(x=(max(list(s.index)) + 1))
        if method == "cosine":
            res = fit_cosine_function(s=s)
        return res
    
def apply_extrapolation(s: pd.Series, method='linear'):
    return s.fillna(s.apply((lambda h: h.expanding().apply(lambda l: extrapolate_temporal(s=l, method=method))), axis=0))

# define extrapolation methods 

def extrapolate_temporal(s, method):
    """ apply extrapolation with past data points """
    s = s.dropna()
    if method in ['linear', 'quadratic']:
        k = ['linear', 'quadratic'].index(method) + 1
        if len(s) < k + 1:
            res = s if type(s) != type(pd.Series([0.001])) else s.iloc[-1]
        else:
            spline = InterpolatedUnivariateSpline((list(s.index)), s, k=k)
            res = spline(x=(max(list(s.index)) + 1))
        if method == "cosine":
            res = fit_cosine_function(s=s)
        return res
    
def apply_extrapolation(s: pd.Series, method='linear'):
    return s.fillna(s.apply((lambda h: h.expanding().apply(lambda l: extrapolate_temporal(s=l, method=method))), axis=0))


# apply linear and quadratic interpolation per sequence group but also apply extrapolation also applying the first with mean 

def apply_interpolation(induced_df, training_df, regress = "linear", identifier="sequence_id", type_polation="inter"):

    if type_polation == "inter":
    
        if regress == "linear":
            df_res =  induced_df.groupby(identifier)[FEATURES]\
                .apply(lambda s: s.interpolate(method="linear")).reset_index()
            
        # when not more than 2 non-missing values, fall back to the interpolation of n-1 order
        if regress == "quadratic":
            df_res =  induced_df.groupby(identifier)[FEATURES]\
                .apply(lambda s: s.interpolate(method="quadratic") if len(s.dropna()) > 2 else s.interpolate(method="linear"))\
                .reset_index()
                
        # nearest interpolation
        if regress == "nearest":
            df_res =  induced_df.groupby(identifier)[FEATURES]\
                .apply(lambda s: s.interpolate(method="nearest"))\
                .reset_index()

        # drop level 1 if needed
        if "level_1" in df_res.columns:
            df_res = df_res.drop(columns=["level_1"])
            
        # remaining first missing values with population mean 
        return apply_averages(induced_df=df_res, training_df=training_df, aggr="mean", fill="pop")\
            .assign(time_index = induced_df.reset_index().time_index)
                
    # apply extrapolation
    if type_polation == "extra":
        df_res =  induced_df.groupby(identifier)[FEATURES].apply(lambda s: apply_extrapolation(s, regress)).reset_index() #apply_extrapolation(s, "cosine")
        
        # remaining first missing values with 1st time step means
        return apply_averages(induced_df=df_res.assign(time_index = induced_df.reset_index().time_index), training_df=training_df, aggr="mean", fill="1st")

def apply_mice_lr_stats(induced_df, identifier="sequence_id", with_static = False):

    induced_df.loc[:, FEATURES] = induced_df[FEATURES].fillna(induced_df[FEATURES].mean())
    
    imp = MICEData(induced_df)
    fml = 'rr ~ hr + bp_sys + bp_dia + spo2'
    if with_static:
        fml = 'rr ~ hr + bp_sys + bp_dia + spo2 + age + gender'
    mice = MICE(fml, OLS, imp)
    results = []
    for k in range(10):
        x = mice.next_sample()
        results.append(x)
            
    return imp.next_sample().reset_index().assign(sequence_id = induced_df.reset_index().sequence_id)


def apply_mice_temporal(induced_df, identifier="sequence_id", with_static = False):
    
    # iterate through all sequence lengths and limit feature space to previous values only 
    max_ts = induced_df.groupby(identifier)["time_index"].count().max()
    
    induced_df_orig = induced_df.copy()
    
    print("Maximum time index ", max_ts)
    
    imputed_sets = []
    
    for ts in range(1, max_ts+1):
        
        induced_df_temp = induced_df_orig[(induced_df_orig.time_index < ts)]

        kds = mf.KernelDataSet(
                induced_df_temp.drop(columns=["sequence_id", "time_index"]),
                random_state=0)

        df_imp = kds.complete_data()

        df_imp["sequence_id"] = induced_df_temp["sequence_id"]
        df_imp["time_index"] = induced_df_temp["time_index"]
        
        imputed_sets.append(df_imp[df_imp.time_index == (ts-1)])
    
    imp_all = pd.concat(imputed_sets).sort_values(["sequence_id", "time_index"])
    
    assert len(imp_all) == len(induced_df_orig)
        
    return imp_all