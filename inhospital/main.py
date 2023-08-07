
import json
from data_load import load_raw_data, load_miss_values, load_miss_fractions
from desc_stats import calc_sum_columns
from miss_inducer import induce_mar, induce_mnar, induce_mcar
from conf_vars import time_vars, store_paths_values, imp_methods, error_functions
from eval_metrics import calc_error
from functools import reduce
from desc_stats import plot_miss_patterns_values, plot_miss_patterns_fractions, calculate_missingness, mean_confidence_interval
import pandas as pd
# defining main method

# usage of defined functions 
def induce(store: bool = False):
    df = load_raw_data()
    print("raw data loaded")
    df_mcar = induce_mcar(df = df, time_colums= time_vars, static_columns=["case_id","timestamp", "seq_no", "age"], store_path= store_paths_values["mcar"] if store else "")
    print("mcar induced")
    df_mar = induce_mar(df=df, cond_var = "age", miss_range = [0.1, 0.9], 
                identifier = "case_id", static_columns = ["case_id", "timestamp", "seq_no", "age"], time_colums=time_vars, store_path= store_paths_values["mar"] if store else "")
    print("mar induced")
    df_mnar = induce_mnar(df=df, static_columns=["case_id", "timestamp", "seq_no", "age"], time_colums=time_vars, miss_range=[0.1, 0.9], store_path= store_paths_values["mnar"] if store else "")
    print("mnar induced")
    return {
        "complete": df,
        "mcar": df_mcar,
        "mar": df_mar,
        "mnar": df_mnar
    }

def missing():
    ## calculating missingness and storing result
    pat_data = load_miss_values()
    _ = calculate_missingness(df=pat_data["mcar"], store_path="/home/giesan/data/mcar_miss.csv")
    _ = calculate_missingness(df=pat_data["mar"], store_path="/home/giesan/data/mar_miss.csv")
    _ = calculate_missingness(df=pat_data["mnar"], store_path="/home/giesan/data/mnar_miss.csv")

def plot():
    plot_miss_patterns_values(df_dict=load_miss_values(), is_in_range=True)
    plot_miss_patterns_values(df_dict=load_miss_values(), is_in_range=False)
    plot_miss_patterns_fractions(df_dict=load_miss_fractions())

def confidence(n_ci: int = 3):
    res_dfs = []
    # induce missingness patterns n_ci times
    ci_patterns = [induce() for i in range(0, n_ci)]
    # apply imputation methods
    res_dict, j = {}, 0
    for miss_pat in ci_patterns[0].keys():
        if miss_pat == "complete":
            # skip complete data
            continue
        for imp_m in imp_methods.keys():
            # impute ci data
            imp_ci = [imp_methods[imp_m](df_in = dic[miss_pat], time_vars = time_vars) for dic in ci_patterns]
            res_dict[j] = {
                    "missing_pattern": miss_pat,
                    "imputation_method": imp_m
                }
            # calculate error
            err_dict = {}
            for metric in error_functions.keys():
                err_dict[metric] = [calc_error(df_gt=ci_patterns[i]["complete"],
                                                  df_imp = imp_ci[i],
                                                  metric=error_functions[metric],
                                                  df_miss=ci_patterns[i][miss_pat],
                                                  time_vars=time_vars) 
                                                  for i in range(0, n_ci)]
                # create dataframe and append to res df
                df = pd.concat([pd.DataFrame(err_dict[metric][x]).set_index("variable").T for x in range(0, n_ci)])
                df[["missing_pattern", "imputation_method", "metric", "exp"]] = miss_pat, imp_m, metric, j
                res_dfs.append(df)
            res_dict[j]["metrics"] = err_dict
            j = j + 1
    # write results to dict and save to local
    with open("./res_ci_imp.json", "w") as f:
        json.dump(res_dict, f)
    # store res dfs
    res_df = pd.concat(res_dfs)
    gr_cols = ["missing_pattern", "imputation_method", "metric"]
    res_df = calc_sum_columns(res_df, gr_cols=gr_cols, val_cols=time_vars)
    res_df.to_csv("./res_ci_raw_data.csv")
    dfs = [res_df.groupby(gr_cols)[var]\
    .apply(lambda x: mean_confidence_interval(x)).reset_index() for var in time_vars + ["sum_error"]]
    reduce(lambda df1, df2: df1.merge(df2, on=["missing_pattern", "imputation_method", "metric"] ,how="inner"), dfs)\
        .to_csv("./res_ci_error_95_ci.csv")


if __name__ == '__main__':
    #plot_distr(df)
    #get_desc_stats(df)
    confidence(5)
    #induce()
    #missing()
    #plot()