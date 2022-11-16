from conf_vars import time_vars
from sklearn.utils import shuffle
from functools import reduce
import numpy as np
import pandas as pd
from data_load import load_data
from desc_stats import plot_miss_patterns

# cutting sequences on same length if needed
def cut_seq_lengths(df=None, time_steps=24):
    return df.groupby(['case_id', 'seq_no']).head(time_steps).reset_index(drop=True)

# drop random values from df to simulate MCAR missingness is evenly distributed, can be different
def induce_mcar(df = None, miss_fraction = 0.1):
    dfs = []
    for v in time_vars:
        dfs.append(shuffle(df[["case_id","timestamp", "seq_no", "age", v]]).sample(frac = 1 - miss_fraction))
    df_join = reduce(lambda df1, df2: df1.merge(df2, on=["case_id", "timestamp", "seq_no", "age"], how = "outer"), dfs).drop_duplicates()
    return df[["case_id","timestamp", "seq_no", "age"]].merge(df_join, on=["case_id", "timestamp", "seq_no", "age"], how = "left")

# drop random values conditionally on one dependent variable per case
def induce_mar(df = None, cond_var = "age", s_range = [0.1, 0.9]):
    df_gr = df.groupby("case_id")[cond_var].median().reset_index()
    # asssign missing rate to cases normalized with norm distr added
    miss_fractions =  df_gr[cond_var]
    # normalization between [s_min : s_max]
    s_min, s_max = s_range[0], s_range[1]
    miss_fractions = (miss_fractions - min(miss_fractions))/ \
        (max(miss_fractions) - min(miss_fractions)) * (s_max - s_min) + s_min
    df_gr = df_gr.assign(miss_fraction = miss_fractions)
    # merge missing rates to actual values 
    df = df.merge(df_gr[["case_id", "miss_fraction"]], on=["case_id"])
    # going through variables and cases on conditioning on dependent variable
    df_vars = []
    for time_v in time_vars:
        dfs_cases = []
        for i, row in df_gr.iterrows():
            miss_fraction = row["miss_fraction"]
            dfs_cases.append(df[(df.case_id == row["case_id"])] \
                [["case_id","timestamp","seq_no","age", time_v]].sample(frac = 1 - miss_fraction))
        df_vars.append(pd.concat(dfs_cases))
    # outerjoining results containing missingness per variable to one dataframe
    df_join = reduce(lambda df1, df2: df1.merge(df2, on=["case_id", "timestamp", "seq_no", "age"], how = "outer"), df_vars).drop_duplicates()
    return df[["case_id","timestamp", "seq_no", "age"]].merge(df_join, on=["case_id", "timestamp", "seq_no", "age"], how = "left")

def induce_and_plot():
    df = load_data()
    df_mcar = induce_mcar(df)
    df_mar = induce_mar(df)
    plot_miss_patterns(df=df, df_mar=df_mar, df_mcar=df_mcar)

