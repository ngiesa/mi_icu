import pandas as pd
import seaborn as sns
import math
from matplotlib import pyplot as plt
from conf_vars import variables, drop_cols, time_vars, var_ranges
from pandas.core.frame import DataFrame

def plot_distr(df, name = "distr_values"):
    # get counts
    no_rows = len(df)
    no_cases = len(df.case_id.drop_duplicates())
    no_seq = len(df[["case_id", "seq_no"]].drop_duplicates())
    # define drop cols
    drop = [x for x in drop_cols if x in list(df.columns)]
    # plot distributions
    f, axs = plt.subplots(len(df.drop(columns=drop).columns), 1, figsize= (10, 35))
    f.suptitle("Distribution for {} datapoints with {} sequences \n for {} cases with EHR sampled in 2h intervals ".format(no_rows, no_seq, no_cases))
    # go though variables and plot on axis
    for i, v in enumerate(sorted(list(df.drop(columns=drop).columns))):
        df_plot = df[v]
        if v == "age":
            df_plot = df.groupby("case_id")["age"].median().reset_index()[v]
        if v == "seq_len":
            df_plot = df.groupby(["case_id", "seq_no"])["seq_len"].median().reset_index()[v]
        sns.histplot(df_plot, kde=True, ax=axs[i], stat="density")
        axs[i].set_title("n = {} ".format(len(df_plot)))
    # store plots
    f.tight_layout(pad=4)
    f.savefig("./{}.png".format(name))


def get_desc_stats(df, name = "desc_values"):
    pd.DataFrame(df[variables].describe()).to_csv("./{}.csv".format(name))

def calculate_missingness(df: DataFrame, identifier: str = "case_id", store_path = ""):
    # get number of nans and general row counts
    df_count_na = df.set_index(identifier)[time_vars].isna().groupby(level=0).sum()
    df_count_all = df.fillna(0).groupby(identifier)[time_vars].count()
    # caluate the missingness per variable and per case
    res = (df_count_na / df_count_all).reset_index()
    if store_path != "":
        res.to_csv(store_path)
    return res
    
def plot_miss_patterns_values(df_dict: dict = None, name: str ="miss_pattern_val", is_in_range = True):
    rows = (math.ceil(len(time_vars)/2))
    f, axs = plt.subplots((rows), 2, figsize=(15, rows*4))
    f.suptitle("Value Distributions for Missingness Patterns")
    for i, v in enumerate(time_vars):
        for k in df_dict.keys():
            vals = pd.DataFrame({k: df_dict[k]["data"][v]})
            if is_in_range:
                vals[k] = [x for x in vals if (x >= var_ranges[v][0]) & (x <= var_ranges[v][1])]
            sns.histplot(vals[k], ax=axs[i%3][int(i/rows)], alpha=0.5, color=df_dict[k]["color"], kde=True, label=k)
        if i == 0:
            axs[i%3][int(i/rows)].legend()
    f.tight_layout(pad=4)
    f.savefig("./{}_range_{}.png".format(name, is_in_range))

def plot_miss_patterns_fractions(df_dict, name="miss_pattern_miss"):
    rows = (math.ceil(len(time_vars)/2))
    for k in df_dict.keys():
        f, axs = plt.subplots((rows), 2, figsize=(15, rows*4))
        f.suptitle("Variable Missingness per Case for Missingness Pattern : {}".format(k.upper()))
        df_dict[k]
        for i, v in enumerate(time_vars):
            sns.histplot(df_dict[k]["data"][v], ax=axs[i%3][int(i/rows)], alpha=0.5, color=df_dict[k]["color"], kde=True, label=k)
            if i == 0:
                axs[i%3][int(i/rows)].legend()
        f.tight_layout(pad=4)
        f.savefig("./{}_pat_{}.png".format(name, k))