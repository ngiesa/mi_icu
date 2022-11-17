from conf_vars import time_vars
from sklearn.utils import shuffle
from functools import reduce
import numpy as np
import pandas as pd
from data_load import load_data
from desc_stats import plot_miss_patterns
from pandas.core.frame import DataFrame

# cutting sequences on same length if needed
def cut_seq_lengths(df: DataFrame = None, time_steps: int = 24, gr_columns: list = ['case_id', 'seq_no']):
    '''
    Selects the first n (time_steps) rows within a group 
            Parameters:
                    df (DataFrame): A time variant df with grouping and value columns
                    time_steps (int): First n rows which are selected within groups
                    gr_columns (list): Column names for group by condition

            Returns:
                    df (DataFrame): Df with first n (time_steps) per group
    '''
    return df.groupby([gr_columns]).head(time_steps).reset_index(drop=True)

# drop random values from df to simulate MCAR missingness is evenly distributed, can be different
def induce_mcar(df: DataFrame = None, miss_prop: float = 0.1, time_colums: list = [], static_columns: list = []):
    '''
    Induces MCAR pattern for time variant variables under missing probability
            Parameters:
                    df (DataFrame): A time variant df with time related variables
                    miss_prop (float): Probability for each value v being missing 
                    time_colums (list): Column names for time related variables
                    static_coluns (list): Column names for static variables 
            Returns:
                    df (DataFrame): Df with MCAR pattern
    '''
    # drawing 1 - miss_prop values as subsample per variable 
    dfs = [shuffle(df[static_columns + [v]]).sample(frac = 1 - miss_prop) for v in time_colums]
    # joining subsampled values per variable back to one df and return result
    df_join = reduce(lambda df1, df2: df1.merge(df2, on=static_columns, how = "outer"), dfs).drop_duplicates()
    return df[static_columns].merge(df_join, on=static_columns, how = "left")

# drop random values conditionally on one dependent variable per case
def induce_mar(df: DataFrame = None, cond_var: str = "age", miss_range: list = [0.1, 0.9], 
                identifier: str = "case_id", static_columns: list = []):
    '''
    Induces MAR pattern for time variant variables with range of missing probabilities under a conditioning (dependent) variable
    The direction of missing probabilities is positive (increases when dependent variable increases)
            Parameters:
                    df (DataFrame): A time variant df with time related variables
                    cond_var (float): The dependent variable for inducing missingness
                    time_colums (list): Column names for time related variables
                    miss_range (list): Range of missing probabilities
                    identifier (str): Identifier of a case / patient / sequence
                    static_columns (list): Column names for group by condition being static
            Returns:
                    df (DataFrame): Df with MAR pattern
    '''
    # group by identifier and get medians per group for conditioning 
    df_gr = df.groupby(identifier)[cond_var].median().reset_index()
    # asssign values of cond var to tmp var
    miss_fractions =  df_gr[cond_var]
    # normalization of cond values between [s_min : s_max]
    s_min, s_max = miss_range[0], miss_range[1]
    miss_fractions = (miss_fractions - min(miss_fractions))/ \
        (max(miss_fractions) - min(miss_fractions)) * (s_max - s_min) + s_min
    df_gr = df_gr.assign(miss_fraction = miss_fractions)
    # merge missing rates to actual values 
    df = df.merge(df_gr[[identifier] + ["miss_fraction"]], on=identifier)
    # going through variables and cases and conditioning on dependent variable
    df_vars = []
    for time_v in time_vars:
        dfs_cases = []
        for i, row in df_gr.iterrows():
            miss_fraction = row["miss_fraction"]
            dfs_cases.append(df[df.case_id == row[identifier]] \
                [static_columns + [time_v]].sample(frac = 1 - miss_fraction))
        df_vars.append(pd.concat(dfs_cases))
    # outerjoining results containing missingness per variable to one dataframe
    df_join = reduce(lambda df1, df2: df1.merge(df2, on=static_columns, how = "outer"), df_vars).drop_duplicates()
    return df[static_columns].merge(df_join, on=static_columns, how = "left")

# apply missingness depending on the acutal value of one feature / high value = high missingness in a range 
def induce_mnar(df: DataFrame = None, static_columns:list = [], time_colums:list = [], miss_range: list = [0.1, 0.9]):
        s_min, s_max, df_per_var = miss_range[0], miss_range[1], []
        # get min max information for time related vars and store into df
        min_max = df.describe().loc[["min", "max"], time_colums]
        # get missing rates normalized in range and dependent on value levels of variables
        missing_rates_per_var = (df[time_colums] - min_max.loc[:, time_colums].loc["min"])/ \
                                (min_max.loc[:, time_colums].loc["max"] - \
                                min_max.loc[:, time_colums].loc["min"]) * (s_max - s_min) + s_min
        # go through each variable and levels to induce MNAR
        for var in time_colums:
                # getting missing rates and counts per feature level 
                miss_val = pd.concat([df[var].to_frame("value"), 
                                missing_rates_per_var[var].to_frame("miss")], axis=1)\
                                .value_counts().reset_index()
                # reducing to levels occuring at least 10 times (otherwise subsampling may not be sufficient)
                miss_val, ds = miss_val[miss_val[0] > 10], []
                # going through feature levels and inducing missingness per level
                for i, row in miss_val.iterrows():
                        d = df[static_columns + [var]]
                        ds.append(d[d[var] == row.value].sample(frac=1-row.miss))
                df_per_var.append(pd.concat(ds))
        # merging dfs per variable outer for preserving missing columns
        df_join = reduce(lambda df1, df2: df1.merge(df2, on=static_columns, how = "outer"), df_per_var).drop_duplicates()
        return df[static_columns].merge(df_join, on=static_columns, how = "left")

# usage of defined functions 
def induce_and_plot():
    df = load_data()
    df_mcar = induce_mcar(df = df, time_colums= time_vars, static_columns=["case_id","timestamp", "seq_no", "age"])
    df_mar = induce_mar(df=df, cond_var = "age", miss_range = [0.1, 0.9], 
                identifier = "case_id", static_columns = ["case_id", "timestamp", "seq_no", "age"])
    df_mnar = induce_mnar(df=df, static_columns=["case_id", "timestamp", "seq_no", "age"], time_colums=time_vars)
    plot_miss_patterns(df=df, df_mar=df_mar, df_mcar=df_mcar, df_mnar = df_mnar)

induce_and_plot()