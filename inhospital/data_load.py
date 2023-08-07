import pandas as pd
from conf_vars import col_names, store_paths_values, store_paths_miss

def load_raw_data():
    df = pd.read_csv("/home/giesan/data/mi_icu_full_data_07_10_22.csv", header= None, names=col_names, index_col=0).reset_index()
    # basic preprocess age because original file lags in consistency here
    df.loc[:,"age"] = df.loc[:,"age"].ffill()
    return df.drop_duplicates()

def load_miss_values():
    return {
        "complete": {
            "data": load_raw_data(),
            "color": "red"
        },
        "mcar": {
            "data": pd.read_csv(store_paths_values["mcar"], index_col=0),
            "color": "green"
        },
        "mar": {
            "data": pd.read_csv(store_paths_values["mar"], index_col=0),
            "color": "blue"
        }, 
        "mnar": {
            "data":  pd.read_csv(store_paths_values["mnar"], index_col=0),
            "color": "yellow"
        }
    }

def load_miss_fractions():
    return {
        "mcar": {
            "data": pd.read_csv(store_paths_miss["mcar"], index_col=0),
            "color": "green"
        },
        "mar": {
            "data": pd.read_csv(store_paths_miss["mar"], index_col=0),
            "color": "blue"
        }, 
        "mnar": {
            "data":  pd.read_csv(store_paths_miss["mnar"], index_col=0),
            "color": "orange"
        }
    }
