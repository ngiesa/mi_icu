from imp_methods import impute_ffil, impute_median
from sklearn.metrics import mean_absolute_error, mean_squared_error

variables = ["bp", "hr", "rr", "spo2", "tmp", "age", "seq_len"]
col_names = ["case_id", "timestamp", "bp", "hr", "rr", "spo2", "tmp", "time_index", "seq_no", "seq_len", "age"]
time_vars = ["bp", "hr", "rr", "spo2", "tmp"]
drop_cols = ["case_id", "timestamp", "seq_no", "time_index"]

var_ranges = {
    "bp": [2, 140],
    "hr": [30, 200],
    "rr": [2, 90],
    "spo2": [80, 100],
    "tmp": [34, 45]
}

store_paths_values = {
    "complete": "/home/giesan/data/mi_icu_full_data_07_10_22.csv",
    "mcar": "/home/giesan/data/mcar_val.csv",
    "mar": "/home/giesan/data/mar_val.csv",
    "mnar": "/home/giesan/data/mnar_val.csv",
}

store_paths_miss = {
    "mcar": "/home/giesan/data/mcar_miss.csv",
    "mar": "/home/giesan/data/mar_miss.csv",
    "mnar": "/home/giesan/data/mnar_miss.csv",
}

imp_methods = {
    "locf": impute_ffil,
    "median": impute_median
}

error_functions = {
    "mae": mean_absolute_error,
    "mse": mean_squared_error
}

# sys, dia, hr, rr, spo2 (tmp?) 2h sampling interval 
# wie 2h features, mehr patienten 4h
# 6h wie 2h + tmp 9000 patienten  