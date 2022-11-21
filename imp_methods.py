from conf_vars import time_vars

def impute_ffil(df  = None, group_cols = ["case_id"], time_vars = time_vars):
    stats = df.describe().reset_index() # TODO check on these two methods and why they yield same results
    medi_fill = stats[stats["index"] == "50%"][time_vars].iloc[0]
    # apply forward fill, impute values with median and remaining with 0 in edge cases
    df[time_vars] = df.groupby(group_cols).\
        transform(lambda x: x.ffill())[time_vars].fillna(medi_fill).fillna(0)
    return df

def impute_median(df  = None, time_vars = time_vars):
    stats = df.describe().reset_index()
    medi_fill = stats[stats["index"] == "50%"][time_vars].iloc[0]
    # apply forward fill, impute values with median and remaining with 0 in edge cases
    df[time_vars] = df[time_vars].fillna(medi_fill).fillna(0)
    return df
