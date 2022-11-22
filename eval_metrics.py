from pandas.core.frame import DataFrame
import pandas as pd

def calc_error(df_gt: DataFrame = None, df_imp: DataFrame = None, time_vars: list = [], metric = None, df_miss = None):
    '''
    Calculates the error per variable between actual and imputed values
            Parameters:
                    df_gt (DataFrame): Ground thruth dataframe with real values
                    df_imp (DataFrame): Imputed dataframe 
                    time_vars (list): Column names for imp variables
                    metric (metric): (Sklearn) Metric to calculate the error
                    df_miss (DataFrame): df with miss pattenr, if not None just include imputed vales

            Returns:
                    df (DataFrame): Df with error per variable
    '''
    if df_miss is not None:
        error = [metric(df_gt.iloc[df_miss[df_miss[v].isna()].index][v], 
                        df_imp.iloc[df_miss[df_miss[v].isna()].index][v]) 
                        for v in time_vars]
    else:
        error = [metric(df_gt[v], df_imp[v]) for v in time_vars]
    return pd.DataFrame({"variable": time_vars, "error": error})
    