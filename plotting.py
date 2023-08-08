import pandas as pd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import seaborn as sns
import random

from data_loader import DataLoader

def plot_frequency_lens(data_loader: DataLoader = None):

    ''' plots sequence lenghts of created complete case datasets '''

    conf_items = data_loader.conf_items
    data_meas = data_loader.data_meas

    # prepare and concat dataframes as long format for hist and boxplot
    dfs_plotified = [(conf_items[item]["data"].groupby(conf_items[item]["patient_id"])\
        ['{}h_index'.format(str(conf_items[item]["sampling_interval"]))]\
            .count() * conf_items[item]["sampling_interval"])\
                .reset_index(name="seq_lens").assign(hourly_sampling=str(conf_items[item]["sampling_interval"])) for item in conf_items.keys()] + \
                    [data_meas.groupby("subject_id").hours_in.count().reset_index(name="seq_lens").assign(hourly_sampling=str(1))]

    df_plot = pd.concat(dfs_plotified).reset_index()

    fig = plt.figure(constrained_layout=True, figsize=(12, 2 * len(df_plot.sampling.unique())))
    subf = fig.subfigures(2,1)
    ax1, ax2 = subf[0].subplots(1,1), subf[1].subplots(1,1)
    my_pal, rand_color = {}, random.sample(list(mcolors.TABLEAU_COLORS.keys()), k=len(df_plot.sampling.unique()))
    for i, sampl in enumerate(df_plot.sampling.unique()):
        my_pal[sampl] = rand_color[i]
    sns.histplot(data=df_plot.sort_values("hourly_sampling", ascending=True),  x="seq_lens", ax=ax1, hue="hourly_sampling", palette=my_pal, bins=100)
    sns.boxplot(data=df_plot.sort_values("hourly_sampling", ascending=False),  xs="seq_lens", ax=ax2,  y="hourly_sampling", palette=my_pal)
    fig.savefig("./sequence_lengths.png", dpi=320)

def plot_missing_rates_heatmap(data_loader: DataLoader = None, sampling_intervals: list =[]):

    '''plot missing rates per sampling interval '''

    data = data_loader.data_meas
    features = data_loader.features

    df_plot = None
    for i, interv in enumerate(sampling_intervals):
        # iterate through sampling intervals, aggregate and derive missingness
        df_sampl = data.assign(sample_index = data.groupby(["subject_id"]).cumcount()//interv)
        df_sampl = df_sampl.groupby(["subject_id", "sample_index"]).mean().reset_index()
        df_sampl = (df_sampl[features].isna().sum()/len(df_sampl))\
            .reset_index(name="sampling_{}h".format(interv))\
            .reset_index().drop(columns=["index"])
        
        if i == 0:
            df_plot = df_sampl
        else:
            df_plot = df_plot.merge(df_sampl)

    df_plot = df_plot.sort_values(by="sampling_{}h".format(interv))\
        .reset_index().drop(columns=["index"])
    
    df_plot.to_csv("./missing_rates.csv")

    fig, ax = plt.subplots(figsize=(30, i*2)) # TODO
    sns.heatmap(df_plot.drop(columns="LEVEL2").T, ax=ax, cbar_kws={"shrink": 0.7})

    # store figure
    fig.savefig("./missing_rates.png", dpi=340)



