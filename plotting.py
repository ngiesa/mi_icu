import pandas as pd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import seaborn as sns
import random

sns.set_theme(style="whitegrid")

from data_loader import DataLoader

def plot_frequency_lens(data_loader: DataLoader = None):

    ''' plots sequence lenghts of created complete case datasets '''

    conf_items = data_loader.conf_items
    data_meas = data_loader.data_meas

    # prepare and concat dataframes as long format for hist and boxplot
    dfs_plotified = [(conf_items[item]["data"].groupby(conf_items[item]["patient_id"])\
        ['{}h_index'.format(str(conf_items[item]["sampling_interval"]))]\
            .count() * conf_items[item]["sampling_interval"])\
                .reset_index(name="seq_lens").assign(sampling=str(conf_items[item]["sampling_interval"]) + "h") for item in conf_items.keys()] + \
                    [data_meas.groupby("subject_id").hours_in.count().reset_index(name="seq_lens").assign(sampling=str(1) + "h")]

    df_plot = pd.concat(dfs_plotified).reset_index()
    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    ax1, ax2 = axs[0], axs[1]
    my_pal, rand_color = {}, random.sample(list(mcolors.TABLEAU_COLORS.keys()), k=len(df_plot.sampling.unique()))
    for i, sampl in enumerate(df_plot.sampling.unique()):
        my_pal[sampl] = rand_color[i]

    sns.histplot(data=df_plot.sort_values("sampling", ascending=False),  x="seq_lens", ax=ax1, hue="sampling",bins=100, palette=my_pal)
    sns.boxplot(data=df_plot.sort_values("sampling", ascending=False), x="seq_lens", y="sampling", ax=ax2, palette=my_pal)

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

    df_plot = df_plot.sort_values(by="sampling_1h")\
        .reset_index().drop(columns=["index"])
    
    df_plot.to_csv("./missing_rates.csv")

    fig, ax = plt.subplots(figsize=(30, i*2))
    sns.heatmap(df_plot.drop(columns="LEVEL2").T, ax=ax, cbar_kws={"shrink": 0.7})

    # store figure
    fig.savefig("./missing_rates.png", dpi=340)



# scatterplot with number of feats, sampling interval and number of patients 
def plot_scatter_feat_combo():

    f, ax = plt.subplots(1,1,figsize=(12, 8))

    df_combo_sets = pd.read_csv("./combo_miss_rate_feats.csv", index_col=0)

    cmap = sns.cubehelix_palette(rot=-.2, as_cmap=True)
    f = sns.relplot(
        data=df_combo_sets,
        x="sampling", y="combo_set",
        size="n_patients", hue="n_patients",
        sizes=(50, 400),
        edgecolor='grey',
        palette=cmap,
    )
    f.ax.xaxis.grid(True, "minor", linewidth=.25)
    f.ax.yaxis.grid(True, "minor", linewidth=.25)
    f.despine(left=True, bottom=True)
    f.tight_layout()
    f.savefig("./scatter_combo_feat.png", dpi=320)