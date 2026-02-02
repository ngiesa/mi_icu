import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.colors import LogNorm, Normalize
from static_variables import DATASETS, FEATURES, load_resampled_sequences, load_complete_sequences
from imputation import prepare_bootstrapps


def plot_baseline_infos(plotting_data, DATASETS, clean):
    f, axs_occ = plt.subplots(3, 3, figsize=(14, 9))

    for d, dataset in enumerate(DATASETS):
        
        corr = plotting_data[dataset.upper()]["corr"]
                
        sns.barplot(plotting_data[dataset.upper()]["acf"], x="feature", y="acf_coef", 
                    ax=axs_occ[1][d], color="darkorange", linewidth=1, edgecolor=".5")
        sns.barplot(plotting_data[dataset.upper()]["miss"], x="feature", y="single_miss", 
                    ax=axs_occ[2][d], color="darkred", linewidth=1, edgecolor=".5")
        sns.heatmap(corr, annot=True, mask=np.triu(np.ones_like(corr)).astype(bool),
                vmin=0.0, vmax=1, cbar_kws={"shrink": 0.6}, ax=axs_occ[0][d], cbar=False, cmap="Blues")
    
        axs_occ[1][d].set_ylim(0, 1)
        axs_occ[2][d].set_ylim(0, 1)
        axs_occ[1][d].grid()
        axs_occ[2][d].grid()
        
        axs_occ[0][d].set_title(dataset, fontweight="bold", fontsize=18)
        axs_occ[0][d].set_xlabel("")
        
        axs_occ[1][d].set_xlabel("")
        axs_occ[1][0].set_ylabel("auto-correlation (lag=1)\n", fontweight="bold", fontsize=12)
        axs_occ[1][1].set_ylabel("")
        axs_occ[1][2].set_ylabel("")
        
        axs_occ[2][d].set_xlabel("")
        axs_occ[2][d].set_ylabel("")
        axs_occ[2][0].set_ylabel("single missingness\n", fontweight="bold", fontsize=12)
        axs_occ[2][d].set_xlabel("feature", fontweight="bold", fontsize=12)
        axs_occ[0][0].set_ylabel("cross-correlation", fontweight="bold", fontsize=12)
        axs_occ[0][1].set_ylabel("")
        axs_occ[0][2].set_ylabel("")
        
    f.savefig("./plots/missing_corr_missing_{}.png".format(clean), dpi=300)
    
    
def plot_demographics(list_complete_sequences, list_resampled_sequences, DATASETS, clean):
    fig_occ, axs_occ = plt.subplots(3, 3, figsize= (14, 9))

    upper_limits = [200, 400, 50]

    ids = ["subject_id", "patientid", "sequence_id"]

    reduce_time_seq = True
    
    df_desc = []


    for d, dataset in enumerate(DATASETS):
        static_data = pd.read_csv("./desc/static_data_{}.csv".format(dataset.lower()))
        
        df_lens_compl = list_complete_sequences[d].groupby("sequence_id")["hr"].count().reset_index()
        # reduce the data to all sequences longer than 3 time steps 
        if reduce_time_seq:
            df_lens_compl = df_lens_compl[df_lens_compl.hr > 2]
        print("seq lens ", len(df_lens_compl))
        print("complete desc")
        df_desc.append(df_lens_compl.hr.describe().reset_index(name="value").assign(clean=clean).assign(dataset=dataset).assign(feature="seq_lens").assign(data="complete"))
        df_lens_org = list_resampled_sequences[d].groupby("sequence_id")["hr"].count().reset_index()
        df_desc.append(df_lens_org.hr.describe().reset_index(name="value").assign(clean=clean).assign(dataset=dataset).assign(feature="seq_lens").assign(data="original"))


        # reduce static data to included ones
        static_compl = static_data.reset_index().assign(sequence_id = static_data.reset_index()[ids[d]]).merge(df_lens_compl, on="sequence_id")
        sns.histplot(x = list(df_lens_org[df_lens_org.hr < upper_limits[d]].hr), stat="proportion", ax=axs_occ[0][d],  bins=30, label="ORI")
        sns.histplot(x = list(df_lens_compl[df_lens_compl.hr < upper_limits[d]].hr), stat="proportion", ax=axs_occ[0][d], bins=30, label="CCD")
        sns.histplot(x = pd.to_numeric(static_data.reset_index()["age"]), bins=30, stat="proportion", ax=axs_occ[1][d])
        sns.histplot(x = pd.to_numeric(static_compl["age"]), bins=30, stat="proportion", ax=axs_occ[1][d])
        sns.histplot(x = static_data.reset_index()["gender"].sort_values(), bins=30, stat="proportion", ax=axs_occ[2][d])
        sns.histplot(x = static_compl["gender"].sort_values(), bins=30, stat="proportion", ax=axs_occ[2][d])
        
        df_desc.append(static_compl["age"].describe().reset_index(name="value").assign(clean=clean).assign(dataset=dataset).assign(feature="age").assign(data="complete"))
        df_desc.append(static_compl["gender"].describe().reset_index(name="value").assign(clean=clean).assign(dataset=dataset).assign(feature="gender").assign(data="complete"))
        df_desc.append(static_data.reset_index()["age"].describe().reset_index(name="value").assign(clean=clean).assign(dataset=dataset).assign(feature="age").assign(data="original"))
        df_desc.append(static_data.reset_index()["gender"].describe().reset_index(name="value").assign(clean=clean).assign(dataset=dataset).assign(feature="gender").assign(data="original"))
        
        axs_occ[0][d].grid()
        axs_occ[1][d].grid()
        axs_occ[2][d].grid()
        
            
        for a in range(0, 3):
            axs_occ[2][a].set_xlabel("sex")
            axs_occ[0][a].set_xlabel("seq len")
            if d != 0:
                axs_occ[a][d].set_ylabel("")
                
            
        axs_occ[0][d].set_title(dataset, fontweight="bold", fontsize=18)
        
    axs_occ[0][0].legend()
    fig_occ.tight_layout()
    
    pd.concat(df_desc).to_csv("./desc/demo_decs_{}.csv".format(clean))
    
    fig_occ.savefig("./plots/demographics_complete_{}.png".format(clean), dpi=400)
    
    
def plot_age_missing_dependency(origin = "mcar", clean="pres", show_legend = True, direction= "normal"):

    # get common quartiles 
    
    """ origin can have values of mcar or mar """

    all_ages_df = []
    features = FEATURES

    for d, dataset in enumerate(DATASETS):
        static_data = pd.read_csv("./desc/static_data_{}.csv".format(dataset.lower()))
        age_stat = static_data.reset_index()[["age", "sequence_id"]].drop_duplicates()
        all_ages_df.append(age_stat.assign(dataset=dataset))
        
    age_all_df = pd.concat(all_ages_df)
    age_all_df = age_all_df[age_all_df.age >= 18]
    binning = pd.qcut(age_all_df.age, 3, labels=False, retbins=True, duplicates='drop')

    bin_gr = []
    for i, b in enumerate(binning[1]):
        if i+1 == len(binning[1]):
            break
        bin_gr.append("({},{}]".format(int(b), int(binning[1][i+1])))
        
    age_all_df = age_all_df.assign(age_bin = [bin_gr[j] for j in binning[0]]).assign(j = [j for j in binning[0]])
    
    #age_all_df.to_csv("./desc/age_groups_sequence.csv")
        
    list_resampled_sequences = []
    
    if origin == "mcar":
    
        list_resampled_sequences = load_resampled_sequences(clean=clean)

    fig_occ, axs_occ = plt.subplots(1, 3, figsize=(10, 12))
    fig_miss, axs_miss = plt.subplots(1, 3, figsize=(11.2, 3.2), sharey=True)
    sorted_states = pd.read_csv("/home/giesan/mi_icu/spatio_temporal_pattern/sorted_states_for_heatmaps.csv", dtype=object)
    sorting_states, sequences = [], []
    for d, dataset in enumerate(DATASETS):
        # get sequences as missing indicator tables and assign bins
        if origin == "mcar":
            sequences = list_resampled_sequences[d][features].isnull().astype(int)\
                .assign(sequence_id = list_resampled_sequences[d]["sequence_id"])
        if origin == "mar":
            sequences = pd.read_csv("./induced_sequences/{}/missing_matrix_{}_boot_0_mar_{}.csv".format(dataset.lower(), dataset.lower(), clean), index_col = 0)
            if direction == "reverse":
                import os
                dirs = os.listdir("./induced_sequences/{}/".format(dataset.lower()))
                if "missing_matrix_{}_boot_0_mar_{}_rev.csv".format(dataset.lower(), clean) in dirs:
                    print("rev open")
                    sequences = pd.read_csv("./induced_sequences/{}/missing_matrix_{}_boot_0_mar_{}_rev.csv".format(dataset.lower(), dataset.lower(), clean), index_col = 0)   
        # retrieve counts as proportions
        age_occurrence = age_all_df[age_all_df.dataset == dataset][["sequence_id", "age_bin"]].merge(sequences, on="sequence_id").groupby("age_bin")[features]\
            .value_counts(normalize=True).reset_index()
        # top 5 and combine feats
        top5_comb = age_occurrence#.groupby("bin")
        df_plot = top5_comb.assign(occurrence = top5_comb.hr.astype(str) + 
                                top5_comb.bp_sys.astype(str) + 
                                top5_comb.bp_dia.astype(str)+ 
                                top5_comb.spo2.astype(str)+ 
                                top5_comb.rr.astype(str))
        # get all distinct combinations and join
        all_comb = df_plot[["occurrence"]].drop_duplicates()
        # iterate combinations show age differences in occurrence matrix (original)
        for i, gr in df_plot.groupby("age_bin"):
            all_comb = all_comb.merge(gr[["occurrence", "proportion"]].rename(columns={"proportion": i}), on="occurrence", how="outer")
        # alter index and plot
        all_comb.columns = ["occurrence"] + bin_gr
        # merge with sorted states
        all_comb = sorted_states[["occurrence"]].merge(all_comb, how="outer")
        sns.heatmap(all_comb.set_index("occurrence"), annot=True, cbar=None, cmap="viridis", ax=axs_occ[d], norm=LogNorm(), vmin=0, vmax=0.83, linewidths=0.5, linecolor="white")
        axs_occ[d].set_title(dataset, fontweight="bold", fontsize=18)
        axs_occ[d].set_yticklabels(list(all_comb["occurrence"]))
        axs_occ[d].set_xlabel("")
        axs_occ[d].set_ylabel("")
        
        # store for sorting purposes 
        sort_df = all_comb.iloc[:,0:2]
        sort_df.columns=["occurrence", "prop"]
        sorting_states.append(sort_df.assign(dataset=dataset))
        
        # plotting also the miss per feature and age group
        df_plot_miss_age = age_all_df[age_all_df.dataset == dataset][["sequence_id", "age_bin"]]\
            .merge(sequences, on="sequence_id").groupby("age_bin")[features].mean().reset_index().drop(columns=["age_bin"]).T.reset_index()
        # transforming the bins
        df_plot = pd.concat([
            df_plot_miss_age[["index", 0]].rename(columns={0: "mr"}).assign(age=bin_gr[0]), 
            df_plot_miss_age[["index", 1]].rename(columns={1: "mr"}).assign(age=bin_gr[1]),
            df_plot_miss_age[["index", 2]].rename(columns={2: "mr"}).assign(age=bin_gr[2]),
        ], axis=0)
        # adding all 
        df_plot = pd.concat([df_plot, df_plot.groupby("age")["mr"]\
            .mean().reset_index().assign(index="all")])
        sns.barplot(df_plot, y="mr", x="index", hue="age",  ax=axs_miss[d], palette="mako", linewidth=1, edgecolor=".5") # palette="PuRd",
        
        if origin == "mcar":
            axs_miss[d].set_ylim(0.0, 0.35)
        
        if origin == "mar":
            axs_miss[d].set_ylim(0.0, 0.9)
        
        axs_miss[d].set_xlabel("")
        axs_miss[d].set_ylabel("")
        if (d == 0) & show_legend:
            axs_miss[d].legend(loc ="upper left", title="age")
        else:
            axs_miss[d].legend().remove()
        axs_miss[d].grid()
        axs_miss[d].set_title(dataset, fontweight="bold", fontsize=18)
    
    clean_s = clean
    if direction == "reverse":
        clean_s = clean + "_rev"    
    
    fig_miss.supxlabel("feature", fontsize=12, fontweight="bold")
    fig_miss.supylabel("single missingness\n", fontsize=12, fontweight="bold")
    fig_occ.supxlabel("age bin (lower bound, upper bound]", fontsize=12)
    fig_occ.supylabel("occurrence state\n", fontsize=12)
    fig_occ.tight_layout()
    fig_miss.tight_layout()
    fig_occ.savefig("./plots/occurrence_per_age_bin_{}_{}.png".format(clean_s, origin), dpi=300)
    fig_miss.savefig("./plots/missing_rate_per_age_bin_{}_{}.png".format(clean_s, origin), dpi=300)
        
def plot_feature_bins(datasets, features):
    
    for clean in ["drop", "pres"]:
        list_resampled_sequences = load_resampled_sequences(clean=clean)

        all_vitals = []
        for s, seq in enumerate(list_resampled_sequences):
            all_vitals.append(seq[features + ["sequence_id"]].assign(dataset = datasets[s]))
            
        original_vitals = pd.concat(all_vitals)
        
        # apply cleansing with valid ranges

        bin_group_legends = []
        df_plots = []

        units = ["1/min", "mmHg", "mmHg", "%", "1/min"]

        # go through all features and look at single missingness across these bin groups
        for l, f in enumerate(features):
            binning = pd.qcut(original_vitals[f], 3, labels=False, retbins=True, duplicates='drop')
            original_vitals[f + "_bin"] = binning[0]
            bin_gr = []
            for i, b in enumerate(binning[1]):
                if i+1 == len(binning[1]):
                    break
                bin_gr.append("({},{}]".format(int(b), int(binning[1][i+1])))
            bin_group_legends.append(pd.DataFrame({"feature": f, "unit":units[l], "bins": str(bin_gr)}, index=[f]))
            df_plots.append(original_vitals[[f, f+"_bin", "dataset"]]\
                .assign(miss=original_vitals[f].isna().astype(int).shift(1))\
                    .groupby([f+"_bin", "dataset"])["miss"].mean().reset_index().rename(columns={f+"_bin": "bin"}).assign(feature=f))
            
        pd.concat(bin_group_legends).to_csv("./desc/feature_bins_{}.csv".format(clean))
        
        fig_miss, axs_miss = plt.subplots(1, 3, figsize=(14, 3.8), sharey=True)

        # iterating through datasets
        for d, dataset in enumerate(datasets):

            df_plot = original_vitals[original_vitals.dataset == dataset]
            
            # what is missingness rate in each value group??? 
            
            df_plot = pd.concat(df_plots)
            df_plot = df_plot[df_plot.dataset == dataset].drop(columns=["dataset"])
            df_plot = pd.concat([df_plot, df_plot.groupby("bin")["miss"].mean().reset_index().assign(feature="all")])
            sns.barplot(df_plot, y="miss", x="feature", hue="bin",  ax=axs_miss[d], palette="summer", linewidth=1, edgecolor=".5")
            axs_miss[d].set_ylim(0.0, 0.25)
            axs_miss[d].set_xlabel("")
            axs_miss[d].set_ylabel("")
            if d == 0:
                axs_miss[d].legend(loc ="upper left", title="bin")
            else:
                axs_miss[d].legend().remove()
            axs_miss[d].grid()
            axs_miss[d].set_title(dataset, fontweight="bold", fontsize=18)
            
            
        fig_miss.supxlabel("feature", fontsize=12, fontweight="bold")
        fig_miss.supylabel("single missingness\n", fontsize=12, fontweight="bold")

        fig_miss.tight_layout()

        fig_miss.savefig("./plots/missing_rate_per_value_bin_{}.png".format(clean), dpi=300)
        
        
def plot_mnar_occurrences(boot = 0, clean = "pres"):
    
    fig_mcar, axs_mcar = plt.subplots(1, 3, figsize=(10, 12))
    fig_mnar, axs_mnar = plt.subplots(1, 3, figsize=(10, 12))

    from matplotlib.colors import LogNorm
    import seaborn as sns

    def assign_state_combined(df):
        
        """ combining all occurrence states to one label """
        
        return df.assign(state = df[FEATURES].astype(str).values.sum(axis=1))

    for d, dataset in enumerate(DATASETS):
        
        # reading data
        df_range = pd.read_csv("/home/giesan/mi_icu/spatio_temporal_pattern/desc/complete_value_ranges.csv", index_col = 0)
        df_induced_mnar = pd.read_csv("/home/giesan/mi_icu/spatio_temporal_pattern/induced_sequences/{}/missing_matrix_{}_boot_{}_mnar_{}.csv".format(dataset.lower(), dataset.lower(), boot, clean), index_col = 0)
        df_induced_mcar = pd.read_csv("/home/giesan/mi_icu/spatio_temporal_pattern/induced_sequences/{}/missing_matrix_{}_boot_{}_mcar_{}.csv".format(dataset.lower(), dataset.lower(), boot, clean), index_col = 0)

        # filter value bins and ranges
        df_value_bins = df_range[df_range.dataset == dataset.upper()]
        
        # get sorted states
        sorted_states = pd.read_csv("/home/giesan/mi_icu/spatio_temporal_pattern/sorted_states_for_heatmaps.csv", dtype=object)
        sorted_states["state"] = sorted_states["occurrence"]
        
        # prepare mcar and mnar plots for differences
        df_plot_mcar = df_value_bins[[f + "_bin" for f in FEATURES] + ["sequence_id", "time_index"]]\
        .merge(df_induced_mcar, on=["sequence_id", "time_index"], suffixes = ["_val", "_bin"])
        
        df_plot_mcar = assign_state_combined(df = df_plot_mcar)
        df_plot_mnar = assign_state_combined(df = df_induced_mnar)

        # get average group assignments 
        df_plot_mcar = df_plot_mcar.assign(bin_avg = df_plot_mcar[[f + "_bin" for f in FEATURES]]\
            .agg(func=np.median, axis=1))
        df_plot_mnar = df_plot_mnar.assign(bin_avg = df_plot_mnar[[f + "_bin" for f in FEATURES]]\
            .agg(func=np.median, axis=1))
        
        # get occurrence states
        df_plot_mcar = df_plot_mcar.groupby("bin_avg")["state"].value_counts(normalize=True).reset_index()
        df_plot_mnar = df_plot_mnar.groupby("bin_avg")["state"].value_counts(normalize=True).reset_index()
        
        # pivot tables
        df_plot_mcar = df_plot_mcar.pivot_table(index="state", columns="bin_avg", values="proportion").reset_index()
        df_plot_mnar = df_plot_mnar.pivot_table(index="state", columns="bin_avg", values="proportion").reset_index()
        
        # merge with all sorted states
        df_plot_mcar = sorted_states[["state"]].merge(df_plot_mcar, how="outer")
        df_plot_mnar = sorted_states[["state"]].merge(df_plot_mnar, how="outer")
        
        df_plot_mcar = df_plot_mcar.set_index("state")
        df_plot_mnar = df_plot_mnar.set_index("state")
        
        df_plot_mcar.columns = [int(c) for c in df_plot_mcar.columns]
        df_plot_mnar.columns = [int(c) for c in df_plot_mnar.columns]
        
        # plot mcar
        sns.heatmap(df_plot_mcar, annot=True, cbar=None, cmap="viridis", 
                ax=axs_mcar[d], norm=LogNorm(), vmin=0, vmax=0.83, linewidths=0.5, linecolor="white")
        
        sns.heatmap(df_plot_mnar, annot=True, cbar=None, cmap="viridis", 
                ax=axs_mnar[d], norm=LogNorm(), vmin=0, vmax=0.83, linewidths=0.5, linecolor="white")
        
        axs_mcar[d].set_xlabel("")
        axs_mcar[d].set_ylabel("")
        axs_mnar[d].set_xlabel("")
        axs_mnar[d].set_ylabel("")
        
        # labels
        axs_mcar[d].set_title(dataset, fontweight="bold", fontsize=18)
        axs_mnar[d].set_title(dataset, fontweight="bold", fontsize=18)
        fig_mcar.supxlabel("value bin", fontsize=12)
        fig_mcar.supylabel("occurrence state\n", fontsize=12)
        fig_mnar.supxlabel("value bin", fontsize=12)
        fig_mnar.supylabel("occurrence state\n", fontsize=12)
        
        fig_mcar.tight_layout()
        fig_mnar.tight_layout()

    fig_mcar.savefig("/home/giesan/mi_icu/spatio_temporal_pattern/plots/occurrence_per_value_bin_{}_mcar.png".format(clean), dpi=400)
    fig_mnar.savefig("/home/giesan/mi_icu/spatio_temporal_pattern/plots/occurrence_per_value_bin_{}_mnar.png".format(clean), dpi=400)
        

def prepare_mnar_plot(compl_ranges, bootstrapp_df):

    """ get bin ranges of feature values and the missingness level """

    df_plot = []
    for f in FEATURES:
        select_f_compl = compl_ranges[["sequence_id", "time_index", f + "_bin"]]
        select_f_ind = bootstrapp_df[["sequence_id", "time_index", f]]
        select_f = select_f_compl.merge(select_f_ind, on=["sequence_id", "time_index"])
        df_plot_miss = select_f[["sequence_id", "time_index"]].assign(bin =  select_f_compl.reset_index()[f + "_bin"])
        df_plot_miss = df_plot_miss.assign(miss = select_f[f].isna().astype(int)).assign(feature = f) #TODO ISNA
        #print(df_plot_miss)
        df_plot.append(df_plot_miss.groupby(["feature", "bin"])["miss"].mean())
        
    return pd.concat(df_plot).reset_index()

def plot_mnar_feature_values(clean = "pres", pattern = "mnar", show_legend=True, direction="normal"):
    
    
    bootstrapps = prepare_bootstrapps(clean=clean, max_boot=1)
    complete_value_ranges = pd.read_csv("/home/giesan/mi_icu/spatio_temporal_pattern/desc/complete_value_ranges.csv", index_col = 0)
    
    fig_miss, axs_miss = plt.subplots(1, 3, figsize=(11.2, 3.2), sharey=True)

    PATTERN = pattern.upper()

    for d, dataset in enumerate(DATASETS):
        
        boot = bootstrapps[dataset]["boots"][0]
        complete = bootstrapps[dataset]["complete"]
        
        suffix = ""
        if direction == "reverse":
            suffix = "_rev"
        
        if PATTERN == "MNAR":
            mnar_pattern = pd.read_csv("/home/giesan/mi_icu/spatio_temporal_pattern/induced_sequences/{}/missing_matrix_{}_boot_0_mnar_pres{}.csv".format(dataset.lower(), dataset.lower(), suffix), index_col = 0)

            dfs_plot = []
            for f in FEATURES:
                miss_stats = mnar_pattern.groupby(f + "_bin")[f].mean().reset_index()
                dfs_plot.append(pd.DataFrame({"feature": f, "miss": miss_stats[f], "bin": miss_stats[f+"_bin"]}))
                
            df_plot = pd.concat(dfs_plot)
        
        else:
        
            compl_ranges = complete_value_ranges[complete_value_ranges.dataset == dataset]
            df_plot = prepare_mnar_plot(compl_ranges, boot)

        
        df_plot = pd.concat([df_plot, df_plot.groupby("bin")["miss"].mean().reset_index().assign(feature="all")])
        
        sns.barplot(df_plot, y="miss", x="feature", hue="bin",  ax=axs_miss[d], palette="summer", linewidth=1, edgecolor=".5")
        
        y_lim = 0.4
        
        if PATTERN == "MNAR":
            y_lim = 0.9
        
        axs_miss[d].set_ylim(0.0, y_lim)
        axs_miss[d].set_xlabel("")
        axs_miss[d].set_ylabel("")
        if (d == 0) & (show_legend):
            axs_miss[d].legend(loc ="upper left", title="bin")
        else:
            axs_miss[d].legend().remove()
        axs_miss[d].grid()
        axs_miss[d].set_title(dataset, fontweight="bold", fontsize=18)
        
    fig_miss.supxlabel("feature", fontsize=12, fontweight="bold")
    fig_miss.supylabel("single missingness\n", fontsize=12, fontweight="bold")

    fig_miss.tight_layout() #TODO Induction of MNAR Pattern while resampling!!!

    fig_miss.savefig("./plots/missing_rate_per_value_bin_{}_{}_.png".format(clean if direction == "normal" else (clean + "_reverse_"), pattern.lower()), dpi=300)
    
    
WINDOW = 10

import numpy as np
from scipy.stats import norm

def get_likelihoods(df):
    
    window_len = WINDOW

    res_temp_window = {}
    likelihoods = []

    for seq_id in df.sequence_id.unique():
        for h_index in range(0, len(df[df.sequence_id == seq_id])+1, int(window_len/2)):
            window = df[h_index: h_index+window_len]
            data = np.array(window[FEATURES])
            if np.sum(np.sum(data)) < int(window_len/2):
                continue
            if not any([1 < d for d in list(np.sum(data, axis=1))]): #TODO
                continue
            m,s = norm.fit(data)
            likelihood = np.product(norm.pdf(data,m,s))
            likelihoods.append(likelihood)
            res_temp_window[len(likelihoods)] = {
                "sequence_id": seq_id,
                "h_index": h_index,
                "likelihood": likelihood,
                "data": data,
                "ones": np.sum(np.sum(data))
            }
    
    return pd.Series(likelihoods), res_temp_window    


def matching_sequences(sim_likelihoods, orig_likelihoods): 
    direct_match = {}

    for sim in pd.Series(sim_likelihoods).unique():
        # 1st order direct match
        for org in pd.Series(orig_likelihoods).unique():
            if sim == org:
                print(sim)#TODO
                direct_match[len(direct_match)] = {"sim": sim, "org": org}
                
    for sim in pd.Series(sim_likelihoods).unique():
        # 2nd order indirect match 
        for org in pd.Series(orig_likelihoods).unique():
            if org not in [direct_match[s]["org"] for s in direct_match.keys()]:
                if sim not in [direct_match[s]["sim"] for s in direct_match.keys()]:
                    if round(org, 9) == round(sim, 9):
                        direct_match[len(direct_match)] = {"sim": sim, "org": org}
                    
    return direct_match
    
def plot_five_example_pattern(clean = "pres", pattern = "mcar"):
    
    list_resampled_sequences  = []

    for link in ["sequence_1_mimic_{}.csv".format(clean), "sequence_2_hirid_{}.csv".format(clean), "sequence_2_icdep_{}.csv".format(clean)]:
        list_resampled_sequences.append(pd.read_csv("./resampled_sequences/" + link, index_col = 0))
        
    list_simulated_sequences = []

    for dataset in DATASETS:
        list_simulated_sequences.append(pd.read_csv("./induced_sequences/{}/missing_matrix_{}_boot_0_{}_pres.csv".format(dataset.lower(), dataset.lower(), pattern), index_col = 0))
        
    import warnings
    warnings.filterwarnings("ignore")
    
    matches = {}

    for d, dataset in enumerate(DATASETS):
        
        print(dataset)

        original_miss = load_resampled_sequences(clean=clean)[d]
        original_miss[FEATURES] = original_miss[FEATURES].isna().astype(int)
        simulated_miss = list_simulated_sequences[d]

        orig_likelihoods, orig_windows = get_likelihoods(original_miss.copy())
        sim_likelihoods, sim_windows = get_likelihoods(simulated_miss.copy())
        
        # get matching pairs
        matches[dataset] = {
            "map": matching_sequences(sim_likelihoods, orig_likelihoods),
            "orig_windows": orig_windows,
            "sim_windows": sim_windows
        }
        
        # retrieve the windows that needs to be plotted 

    f, axs = plt.subplots(5, 6, figsize=(19, 9), sharex=False, sharey=True)
    cmaps = ["Blues", "Greens", "Oranges"]

    for d, dataset in enumerate(DATASETS):
        j = 2 * d
        for e, km in enumerate(matches[dataset]["map"].keys()):
            match = matches[dataset]["map"][km]
            # get windows 
            orig_w = [matches[dataset]["orig_windows"][k] for k in matches[dataset]["orig_windows"].keys() 
                    if matches[dataset]["orig_windows"][k]['likelihood'] == match["org"]][0]
            sim_w = [matches[dataset]["sim_windows"][k] for k in matches[dataset]["sim_windows"].keys() 
                    if matches[dataset]["sim_windows"][k]['likelihood'] == match["sim"]][0]
            
            sns.heatmap(orig_w["data"].T, cmap=cmaps[d], annot=True, linewidth=.5, linecolor="grey", cbar=False, ax=axs[e][j])
            sns.heatmap(sim_w["data"].T, cmap=cmaps[d], annot=True, linewidth=.5, linecolor="grey", cbar=False, ax=axs[e][j+1])
            
            axs[e][j].set_xticklabels([])
            axs[e][j+1].set_xticklabels([])
            axs[e][j].set_yticklabels(FEATURES, rotation=0.45)
            
            if e == 4:
                break
                 
        for t, sample_type in enumerate(["Original", "Simulated"]):
            axs[0][j+t].set_title("{} Patterns in {}".format(sample_type, dataset), fontsize=12, fontweight="bold")
            axs[e][j+1].set_xticklabels(list(range(1, WINDOW+1)))
            axs[e][j].set_xticklabels(list(range(1, WINDOW+1)))  
            
    f.supxlabel("Time Index")
    f.supylabel("Feature\n")
    f.tight_layout(pad=1)

    f.savefig("/home/giesan/mi_icu/spatio_temporal_pattern/plots/five_examples_orig_sim.png", dpi=400)
        
        
def plot_occurrence_states(pattern = "mcar", clean = "drop"):

    STATE_MODE = ["all", "single"]
    sorted_states = pd.read_csv("/home/giesan/mi_icu/spatio_temporal_pattern/sorted_states_for_heatmaps.csv", dtype=object)

    datasets = DATASETS

    ordering_states = []

    for mode in STATE_MODE:
            fig_occ, axs_occ = plt.subplots(1, 3, figsize=(12, 12))
            for d, dataset in enumerate(datasets):
                    
                    bootstrapped_sets = []
                    for b in range(0, 5):
                            
                            df_miss_sim_boot = pd.read_csv("/home/giesan/mi_icu/spatio_temporal_pattern/induced_sequences/{}/missing_matrix_{}_boot_{}_{}_{}.csv".format(
                                    dataset.lower(), dataset.lower(), str(b), pattern, clean), index_col = 0)

                            # condition on what kind of states come after 0000 for transition matrix
                            df_miss_sim_boot['occurrence'] = df_miss_sim_boot['hr'].astype(int).astype(str)\
                                                            + df_miss_sim_boot['bp_sys'].astype(int).astype(str)\
                                                            + df_miss_sim_boot['bp_dia'].astype(int).astype(str)\
                                                            + df_miss_sim_boot['spo2'].astype(int).astype(str)\
                                                            + df_miss_sim_boot['rr'].astype(int).astype(str)
                                                            
                            df_miss_sim_boot["occurrence_shift"] = df_miss_sim_boot.groupby("sequence_id")["occurrence"].shift(1)
                            
                            df_transition_boot = df_miss_sim_boot[df_miss_sim_boot.occurrence == "00000"]\
                                    ["occurrence_shift"].value_counts(normalize=True).reset_index()
                            
                            if mode == "all":
                                    bootstrapped_sets.append(df_miss_sim_boot["occurrence"]\
                                            .value_counts(normalize=True)\
                                                    .reset_index()[["occurrence", "proportion"]])
                            
                            if mode == "single":
                                    bootstrapped_sets.append(df_transition_boot\
                                                    .reset_index().rename(columns={"occurrence_shift": "occurrence"})\
                                                            [["occurrence", "proportion"]])
                                    

                    # get mean per occurrence pattern 
                    resampled = pd.concat(bootstrapped_sets, axis= 0).groupby("occurrence")\
                            ["proportion"].agg(['mean', 'sem']).reset_index()
                    resampled['sampled_ci95_hi'] = resampled['mean'] + 1.96* resampled['sem']
                    resampled['sampled_ci95_lo'] = resampled['mean'] - 1.96* resampled['sem']
                            
                    # compare resampled occurrences with original ones 
                    if mode == "all":
                            original = pd.read_csv("./spatio_temporal_pattern/occurrence_matrices/matrix_{}_{}.csv".format(dataset.lower(), clean))
                            
                            #print(original) #TODO ORIGINAL IN ICDEP??? 
                            
                    if mode == "single":
                            original = pd.read_csv("./spatio_temporal_pattern/transition_matrices/{}/matrix_{}_00000_{}.csv".format(dataset.lower(), dataset.lower(), clean))
                            
                    original['occurrence'] = original['hr'].astype(int).astype(str)\
                                            + original['bp_sys'].astype(int).astype(str)\
                                            + original['bp_dia'].astype(int).astype(str)\
                                            + original['spo2'].astype(int).astype(str)\
                                            + original['rr'].astype(int).astype(str)
                                            
                    original["original"] = original["proportion"]
                    
                    resampled['sampled_mean'] = resampled['mean']
                    
                    # where no CI possible lack of data
                    resampled['sampled_ci95_hi'] = resampled['sampled_ci95_hi'].fillna(resampled['mean'])
                    resampled['sampled_ci95_lo'] = resampled['sampled_ci95_lo'].fillna(resampled['mean'])
                                    
                    df_plot = sorted_states[["occurrence"]].merge(original[["occurrence", "original"]], on="occurrence", how="outer")\
                            .merge(resampled[["occurrence", "sampled_ci95_lo", "sampled_mean", "sampled_ci95_hi"]], on="occurrence",
                            suffixes=["_org", "_sampl"], how="outer").set_index("occurrence")
                            
                    df_plot['sampled_ci95_hi'] = df_plot['sampled_ci95_hi'].fillna(df_plot['original'])
                    df_plot['sampled_ci95_lo'] = df_plot['sampled_ci95_lo'].fillna(df_plot['original'])
                    df_plot['sampled_mean'] = df_plot['sampled_mean'].fillna(df_plot['original'])
                    
                    df_plot['original'] = df_plot['sampled_mean']
                    
                    df_plot = df_plot.rename(columns={
                            "sampled_ci95_hi": "sim_ci95_hi",
                            "sampled_ci95_lo": "sim_ci95_lo",
                            "sampled_mean": "sim_mean",})
                            
                    sns.heatmap(df_plot.abs(), norm=LogNorm(), ax=axs_occ[d], annot=True, cbar=None, cmap="viridis", fmt='.2g', linewidths=0.5, linecolor="white") 
                    axs_occ[d].set_ylabel("")
                    
                    ordering_states.append(df_plot)
                    
                    axs_occ[d].set_title(dataset, fontweight="bold", fontsize=18)
                    axs_occ[d].tick_params(axis='x', labelrotation=45)
                    
            
            if mode == "all":
                    fig_occ.supylabel("occurrence states\n", fontsize=14, fontweight="bold")
                    title = "            Probabilities of 32 occurrence states for original - and simulated missingness\n"
            else: 
                    fig_occ.supylabel("transition states\n", fontsize=14, fontweight="bold")
                    title =  "           Probabilities of 32 transition states subsequent to 00000 for original - and simulated missingness\n"
                    

            fig_occ.suptitle(title, fontsize=14, fontweight="bold")

            fig_occ.tight_layout()

            fig_occ.savefig("/home/giesan/mi_icu/spatio_temporal_pattern/plots/occurrence_states_sampled_boot_{}_{}.png".format(mode, clean), dpi=300)
            
            #pd.concat(ordering_states).reset_index().groupby("occurrence")["original"]\
            #        .mean().fillna(0).reset_index().sort_values(by="original", ascending=False)\
            #                .to_csv("/home/giesan/mi_icu/spatio_temporal_pattern/sorted_states_for_heatmaps.csv")



def plot_feature_miss_per_dataset(datasets = ["MIMIC", "HIRID", "ICDEP"], intervals = list(range(1, 9))):
    
    """
    plot missingness rates per dataset depending on sampling interval
    
    paremeters:
        datasets (dataset): list of datasets MIMIC or HIRID (ICDEP)
        intervals: list of sampling intervals 
    
    """
    
    sampling = {"mimic": 60, "icdep": 5, "hirid": 10}

    f, ax = plt.subplots(len(datasets), 1, figsize=(20, 1.8 * len(intervals)))

    df_meta = pd.read_excel("./desc/common_feat_miss.xlsx")
    comm_features = [   "rr",
                        "temp",
                        "bp_dia",
                        "bp_map",
                        "bp_sys",
                        "spo2",
                        "hr"]

    for i, s in enumerate(datasets):
        df_miss = pd.read_csv("./desc/{}_single_miss_rates.csv".format(s.lower()))
        df_miss = df_miss[df_miss.feature.isin(df_meta["{}_org".format(s.lower())])].sort_values("miss_rate_1", ascending = True)
        
        print(s)
        
        g = sns.heatmap(df_miss[["miss_rate_{}".format(str(i)) for i in intervals]].T,
                        cbar_kws={"shrink": 0.9}, 
                        vmin=0, vmax=1, annot_kws={"size": 16}, ax=ax[i])
        
        cbar = ax[i].collections[0].colorbar
        # here set the labelsize by 20
        cbar.ax.tick_params(labelsize=20)
        
        df_merge = df_meta.rename(columns={"{}_org".format(s.lower()): "feature"})\
            .merge(df_miss, on = "feature")
            
        ax[i].set_xticklabels(list(df_merge["{}_abbr".format(s.lower())]), rotation=45, size=18)
        ax[i].set_yticklabels([sampling[s.lower()] * i for i in list(intervals)], rotation=0, size=18)
        ax[i].set_title("{} dataset".format(s), fontsize=26)
        
        for label in ax[i].get_xticklabels():
            if label.get_text() in comm_features:
                label.set_weight("bold")

    f.supylabel("           sampling interval [min.]\n", size=26, fontweight="bold")
    f.supxlabel("feature                    ", size=26, fontweight="bold")
    f.suptitle("Feature Missingness of Features per Dataset           \n", fontweight="bold", fontsize=26)
    f.tight_layout()
    f.savefig("./home/giesan/mi_icu/spatio_temporal_pattern/plots/common_miss.png", dpi=400) #TODO 

    df_res = []
    for i, s in enumerate(datasets):
        df_meta = pd.read_excel("./desc/common_feat_miss.xlsx")
        df_miss = pd.read_csv("./desc/{}_single_miss_rates.csv".format(s.lower()), index_col = 0)
        df_meta = df_meta.rename(columns={"{}_org".format(s.lower()): "feature"}).rename(columns={"{}_abbr".format(s.lower()): "abbr"})  
        df_meta = df_meta[df_meta.abbr.isin(comm_features)]
        df_merge = df_meta\
                .merge(df_miss, on = "feature").assign(dataset = s)
                
        df_res.append(df_merge[["dataset", "abbr", "miss_rate_1", "miss_rate_2", "miss_rate_3", "miss_rate_4", "miss_rate_5", "miss_rate_6", "miss_rate_7", "miss_rate_8"]])

    pd.concat(df_res).to_csv("/home/giesan/mi_icu/spatio_temporal_pattern/desc/common_miss_rates.csv")
    


plot_mnar_feature_values(clean = "pres", pattern = "mnar", show_legend=False, direction="reverse")