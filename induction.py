
import pandas as pd
import numpy as np
import math
from numpy import random
from functools import reduce
from static_variables import DATASETS, FEATURES, load_resampled_sequences, load_complete_sequences

def getting_occurrence_transition_states():
    
    """ calculating occurrence and transition states """

    for clean in ["pres", "drop"]:

        list_resampled_sequences = load_resampled_sequences(clean)

        for d, dataset in enumerate(DATASETS):
            
            print("getting missing states for ", dataset)
            
            df_sequence = list_resampled_sequences[d]
            miss = df_sequence[FEATURES].isna().astype(int)\
                .assign(sequence_id = df_sequence.sequence_id)
            # calculation of transition states
            transitions = miss.astype(str) + miss.shift(periods=1, axis="rows")\
                .fillna(0).astype(int).astype(str)
                
            # transition states must be derived per sequence and not globally
            transitions["drop"] = [True if x[0:int(len(x)/2)] != x[int(len(x)/2): int(len(x))] \
                else False for x in transitions.sequence_id]
            
            # dropping all first sequence transitions
            transitions = transitions[transitions["drop"] == False]
            
            # get occurrence prob. of resampled data
            occurrence_prob = miss[FEATURES].value_counts(normalize=True).reset_index() 
            
            # iterating through all occurrance probabilities
            conditioned_metrices = {}
            for i, occ in occurrence_prob.iterrows():
                trans_filter_matrix = []
                for f, feat in enumerate(FEATURES):
                    # important condition filter for the transition matrices 
                    feat_filter = transitions[feat].str.startswith(occ[FEATURES][feat].astype(int).astype(str))
                    trans_filter_matrix.append(pd.DataFrame({feat: list(feat_filter)}))
                trans_filtered = transitions[FEATURES].loc[np.array(pd.concat(trans_filter_matrix, axis=1).min(axis=1))]
                key = "".join(list(occ[FEATURES].astype(int).astype(str)))
                conditioned_metrices[key] = trans_filtered.value_counts(normalize=True).reset_index()
                trans_filtered.value_counts(normalize=True).reset_index().to_csv("./transition_matrices/{}/matrix_{}_{}_{}.csv".format(dataset.lower(), dataset.lower(), key, clean))
            
            # investigate occurrence matrices for different age groups    
            print(occurrence_prob.head())
            
            occurrence_prob.to_csv("./occurrence_matrices/matrix_{}_{}.csv".format(dataset.lower(), clean))


def correct_roundings(prob_matrix, prob_type = "occurence"):
    
    """ correction of rounding errors that might occur """
    
    condition_col = "state" if prob_type == "occurence" else "transition_to"
    
    prob_matrix = prob_matrix.round(7)
    
    prob_matrix["proportion"] = prob_matrix["proportion"].astype(float)
    
    if prob_matrix["proportion"].sum() != 1:
    
        diff = 1 - float(prob_matrix.proportion.sum())
        prob_matrix.loc[prob_matrix[condition_col] == "00000", "proportion"] = \
            float(prob_matrix.loc[prob_matrix[condition_col] == "00000", "proportion"].iloc[0] + diff)
        
    return prob_matrix

def mar_condition(prob_matrix, age_group, prob_type = "occurence", direction="reverse"):
    
    """ manipulating the 1st or 2nd order prob matrix MAR induction

        prob_type: either occurrence or transition type
        
        direction: "reverse" or "normal"
    
    """
    
    condition_col = "state" if prob_type == "occurence" else "transition_to"
    
    # determine age factor for resampling strategy
    age_factor = [0.6, 0.2, 0.0][age_group]
    
    if direction == "reverse":
        #print("reverse age groups")
        age_factor = [0.0, 0.2, 0.6][age_group]
    
    # if all absent not in prob then leave unmodified
    if "11111" not in list(prob_matrix[condition_col]):
        return prob_matrix
    
    # manipulate 0000 states and 1111 states
    all_present = prob_matrix[prob_matrix[condition_col] == "00000"]
    all_absent = prob_matrix[prob_matrix[condition_col] == "11111"]

    # get probabilities
    all_present_prob = all_present.iloc[0]["proportion"]
    all_absent_prob = all_absent.iloc[0]["proportion"]

    # assign new proportion
    all_present= all_present.assign(proportion = float(all_present_prob) - float(all_present_prob) * float(age_factor))
    all_absent= all_absent.assign(proportion = float(all_absent_prob) + float(all_present_prob) * float(age_factor))

    return pd.concat(
        [
            all_present,
            all_absent,
            prob_matrix[~prob_matrix[condition_col].isin(["00000", "11111"])]
        ]
    )
    
    
def mnar_condition(complete_value_ranges, dataset, all_miss_matrix, direction):
    
    """ induction of MNAR condition """
    
    if direction == "reverse":
        range_factors = [3.5, 2.5, 1] #  [5, 3.5, 2] MIMIC
    else:
        range_factors = [1, 1.5, 3]
    
    # get value ranges per miss pattern
    value_ranges_dataset = complete_value_ranges[(complete_value_ranges.dataset == dataset)]
    miss_ranges = all_miss_matrix.merge(value_ranges_dataset[[f + "_bin" for f in FEATURES] + ["sequence_id", "time_index"]],
                                        on=["sequence_id", "time_index"], how="left").sort_values(by=["sequence_id", "time_index"]).fillna(0)
    
    print("len miss ranges ", len(miss_ranges))
    
    print(miss_ranges)
    
    dfs_mnar_all = []
    
    for feature in FEATURES:
        
        miss_ranges[feature] = miss_ranges[feature].astype(int)
        
        # get baseline levels and multiply with age factors
        miss_levels = miss_ranges.groupby(feature + "_bin")[feature].mean().reset_index()
        miss_levels[feature] = [range_factors[j] * m for j, m in enumerate(miss_levels[feature])]
        
        print("len miss ranges ", len(miss_ranges))
        
        #print(miss_levels)
        
        dfs_mnar_feat = []

        for bin in range(0, 3):
            range_bin = miss_ranges[miss_ranges[feature + "_bin"] == bin]
            level = miss_levels[miss_levels[feature + "_bin"] == bin][feature].iloc[0]
            print("orig level")
            print(level)
            
            # setting the level of missingness according to bin assignment     
            if ((level == 1) and (direction != "reverse")): #TODO maybe level??? at reverse??? 
                level = 0.8
            
            if ((level != 0) & (direction != "reverse")) or ((level != 2) & (direction == "reverse")): #TODO maybe level at reverse???  WHAT HAPPENS HERE???? 
                if level <= 0:
                    level = 0.01 
                if level >= 1:
                    level = 0.99
                print("sampling")
                print(1-level)
                print(level)
                mnar_miss = random.choice([0, 1], len(range_bin), p=[1-level, level])
                range_bin[feature] = [max(r, m) for r, m in zip(range_bin[feature], mnar_miss)]
                                                    
            dfs_mnar_feat.append(range_bin[[feature, feature + "_bin", "sequence_id", "time_index"]])
        
        dfs_mnar_all.append(pd.concat(dfs_mnar_feat))
        
    return reduce(lambda x, y: x.merge(y, 
                        on=["sequence_id", "time_index"]), dfs_mnar_all)
                    
    
def splitting_transitions(state, index = 0):
    
    """ splitting transition states up in from and to """
    
    return state[index] + state[index+2] + state[index+4] + state[index+6] + state[index+8]

def assign_state_combined(df):
    
    """ combining all occurrence states to one label """
    
    return df.assign(state = df[FEATURES].astype(str).values.sum(axis=1))

def assign_transitions_combined(df):
    
    """ crating from and to transitions for matrix """
    
    df = df.assign(transition = df[FEATURES].astype(str).values.sum(axis=1))
    
    # condition[FEATURES].astype(str).str.slice(1,2)
    
    df = df.assign(transition_from = [splitting_transitions(state, 0) for state in df.transition])
    df = df.assign(transition_to = [splitting_transitions(state, 1) for state in df.transition])
    
    return df
    

def missing_induction(clean = "drop", pattern = "mcar", n_boot = 5, direction="normal"): #TODO for ICDEP!!! and also bootstrapps!!
    
    list_complete_sequences = load_complete_sequences(clean=clean)
    
    age_groups, complete_value_ranges = None, None
    
    if pattern == "mar":
        print("load age groups")
        age_groups = pd.read_csv("./desc/age_groups_sequence.csv", index_col = 0)
        
    if pattern == "mnar":
        print("load feature ranges")
        complete_value_ranges = pd.read_csv("./desc/complete_value_ranges.csv", index_col = 0)
    
    for boot in range(0, n_boot):
        
        for d, dataset in enumerate(DATASETS):
            #TODO

            if dataset.upper() != "ICDEP":
                continue
            
            print("sampling for ", dataset)

            occurrence = pd.read_csv("./occurrence_matrices/matrix_{}_{}.csv"\
                .format(dataset.lower(), clean), index_col = 0).round(7)
            
            # get state from occurrence 
            occurrence = assign_state_combined(occurrence)
            occurrence = correct_roundings(occurrence)
            
            assert occurrence.proportion.sum() == 1

            sampled_missing_matrix_total = []

            df_complete = list_complete_sequences[d]
            occurrence_induct = occurrence
            
            print(len(df_complete))

            for j, sequence_id in enumerate(df_complete.sequence_id.drop_duplicates()):
                
                # init sample miss matrix
                sampled_missing_matrix_seq = []
                
                # iterate through all complete resampled sequences
                df_seq = df_complete[df_complete.sequence_id == sequence_id] 
                
                if pattern == "mar":
                    
                    # if missing mar get age group and alter occurrence matrix for this else get occurrence 
                    seq_age = age_groups[age_groups.dataset == dataset].merge(df_seq[["sequence_id"]], on="sequence_id").drop_duplicates()
                
                    # get age group number for gaining the age factor 
                    age_group = seq_age.iloc[0]["j"]
                
                    occurrence_induct = mar_condition(occurrence, age_group, direction=direction)
                    
                # get occurrence probabilities
                occ_prop = list(occurrence_induct.proportion)
                
                # get initial starting point index 
                sampled_state = np.random.choice(a = list(occurrence_induct.state.astype("str")), p=occ_prop)
                
                # get state 
                init_state = occurrence[occurrence.state == sampled_state]
                sampled_missing_matrix_seq.append(init_state[FEATURES].assign(sequence_id = sequence_id).assign(time_index = 0))
                
                # get the corresponding conditioned transition matrix
                for i in range(1, len(df_seq)):
                    # iterating through the sequences 
                
                    transition_matrix = pd.read_csv("./transition_matrices/{}/matrix_{}_{}_{}.csv"\
                        .format(dataset.lower(), dataset.lower(), str(sampled_state), clean), index_col = 0, dtype=object).round(7)
                                        
                    # assigning state to transition matrix
                    transition_matrix = assign_transitions_combined(transition_matrix)
                    transition_matrix = correct_roundings(transition_matrix, prob_type="transition")
                    
                    if pattern == "mar":
                        transition_matrix = mar_condition(transition_matrix, age_group, prob_type="transition")
                        
                    # sample the next transition
                    next_transition_state = np.random.choice(
                        a = list(transition_matrix.transition_to),
                        p=list(transition_matrix.proportion.astype(float))
                    )

                    # get condition of transition matrix
                    condition = transition_matrix[transition_matrix.transition_to == next_transition_state].iloc[0]
                    
                    # get induced sequential data
                    ind_seq = pd.DataFrame(condition[FEATURES].astype(str).str.slice(1,2))\
                        .T.assign(sequence_id = sequence_id)
                        
                    sampled_missing_matrix_seq.append(ind_seq)
                    # new state comb that act on the conditioned matrix
                    sampled_state = next_transition_state
                
                if j%100 == 0:
                    print("processing sequence ", j)
                
                df_miss_seq = pd.concat(sampled_missing_matrix_seq)\
                    .reset_index(drop=True).assign(time_index = df_seq.reset_index()["time_index"])

                assert len(df_miss_seq) == len(df_seq)
            
                sampled_missing_matrix_total.append(df_miss_seq)
            
            all_miss_matrix = pd.concat(sampled_missing_matrix_total)
            
            print("len induced")
            print(len(all_miss_matrix))
            
            if pattern == "mnar":  
                
                all_miss_matrix = mnar_condition(complete_value_ranges, dataset, all_miss_matrix, direction)
            
            clean_s = clean  
            if direction == "reverse":
                clean_s = clean + "_rev"
            
            print("len induced")
            print(len(all_miss_matrix))
            print("len complete")
            print(len(df_complete))
            
            assert len(df_complete) == len(all_miss_matrix)
                    
            all_miss_matrix\
                .to_csv("./induced_sequences/{}/missing_matrix_{}_boot_{}_{}_{}.csv"\
                    .format(dataset.lower(), dataset.lower(), boot, pattern, clean_s))  #TODO STORE THE WRONG???      
            
            print("./induced_sequences/{}/missing_matrix_{}_boot_{}_{}_{}.csv"\
                    .format(dataset.lower(), dataset.lower(), boot, pattern, clean_s))
            print("stored")
                
    print("Done")