
from preprocess import resample_data, build_complete
from induction import getting_occurrence_transition_states, missing_induction
from plotting import plot_age_missing_dependency, plot_feature_bins, plot_mnar_feature_values, plot_five_example_pattern
from imputation import apply_all_imputation_methods
from validation import validate_all_imputation_methods
from benchmark import benchmark

def main():
    
    # resample data and create complete case sets
    #_ = resample_data()
    #build_complete(plot=False)
    
    # creating binary matrices for occurence and transitions
    # getting_occurrence_transition_states()
    
    #missing_induction(pattern="mcar", clean="pres", n_boot=5)
    # missing_induction(pattern="mcar", clean="drop", n_boot=5)
    # missing_induction(pattern="mar", clean="pres", n_boot=1, direction="reverse")
    
    #missing_induction(pattern="mnar", clean="pres", n_boot=1, direction="reverse")
    
    # plot_age_missing_dependency(origin = "mcar", clean="drop")
    # plot_age_missing_dependency(origin = "mcar", clean="pres")
    # plot_age_missing_dependency(origin = "mar", clean="pres", show_legend = False, direction= "reverse")
    #plot_feature_bins(datasets=DATASETS, features= FEATURES)
    
    # plot_five_example_pattern(clean="pres")

    # missing_induction(pattern="mnar", clean="pres", n_boot=1)
    #missing_induction(pattern="mar", clean="drop", n_boot=1)
    
    #apply_all_imputation_methods(clean="pres", pattern="mnar") #mode="baseline"
    
    #plot_mnar_feature_values(clean = "pres", pattern = "mcar")
    #plot_mnar_feature_values(clean = "pres", pattern = "mnar", show_legend = False)
    
    #validate_all_imputation_methods(dataset = "mimic", method_select="bilstaepreimp")
    #validate_all_imputation_methods(dataset = "hirid", method_select="stgrupreimp") #TODO hirid lstaepreimp TODO 
    #validate_all_imputation_methods(dataset = "icdep", method_select="stgru")
    
    #benchmark(pattern="mcar", dataset = "hirid", is_stae_baselines=False, mode="", normalize=False)
    #benchmark(pattern="mcar", dataset = "icdep", is_stae_baselines=False, mode="", normalize=False)
    #benchmark(pattern="mcar", dataset = "hirid", is_stae_baselines=False, mode="")
    #benchmark(pattern="mcar", dataset = "icdep", is_stae_baselines=True, mode="")
    #benchmark(pattern="mar", dataset = "icdep")
    #benchmark(pattern="mar", dataset = "hirid")
    #benchmark(pattern="mnar", dataset = "mimic")
    #benchmark(pattern="mnar", dataset = "icdep")
    #benchmark(pattern="mnar", dataset = "hirid")
    
    print("finish")

if __name__ == "__main__":
    main()