import pandas as pd
import numpy as np
import torch
import ray
import argparse

import pytorch_lightning as pl

from torch.utils.data import Dataset, DataLoader, RandomSampler
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.metrics import mean_squared_error, mean_absolute_error

from ray import tune
from ray import train
from ray.tune.schedulers import ASHAScheduler
from ray.train.torch import TorchTrainer
from ray.train.lightning import (
    RayDDPStrategy,
    RayLightningEnvironment,
    RayTrainReportCallback,
    prepare_trainer,
)

from ray.train import RunConfig, ScalingConfig, CheckpointConfig

import torch.optim as optim
from tqdm.auto import tqdm


DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(DEVICE)

# fixed non-tunable parameters

FEATURES = ['hr', 'bp_sys', 'bp_dia', 'spo2', 'rr']

# data config parameters
CLEAN = "pres"
PATTERN = "mcar"
DATASET = "mimic"

# missing noise and normalization
MISSING_NOISE = 0
NORMALIZE = True

# parameters part of saved file name
PRE_IMPUTE = False
IS_GRU = True
BIDIRECTIONAL = True
BOOT = 0

NUM_TUNE_SAMPLES = 10
TUNING = False #TODO
WITH_STATIC = True
SELECT_CV = 0

# parse arguments that must be passed like
import sys
DATASET = sys.argv[1]
PRE_IMPUTE = (sys.argv[2] == "True")
IS_GRU = (sys.argv[3]  == "True")
BIDIRECTIONAL = (sys.argv[4]  == "True")
SELECT_CV = int(sys.argv[5]) #TODO ALSO WITH STATIC!!!

# split and max epochs TODO MAYBE ADAPT MAX EPOCH FOR TUNING 
EPOCHS = 100 #TODO

# trainable parameters
N_MODULES = 1
LATENT_SIZE = 64
DROP_OUT = 0.5
L2_PENALTY = 0
LR = 1e-4
BATCH = 64

# default parameters without hyperparameter tuning
parameters = {
        "N_MODULES" : N_MODULES,
        "LATENT_SIZE" : LATENT_SIZE,
        "EPOCHS" : EPOCHS,
        "L2_PENALTY" : L2_PENALTY, 
        "MISSING_NOISE": MISSING_NOISE,
        "PRE_IMPUTE": PRE_IMPUTE,
        "IS_GRU": IS_GRU,
        "DROP_OUT": DROP_OUT,
        "BIDIRECTIONAL": BIDIRECTIONAL,
        "LR": LR,
        "DATASET": DATASET,
        "BATCH": BATCH,
        "WITH_STATIC": WITH_STATIC
    }

print("default params")
print(parameters)

def prepare_static(static_data):
    
    """ preparing the static data points for feeding into network but just for inputs """
    
    # prepared the static data
    static_data = static_data[["age", "gender", "sequence_id"]].sort_values(by="sequence_id")
    static_data = static_data.assign(gender = [-0.5 if x == "M" else 0.5 for x in static_data.gender])
    return static_data.assign(age = [(x-static_data.age.min())/(static_data.age.max()-static_data.age.min()) for x in static_data.age])

def load_data():
    
    if str(DEVICE) == "cpu":
        complete = pd.read_csv("../spatio_temporal_pattern/complete_sequences/sequence_{}_{}.csv".format(DATASET, CLEAN), index_col = 0)
        boot_occurence = pd.read_csv("../spatio_temporal_pattern/induced_sequences/{}/missing_matrix_{}_boot_{}_{}_{}.csv".format(DATASET, DATASET, BOOT, PATTERN, CLEAN), index_col = 0)

    else:
        complete = pd.read_csv("./complete_sequences/sequence_{}_{}.csv".format(DATASET, CLEAN), index_col = 0)
        boot_occurence = pd.read_csv("./induced_sequences/{}/missing_matrix_{}_boot_{}_{}_{}.csv".format(DATASET, DATASET, BOOT, PATTERN, CLEAN), index_col = 0)

    features = FEATURES

    induced_df  = pd.DataFrame(np.array(complete[features]) * np.array(boot_occurence[features].replace(1, float('NaN')).replace(0.0, 1)))
    induced_df.columns = features
    induced_df = induced_df.assign(sequence_id = complete.reset_index().sequence_id)

    complete = complete.assign(time_index = complete.groupby("sequence_id").cumcount())
    induced_df = induced_df.assign(time_index = induced_df.groupby("sequence_id").cumcount())
    
    static_df = None
    
    if WITH_STATIC:
        static_data = pd.read_csv("./desc/static_data_{}.csv".format(DATASET.lower()))
        static_df = prepare_static(static_data=static_data)
    
    print("len complete set")
    print(len(complete))

    return complete, induced_df, static_df

# load data 
COMPLETE, INDUCED, STATIC_DATA = load_data()
print("data loaded")

import torch
from torch import nn

class TEMP_DAE(torch.nn.Module):
    
    """ the temporal autoencoder """
    
    def __init__(self, input_size, batch_first = True, n_modules = 1, size_latent = 32, drop_out = 0.5):
        super().__init__()

        self.input_size = input_size
        self.batch_first = batch_first
        self.n_modules = n_modules
        self.size_latent = size_latent
        self.size_static = 2
        
        self.drop_out = torch.nn.Dropout(p=drop_out)
        
        # configure an optional simple MLP for processing static features
        if WITH_STATIC:
            print("MLP module")
            self.static_model = nn.Sequential(
                nn.Linear(2, int(self.size_static/2)), nn.Sigmoid(), 
                nn.Linear(int(self.size_static/2), self.size_static), nn.Sigmoid())
            self.lin_stat = torch.nn.Linear((self.size_static + self.size_latent), self.size_latent)

        # configure either GRU or LSTM as encoder or decoder
        if IS_GRU:
            print("GRU modules")
            self.encoder = torch.nn.Sequential(
                nn.GRU(input_size = input_size, hidden_size = size_latent, num_layers = self.n_modules, batch_first = batch_first))
            self.decoder = torch.nn.Sequential(
                nn.GRU(input_size = size_latent, hidden_size = input_size, num_layers = self.n_modules, batch_first = batch_first))
        
        else:
            print("LSTM modules")
            self.encoder = torch.nn.Sequential(
                nn.LSTM(input_size = input_size, hidden_size = int(size_latent/ 2) if BIDIRECTIONAL \
                    else size_latent, num_layers =  self.n_modules, batch_first = batch_first, bidirectional = BIDIRECTIONAL))
            self.decoder = torch.nn.Sequential(
                nn.LSTM(input_size = size_latent, hidden_size = input_size, 
                        num_layers = self.n_modules, batch_first = batch_first, bidirectional = False))


    def activate_static(self, encoded, s):
        # feeding MLP with static data
        #print("activate static")
        #print(s.size())
        act_s = self.static_model(s)
        #print(act_s.size())
        # squeezing static so that static is at every time step in latent dimension
        sq_static = act_s.squeeze(1).repeat(encoded.size()[1],1).unsqueeze(0)
        #print(sq_static.size())
        # concat static and latent encoded vectors
        #print(sq_static.size())
        #print(encoded.size())
        cat_stat = torch.cat((sq_static, encoded), 2)
        # linear activation and transfer back to latent space 
        return self.lin_stat(cat_stat)
        
    def forward(self, x, s = None):
        
        """ forward pass of sequence x and optional static data as s"""
        
        drop_out = self.drop_out(x)
        
        if IS_GRU:
            encoded,h0 = self.encoder(drop_out)
            decoded,h0 = self.decoder(encoded)
            
        else:
            encoded,(h0,c0) = self.encoder(drop_out)
                        
            if WITH_STATIC:
                encoded = self.activate_static(s=s, encoded=encoded)
            
            decoded,(h0,c0) = self.decoder(encoded)
            
        return decoded

NUM_WORKERS = 2

# construct the data with following dimensions:
# predictor_sequence, target_sequence, sequence_id, (predictor_static ggf)

# define RNN dataset
class DataSet(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx): 
        if len(self.sequences[idx]) == 4:
            predictor_sequence, target_sequence, sequence_id, predictor_static = self.sequences[idx]
            return {
                "predictor_sequence": torch.Tensor(np.array(predictor_sequence)),
                "predictor_static":  torch.Tensor(np.array(predictor_static)), #TODO MAYBE ADD THE MISS PATTERN INFO FOR WEIGHTING THE LOSS?? 
                "target_sequence": torch.Tensor(np.array(target_sequence)),
            }
        else:
            predictor_sequence, target_sequence, sequence_id = self.sequences[idx]
            return {
                "predictor_sequence": torch.Tensor(np.array(predictor_sequence)),
                "target_sequence": torch.Tensor(np.array(target_sequence)),
            }

class DataModule(pl.LightningDataModule):
    
    """ custom Data Module """
    
    def __init__(
        self,
        train_sequences=None,
        test_sequences=None,
        batch_size=1,
    ):
        super().__init__()
        self.train_sequences = train_sequences
        self.test_sequences = test_sequences
        self.batch_size = int(batch_size)
        self.test_sampler = None

    def setup(self, stage=None):
        self.train_dataset = DataSet(self.train_sequences)
        self.test_dataset = DataSet(self.test_sequences)
        self.test_sampler = RandomSampler(self.test_dataset, replacement=True, num_samples=len(self.test_dataset))

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=NUM_WORKERS,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=NUM_WORKERS,
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=NUM_WORKERS,
        )

    def test_dataloader_sampler(self):
        return DataLoader(
            self.test_dataset, batch_size=len(self.test_dataset), shuffle=False, num_workers=NUM_WORKERS, sampler=self.test_sampler
        )

def build_tuple_seq(df_sequence_predict, df_sequence_target, features = [], id_col = "sequence_id", df_static = None):
    
    # calculate standardization metrics 
    
    list_seq_ids = df_sequence_predict[id_col].unique()
    sequences = []
    for seq_id in list_seq_ids:
        pred_sequence = df_sequence_predict[df_sequence_predict[id_col] == seq_id][features]
        target_sequence = df_sequence_target[df_sequence_target[id_col] == seq_id][features]
        assert len(target_sequence) == len(pred_sequence)
        if df_static is None:
            sequences.append((pred_sequence, target_sequence, seq_id))
        else:
            pred_static = df_static[df_static[id_col] == seq_id][["age", "gender"]]
            sequences.append((pred_sequence, target_sequence, seq_id, pred_static))
    return sequences, list_seq_ids

def flatten_tuple_seq(predicted_sequences, list_ids = [], features = [], id_col = "sequence_id"):
    
    # reconstruct the original dataset 
    dfs_reconstruction = []
    for i, seq_id in enumerate(list_ids):
        seq_i = predicted_sequences[i]
        if WITH_STATIC:
            seq_i = seq_i.squeeze(dim = 0)
        df_reconstruct = pd.DataFrame(seq_i.detach().cpu().numpy())
        df_reconstruct.columns = features
        df_reconstruct[id_col] = seq_id
        dfs_reconstruction.append(df_reconstruct)
    return pd.concat(dfs_reconstruction)

def standardize(df, S_means, S_std):

    df_st = df.copy()
    # z-scale data
    df_st[FEATURES] = (df_st[FEATURES]-S_means)/S_std
    
    return df_st

def de_standardize(sequences, S_means, S_std):
    
    sequences = sequences.copy()
    
    de_sequence = pd.DataFrame(np.array(sequences[FEATURES] * S_std) + np.array(S_means))
    de_sequence.columns = FEATURES
    
    return de_sequence.assign(sequence_id = sequences.reset_index().sequence_id)
    

class ModelTrainer(pl.LightningModule):
    
    """ model training class with logging """
    
    def __init__(self, model, criterion = None, 
                    tunable=False, lr=1e-4, 
                    with_static=False, l2_penalty=0):
        super().__init__()
        self.criterion = criterion
        self.tunable = tunable
        self.lr = lr
        self.with_static = with_static
        self.model = model
        self.losses = []
        self.l2_penalty = l2_penalty

    def forward(self, pred_seq, tar_seq, static = None, model = None):
        if self.with_static:
            out = model(pred_seq, static)
        else:
            out = model(pred_seq)
        loss = 0
        loss = self.criterion(out, tar_seq)
        #print("dim of loss")
        #print(loss.size()) #TODO TUNING??? 
        #print(loss)
        self.losses.append(loss) #TODO CALC LOSS PER BATCH??? MAYBE ALSO INTEGRATE SLICED WASSERSTEIN DISTANCE MULTI-DIM??? 
        return loss, out

    def training_step(self, batch, batch_idx):
        static = None
        pred_seq = batch["predictor_sequence"]
        tar_seq = batch["target_sequence"]
        if self.with_static:
            static = batch["predictor_static"]
        loss, out = self.forward(pred_seq, tar_seq, static, model=self.model)
        out = out.squeeze(1)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return {"loss": loss}
    
    def validation_step(self, batch, batch_idx):
        static = None
        pred_seq = batch["predictor_sequence"]
        tar_seq = batch["target_sequence"]
        if self.with_static:
            static = batch["predictor_static"]
        loss, out = self.forward(pred_seq, tar_seq, static, model=self.model)
        out = out.squeeze(1)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        return {"loss": loss}

    def testing_step(self, batch, batch_idx):
        static = None
        pred_seq = batch["predictor_sequence"]
        tar_seq = batch["target_sequence"]
        if self.with_static:
            static = batch["predictor_static"]
        if torch.cuda.is_available():
            self.model = self.model
        loss, out = self.forward(pred_seq, tar_seq, static, model=self.model)
        out = out.squeeze(1)
        self.log("test_loss", loss, prog_bar=True, logger=True)
        return {"loss": loss}
    
    def test_data(self, sequences):
        print("testing data")
        predictions = {"loss": [], "out": []}
        static = None
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        for item in tqdm(sequences):
            pred_seq = item["predictor_sequence"].cuda()
            tar_seq = item["target_sequence"].cuda()
            if self.with_static:
                print("static validate")
                static = item["predictor_static"].unsqueeze(dim=0).cuda()
                pred_seq = pred_seq.unsqueeze(dim=0).cuda()
                tar_seq = tar_seq.unsqueeze(dim=0).cuda() #TODO
                print(static.size())
            loss, out = self.forward(pred_seq, tar_seq, static, model=self.model)
            predictions["loss"].append(loss)
            predictions["out"].append(out)
            
            #print("predicted sequence")
            #print(out)
            #
            #print("target sequence")
            #print(tar_seq)
            #
            #print("loss")
            #print(loss)
            
        return predictions
            
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.l2_penalty)
    
# init data module and define train and test sets
# configure the model
# training method

def load_cv_data():
    
    """ loading split data for CV process """
    
    if (str(DEVICE) == "cpu"):
        cv_splits = pd.read_csv("/home/giesan/mi_icu/spatio_temporal_pattern/complete_sequences/cv_splits_all.csv", index_col=0)
    else:
        cv_splits = pd.read_csv("./complete_sequences/cv_splits_all.csv", index_col=0)
        
    return cv_splits[(cv_splits["DATASET"] == DATASET.upper()) & ((cv_splits["CLEAN"] == CLEAN.lower()))]
    
# preloading splitting data for hyperparameter tuning 
CV_SPLITS = load_cv_data()

def train_model(config):
    
    """ model training function with optional config files to be tuned """
    
    is_fin_train = "EPOCHS" in list(config.keys())
    
    for CV in range(0, 3):
        
        if CV != SELECT_CV:
            continue
        
        print("CV round {}".format(CV))
        
        cv_seq_ids = CV_SPLITS[CV_SPLITS["CV"] == CV]
        
        # getting training and testing data according to CV splits
        df_test_miss = INDUCED[INDUCED.sequence_id.isin(list(cv_seq_ids[cv_seq_ids["SET"] == "TEST"].sequence_id))]
        df_train_miss = INDUCED[INDUCED.sequence_id.isin(list(cv_seq_ids[cv_seq_ids["SET"] == "TRAIN"].sequence_id))]
        
        df_train_complete = COMPLETE[COMPLETE.sequence_id.isin(list(cv_seq_ids[cv_seq_ids["SET"] == "TRAIN"].sequence_id))]
        df_test_complete = COMPLETE[COMPLETE.sequence_id.isin(list(cv_seq_ids[cv_seq_ids["SET"] == "TEST"].sequence_id))]
        
        df_test_complete_orig = df_test_complete.copy()
        df_test_miss_orig = df_test_miss.copy()
        
        print("test set lengths ")
        print(len(df_test_miss))
        print(len(df_test_complete))
        
        print("train set lengths ")
        print(len(df_train_miss))
        print(len(df_train_complete))
        
        # calculating standardization metrics on training data #TODO THINK ABOUT STANDARDIZATION!!!
        mean_train_miss, mean_train_complete = df_train_miss[FEATURES].mean(), df_train_complete[FEATURES].mean()
        std_train_miss, std_train_complete = df_train_miss[FEATURES].std(), df_train_complete[FEATURES].std()
        
        # normalizing data with summary statistics from training only 
        if NORMALIZE:
            df_train_miss = standardize(df=df_train_miss, S_means=mean_train_complete, S_std=std_train_complete)
            df_test_miss = standardize(df=df_test_miss, S_means=mean_train_complete, S_std=std_train_complete)
            df_train_complete = standardize(df=df_train_complete, S_means=mean_train_complete, S_std=std_train_complete)
            df_test_complete = standardize(df=df_test_complete, S_means=mean_train_complete, S_std=std_train_complete)
            
        if PRE_IMPUTE:
            df_train_complete = df_train_miss.ffill().fillna(0.0)
            df_test_complete = df_test_miss.ffill().fillna(0.0)
        
        # building the training set
        preprocess_train, list_seq_ids_train = build_tuple_seq(df_sequence_predict=df_train_miss.fillna(MISSING_NOISE), 
                                                            df_sequence_target=df_train_complete, features=FEATURES, 
                                                            df_static= STATIC_DATA if WITH_STATIC else None) #TODO static 
        preprocess_test, list_seq_ids_test = build_tuple_seq(df_sequence_predict=df_test_miss.fillna(MISSING_NOISE), 
                                                            df_sequence_target=df_test_complete, features=FEATURES,
                                                            df_static= STATIC_DATA if WITH_STATIC else None) #TODO static
    
        print("one preprocessed train input")
        print(preprocess_train[1][0])
        print("one preprocessed train output")
        print(preprocess_train[1][1])
    
        count_seq_train = len(list_seq_ids_train)
        count_seq_test = len(list_seq_ids_test)

        print("training for ", count_seq_train)
        print("testing for ", count_seq_test)

        model = TEMP_DAE(input_size=len(FEATURES), 
                        batch_first=True, n_modules=config["N_MODULES"], 
                        size_latent=config["LATENT_SIZE"], drop_out=config["DROP_OUT"])

        # init data module
        data_module = DataModule(
            batch_size=1, 
            train_sequences=preprocess_train,
            test_sequences=preprocess_test)
        
        # init model trainer
        model_trainer = ModelTrainer(model=model, tunable=False, criterion=torch.nn.MSELoss(), #TODO DOES NOT FIT ANYMORE??? WITH STATIC??? 
                                    l2_penalty=config["L2_PENALTY"], lr=config["LR"], with_static=WITH_STATIC)

        
        batch = config["BATCH"]

        # model checkpoint for best model with default
        if is_fin_train:
            checkpoint_callback = ModelCheckpoint(
                            dirpath="./checkpoints/",
                            filename="best_checkpoint",
                            verbose=True,
                            save_top_k=1,
                            monitor="val_loss",
                            mode="min")
            trainer = pl.Trainer(
                accumulate_grad_batches=batch, 
                max_epochs = EPOCHS, 
                log_every_n_steps = 20,
                devices = 1,
                callbacks=[checkpoint_callback]
            )
        # config with ray hyperparameter tuning
        else:
            trainer = pl.Trainer(
                accumulate_grad_batches=batch,
                devices="auto",
                accelerator="auto",
                strategy=RayDDPStrategy(),
                callbacks=[RayTrainReportCallback()],
                plugins=[RayLightningEnvironment()],
                enable_progress_bar=False,
            )
            trainer = prepare_trainer(trainer)
            
        # fit trainer wither with ray or normally
        trainer.fit(model_trainer, data_module)
    
        # break after fitting for hyperparameter tuning only
        if not is_fin_train:
            return
    
        # predict
        torch.save(model.state_dict(), "./best_gru_ae/best_model_{}.pt".format("gru" if IS_GRU else ("bilstm" if BIDIRECTIONAL else "lstm")))

        # checkpoint_callback.best_model_path
        best_trainer = ModelTrainer.load_from_checkpoint(trainer.checkpoint_callback.best_model_path, model=model.cuda()) #TODO int?? with_static = WITH_STATIC 
        
        
        # config best trainer
        best_trainer.freeze()
        best_trainer.criterion = torch.nn.MSELoss()
        best_trainer.with_static = WITH_STATIC

        pred_train = best_trainer.test_data(data_module.train_dataloader().dataset)
        pred_test = best_trainer.test_data(data_module.test_dataloader().dataset)

        
        # reconstruction and de-normalization
        rec_test = flatten_tuple_seq(predicted_sequences=pred_test["out"], list_ids=list_seq_ids_test, features=FEATURES)
        rec_train = flatten_tuple_seq(predicted_sequences=pred_train["out"], list_ids=list_seq_ids_train, features=FEATURES)
            
        print("len rec test ", len(rec_test))
        print("len rec train ", len(rec_train))
        
        import gc
        gc.collect()
        torch.cuda.empty_cache()

        if NORMALIZE:
            rec_test = de_standardize(sequences=rec_test, S_means=mean_train_complete, S_std=std_train_complete)
            rec_train = de_standardize(sequences=rec_train, S_means=mean_train_complete, S_std=std_train_complete)

        # res = pd.concat([rec_test, rec_train])
        res = rec_test
    
        print("prediction results")
        print(res.head())
        
        # reconstruction of prediction results
        print("replacing miss")
        res["time_index"] = res.groupby("sequence_id").cumcount()
    
        imputed = df_test_miss_orig
        imputed = imputed.sort_values(["sequence_id", "time_index"]).reset_index(drop=True)
        res = res.sort_values(["sequence_id", "time_index"]).reset_index(drop=True)
        imputed[imputed.isnull()] = res

        print("len imputed")
        print(len(imputed))
        
        print("imputed file")
        print(imputed.head())
    
        # get errors
        nrms, nmaes = [], []
        
        assert len(df_test_complete_orig) == len(imputed)
        
        for f in FEATURES:
            f_complete, f_imp, norm = df_test_complete_orig[f], imputed[f], np.mean(df_test_complete_orig[f])
            nrms.append(np.sqrt(mean_squared_error(f_complete, f_imp)) / norm)
            nmaes.append(mean_absolute_error(f_complete, f_imp) / norm)
            
            
        if (str(DEVICE) != "cpu") & (is_fin_train):
            path = "./{}/{}{}_{}_{}_{}_cv_{}.csv".format(
                DATASET.lower(),
                "gruae" if IS_GRU else ("bilstae" if BIDIRECTIONAL else "lstae"),
                ("preimp" if PRE_IMPUTE else "") + ("stat" if WITH_STATIC else ""),
                BOOT,
                PATTERN,
                CLEAN,
                CV
                )
            imputed.to_csv(path)
            
            print(path)
            
            parameters["DATASET"] = DATASET
            
            print("Current db: ", DATASET)
            print("baselines:")
            print("MIMIC global NRMSE: 0.043, NMAE: 0.010")
            print("HIRID global NRMSE: 0.202, NMAE: 0.021")
            print("ICDEP global NRMSE: 0.157, NMAE: 0.014")
            print("mean NRMSE: ", np.mean(nrms))
            print("mean NMSE: ", np.mean(nmaes))
            print("parameters ")
            print(parameters)

# tune search space configuration
CONFIG_SPACE = {
        "LATENT_SIZE": tune.choice([32, 64, 128, 256]), #TODO INCREASING LATENT FOR HIRID??? 
        "N_MODULES": tune.choice([1, 2]),
        "DROP_OUT": tune.choice([0.5, 0.75, 0.9]),
        "LR": tune.choice([1e-4,1e-5,1e-6]),
        "L2_PENALTY": tune.choice([0, 1e-1, 1e-2]),
        "BATCH": tune.choice([64, 32, 128])
    }

def ray_hypertune():
    
    """ function for ray hyperparameter tuning """
    
    print("start ray tuning")
    
    ray.init(ignore_reinit_error=True, num_cpus=2, num_gpus=2)
    
    # config hyperband like scheduler with successive halving
    scheduler = ASHAScheduler(max_t=EPOCHS, grace_period=1, reduction_factor=2)

    # setup tuning environment
    scaling_config = ScalingConfig(
        num_workers=1, use_gpu=True, resources_per_worker={"CPU": 1, "GPU": 2}
    )

    # init run config to access resources
    run_config = RunConfig(
        checkpoint_config=CheckpointConfig(
            num_to_keep=2,
            checkpoint_score_attribute="val_loss",
            checkpoint_score_order="min",
        ),
    )
    
    # configure tuning Trainer 
    ray_trainer = TorchTrainer(
        train_model,
        scaling_config=scaling_config,
        run_config=run_config,
    )
    
    tuner = tune.Tuner(
        ray_trainer,
        param_space={"train_loop_config": CONFIG_SPACE},
        tune_config=tune.TuneConfig(
            metric="val_loss",
            mode="min",
            num_samples=NUM_TUNE_SAMPLES,
            scheduler=scheduler,
        ),
    )
    
    print("tuner config")
    
    return tuner.fit()

if TUNING:
    # hyperparameter tuning and fitting
    results = ray_hypertune()

    # get best hyperparameters
    best_conf = results.get_best_result(metric="val_loss", mode="min").config

    # get best params and overwrite default param map
    for key in CONFIG_SPACE.keys():
        parameters[key] = best_conf['train_loop_config'][key]
    
    
    print("best params")
    print(parameters)

# train model with final parameters
train_model(parameters)