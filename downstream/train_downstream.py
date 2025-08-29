
import pandas as pd
import numpy as np
import pytorch_lightning as pl
import torch.optim as optim
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader, RandomSampler
from transf_model import TimeSeriesTransformer
import torch
from losses import weighted_binary_cross_entropy
from ray import tune
from ray import train
from ray.tune.schedulers import ASHAScheduler
from ray.train.torch import TorchTrainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torchmetrics import AUROC, AveragePrecision
from ray.tune.schedulers import ASHAScheduler
from ray.train import RunConfig, ScalingConfig, CheckpointConfig
import ray
from ray.train.lightning import (
    RayDDPStrategy,
    RayLightningEnvironment,
    RayTrainReportCallback,
    prepare_trainer,
)
import random

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(DEVICE)

# fixed non-tunable parameters

FEATURES = ['hr', 'bp_sys', 'bp_dia', 'spo2', 'rr']
CLEAN = "pres"
PATTERN = "mcar"
DATASET = "icdep"
COMPLETE_ONLY = True
WITH_STATIC = False
BOOT = 0
BATCH_SIZE = 32
NUM_WORKERS = 4
EPOCHS = 100
NUM_TUNE_SAMPLES = 10
HYPERTUNE = False
CV_X_ONYL = True
CV_SELECT = 1

HYPERPARAMS = {
    "ICDEP": {
            "BATCH": 64,
            "L2_PENALTY": 0,
            "LR": 1e-4,
            "DIM_HIDDEN": 32,
            "DIM_FFN": 32,
            "MODE": "FINAL"
            },
    "MIMIC": {
            "BATCH": 32,
            "L2_PENALTY": 0,
            "LR": 1e-4,
            "DIM_HIDDEN": 32,
            "DIM_FFN": 16,
            "MODE": "FINAL"
    },
    "HIRID": {
            "BATCH": 64,
            "L2_PENALTY": 0,
            "LR": 1e-4,
            "DIM_HIDDEN": 32,
            "DIM_FFN": 32,
            "MODE": "FINAL"
    }
}

endpoint_cols = {
    "mimic": "hospital_expire_flag",
    "hirid": "death",
    "icdep": "c_target"
}

numb_timepoints = {
    "MIMIC": 12,
    "HIRID": 3 * 12,
    "ICDEP": 9 # one hour and a half???
}

RES, MODEL_NAME = [], ""

MAX_SEQ_LEN = numb_timepoints[DATASET.upper()]

class DataSet(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx): 
            predictor_sequence, target = self.sequences[idx]
            return {
                "predictor_sequence": torch.Tensor(np.array(predictor_sequence)),
                "target": torch.Tensor([target])
            }

class DataModule(pl.LightningDataModule):
    
    """ custom Data Module """
    
    def __init__(
        self,
        train_sequences=None,
        test_sequences=None,
        batch_size=BATCH_SIZE,
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


class ModelTrainer(pl.LightningModule):
    
    """ model training class with logging """
    
    def __init__(self, model, criterion = None, prevalence = 0.1,
                    tunable=False, lr=1e-4, l2_penalty=0):
        super().__init__()
        self.criterion = criterion
        self.tunable = tunable
        self.lr = lr
        self.model = model
        self.losses = []
        self.l2_penalty = l2_penalty
        self.prevalence = prevalence

    def forward(self, pred_seq, target, model = None): #TODO modify the model??? 
        out = model(pred_seq)
        loss = 0
        loss = self.criterion(out, target, self.prevalence)
        return loss, out

    def training_step(self, batch, batch_idx):
        pred_seq = batch["predictor_sequence"]
        target = batch["target"]
        #print(batch)#TODO
        loss, out = self.forward(pred_seq, target, model=self.model)
        out = out.squeeze(1)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return {"loss": loss}
    
    def validation_step(self, batch, batch_idx):
        pred_seq = batch["predictor_sequence"]
        target = batch["target"]
        #print(batch)#TODO
        loss, out = self.forward(pred_seq, target, model=self.model)
        out = out.squeeze(1)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        return {"loss": loss}

    def testing_step(self, batch, batch_idx):
        pred_seq = batch["predictor_sequence"]
        target = batch["target"]
        #print(batch)#TODO
        loss, out = self.forward(pred_seq, target, model=self.model)
        out = out.squeeze(1)
        self.log("test_loss", loss, prog_bar=True, logger=True)
        return {"loss": loss}
    
    def test_data(self, sequences, n_bootstrap=100):
        
        # Compute 95% confidence intervals
        def confidence_interval(data, confidence=0.95):
            lower = np.percentile(data, (1 - confidence) / 2 * 100)
            upper = np.percentile(data, (1 + confidence) / 2 * 100)
            return lower, upper
        
        # define metrics
        auroc_metric = AUROC()
        auprc_metric = AveragePrecision()
            
        print("testing data")
        predictions = {"loss": [], "out": [], "target": []}
        
        # calculate the predictions according to targets
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        for item in tqdm(sequences):
            if torch.cuda.is_available():
                pred_seq = item["predictor_sequence"].cuda()
                target = item["target"].cuda()
            else:
                pred_seq = item["predictor_sequence"].unsqueeze(dim=0)
                target = item["target"].unsqueeze(dim=0)
            
            # feed into model
            loss, out = self.forward(pred_seq, target, model=self.model)
            
            # store results
            predictions["loss"].append(loss)
            predictions["out"].append(out)
            predictions["target"].append(int(target))
        
        # get the metrics     
        auroc = auroc_metric(torch.tensor(predictions["out"]).unsqueeze(dim=1), torch.tensor(predictions["target"]).unsqueeze(dim=1))
        auprc = auprc_metric(torch.tensor(predictions["out"]).unsqueeze(dim=1), torch.tensor(predictions["target"]).unsqueeze(dim=1))
        
        # Convert to tensors
        outs = torch.tensor(predictions["out"])
        targets = torch.tensor(predictions["target"])

        # Bootstrapping
        auroc_scores, auprc_scores, num_samples = [], [], len(targets)

        for _ in range(n_bootstrap):
            idxs = np.random.choice(num_samples, size=num_samples, replace=True)
            sampled_outs = outs[idxs]
            sampled_targets = targets[idxs]
            
            # get the eval metrics per sample
            auroc = auroc_metric(sampled_outs, sampled_targets)
            auprc = auprc_metric(sampled_outs, sampled_targets)

            auroc_scores.append(auroc.item())
            auprc_scores.append(auprc.item())
        
        # get 95% confidence interval    
        auroc_lower, auroc_upper = confidence_interval(auroc_scores)
        auprc_lower, auprc_upper = confidence_interval(auprc_scores)

        # aggregate bootstrapping results
        bootstrapped_metrics = {
            "auroc_mean": np.mean(auroc_scores),
            "auroc_CI": auroc_upper - np.mean(auroc_scores),
            "auprc_mean": np.mean(auprc_scores),
            "auprc_CI": auprc_upper - np.std(auprc_scores)
        }
            
        # get confidence by bootstrapping with 100 samples
        
        return predictions, bootstrapped_metrics, {
            "auroc": auroc,
            "auprc": auprc
        }
            
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.l2_penalty)

def load_cv_data():
    
    """ loading split data for CV process """
    
    if (str(DEVICE) == "cpu"):
        cv_splits = pd.read_csv("/home/giesan/mi_icu/spatio_temporal_pattern/complete_sequences/cv_splits_all.csv", index_col=0)
    else:
        cv_splits = pd.read_csv("./complete_sequences/cv_splits_all.csv", index_col=0)
        
    return cv_splits[(cv_splits["DATASET"] == DATASET.upper()) & ((cv_splits["CLEAN"] == CLEAN.lower()))]

def prepare_static(static_data):
    
    """ preparing the static data points for feeding into network but just for inputs """
    
    # prepared the static data
    #static_data = static_data[["age", "gender", "sequence_id"]].sort_values(by="sequence_id")
    static_data = static_data.assign(gender = [-0.5 if x == "M" else 0.5 for x in static_data.gender])
    return static_data.assign(age = [(x-static_data.age.min())/(static_data.age.max()-static_data.age.min()) for x in static_data.age])

def build_tuple_seq(df_sequence_predict, id_col = "sequence_id", target_column = "", df_static = None):
    
    # calculate standardization metrics 
    list_seq_ids = df_sequence_predict[id_col].unique()
    sequences = []
    select_cols = FEATURES + ["age", "gender"] if WITH_STATIC else FEATURES
    for seq_id in list_seq_ids:
        single_seq = df_sequence_predict[df_sequence_predict[id_col] == seq_id].head(MAX_SEQ_LEN) # 10s
        if len(single_seq) == MAX_SEQ_LEN:
            single_stat = df_static[df_static[id_col] == seq_id]
            target = single_stat[target_column].iloc[0]
            sequences.append((single_seq[select_cols], target))
    return sequences

def standardize(df, S_means, S_std):

    df_st = df.copy()
    # z-scale data
    df_st[FEATURES] = (df_st[FEATURES]-S_means)/S_std
    
    return df_st

def load_data():
    
    """ loading data according to CV splits always complete or with respect to imp. methods """
    
    CV_SPLITS = load_cv_data()
    
    complete_cvs, imputed_cvs = [], []
        
    if str(DEVICE) == "cpu":
        complete = pd.read_csv("/home/giesan/mi_icu/spatio_temporal_pattern/complete_sequences/sequence_{}_{}.csv".format(DATASET, CLEAN), index_col = 0)
        complete = complete.assign(time_index = complete.groupby("sequence_id").cumcount())
        for CV in range(0, 3):
            imputed_cvs.append(pd.read_csv("/home/giesan/mi_icu/spatio_temporal_pattern/imputed_sequences/{}/{}_{}_{}_{}_cv_{}.csv".format(DATASET, IMPUTATION_METHOD.lower(), BOOT, PATTERN, CLEAN, str(CV)), index_col = 0))
            cv_seq_ids = CV_SPLITS[CV_SPLITS["CV"] == CV]
            complete_cvs.append(complete[complete.sequence_id.isin(list(cv_seq_ids[cv_seq_ids["SET"] == "TEST"].sequence_id))])

    else:
        complete = pd.read_csv("./complete_sequences/sequence_{}_{}.csv".format(DATASET, CLEAN), index_col = 0)
        complete = complete.assign(time_index = complete.groupby("sequence_id").cumcount())
        for CV in range(0, 3):
            imputed_cvs.append(pd.read_csv("./imputed_sequences{}/{}_{}_{}_{}_cv_{}.csv".format(DATASET, IMPUTATION_METHOD.lower(), BOOT, PATTERN, CLEAN, str(CV)), index_col = 0))
            cv_seq_ids = CV_SPLITS[CV_SPLITS["CV"] == CV]
            complete_cvs.append(complete[complete.sequence_id.isin(list(cv_seq_ids[cv_seq_ids["SET"] == "TEST"].sequence_id))])
    
    static_df = None
    
    if (str(DEVICE) == "cpu"):
        static_data = pd.read_csv("/home/giesan/mi_icu/spatio_temporal_pattern/desc/static_data_{}.csv".format(DATASET.lower()))
        static_df = prepare_static(static_data=static_data)
    else:
        static_data = pd.read_csv("./desc/static_data_{}.csv".format(DATASET.lower()))
        static_df = prepare_static(static_data=static_data)

    return complete_cvs, imputed_cvs, static_df.fillna(0)

def train_downstream(config):
    
    """ main method to train TRAPOD architecture for downstream prediction task """
    
    complete_cvs, imputed_cvs, static_df = load_data()
    
    # reading imputation methods
    IMPUTATION_METHOD = config["IMPUTATION_METHOD"]
    
    i = config["CV"]
    # get train indexes for all others than current one
    train_indexes = [j for j in range(0,3) if j != i]
    
    # reading CV daata
    set_compl = complete_cvs[i]
    set_imp = imputed_cvs[i]
    
    # splitting the imputed test folds only into train / test for downstream 
    train_seqs = random.sample(list(set_compl.sequence_id.unique()), k=round(len(set_compl.sequence_id.unique()) * 0.8))
    test_seqs = list(set_compl[~set_compl.sequence_id.isin(train_seqs)].sequence_id.unique())
    
    print("len train seq: ", len(train_seqs))
    print("len test seq: ", len(test_seqs))
    
    # build test sets
    test_set_compl = set_compl[set_compl.sequence_id.isin(test_seqs)]
    test_set_imp = set_imp[set_imp.sequence_id.isin(test_seqs)]
    
    # build train sets
    train_set_compl = set_compl[set_compl.sequence_id.isin(train_seqs)]
    train_set_imp = set_imp[set_imp.sequence_id.isin(train_seqs)]
    
    # set sized must be equal 
    assert len(train_set_imp) == len(train_set_compl)
    
    # apply standardization 
    mean_train_compl, mean_train_imp = train_set_compl[FEATURES].mean(), train_set_imp[FEATURES].mean()
    std_train_compl, std_train_imp = train_set_compl[FEATURES].std(), train_set_imp[FEATURES].std()
    
    # apply standardization 
    df_train_imp = standardize(df=train_set_imp, S_means=mean_train_imp, S_std=std_train_imp)
    df_test_imp = standardize(df=test_set_imp, S_means=mean_train_imp, S_std=std_train_imp)
    df_train_compl = standardize(df=train_set_compl, S_means=mean_train_compl, S_std=std_train_compl)
    df_test_compl = standardize(df=test_set_compl, S_means=mean_train_compl, S_std=std_train_compl)
    
    print(len(df_test_compl))
    
    # build tuples for complete sets
    train_sequences_compl = build_tuple_seq(df_sequence_predict=df_train_compl\
        .merge(static_df[["age", "gender", "sequence_id"]].drop_duplicates(), 
            on="sequence_id"), df_static=static_df, target_column=endpoint_cols[DATASET])
    test_sequences_compl = build_tuple_seq(df_sequence_predict=df_test_compl\
        .merge(static_df[["age", "gender", "sequence_id"]].drop_duplicates(), 
            on="sequence_id"), df_static=static_df, target_column=endpoint_cols[DATASET])
    
    # build tuples for imputed sets
    train_sequences_imp = build_tuple_seq(df_sequence_predict=df_train_imp\
        .merge(static_df[["age", "gender", "sequence_id"]].drop_duplicates(), 
            on="sequence_id"), df_static=static_df, target_column=endpoint_cols[DATASET])
    test_sequences_imp = build_tuple_seq(df_sequence_predict=df_test_imp\
        .merge(static_df[["age", "gender", "sequence_id"]].drop_duplicates(), 
            on="sequence_id"), df_static=static_df, target_column=endpoint_cols[DATASET])
    
    print("length of testing set: ", len(test_sequences_imp))
    print("length of training set: ", len(train_sequences_imp))
    
    preval = static_df[endpoint_cols[DATASET]].value_counts(normalize=True).iloc[1]
    
    print("prevalence ", preval)
    
    if COMPLETE_ONLY:
        data_module = DataModule(
            batch_size=config["BATCH"], 
            train_sequences=train_sequences_compl,
            test_sequences=test_sequences_compl)
    else:
        data_module = DataModule(
            batch_size=config["BATCH"],
            train_sequences=train_sequences_imp,
            test_sequences=test_sequences_imp)
    
    model = TimeSeriesTransformer(input_size=len(FEATURES) if WITH_STATIC == False else len(FEATURES) + 2, 
                                max_seq_len=MAX_SEQ_LEN, dim_val = config["DIM_HIDDEN"], dim_feedforward_encoder=config["DIM_FFN"])
    
    model_trainer = ModelTrainer(model=model, tunable=False, criterion=weighted_binary_cross_entropy,
                                l2_penalty=config["L2_PENALTY"], prevalence=preval,
                                lr=config["LR"])
    
    early_stop_callback = EarlyStopping(monitor="val_loss", patience=7, verbose=False, mode="min")
    checkpoint_callback = ModelCheckpoint(
                        dirpath="./checkpoints/",
                        filename="best_checkpoint",
                        verbose=True,
                        save_top_k=1,
                        monitor="val_loss",
                        mode="min")
    
    if config["MODE"] == "TUNE":
        trainer = pl.Trainer(
            devices="auto",
            accelerator="auto",
            strategy=RayDDPStrategy(),
            callbacks=[RayTrainReportCallback()],
            plugins=[RayLightningEnvironment()],
            enable_progress_bar=False,
        )
        trainer = prepare_trainer(trainer)
    else:
        trainer = pl.Trainer(
                max_epochs = EPOCHS,
                log_every_n_steps = 20,
                devices = 1,
                callbacks=[early_stop_callback]
            )
        
    #condition if we want to fit on imputed only
    trainer.fit(model_trainer, data_module)
    
    if config["MODE"] == "TUNE":
        return
    MODEL_NAME = "transform_{}_{}_{}_{}".format(DATASET, CLEAN, PATTERN, "COMPL" if COMPLETE_ONLY else IMPUTATION_METHOD)
    torch.save(model.state_dict(), "./results/{}/".format(DATASET.lower()) + "./{}.pt".format(MODEL_NAME))
    
    best_trainer = ModelTrainer.load_from_checkpoint(trainer.checkpoint_callback.best_model_path, model=model) #.cuda()
    best_trainer.freeze()
    
    best_trainer.criterion = weighted_binary_cross_entropy
                    
    #pred_train = best_trainer.test_data(data_module_imp.train_dataloader().dataset)
    pred_test, boot_test, metric_test = best_trainer.test_data(data_module.test_dataloader().dataset)
    pred_train, boot_train, metric_train = best_trainer.test_data(data_module.train_dataloader().dataset)
    
    print(metric_test)
    
    RES.append(pd.DataFrame({
        "n": len(test_sequences_imp),
        "val_auroc": float(metric_test["auroc"].float()),
        "val_auroc_CI": boot_test["auroc_CI"],
        "val_auprc": float(metric_test["auprc"].float()),
        "val_auprc_CI": boot_test["auprc_CI"],
        "train_auroc": float(metric_train["auroc"].float()),
        "train_auroc_CI": boot_train["auroc_CI"],
        "train_auprc": float(metric_train["auprc"].float()),
        "train_auprc_CI": boot_train["auprc_CI"],
        "val_predic" : str([float(x[0][0]) for x in pred_test["out"]]),
        "val_target": str(pred_test["target"]),
        "CV": config["CV"]
        }, index=[config["CV"]]))
    
    #TODO MAYBE STORE ALSO PRED AND TARGET FOR AUROC CURVES!!! TODO also implement bootstrapping!!!
    
    #print(df_train_compl.merge(static_df[["age", "gender", "sequence_id"]].drop_duplicates(), on="sequence_id"))
    #build_tuple_seq()
    #print(df_test_compl) #TODO enhance downstream task and ...

DEF_CONF = {
    "BATCH": 32,
    "L2_PENALTY": 0,
    "LR": 1e-4,
    "DIM_HIDDEN": 16, #16 TODO good for complete dataset
    "DIM_FFN": 16, #16
    "MODE": "FINAL"
}

DEF_CONF = HYPERPARAMS[DATASET.upper()]

CONFIG_SPACE = {
        "DIM_FFN": tune.choice([16, 32, 64, 128]),
        "DIM_HIDDEN": tune.choice([16, 32, 64, 128]),
        "LR": tune.choice([1e-3, 1e-4]),
        "L2_PENALTY": tune.choice([0, 1e-1]),
        "BATCH": tune.choice([64, 32])
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
        train_downstream,
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

# LINPOL MCAR 
#BILSTM MAR
for IMPUTATION_METHOD in ["BILSTAELOC"]: #"LSTAELOC", "LINPOL", "BILSTAELOC"  "LINPOL", "BILSTAELOC"
        
    RES = []
    
    DEF_CONF["IMPUTATION_METHOD"] = IMPUTATION_METHOD
    
    print("training model for ", IMPUTATION_METHOD)

    for CV in range(0, 3):
        
        if CV_X_ONYL:
            if CV != CV_SELECT: #TODO
                continue
        
        DEF_CONF["CV"] = CV

        if HYPERTUNE:
            
            # hyperparameter tuning and fitting
            DEF_CONF["MODE"] = "TUNE"
            
            results = ray_hypertune()

            # get best hyperparameters
            best_conf = results.get_best_result(metric="val_loss", mode="min").config

            # get best params and overwrite default param map
            for key in CONFIG_SPACE.keys():
                DEF_CONF[key] = best_conf['train_loop_config'][key]
            
            DEF_CONF["MODE"] = "FINAL"
            
            print("best params")
            print(DEF_CONF)

        train_downstream(DEF_CONF)

    RES_PATH = "./results/{}/res_{}_{}_{}_{}_".format(DATASET.lower(), DATASET, CLEAN, PATTERN, "COMPL" if COMPLETE_ONLY else IMPUTATION_METHOD)
    pd.concat(RES).to_csv(RES_PATH + (".csv" if not CV_X_ONYL else "cv_{}.csv".format(CV_SELECT)))
    
    # show stored path
    print(RES_PATH)
    
    # break when GT values are used for training
    if COMPLETE_ONLY:
        break

print("finish")