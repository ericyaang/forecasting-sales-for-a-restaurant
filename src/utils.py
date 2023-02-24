import pandas as pd 
import os
from omegaconf import DictConfig, OmegaConf
import joblib 

def load_data(cfg: DictConfig):
    X_train = pd.read_parquet(os.path.join(cfg.processed.dir_nb, cfg.processed.X_train.name))
    y_train = pd.read_parquet(os.path.join(cfg.processed.dir_nb, cfg.processed.y_train.name))
    X_test = pd.read_parquet(os.path.join(cfg.processed.dir_nb, cfg.processed.X_test.name))
    y_test = pd.read_parquet(os.path.join(cfg.processed.dir_nb, cfg.processed.y_test.name))
    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
    }

def load_model(data: dict, cfg: DictConfig):
    model = joblib.load(os.path.join(cfg.trained_model.dir_nb, cfg.trained_model.name))
    return model.fit(data['X_train'], data['y_train'])
