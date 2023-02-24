import pandas as pd
import mlflow 
import os
from sklearn.model_selection import cross_validate, TimeSeriesSplit
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.linear_model import Lasso, Ridge, LinearRegression, ElasticNet
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import hydra
from hydra.utils import to_absolute_path as abspath
from omegaconf import DictConfig
import logging

def load_train(cfg: DictConfig):
    X_train = pd.read_parquet(abspath(cfg.processed.X_train.path))
    y_train = pd.read_parquet(abspath(cfg.processed.y_train.path))
    return X_train, y_train

def get_experiment_id(name):
    exp = mlflow.get_experiment_by_name(name)
    if exp is None:
      exp_id = mlflow.create_experiment(name)
      return exp_id
    return exp.experiment_id

def run_cv(
    X: pd.DataFrame,
    y: pd.DataFrame,
    cfg:DictConfig,
    model,
    model_name: str, experiment_name: str):
    os.environ['MLFLOW_TRACKING_USERNAME'] = cfg.mlflow.username
    os.environ['MLFLOW_TRACKING_PASSWORD'] = cfg.mlflow.password
    mlflow.set_tracking_uri(cfg.mlflow.uri)
    exp_id = get_experiment_id(experiment_name)
    
    with mlflow.start_run(run_name=model_name, experiment_id=exp_id):

        ts_cv = TimeSeriesSplit(
            n_splits=cfg.params.ts_split,
            gap=7, # 7 DAYS GAPE BETWEEN SPLITS
            max_train_size=None,
            test_size=None,
        )        
        cv_results = cross_validate(
        model,
        X,
        y,
        cv=ts_cv,
        scoring=["neg_mean_absolute_error", "neg_root_mean_squared_error","neg_mean_absolute_percentage_error", "r2"])


        mae = -cv_results["test_neg_mean_absolute_error"]
        rmse = -cv_results["test_neg_root_mean_squared_error"]
        mape = -cv_results["test_neg_mean_absolute_percentage_error"]
        r2 = cv_results["test_r2"]

        mlflow.log_metric("rmse", rmse.mean().round(2))
        mlflow.log_metric("r2", r2.mean().round(2))
        mlflow.log_metric("mae", mae.mean().round(2))
        mlflow.log_metric("mape", mape.mean().round(2))


def get_regressors(cfg:DictConfig) -> dict:

    return {
        "Random Forest": RandomForestRegressor(
            n_estimators=20,                
            random_state=cfg.params.random_state,

        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=20,
            random_state=cfg.params.random_state,

        ),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            random_state=cfg.params.random_state,
        ),
        "ElasticNet": ElasticNet(

            random_state=cfg.params.random_state,

        ),
        "Linear": LinearRegression(

        ),
        "Lasso": Lasso(

        ),
        "Ridge": Ridge(

            random_state=cfg.params.random_state,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=20,
            random_state=cfg.params.random_state,
        ),
        "LightGBM": LGBMRegressor(
            n_estimators=50,
            random_state=cfg.params.random_state,

        ),
    }

@hydra.main(version_base=None, config_path="../config", config_name="main")
def main(cfg: DictConfig):
  
  X, y = load_train(cfg)
  models = get_regressors(cfg)
  for model_name, model in models.items():
    print(f"Running cross-validation for {model_name}...")
    run_cv(X, y.values.ravel(), cfg, model, model_name, experiment_name='select_models_null')
  

if __name__ == '__main__':
    logging.info("\n********************")
    logging.info(">>>>> Training with CV started <<<<<")
    try:
        main()
    except Exception as e:
        logging.exception(e)
        raise e
    else:
        logging.info(">>>>> The training with CV has been completed and recorded on DagsHub! <<<<<\n")
