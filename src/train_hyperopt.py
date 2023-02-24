import logging
import hydra
import os
import joblib
import mlflow
import numpy as np
import pandas as pd
from hydra.utils import to_absolute_path as abspath
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from omegaconf import DictConfig
from sklearn.model_selection import TimeSeriesSplit, cross_validate
from xgboost import XGBRegressor
import warnings

warnings.filterwarnings(action="ignore")
logger = logging.getLogger(__name__)

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

@hydra.main(version_base=None, config_path="../config", config_name="main")
def main(cfg: DictConfig):

    logger.info("Loading data...")
    X, y = load_train(cfg)
    

    def objective(params):
        """
        Define a função objetivo para otimização de hiperparâmetros usando o Hyperopt.
        """
        model = XGBRegressor(
            random_state=cfg.params.random_state,
            max_depth=int(params["max_depth"]),
            learning_rate=params["learning_rate"],
            n_estimators=int(params["n_estimators"]),
            subsample=params["subsample"],
            colsample_bytree=params["colsample_bytree"],
            gamma=params["gamma"],
            reg_alpha=params["reg_alpha"],
            reg_lambda=params["reg_lambda"],
        )

        # Validação cruzada com 5 splits e gap de 7 dias
        ts_cv = TimeSeriesSplit(
            n_splits=5,
            gap=7,
            max_train_size=None,
            test_size=None,
        )
        cv_results = cross_validate(
            model,
            X,
            y.values.ravel(),
            cv=ts_cv,
            return_estimator=True,
            return_train_score=True,
            scoring=[
                "neg_mean_absolute_error",
                "neg_root_mean_squared_error",
                "neg_mean_absolute_percentage_error",
                "r2",
            ],
            n_jobs=-1,
        )
        mae = -cv_results["test_neg_mean_absolute_error"].mean().round(2)
        rmse = -cv_results["test_neg_root_mean_squared_error"].mean().round(2)
        mape = -cv_results["test_neg_mean_absolute_percentage_error"].mean().round(2)
        r2 = cv_results["test_r2"].mean().round(2) 

        # keep only relevant parameters
        all_params = model.get_params()
        params_to_keep = ['max_depth', 'learning_rate', 'n_estimators', 'subsample', 'colsample_bytree', 'gamma', 'reg_alpha', 'reg_lambda']
        params = {k: v for k, v in all_params.items() if k in params_to_keep}

        return {"loss": rmse, "model": params, "status": STATUS_OK, "metric": {"rmse": rmse, "mae": mae, "mape": mape, "r2": r2}}       
    
    # Define space
    space = {
        "max_depth": hp.quniform("max_depth", 3, 10, 1),
        "learning_rate": hp.uniform("learning_rate", 0.01, 0.2),
        "n_estimators": hp.quniform("n_estimators", 50, 500, 25),
        "subsample": hp.uniform("subsample", 0.5, 1),
        "colsample_bytree": hp.uniform("colsample_bytree", 0.5, 1),
        "gamma": hp.uniform("gamma", 0, 10),
        "reg_alpha": hp.uniform("reg_alpha", 0, 1),
        "reg_lambda": hp.uniform("reg_lambda", 0, 1),
    }

    # Set environment variables and MLflow tracking URI from configuration
    os.environ['MLFLOW_TRACKING_USERNAME'] = cfg.mlflow.username
    os.environ['MLFLOW_TRACKING_PASSWORD'] = cfg.mlflow.password
    mlflow.set_tracking_uri(cfg.mlflow.uri)

    # Get experiment ID for 'best_model' experiment
    #exp_id = mlflow.get_experiment_by_name('best_model').experiment_id
    exp_id = get_experiment_id('best_model')
    # Start an MLflow run with a given name and experiment ID
    with mlflow.start_run(run_name='hyperopt', experiment_id=exp_id):
        
        # Optimize the objective function using Hyperopt and get the best model parameters with 100 evaluations
        trials = Trials()
        best_model = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=100, trials=trials)
        
        # Get the metrics of the best model
        best_model_metrics = trials.results[np.argmin([r["loss"] for r in trials.results])]['metric']
        
        # Log the best model parameters and metrics to MLflow
        mlflow.log_params(best_model)
        mlflow.log_metrics(best_model_metrics)
        
        # Log a message to the logger
        logger.info("Best model parameters and metrics are saved to MLflow")

        # Save the best model to a joblib file
        joblib.dump(best_model, abspath(cfg.model.path))
        logger.info("model saved successfully!")
        
if __name__ == "__main__":
    main()