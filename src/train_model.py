from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error,
)
import numpy as np
import pandas as pd
from hydra.utils import to_absolute_path as abspath
from omegaconf import DictConfig
import hydra
import joblib
from xgboost import XGBRegressor
import os
import mlflow
import logging 

logger = logging.getLogger(__name__)

def load_data(cfg: DictConfig):
    X_train = pd.read_parquet(abspath(cfg.processed.X_train.path))
    y_train = pd.read_parquet(abspath(cfg.processed.y_train.path))
    X_test = pd.read_parquet(abspath(cfg.processed.X_test.path))
    y_test = pd.read_parquet(abspath(cfg.processed.y_test.path))
    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
    }


def train_model(cfg: DictConfig, data: dict):
    # load best params
    params = joblib.load(cfg.model.path)

    # convert max_depth, n_estimators to integer type
    params = {
        key: int(value) if key in ["max_depth", "n_estimators"] else value
        for key, value in params.items()
    }

    # fit the model
    clf = XGBRegressor(**params)
    clf.fit(
        data["X_train"],
        data["y_train"],
        eval_set=[
            (data["X_train"], data["y_train"]),
            (data["X_test"], data["y_test"]),
        ],
        eval_metric="rmse",
        early_stopping_rounds=10,
    )
    return clf


def get_prediction(model: XGBRegressor, data: dict):
    return model.predict(data["X_test"])


def evaluate_model(prediction: pd.DataFrame, data: dict):
    return {
        "mape": mean_absolute_percentage_error(
            data["y_test"], prediction
        ).round(2),
        "mae": mean_absolute_error(data["y_test"], prediction).round(2),
        "rmse": np.sqrt(mean_squared_error(data["y_test"], prediction)).round(
            2
        ),
        "r2": r2_score(data["y_test"], prediction).round(2),
    }


def save_model(cfg: DictConfig, model: XGBRegressor):
    joblib.dump(model, cfg.trained_model.path)


def get_experiment_id(name):
    exp = mlflow.get_experiment_by_name(name)
    if exp is None:
        exp_id = mlflow.create_experiment(name)
        return exp_id
    return exp.experiment_id


def log_metrics_to_mlflow(cfg: DictConfig, metrics: dict):
    # Set environment variables and MLflow tracking URI from configuration
    os.environ["MLFLOW_TRACKING_USERNAME"] = cfg.mlflow.username
    os.environ["MLFLOW_TRACKING_PASSWORD"] = cfg.mlflow.password
    mlflow.set_tracking_uri(cfg.mlflow.uri)
    exp_id = get_experiment_id("final_evaluation")
    with mlflow.start_run(run_name="evaluation", experiment_id=exp_id):
        mlflow.log_metrics(metrics)


@hydra.main(version_base=None, config_path="../config", config_name="main")
def train(cfg: DictConfig):
    logger.info("Loading data...")
    data = load_data(cfg)
    logger.info("Training model...")
    clf = train_model(cfg, data)
    prediction = get_prediction(clf, data)
    logger.info("Getting model results...")
    results = evaluate_model(prediction, data)
    log_metrics_to_mlflow(cfg, results)
    logger.info("Saving model...")
    save_model(cfg, clf)
    logger.info("Done!")


if __name__ == "__main__":
    train()
