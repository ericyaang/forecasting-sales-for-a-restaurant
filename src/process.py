import warnings

import hydra
import pandas as pd
from hydra.utils import to_absolute_path as abspath
from omegaconf import DictConfig
from sklearn import set_config
import logging
from pipeline import preprocessor_1_a, preprocessor_2_c

warnings.filterwarnings(action="ignore")
set_config(transform_output="pandas")
logger = logging.getLogger(__name__)

#### Helper functions:
def split_label(data: pd.DataFrame, target: str):
    X = data.drop([target], axis=1)
    y = data[[target]]
    return X, y


def split_data(data: pd.DataFrame, split_size: float):
    """Split the data based on a test split factor."""
    length = len(data)
    t_idx = round(length * (1 - split_size))
    train, test = data[:t_idx], data[t_idx:]
    return train, test


###### TO DO: fazer docstrings ######


def get_data(data_location: str):
    # read parquet data
    df = pd.read_parquet(data_location, columns=["net_sales", "date"])

    # set date as index
    df = df.set_index("date")

    # select date from 1 jan 2022
    df = df.loc["2021-06-01":]

    return df


def add_climate_features(df: pd.DataFrame, data_location: str):
    # load weather data
    weather_data = pd.read_parquet(
        data_location,  # columns=['temperature', 'sunshine', 'cloud_cover']
    )
    # Make sure values are numeric
    weather_data = weather_data.apply(pd.to_numeric, errors="coerce")

    # filter dates from the original dataset
    weather_data = weather_data.loc[
        str(df.index.date.min()) : str(df.index.date.max())
    ]
    return df.join(weather_data, how="left")


@hydra.main(version_base=None, config_path="../config", config_name="main")
def process(cfg: DictConfig):
    # load data
    logger.info("Loading data...")
    data = get_data(cfg.raw_data.path)

    # Add climate features
    logger.info("Adding climate features...")
    data = add_climate_features(data, cfg.weather_data.path)

    # load and apply preprocessor 1
    logger.info("Applying preprocessor 1...")
    prep_1 = preprocessor_1_a(
        cfg.params.target_col, cfg.params.holiday_id
    )  # CHANGE THIS
    data = prep_1.fit_transform(data)

    # Split data
    logger.info("Splitting data...")
    train, test = split_data(data, cfg.params.split_size)

    # Split labels
    X_train, y_train = split_label(train, cfg.params.target_col)
    X_test, y_test = split_label(test, cfg.params.target_col)

    # Load and apply preprocessor 2
    logger.info("Applying preprocessor 2...")
    prep_2 = preprocessor_2_c()  # CHANGE THIS
    X_train = prep_2.fit_transform(X_train)
    X_test = prep_2.transform(X_test)

    # Save data
    logger.info("Saving data...")
    X_train.to_parquet(abspath(cfg.processed.X_train.path))
    X_test.to_parquet(abspath(cfg.processed.X_test.path))
    y_train.to_parquet(abspath(cfg.processed.y_train.path))
    y_test.to_parquet(abspath(cfg.processed.y_test.path))
    logger.info("Data processing completed successfully.")

if __name__ == "__main__":
    process()
