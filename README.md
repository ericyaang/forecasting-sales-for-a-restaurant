[![](https://img.shields.io/badge/DagsHub-Link%20to%20DagsHub-red)](https://dagshub.com/ericyaang/forecasting-sales-for-a-restaurant)
# Machine Learning-Based Sales Prediction for a Top-Rated Steakhouse

This project is a Proof of Concept (PoC) prototype that aims to help a restaurant manager scale the number of employees needed to meet the establishment's daily demand.

Predicting sales enables businesses to allocate resources efficiently, optimize cash flow and production, and make well-informed and effective business plans.

However, the goal here is to predict the trend of the level of demand using the log of sales as a proxy. Therefore, it is not necessary to predict every value with meticulous accuracy.

The Extreme Gradient Boosting (XGBoost) algorithm was chosen for predicting future sales due to its superior overall performance and faster processing time. It also requires less effort in feature engineering.

## Overview

**Objective**: Optimize employee scheduling at the establishment.

**What is needed**: Daily sales and weather data.

**Horizon**: Sales for the next 7 days.

**Metrics**:
- mape (main reference)
- mae
- rmse
- r2
  
**Data**:

  - Historical daily sales data.
  - Historical climate data for the region.

  The ideal minimum period would be 1 year, in order to observe the annual seasonality that encompasses the complete cycle of seasons, as temperature varies depending on the season.

## Requirements

* **Data:** The app only requires the total value of daily sales, as follows in this example:

date | sales | 
--- | --- |
2021-06-01 | 3540.91 |
2021-06-02 | 5049.35 |
2021-06-03 | 7655.55 |
2021-06-04 | 5885.45 |

However, it is ideal to provide sales data hourly to identify the flow by shifts. Additional information usually improves the model's performance. The ideal minimum required period is at least 365 days.


## Tools used 
- [Poetry](https://python-poetry.org/): packaging and dependency management
- [MLflow](https://mlflow.org/): experiment tracking
- [DagsHub](https://dagshub.com): managing and tracking machine learning workflows
- [Hydra](https://hydra.cc/): simplifying the configuration of complex applications
  

## Project structure

- `src`:  contains the source code of the project, usually written in Python scripts
- `config`: contains configuration files for the project, which can include settings for data sources, model parameters, and other variables
- `data`: contains the data used in the project, which may be organized into subdirectories for different datasets or versions
- `notebook`: contains Jupyter Notebooks used for reporting results
- `tests`: contains test files to ensure the code is functioning as intended
- `models`: stores the fitted models

## Set Up the Project

```bash
make install_all
```

On `config\main.yaml` paste your credentials from dagshub repository to your track metrics with mlflow

See more in [documentation](https://dagshub.com/docs/integration_guide/mlflow_tracking/index.html)

```yaml

mlflow:
  uri: MLFLOW_TRACKING_URI
  username: MLFLOW_TRACKING_USERNAME
  password: MLFLOW_TRACKING_PASSWORD
  experiment_name: XGBoost Baseline

```


##  Run the Project

```bash
make pipeline
```
### Machine learning pipeline

Scripts:

* `src/process.py`: contains code for preprocessing the data.
* `src/select_regressor.py`: contains code for selecting the appropriate regressor for the model.
* `src/train_hyperopt.py`: contains code for training the model using hyperopt and cross-validation to find the optimal hyperparameters.
* `src/train_model.py`: contains code for training the final model and saving it as an object.


![pipeline](img\pipeline.png)

Once the pipeline run is completed, you can access to MLflow's server in your dagshub repository like this:

![dagshub+mlflow](img\dagshub.png)
