import datetime
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from workalendar.registry import registry


def get_de_holidays(date_init: datetime.date, date_end: datetime.date, cal_id: str):
    CalendarClass = registry.get(cal_id)
    de_sh_holidays = CalendarClass()
    _list = []
    year_init = date_init.year
    year_end = date_end.year
    for year in range(year_init, year_end + 1):
        temp = de_sh_holidays.holidays(year)
        _list.extend(temp)
        _dict = dict((x, y) for x, y in _list)
        _series = pd.Series(
            _dict, name="holiday"
        ).to_frame()  ## ARRUMAR AQUI EM BAIXO DEPOIS
        mask = (_series.index < date_init) | (
            _series.index > date_end
        )  # https://stackoverflow.com/questions/51474263/typeerror-cannot-compare-type-timestamp-with-type-date
        _series = _series[~mask]
        _series.index.names = ["date"]
    return _series


class GetHolidays(BaseEstimator, TransformerMixin):
    """Generate holidays"""

    def __init__(self, cal_id):
        self.cal_id = cal_id

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        de_holidays = get_de_holidays(X.index.min(), X.index.max(), self.cal_id)
        X = X.copy()
        X = X.join(de_holidays, how="left")
        X["holiday"] = X["holiday"].fillna("None")
        X["holiday"] = np.where(X["holiday"] == "None", 0, 1).astype("int8")
        return X


class AddLags(BaseEstimator, TransformerMixin):
    """Generate holidays"""

    def __init__(self, col):
        self.col = col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        return pd.concat(
            [
                X,
                X[self.col].shift(1).rename("lagged_data_1d"),
                X[self.col].shift(2).rename("lagged_data_2d"),
                X[self.col].shift(3).rename("lagged_data_3d"),
                X[self.col].shift(7).rename("lagged_data_1w"),
                X[self.col].shift(14).rename("lagged_data_2w"),
                X[self.col].shift(1).rolling(7).mean().rename("lagged_mean_1w"),
                X[self.col].shift(1).rolling(7).max().rename("lagged_max_1w"),
                X[self.col].shift(1).rolling(7).min().rename("lagged_min_1w"),
                X[self.col].shift(1).rolling(7 * 2).mean().rename("lagged_mean_2w"),
                X[self.col].shift(1).rolling(7 * 2).max().rename("lagged_max_2w"),
                X[self.col].shift(1).rolling(7 * 2).min().rename("lagged_min_2w"),
            ],
            axis="columns",
        ).fillna(0)


class DateTimeFeaturesExtractor(BaseEstimator, TransformerMixin):
    """Extract Datetime features"""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X["day"] = X.index.day.values.astype("int32")
        X["month"] = X.index.month.values.astype("int32")
        X["year"] = X.index.year.values.astype("int32")
        X["week"] = X.index.isocalendar().week.astype("int32")
        X["dow"] = X.index.dayofweek.values.astype("int32")
        X["month_end"] = X.index.is_month_end.astype("int8")
        X["weekend"] = (X.index.weekday.values >= 5).astype("int8")

        # days of th week
        days = {
            0: "monday",
            1: "tuesday",
            2: "wednesday",
            3: "thursday",
            4: "friday",
            5: "saturday",
            6: "sunday",
        }
        X["dow"] = X.dow.map(days).astype("category")

        # Seasons
        # dictionary for the future replacement of months with seasons
        ds = {
            12: "winter",
            1: "winter",
            2: "winter",
            3: "spring",
            4: "spring",
            5: "spring",
            6: "summer",
            7: "summer",
            8: "summer",
            9: "autumn",
            10: "autumn",
            11: "autumn",
        }

        X["season"] = X["month"].replace(ds).astype("category")

        X.sort_values(by="date", inplace=True)
        return X


class CyclicalFeatures(TransformerMixin, BaseEstimator):
    """CyclicalFeatures transformer."""

    def __init__(self, max_vals: Dict[str, float] = {}):
        """Nothing much to do."""
        super().__init__()
        self.feature_names: List[str] = []
        self.max_vals = max_vals

    def get_feature_names(self):
        """Feature names."""
        return self.feature_names

    def transform(self, df: pd.DataFrame):
        """Annotate date features."""
        Xt = []
        for col in df.columns:
            if col in self.max_vals:
                max_val = self.max_vals[col]
            else:
                max_val = df[col].max()
            for fun_name, fun in [("cos", np.cos), ("sin", np.sin)]:
                date_feature = fun(2 * np.pi * df[col] / max_val)
                date_feature.name = f"{col}_{fun_name}"
                Xt.append(date_feature)

        df2 = pd.concat(Xt, axis=1)
        self.feature_names = list(df2.columns)
        return df2

    def fit(self, df: pd.DataFrame, y=None, **fit_params):
        """No fitting needed."""
        return self


class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns: list):
        self.columns = columns

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        return self

    def transform(self, X: pd.DataFrame):
        data = X.copy()
        data = data.drop(columns=self.columns)
        return data


class Interpolate(BaseEstimator, TransformerMixin):
    def __init__(self, columns, method="linear"):
        self.columns = columns
        self.method = method

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X[self.columns] = X[self.columns].interpolate(method=self.method)
        return X


class ZeroFill(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X[self.columns] = X[self.columns].fillna(value=0)
        return X


class Log1pTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns: list):
        self.columns = columns

    def fit(self, X, y=None):
        # stateless transformer
        return self

    def transform(self, X):
        # assumes X is a DataFrame
        X[self.columns] = np.log1p(X[self.columns])
        return X
