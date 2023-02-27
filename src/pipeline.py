from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder

from transformers import (CyclicalFeatures, DateTimeFeaturesExtractor,
                          GetHolidays, Interpolate, Log1pTransformer, ZeroFill)

###### Check data types before applying this transformation ######


# Preprocessor applied to all data


# 1.a: Baseline
def preprocessor_1_a(target: str, holiday_id: str, imputer_type: str):
    if imputer_type == "interpolate":
        imputer = Interpolate(target)
    elif imputer_type == "null":
        imputer = ZeroFill(target)
    else:
        raise ValueError("Invalid imputer type. Choose either 'interpolate' or 'null'.")

    return Pipeline(
        steps=[
            ("imputer", imputer),
            ("dt", DateTimeFeaturesExtractor()),
            ("holy", GetHolidays(holiday_id)),
            ("log", Log1pTransformer(target)),
        ]
    )


def preprocessor_1_b(target: str, holiday_id: str):
    return Pipeline(
        steps=[
            ("intp", Interpolate(target)),
            ("dt", DateTimeFeaturesExtractor()),
            ("holy", GetHolidays(holiday_id)),
            ("log", Log1pTransformer(target)),
            # ("lag", AddLags(target)),
        ]
    )


# 1.b: For new data
def preprocessor_1_c(holiday_id: str):
    return Pipeline(
        steps=[
            ("dt", DateTimeFeaturesExtractor()),
            ("holy", GetHolidays(holiday_id)),
        ]
    )


def preprocessor_1_d(target: str, holiday_id: str):
    return Pipeline(
        steps=[
            ("intp", ZeroFill(target)),
            ("dt", DateTimeFeaturesExtractor()),
            ("holy", GetHolidays(holiday_id)),
            ("log", Log1pTransformer(target)),
            # ("lag", transformers.AddLags(target)),
        ]
    )


# Preprocessor applied to splits


# 2.a: ohe + cyc + sc (float)
def preprocessor_2_a():
    return ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                make_column_selector(dtype_include="category"),
            ),
            (
                "cyc",
                CyclicalFeatures(),
                make_column_selector(dtype_include="int32"),
            ),
            (
                "sc",
                MinMaxScaler(),
                make_column_selector(dtype_include="float64"),
            ),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )


# 2.b: cat + sc (int+float)
def preprocessor_2_b():
    return ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                make_column_selector(dtype_include="category"),
            ),
            (
                "sc",
                MinMaxScaler(),
                make_column_selector(dtype_include=["float64", "int32"]),
            ),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )


# 2.c: only ordinal on categorical cols
def preprocessor_2_c():
    return ColumnTransformer(
        transformers=[
            (
                "cat",
                OrdinalEncoder(),
                make_column_selector(dtype_include="category"),
            ),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )


# 2.d cat on all (int+cat)
def preprocessor_2_d():
    return ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                make_column_selector(dtype_include="category"),
            ),
            (
                "cat_int",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                make_column_selector(dtype_include="int32"),
            ),
        ],
        remainder=MinMaxScaler(),
        verbose_feature_names_out=False,
    )
