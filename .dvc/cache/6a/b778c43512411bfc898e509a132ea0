# coding: utf-8
import os
import re
import warnings
from collections import defaultdict
from pathlib import Path

import pandas as pd

from config import config
from config.config import logger

warnings.filterwarnings("ignore")


def read_flat(path):
    with open(path, mode="r") as f:
        _list = []
        # read all the lines of a file in a list
        for line in f.readlines():
            # removing white spaces
            _list.append(line.rstrip())
    # return list without empty items
    return list(filter(None, _list))


def filter_list(_list):
    """Return only items before string '###'.
    This assures those only items the first occurrence of all items is considered.
    """
    hashtags = [re.findall("\#{2,}", item) for item in _list]
    hashtags = list(filter(None, hashtags))
    # identify more than one line with a string that contains '###'
    if len(hashtags) > 1:
        hashtag_idx = [
            i for i, item in enumerate(_list) if re.search(r"\#{2,}", item)
        ]
        # return sliced list
        return _list[: hashtag_idx[0]]
    else:
        # slice first item that is not present
        return _list


def clean_file(_list):
    # delete some strings string
    _list = [x for x in _list if x != "PORTERHOUSE"]
    # insert ID and Date
    id_str = "id: " + _list[1].split()[4]

    date_str = "date: " + _list[1].split()[-1]

    _list.insert(0, id_str)
    _list.insert(1, date_str)

    # delete more than one repeated character
    _list = [
        re.sub(r"\.{2,}|\={2,}|\#{2,}|\-{2,}", "", item) for item in _list
    ]
    # Remove additional white spaces
    _list = [
        re.sub("[\s]+", "", item).strip().replace("EUR", "") for item in _list
    ]
    # delete empty items
    _list = list(filter(None, _list))
    # split items with ':'
    _list = [item.split(":") for item in _list]

    return _list


def listToDict(lst):
    return {lst[i]: lst[i + 1] for i in range(0, len(lst), 2)}


def return_dict(path):
    dir_list = [file for file in os.listdir(path) if file.endswith(".fiscal")]
    # Select features
    var_list = [
        "AbgerechneteTische",
        "GesamtBestellungBrutto.",
        "GesamtBestellungNetto",
    ]

    # Append generated values from the function clean_file()
    var_list.append("id")
    var_list.append("date")
    # create an empty list
    df_array = []

    # apply and append all files
    for file in dir_list:
        file_path = os.path.join(path, file)
        _temp = read_flat(file_path)
        _temp = filter_list(_temp)
        _temp = clean_file(_temp)
        _temp = [x for x in _temp if set(x).intersection(var_list)]
        _temp = [listToDict(list) for list in _temp]
        result = {}
        for d in _temp:
            result.update(d)
        for key, val in result.items():
            if key in d:
                d[key] = [d[key], val]
                df_array.append(result)

    # merge dicts
    df_all = defaultdict(list)

    for d in df_array:
        for k, v in d.items():
            df_all[k].append(v)
    return df_all


def dict_to_pandas(df_dict):
    initial_date = df_dict["date"][0].replace(".", "-")
    final_date = df_dict["date"][-1].replace(".", "-")

    _dates = pd.DataFrame(
        pd.date_range(initial_date, final_date), columns=["date"]
    )

    _df = pd.DataFrame(df_dict)
    # convert date to a valid format
    _df["date"] = pd.to_datetime(_df["date"], format="%d.%m.%Y")
    # rename columns
    _df = _df.rename(
        columns={
            "GesamtBestellungBrutto.": "gross_sales",
            "GesamtBestellungNetto": "net_sales",
            "AbgerechneteTische": "n_tables",
        }
    )

    # convert data types to numeric
    _df["gross_sales"] = _df["gross_sales"].astype("float")
    _df["net_sales"] = _df["net_sales"].astype("float")
    _df["n_tables"] = _df["n_tables"].astype("int")
    ## from the all_dates DataFrame, left join onto the DataFrame with missing dates
    return _dates.merge(right=_df, how="left", on="date")


def save_to_parquet():
    """Save dataframe"""
    df_dict = return_dict(config.PRIVATE_DIR)
    df_pd = dict_to_pandas(df_dict)
    df_pd.to_parquet(Path((config.DATA_DIR), "raw.parquet"), engine="pyarrow")
    logger.info("✅ Saved data!")


if __name__ == "__main__":
    save_to_parquet()
