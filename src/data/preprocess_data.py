"""
Module contains functions for preprocessing the dataset into a Pandas DataFrame.

NOTE:
    Ensure the that the scripts are ran from the root directory '/IPAGE'

The absolute path should always point to the root folder '~/*/IPAGE' where the
'data' folder is found.
"""
import pandas as pd
from typing import Union
from .load_data import load_data
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderServiceError

COLS_TO_DROP = ["longitude", "latitude", "Soil group", "Land class", "Soil type"]

def remove_outliers(df: pd.DataFrame, feature: str) -> pd.DataFrame:
    # REFERENCE:
    # 1. https://github.com/OmdenaAI/IPAGE/blob/main/notebooks/task_3_model_development_and_training/by-tasfiq/tasfiq_tf_v3_train_all_targets.ipynb
    Q1: Union[int, float] = df[feature].quantile(0.05)  # First quartile (5th percentile)
    Q3: Union[int, float] = df[feature].quantile(0.95)  # Third quartile (95th percentile)
    IQR: Union[int, float] = Q3 - Q1  # Interquartile Range
    lower_bound: Union[int, float] = Q1 - 1.5 * IQR
    upper_bound: Union[int, float] = Q3 + 1.5 * IQR

    # Filter the dataset to include only non-outliers
    df: pd.DataFrame = df[(df[feature] >= lower_bound) & (df[feature] <= upper_bound)]
    return df

def remove_duplicates(df: pd.DataFrame, exclude_cols=["date"]) -> None:
    df.drop_duplicates(
        subset=[col for col in df.columns if col not in exclude_cols],
        inplace=True
    )

def get_geospacial_details(row: pd.Series) -> list[list[str,float]]:
    # REFERENECE:
    # 1. https://github.com/OmdenaAI/IPAGE/blob/main/notebooks/task_1_data_collection_and_preprocessing/by-tasfiq/geospatial_processing.ipynb
    # 2. https://github.com/OmdenaAI/IPAGE/blob/main/notebooks/task_1_data_collection_and_preprocessing/by-Gideon/geospacial_preprocessing.ipynb
    
    longitude = str(row.loc["longitude"])
    latitude = str(row.loc["latitude"])
    address = ",".join([latitude, longitude])

    geolocator = Nominatim(user_agent="my_geopy_app")
    
    try:
        location = geolocator.reverse(address)
        address = location.raw['address']
    
        # Traverse the data
        state = address.get('state', float("nan"))
    
    except GeocoderServiceError as e:
        state = float("nan")
    return state

