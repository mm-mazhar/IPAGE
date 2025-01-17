from src.data.load_data import load_data
import numpy as np
from src.data.preprocess_data import remove_duplicates, remove_outliers, get_geospacial_details
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, BaggingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
import joblib
from pathlib import Path
import pandas as pd
from typing import Union, Iterable

COLS_TO_DROP: list = ["longitude", "latitude", "Soil group", "Land class", "Soil type"]
ALL_TARGETS: list = ["Boron", "Zinc", "SOC"]
DATASET_VERSION: str = "v3" # using the version 3, v3 dataset
RANDOM_STATE: int = 42
TRAIN_SIZE: float = 0.7
TEST_SIZE: float = 0.2
VAL_SIZE: float = 0.1

MODEL_FILE_PATH = Path().absolute() / "data" / "models" # current working directory(i.e root 'IPAGE') + data + models
MODEL_FILE_PATH.mkdir(parents=True, exist_ok=True)


class DataPreprocessor:
    def __init__(
            self, col_to_drop: Iterable = COLS_TO_DROP,
            filename: str = f"merged_{DATASET_VERSION}.csv", filedir: str = "data"
        ):
        self.col_to_drop: Iterable = col_to_drop
        self.data: pd.DataFrame = load_data(filename, filedir)

    def preprocess(self):
        self.data.drop(COLS_TO_DROP, axis=1, inplace=True)
        for col in self.data.select_dtypes(exclude=["object"]):
            self.data = remove_outliers(self.data, col)
        return self.data


class BaseModelData:
    data = None
    X_train = None
    X_test = None
    y_train = None
    y_test = None
    # Code for Modeling pipeline, Model fiting goes here
    def __init__(self, target_variables):
        self.target_variables = target_variables
        self.best_model = None
        self.pipeline = None

    @classmethod
    def split_data(cls, data: pd.DataFrame, target_variables):
        cls.data = data
        # Remove any column in `ALL_TARGET` that has not been selected
        # in `self.target_variables` (i.e. target columns that are not been trained)
        for col in cls.data.columns:
            if col not in target_variables and col in ALL_TARGETS:
                cls.data.drop(col, axis=1, inplace=True)
        X = cls.data.drop(target_variables, axis=1)
        y = cls.data[target_variables]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        cls.X_train = pd.DataFrame(X_train, columns=X.columns)
        cls.X_test = pd.DataFrame(X_test, columns=X.columns)
        cls.y_train = pd.DataFrame(y_train, columns=y.columns)
        cls.y_test = pd.DataFrame(y_test, columns=y.columns)

    def create_pipeline(self, modelClass):
        numeric_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
        categorical_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, make_column_selector(dtype_include=np.number)),
                ("cat", categorical_transformer, make_column_selector(dtype_exclude=np.number))
            ]
        )
        if hasattr(modelClass,'random_state'):
            model = modelClass(random_state=RANDOM_STATE)
        else:
            model = modelClass()

        self.pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ])

    def train(self, modelClass, data, params=None):

        if not self.data:
            self.split_data(data, self.target_variables)
            
        if not self.pipeline:
            if not isinstance(self.X_train, pd.DataFrame) or not isinstance(self.X_test, pd.DataFrame):
                raise ValueError("Input data to the pipeline must be pandas DataFrames.")
            self.create_pipeline(modelClass)

        if params:
            grid_search = GridSearchCV(
                self.pipeline,
                param_grid=params,
                scoring="neg_mean_squared_error",
                cv=3,
                n_jobs=1
            )
            grid_search.fit(self.X_train, self.y_train)

            self.best_model = grid_search.best_estimator_
            print(self.best_model)
        else:
            self.pipeline.fit(self.X_train, self.y_train)
            self.best_model = self.pipeline

    def evaluate(self):
         if not self.pipeline:
             raise ValueError("Model has not been trained.")
         
         predictions_test = self.best_model.predict(self.X_test)
         predictions_train = self.best_model.predict(self.X_train)

         metrics = {
             "r2_score_train": np.round(r2_score(self.y_train, predictions_train),3),
             "mean_squared_error_train": np.round(mean_squared_error(self.y_train, predictions_train),3),
             "mean_absolute_error_train": np.round(mean_absolute_error(self.y_train, predictions_train),3),
             "r2_score_test": np.round(r2_score(self.y_test, predictions_test),3),
             "mean_squared_error_test": np.round(mean_squared_error(self.y_test, predictions_test),3),
             "mean_absolute_error_test": np.round(mean_absolute_error(self.y_test, predictions_test),3)
         }
         return metrics
    
    def make_prediction(self):
        self.predictions = self.best_model.predict(self.X_test)
    
    def save_model(self, filename):
        joblib.dump(self.best_model, MODEL_FILE_PATH / filename)
