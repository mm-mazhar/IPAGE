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
from typing import List, Dict, Union, Type, Iterable
from dataclasses import dataclass
from src.model.config import RANDOM_STATE, TEST_SIZE, MODEL_FILE_PATH, ALL_TARGETS, COLS_TO_DROP
from sklearn.base import BaseEstimator, TransformerMixin


class OutlierTransformer(BaseEstimator, TransformerMixin):
    """Custom transformer for handling outliers in numeric data."""
    def __init__(self, method="iqr", threshold=1.5):
        self.method = method
        self.threshold = threshold

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if self.method == "iqr":
            for col in X.columns:
                Q1 = X[col].quantile(0.25)
                Q3 = X[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - self.threshold * IQR
                upper_bound = Q3 + self.threshold * IQR
                X[col] = X[col].clip(lower_bound, upper_bound)
        return X


@dataclass
class DataPreprocessor:
    """Handles preprocessing of the dataset"""
    filename: str
    filedir: str
    cols_to_drop: Union[List[str], None] = None

    def __post_init__(self):
        self.data = load_data(self.filename, self.filedir)
        if not self.cols_to_drop:
            self.cols_to_drop = COLS_TO_DROP
    
    def preprocess(self):
        """Preprocess the dataset"""
        self.data.drop(self.cols_to_drop, axis=1, inplace=True)
        return self.data


@dataclass
class BaseModel:
    """Handles the machine learning pipeline, training, and evaluation."""
    target_variables: List[str]
    model_class: Type = RandomForestRegressor
    params: Dict = None
    best_model: Union[Pipeline, None] = None
    pipeline: Union[Pipeline, None] = None


    def __post_init__(self):
        self.X_train = pd.DataFrame()
        self.X_test = pd.DataFrame()
        self.y_train = pd.DataFrame()
        self.y_test = pd.DataFrame()

    def split_data(self, data: pd.DataFrame, target_variables):
        """ Split the dataset into training and testing sets """
        # Remove any column in `ALL_TARGETS` that has not been selected
        # in `self.target_variables` (i.e. target columns that are not been trained)
        drop_targets = [col for col in ALL_TARGETS if col not in target_variables]
        data.drop(drop_targets, axis=1, inplace=True)

        X = data.drop(target_variables, axis=1)
        y = data[target_variables]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )

    def create_pipeline(self):
        """Create a machine learning pipeline"""
        numeric_transformer = Pipeline(steps=[
            ("outlier", OutlierTransformer()),
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
        categorical_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, make_column_selector(dtype_include=["int", "float"])),
                ("cat", categorical_transformer, make_column_selector(dtype_exclude=["int", "float"]))
            ]
        )
        if hasattr(modelClass,'random_state'):
            model = modelClass(random_state=RANDOM_STATE)
        else:
            model = modelClass()

        self.pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", self.model_class(random_state=RANDOM_STATE))
        ])

    def train(self, data, modelClass=RandomForestRegressor, params=None):
        
        if self.X_train.empty or self.X_test.empty or self.y_train.empty or self.y_test.empty:
            self.split_data(data, self.target_variables)
            
        if not self.pipeline:
            self.create_pipeline()

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

    def evaluate(self) -> Dict[str, float]:
        """Evaluate the trained model"""
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
    
    def save_model(self, filename):
        joblib.dump(self.best_model, MODEL_FILE_PATH / filename)
