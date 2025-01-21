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
from src.logs.logger import setLogger
from src.logs.config import DATA_LOG_FILE, MODEL_LOG_FILE
import sys

# Set up loggers
data_logger = setLogger(__name__, DATA_LOG_FILE, display_console=False)
model_logger = setLogger(__name__, MODEL_LOG_FILE, display_console=False)


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
            self.cols_to_drop = [col for col in COLS_TO_DROP if col in self.data.columns]
    
    def preprocess(self):
        """Preprocess the dataset"""
        self.data.drop(self.cols_to_drop, axis=1, inplace=True)
        data_logger.info(
            f'Columns {", ".join(self.cols_to_drop)} have been dropped from the dataset {self.filename}'
        )
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
        model_logger.info("Data split into training and testing sets")

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

        self.pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", self.model_class(random_state=RANDOM_STATE))
        ])
        model_logger.info(f"Pipeline created with {self.model_class.__name__}")

    def train(self, data, modelClass=RandomForestRegressor, params=None):
        
        if self.X_train.empty or self.X_test.empty or self.y_train.empty or self.y_test.empty:
            self.split_data(data, self.target_variables)
            
        if not self.pipeline:
            model_logger.info("Creating pipeline")
            self.create_pipeline()

        if params:
            model_logger.info("Hyperparameter tuning")
            model_logger.info(f"\nParameters: {params}")
            grid_search = GridSearchCV(
                self.pipeline,
                param_grid=params,
                scoring="neg_mean_squared_error",
                cv=3,
                n_jobs=1
            )
            grid_search.fit(self.X_train, self.y_train)

            self.best_model = grid_search.best_estimator_
            model_logger.info("Best model selected")
            model_logger.info(f"\nBest model: {self.best_model}")
        else:
            self.pipeline.fit(self.X_train, self.y_train)
            self.best_model = self.pipeline

    def evaluate(self) -> Dict[str, float]:
        """Evaluate the trained model"""
        if not self.pipeline:
             model_logger.error("Model has not been trained.")
             raise ValueError("Model has not been trained.")
         
        predictions_test = self.best_model.predict(self.X_test)
        predictions_train = self.best_model.predict(self.X_train)

        metrics = [
            {
                "subject": "Test",
                "r2_score": np.round(r2_score(self.y_test, predictions_test),3),
                "mean_squared_error": np.round(mean_squared_error(self.y_test, predictions_test),3),
                "mean_absolute_error": np.round(mean_absolute_error(self.y_test, predictions_test),3)
            },
            {
                "subject": "Train",
                "r2_score": np.round(r2_score(self.y_train, predictions_train),3),
                "mean_squared_error": np.round(mean_squared_error(self.y_train, predictions_train),3),
                "mean_absolute_error": np.round(mean_absolute_error(self.y_train, predictions_train),3)
            }
        ]
        return metrics

    def save_model(self, filename):
        """Save the best model"""
        data_logger.info(f"Saving model to {MODEL_FILE_PATH / filename}")
        model_logger.info(f"Saving model to {MODEL_FILE_PATH / filename}")
        joblib.dump(self.best_model, MODEL_FILE_PATH / filename)
        model_logger.info("Model saved successfully")
        data_logger.info("Model saved successfully")

    @staticmethod
    def load_model(filename):
        """Load a saved model"""
        try:
            model = joblib.load(MODEL_FILE_PATH / filename)
            model_logger.info(f"Model loaded from {MODEL_FILE_PATH / filename}")
            return model
        except FileNotFoundError:
            model_logger.error(f"Model not found at {MODEL_FILE_PATH / filename}")
            raise FileNotFoundError(f"Model not found at {MODEL_FILE_PATH / filename}")
        except Exception as e:
            model_logger.error(f"An error occurred: {e}")
            raise e