from src.data.load_data import load_data
from src.data.preprocess_data import remove_duplicates, remove_outliers, get_geospacial_details
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.imputer import SimpleImputer
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
            self, col_to_drop: Iterable,
            filename: str = f"merged_{DATASET_VERSION}.csv", filedir: str = "data"
        ):
        self.col_to_drop: Iterable = col_to_drop
        self.data: pd.DataFrame = load_data(filename, filedir)

    def preprocess(self):
        self.data.drop(COLS_TO_DROP, axis=1, inplace=True)
        for col in self.data.select_dtypes(exclude=["object"]):
            self.data = remove_outliers(self.data, col)
        return self.data


class ModelData:
    # Code for Modeling pipeline, Model fiting goes here
    def __init__(self, target_variables, random_state=RANDOM_STATE):
        self.target_variables = target_variables
        self.random_state = random_state
        self.best_model = None
        self.pipeline = None

    def create_pipeline(self):

        # create list of numeric and categorical features
        numeric_features = [col for col in self.data if col not in self.target_variables and self.data[col].dtype != 'object']
        categorical_features = [col for col in self.data if col not in self.target_variables and self.data[col].dtype == 'object']

        numeric_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(handle_unknown="ignore")

        missing_value_imputer = ColumnTransformer(
            transformers=[
                ("num", SimpleImputer(strategy="median"), numeric_features),
                ("cat", SimpleImputer(strategy="constant", fill_value="missing"), categorical_features)
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features)
            ]
        )

        # Create a simple base model
        model = RandomForestRegressor(random_state=self.random_state)

        self.pipeline = Pipeline(steps=[
            ("imputer", missing_value_imputer),
            ("preprocessor", preprocessor),
            ("model", model)
        ])

    def train(self, X_train, y_train, params=None):
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
            grid_search.fit(X_train, y_train)

            self.best_model = grid_search.best_estimator_
        else:
            self.pipeline.fit(X_train, y_train)
            self.best_model = self.pipeline

    def evaluate(self, X_test, y_test):
         predictions = self.best_model.predict(X_test)

         metrics = {
             "r2_score": r2_score(X_test, predictions),
             "mean_squared_error": mean_squared_error(X_test, predictions),
             "mean_absolute_error": mean_absolute_error(X_test, predictions)
         }
         return metrics
    
    def save_model(self, filename):
        joblib.dump(self.best_model, MODEL_FILE_PATH / filename)
