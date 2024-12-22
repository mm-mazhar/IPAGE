# -*- coding: utf-8 -*-
# """
# boron_train_v1.py
# Created on Dec 22, 2024
# Description: Implementation of Multiple Models for Boron Prediction
# with MLflow Integration and Hyperparameter Refinement
# feature selection using permutation importance all models
# @ Author: Mazhar
# """

from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import AdaBoostRegressor, BaggingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils import Bunch
from xgboost import XGBRegressor

# Hyperparameter definitions
MODEL_ESTIMATOR_ALPHA: list = [0.0001, 0.001, 0.01, 0.1, 1, 10]
MODEL_ESTIMATOR_DEPTH: list[int] = [4, 6, 8]
MODEL_ESTIMATOR_NUM_LEAVES: list[int] = [31, 50]
MODEL_ESTIMATOR_MAX_DEPTH: list[int] = [5, 10, 15]
MODEL_ESTIMATOR_N_ESTIMATORS_RF: list[int] = [100, 200]
MODEL_ESTIMATOR_N_ESTIMATORS_ADA: list[int] = [50, 100]
MODEL_ESTIMATOR_N_ESTIMATORS_BAGGING: list[int] = [10, 50]
MODEL_ESTIMATOR_N_NEIGHBORS: list[int] = [5, 10]
MODEL_ESTIMATOR_C_SVR: list = [0.1, 1, 10]

NUM_SIMPLE_IMPUTER: str = "mean"
CAT_SIMPLE_IMPUTER: str = "most_frequent"
ONE_HOT_ENCODER_HANDLE_UNKNOWN: str = "ignore"
MIN_SAMPLE_LEAF: list[int] = [1, 3, 5, 7]
POWER_TRANSFORMER_METHOD: str = "yeo-johnson"
CV: int = 10
# GRIDSEARCHCV_SCORING: str = "neg_mean_absolute_error"
GRIDSEARCHCV_SCORING: str = "neg_root_mean_squared_error"
RANDOM_STATE: int = 0
TEST_SIZE: float = 0.2
N_REPEATS: int = 10

# Load the dataset
# file_path = Path("../../../data/merged_v2.csv")
file_path = Path("./data/merged_v2.csv")
data: pd.DataFrame = pd.read_csv(file_path)

# Define target variable and features
target = "Boron"
features: list[str] = [
    col for col in data.columns if col not in ["SOC", "Zinc", target]
]

# Ensure droping raw longitude and latitude if present
if "longitude" in features and "latitude" in features:
    features.remove("longitude")
    features.remove("latitude")

# Dynamically identify categorical and numerical features
categorical_features: list[str] = (
    data[features].select_dtypes(include=["object", "category"]).columns.tolist()
)
numerical_features: list[str] = (
    data[features].select_dtypes(include=["number"]).columns.tolist()
)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    data[features], data[target], test_size=TEST_SIZE, random_state=RANDOM_STATE
)

# # Ensure y_train and y_test are 1D arrays
# y_train = y_train.values.ravel()  # Or y_train = y_train.values.flatten()
# y_test = y_test.values.ravel()

# Define preprocessing steps
numerical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy=NUM_SIMPLE_IMPUTER)),
        ("scaler", StandardScaler()),
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy=CAT_SIMPLE_IMPUTER)),
        ("onehot", OneHotEncoder(handle_unknown=ONE_HOT_ENCODER_HANDLE_UNKNOWN)),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# Feature selection using permutation importance
pipeline_for_perm_imp = Pipeline(
    [
        ("preprocessor", preprocessor),
        ("rf", RandomForestRegressor(random_state=RANDOM_STATE)),
    ]
)

pipeline_for_perm_imp.fit(X_train, y_train)
perm_importance: Bunch | dict[str, Bunch] = permutation_importance(
    pipeline_for_perm_imp,
    X_train,
    y_train,
    n_repeats=N_REPEATS,
    random_state=RANDOM_STATE,
)
importances: Any = perm_importance.importances_mean

# Sort the feature importances in descending order
feature_importances: list = sorted(
    zip(features, importances), key=lambda x: x[1], reverse=True
)
# top_n_features: list = [
#     feature for feature, score in feature_importances[: len(feature_importances)]
# ]
top_n_features: list = [feature for feature, score in feature_importances[:6]]

# Update X data with top features
X: pd.DataFrame = data[top_n_features]
X_train, X_test, y_train, y_test = train_test_split(
    X, data[target], test_size=TEST_SIZE, random_state=RANDOM_STATE
)

# # Ensure y_train and y_test are 1D arrays
# y_train = y_train.values.ravel()  # Or y_train = y_train.values.flatten()
# y_test = y_test.values.ravel()

# Dynamically identify updated categorical and numerical features
categorical_features_top_features: Any = X_train.select_dtypes(
    include=["object", "category"]
).columns.tolist()
numerical_features_top_features: Any = X_train.select_dtypes(
    include=["number"]
).columns.tolist()

# Define preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_features_top_features),
        ("cat", categorical_transformer, categorical_features_top_features),
    ]
)

# Define models and hyperparameter grids
models: dict[str, Any] = {
    "Lasso": (
        Lasso(),
        {"model__alpha": MODEL_ESTIMATOR_ALPHA},
    ),
    "CatBoost Regression": (
        CatBoostRegressor(verbose=0),
        {"model__depth": MODEL_ESTIMATOR_DEPTH},
    ),
    "LightGBM Regressor": (
        LGBMRegressor(),
        {"model__num_leaves": MODEL_ESTIMATOR_NUM_LEAVES},
    ),
    "LinearRegression": (LinearRegression(), {}),
    "DecisionTree": (
        DecisionTreeRegressor(),
        {"model__max_depth": MODEL_ESTIMATOR_MAX_DEPTH},
    ),
    "RandomForest": (
        RandomForestRegressor(random_state=RANDOM_STATE),
        {
            "model__n_estimators": MODEL_ESTIMATOR_N_ESTIMATORS_RF,
            "model__max_depth": MODEL_ESTIMATOR_MAX_DEPTH,
        },
    ),
    "AdaBoost": (
        AdaBoostRegressor(random_state=RANDOM_STATE),
        {"model__n_estimators": MODEL_ESTIMATOR_N_ESTIMATORS_ADA},
    ),
    "Bagging": (
        BaggingRegressor(random_state=RANDOM_STATE),
        {"model__n_estimators": MODEL_ESTIMATOR_N_ESTIMATORS_BAGGING},
    ),
    "KNeighbors": (
        KNeighborsRegressor(),
        {"model__n_neighbors": MODEL_ESTIMATOR_N_NEIGHBORS},
    ),
    "SVR": (
        SVR(),
        {"model__C": MODEL_ESTIMATOR_C_SVR},
    ),
    "XGB": (
        XGBRegressor(random_state=RANDOM_STATE),
        {"model__n_estimators": MODEL_ESTIMATOR_N_ESTIMATORS_RF},
    ),
}

print(f"Top Numerical Features: {numerical_features_top_features}")
print(f"Top Numerical Features: {categorical_features_top_features}")

# Start MLflow experiment
mlflow.set_experiment(f"{target} Prediction with Multiple Models (V1)")

best_model_path = None
best_r2_score: float = -np.inf

for model_name, (model, param_grid) in models.items():
    print(f"Running GridSearchCV for model: {model_name}")
    print("*" * 100, "\n")

    # Create pipeline with the current model
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

    # Define GridSearchCV
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=CV,
        scoring=GRIDSEARCHCV_SCORING,
        verbose=1,
        n_jobs=-1,
    )

    with mlflow.start_run(run_name=model_name):
        # Train the model with hyperparameter tuning
        grid_search.fit(X_train, y_train)

        # Get the best estimator and its parameters
        best_boron_pipeline_v1: Pipeline = grid_search.best_estimator_
        best_params: dict = grid_search.best_params_

        # Log the best parameters
        mlflow.log_params(best_params)

        # Log the best score
        mlflow.log_metric("best_cv_score", -grid_search.best_score_)

        # Provide an input example for the model
        input_example = pd.DataFrame([X_test.iloc[0].values], columns=X_test.columns)

        # Log the model with input example
        mlflow.sklearn.log_model(
            best_boron_pipeline_v1,
            artifact_path="best_boron_pipeline_v1",
            input_example=input_example,
        )

        # Make predictions using the trained model
        predictions = best_boron_pipeline_v1.predict(X_test)

        # Evaluate the model
        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        # Log metrics
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2", r2)

        print(f"{model_name} Results:")
        print("*" * 50, "\n")
        print(f"Mean Absolute Error (MAE): {mae:.4f}")
        print(f"Mean Squared Error (MSE): {mse:.4f}")
        print(f"R-Squared (R²): {r2:.4f}")
        print(f"Best parameters: {best_params}")
        print("\n", "*" * 100)

        # Save the best model based on R-squared score
        if r2 > best_r2_score:
            best_r2_score = r2
            best_r2_score = round(best_r2_score, 4)
            best_model_path = Path(
                f"./notebooks/task_3_model_development_and_training/by-Maz/checkpoints/boron_{model_name}_r2_{best_r2_score}_v1.pkl"
            )
            joblib.dump(best_boron_pipeline_v1, best_model_path)

if best_model_path:
    print(f"Best model saved to {best_model_path}")
