# -*- coding: utf-8 -*-
# """
# model_utils.py
# Created on Dec 25, 2024
# @ Author: Mazhar
# """

from typing import Any

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import boxcox
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PowerTransformer


def plot_geographical_features(
    df, targets, lat_col="latitude", lon_col="longitude", figsize=(18, 6)
):
    """
    Plot geographical distribution of targets on a map.

    Parameters:
    - df (pd.DataFrame): DataFrame containing data.
    - features (list of str): List of feature column names to plot.
    - lat_col (str): Column name for latitude.
    - lon_col (str): Column name for longitude.
    - figsize (tuple): Figure size.
    """
    # Number of features determines the number of subplots
    num_targets = len(targets)

    # Create subplots
    fig, axes = plt.subplots(
        1, num_targets, figsize=figsize, subplot_kw={"projection": ccrs.PlateCarree()}
    )

    # Ensure axes is iterable for a single subplot
    if num_targets == 1:
        axes = [axes]

    # Loop through each feature and corresponding axis
    for i, (target, ax) in enumerate(zip(targets, axes)):
        # Add geographical features
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linestyle=":")
        ax.add_feature(cfeature.LAND, edgecolor="black")
        ax.add_feature(cfeature.LAKES, edgecolor="black")
        ax.add_feature(cfeature.RIVERS, edgecolor="blue")
        ax.add_feature(cfeature.STATES, edgecolor="red")

        # Scatter plot for the feature
        scatter = ax.scatter(
            df[lon_col],
            df[lat_col],
            c=df[target],
            cmap="viridis",
            s=50,
            alpha=0.7,
            transform=ccrs.PlateCarree(),
        )

        # Add a title for each subplot
        ax.set_title(f"{target} Distribution")

        # Add a colorbar for each subplot
        cbar = fig.colorbar(
            scatter, ax=ax, orientation="vertical", shrink=0.7, label=target
        )

    # Add a main title for the figure
    fig.suptitle("Geographical Distribution", fontsize=16)

    # Adjust spacing between subplots
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Show the plots
    plt.show()


def get_overfitting_status(val_r2, test_r2):
    """
    Determines the overfitting status based on the difference between validation R² and test R².

    Args:
        val_r2 (float): Validation R² score.
        test_r2 (float): Test R² score.

    Returns:
        tuple: (overfitting_status (str), overfitting_numeric (int))
    """
    diff = abs(val_r2 - test_r2)
    if diff <= 0.02:
        return "Not Overfitting", 0
    elif 0.02 < diff <= 0.06:
        return "Slight Overfitting", 1
    else:
        return "High Overfitting", 2


def categorical_value_counts_to_df(df) -> pd.DataFrame:
    """
    Calculates value counts for each categorical column in a DataFrame and
    returns the results in a new DataFrame.

    Args:
        df: Pandas DataFrame.

    Returns:
        Pandas DataFrame: A DataFrame containing value counts, unique counts,
                         and column names.
    """
    results: list = []
    for column in df.select_dtypes(include=["object"]):
        value_counts: Any = df[column].value_counts()
        unique_count: Any = df[column].nunique()

        for value, count in value_counts.items():
            results.append(
                {
                    "Categorical Feature": column,
                    "Sub-Category": value,
                    "Each Count": count,
                    "Total Unique Count": unique_count,
                }
            )

    return pd.DataFrame(results)


def group_low_frequency_categories(df, threshold=5) -> pd.DataFrame:
    """
    Groups low-frequency categories in all categorical columns of a DataFrame into 'Other'.

    Args:
        df: Pandas DataFrame.
        threshold: The minimum count for a category to be kept as is. Categories with counts
                 less than or equal to this value will be replaced by 'Other'.

    Returns:
        Pandas DataFrame: The modified DataFrame.
    """

    for column in df.select_dtypes(include=["object"]):
        value_counts = df[column].value_counts()
        df[column] = df[column].apply(
            lambda x: x if value_counts[x] > threshold else "Others"
        )
    return df


# Function to calculate the range without outliers
def range_without_outliers(df, feature) -> tuple:
    Q1: Any = df[feature].quantile(0.25)  # First quartile (25th percentile)
    Q3: Any = df[feature].quantile(0.75)  # Third quartile (75th percentile)
    IQR: Any = Q3 - Q1  # Interquartile Range
    lower_bound: Any = Q1 - 1.5 * IQR
    upper_bound: Any = Q3 + 1.5 * IQR

    # Filter out outliers
    filtered_data: Any = df[(df[feature] >= lower_bound) & (df[feature] <= upper_bound)]

    # Return the range
    min_val: Any = filtered_data[feature].min()
    max_val: Any = filtered_data[feature].max()

    return min_val, max_val


# Function to remove outliers based on IQR
def remove_outliers(df, feature) -> pd.DataFrame:
    # Q1 = df[feature].quantile(0.25)  # First quartile (25th percentile)
    # Q3 = df[feature].quantile(0.75)  # Third quartile (75th percentile)
    Q1: Any = df[feature].quantile(0.05)  # First quartile (5th percentile)
    Q3: Any = df[feature].quantile(0.95)  # Third quartile (95th percentile)
    IQR: Any = Q3 - Q1  # Interquartile Range
    lower_bound: Any = Q1 - 1.5 * IQR
    upper_bound: Any = Q3 + 1.5 * IQR

    # Filter the dataset to include only non-outliers
    df: pd.DataFrame = df[(df[feature] >= lower_bound) & (df[feature] <= upper_bound)]
    return df


# Function to classify skewness
def classify_skewness(skewness) -> Any:
    if abs(skewness) > 2:
        return (
            "Highly Skewed",
            "Recommend Log Transformation (if positive) or Yeo-Johnson",
        )
    elif abs(skewness) > 1:
        return "Moderately Skewed", "Recommend Yeo-Johnson Transformation"
    elif abs(skewness) > 0.5:
        return "Slightly Skewed", "Transformation optional"
    else:
        return "Symmetrical", "No transformation needed"


def transform_features_for_skewness(
    features, df, transformation_methods=["log1p", "sqrt", "yeo-johnson", "boxcox"]
) -> tuple[pd.DataFrame, dict]:
    """
    Check and transform skewness of features in a DataFrame.

    Parameters:
        features (list): List of numerical features to check and transform.
        df (pd.DataFrame): DataFrame containing the features.
        transformation_methods (list): List of transformations to try (log1p, sqrt, yeo-johnson, boxcox).

    Returns:
        pd.DataFrame: Transformed features DataFrame.
        dict: Skewness report.
    """
    transformed_features = pd.DataFrame()
    skewness_report: dict = {}

    for feature in features:
        original_skewness: Any = df[feature].skew()
        skewness_report[feature] = {"Original Skewness": original_skewness}

        # Initialize best transformation
        best_transformation = None
        best_skewness: Any = abs(original_skewness)
        best_transformed_data: Any = df[feature]

        # Apply transformations
        for method in transformation_methods:
            transformed_data = None
            if method == "log1p" and (df[feature] > 0).all():
                transformed_data: Any = np.log1p(df[feature])
            elif method == "sqrt" and (df[feature] >= 0).all():
                transformed_data = np.sqrt(df[feature])
            elif method == "yeo-johnson":
                yeo_transformer = PowerTransformer(method="yeo-johnson")
                transformed_data = yeo_transformer.fit_transform(
                    df[[feature]]
                ).flatten()
            elif method == "boxcox" and (df[feature] > 0).all():
                transformed_data, _ = boxcox(df[feature])
            else:
                continue  # Skip if conditions are not met

            # Calculate skewness
            if transformed_data is not None:
                transformed_skewness = pd.Series(transformed_data).skew()

                # Update best transformation if skewness is improved
                if abs(transformed_skewness) < best_skewness:
                    best_skewness = abs(transformed_skewness)
                    best_transformation = method
                    best_transformed_data = transformed_data

        # Save transformed data and skewness
        transformed_features[feature] = best_transformed_data
        skewness_report[feature]["Best Skewness"] = best_skewness
        skewness_report[feature]["Best Transformation"] = (
            best_transformation or "None (No improvement)"
        )
        skewness_report = pd.DataFrame(skewness_report)

    return transformed_features, skewness_report


# # Example usage
# numerical_features = ["Feature1", "Feature2", "Feature3"]  # Replace with actual numerical feature names
# transformed_features, skewness_report = transform_features_for_skewness(numerical_features, df)

# # Display skewness report
# for feature, report in skewness_report.items():
#     print(f"Feature: {feature}")
#     print(f"  Original Skewness: {report['Original Skewness']:.4f}")
#     print(f"  Best Skewness: {report['Best Skewness']:.4f}")
#     print(f"  Best Transformation: {report['Best Transformation']}")
#     print("-" * 50)


def transform_targets(
    targets, skewness_threshold=0.75, specific_transformations=None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Applies transformations (Log, Yeo-Johnson, Box-Cox) to reduce skewness of target variables.

    Parameters:
    - targets: pd.DataFrame or pd.Series. Target variables to transform.
    - skewness_threshold: float. Threshold of skewness to decide if transformation is required.
    - specific_transformations: dict. Specify a transformation for specific targets (e.g., {'Boron': 'boxcox'}).

    Returns:
    - transformed_targets: pd.DataFrame. Transformed target variables.
    - transformation_report: pd.DataFrame. Report of skewness and transformations applied.
    """
    # Ensure input is DataFrame for consistency
    if isinstance(targets, pd.Series):
        targets: pd.DataFrame = targets.to_frame()

    # Initialize transformers
    log_transform = lambda x: np.log1p(x)
    yeo_johnson = PowerTransformer(method="yeo-johnson")

    # Containers for transformed targets and the report
    transformed_targets = pd.DataFrame(index=targets.index)
    transformation_report: dict = {
        "Target": [],
        "Original Skewness": [],
        "Final Skewness": [],
        "Selected Transformation": [],
        "Lambda (if applicable)": [],
    }

    # Iterate over each target
    for target in targets.columns:
        original_skew = targets[target].skew()
        transformation_report["Target"].append(target)
        transformation_report["Original Skewness"].append(original_skew)

        if abs(original_skew) < skewness_threshold:
            # Skewness is within acceptable range; no transformation needed
            transformed_targets[target] = targets[target]
            transformation_report["Final Skewness"].append(original_skew)
            transformation_report["Selected Transformation"].append("None")
            transformation_report["Lambda (if applicable)"].append(None)
            continue

        # Apply transformations and compute skewness
        transformed_data = {}
        skewness_results = {}

        # Log Transformation (only if all values > 0)
        if (targets[target] > 0).all():
            transformed_data["log"] = log_transform(targets[target])
            skewness_results["log"] = transformed_data["log"].skew()

        # Yeo-Johnson Transformation
        transformed_data["yeo_johnson"] = yeo_johnson.fit_transform(
            targets[[target]]
        ).flatten()
        skewness_results["yeo_johnson"] = pd.Series(
            transformed_data["yeo_johnson"]
        ).skew()

        # Box-Cox Transformation (only if all values > 0 and specified in specific_transformations)
        if (
            specific_transformations is not None
            and target in specific_transformations
            and specific_transformations[target] == "boxcox"
            and (targets[target] > 0).all()
        ):
            transformed_data["boxcox"], fitted_lambda = boxcox(targets[target])
            skewness_results["boxcox"] = pd.Series(transformed_data["boxcox"]).skew()
        else:
            fitted_lambda = None

        # Select the transformation with the least skewness
        selected_transformation = min(
            skewness_results, key=lambda k: abs(skewness_results[k])
        )
        transformed_targets[target] = transformed_data[selected_transformation]

        # Populate the report
        transformation_report["Final Skewness"].append(
            skewness_results[selected_transformation]
        )
        transformation_report["Selected Transformation"].append(selected_transformation)
        transformation_report["Lambda (if applicable)"].append(
            fitted_lambda if selected_transformation == "boxcox" else None
        )

    # Create report as a DataFrame
    transformation_report_df = pd.DataFrame(transformation_report)

    return transformed_targets, transformation_report_df


class SkewnessTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self, transformation_methods=["log1p", "sqrt", "yeo-johnson", "boxcox"]
    ) -> None:
        self.transformation_methods: Any = transformation_methods
        self.skewness_report: dict = {}

    def fit(self, X, y=None):
        return self  # No fitting required for transformations

    def transform(self, X) -> pd.DataFrame:
        # Apply the `transform_features_for_skewness` function
        transformed_features, skewness_report = transform_features_for_skewness(
            features=X.columns.tolist(),
            df=X,
            transformation_methods=self.transformation_methods,
        )
        self.skewness_report = skewness_report
        return transformed_features
