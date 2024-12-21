# -*- coding: utf-8 -*-
# """
# kmeans_pca_sidebar.py
# Created on Dec 17, 2024
# @ Author: Mazhar
# """

import pandas as pd
import streamlit as st


def config_kmeans_pca_sidebar(
    data: pd.DataFrame,
    exclude_cols: list[str],
    get_numeric_columns: callable,
    get_categorical_columns: callable,
) -> tuple:
    """
    Configure the Streamlit sidebar for the Dashboard page.

    Args:
        data (pd.DataFrame): The input dataset.
        exclude_cols (List[str]): List of columns to exclude from numeric features.
        get_numeric_columns (callable): Function to get numeric columns.
        get_categorical_columns (callable): Function to get categorical columns.

    Returns:
        Dict[str, any]: A dictionary containing selected filters and configuration values.
    """
    with st.sidebar:
        # Get Numeric Columns
        numerical_features: list[str] = get_numeric_columns(
            data, exclude_columns=exclude_cols
        )

        # Get Categorical Columns
        categorical_features: list[str] = get_categorical_columns(data)

        # Sidebar Title
        st.subheader("Filters")

        # Select Numeric Feature
        numeric_feature_kmeans_pca: str = st.selectbox(
            "Select Numeric Feature",
            numerical_features,
            index=0,  # Adjust index as needed
            key="numeric_kmeans_pca",
        )

        # Select Categorical Feature
        categorical_feature_kmeans_pca: str = st.selectbox(
            "Select Categorical Feature",
            categorical_features,
            index=1,  # Adjust index as needed
            key="categorical_kmeans_pca",
        )

    return numeric_feature_kmeans_pca, categorical_feature_kmeans_pca
