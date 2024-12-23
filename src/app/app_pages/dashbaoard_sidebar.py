# -*- coding: utf-8 -*-
# """
# dashboard_sidebar.py
# Created on Dec 17, 2024
# @ Author: Mazhar
# """

import pandas as pd
import streamlit as st


def config_dashboard_sidebar(
    data: pd.DataFrame,
    exclude_cols: list[str],
    get_numeric_columns: callable,
    get_categorical_columns: callable,
) -> dict[str, any]:
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
        # Bar Plot Methods
        methods: list[str] = ["mean", "sum", "count"]

        # Get Numeric and Categorical Columns
        numerical_features: list[str] = get_numeric_columns(
            data, exclude_columns=exclude_cols
        )
        categorical_features: list[str] = get_categorical_columns(data)

        # Filters Section
        st.subheader("Filters")

        # Method Selection
        selected_method: str = st.selectbox(
            "Select Method", methods, index=0, key="method"
        )

        # Numeric Feature Selection for Y-axis
        selected_numeric_feature: str = st.selectbox(
            # "Select Numeric Feature for Y-axis",
            "Select Numeric Feature",
            numerical_features,
            index=0,
            key="numeric",
        )

        # Categorical Feature Selection
        selected_categorical_feature: str = st.selectbox(
            "Select Categorical Feature",
            categorical_features,
            index=1,
            key="categorical",
        )

        # Numeric Feature Selection for X-axis
        selected_numeric_feature_x: str = st.selectbox(
            "Select Numeric Feature for X-axis (Scatter Plot)",
            numerical_features,
            index=1,
            key="numeric_x",
        )

        # Stacked Bar Chart Filter
        st.subheader("Stacked Bar Chart Filter")
        selected_features: list[str] = st.multiselect(
            "Select Categorical Features",
            categorical_features,
            default=categorical_features[:2],
            key="plot_stacked_bar_chart_categorical",
        )

        # Data Table Filter
        st.subheader("Data Table Filter")
        all_columns: list[str] = data.columns.tolist()
        selected_columns: list[str] = st.multiselect(
            "Select columns to display",
            all_columns,
            default=all_columns,
            key="data_table_columns",
        )

    # Return all selected values and configurations
    return {
        "selected_method": selected_method,
        "selected_numeric_feature": selected_numeric_feature,
        "selected_numeric_feature_x": selected_numeric_feature_x,
        "selected_categorical_feature": selected_categorical_feature,
        "selected_features": selected_features,
        "selected_columns": selected_columns,
    }
