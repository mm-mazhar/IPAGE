# -*- coding: utf-8 -*-
# """
# utils.py
# Created on Nov 07, 2024
# @ Author: Mazhar
# """

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import streamlit as st
import streamlit.components.v1 as components
from plotly.graph_objs._figure import Figure


# Load dataset
@st.cache_data
def load_data(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path)


# Display centered title H1
def display_centered_title(title: str, color: str = "black") -> None:
    """Display a centered title in Streamlit with an optional color."""
    st.markdown(
        f"<h1 style='text-align: center; color: {color};'>{title}</h1>",
        unsafe_allow_html=True,
    )

# Display centered title H3
def title_h3(title: str, color: str = "black") -> None:
    """Display a centered title in Streamlit with an optional color."""
    st.markdown(
        f"<h3 style='text-align: center; color: {color};'>{title}</h3>",
        unsafe_allow_html=True,
    )


def color_ideal_values(val, min_val, max_val, color) -> str:
    return f"color: {color};" if min_val <= val <= max_val else ""


def highlight_ideal_values(val, min_val, max_val, color) -> str:
    color = f"background-color: {color}" if min_val <= val <= max_val else ""
    return color


def get_categorical_columns(df: pd.DataFrame) -> list[str]:
    """Return a list of categorical column names from the DataFrame."""
    return df.select_dtypes(include=["object", "category"]).columns.tolist()


def get_numeric_columns(df: pd.DataFrame) -> list[str]:
    """Return a list of numeric column names from the DataFrame."""
    return df.select_dtypes(include=[np.number]).columns.tolist()


def visualize_distributions(
    df: pd.DataFrame, numeric_cols: list[str], categorical_cols: list[str]
) -> None:
    """Visualize the distribution of selected numeric and categorical columns using Plotly."""
    st.subheader("Feature Selection")

    # Create two columns
    col1, col2 = st.columns(2)

    # Use the first column for numeric feature selection
    with col1:
        st.write("Numeric Features")
        selected_numeric_features: list[str] = st.multiselect(
            "Select numeric features to visualize", numeric_cols, index=1
        )

    # Use the second column for categorical feature selection
    with col2:
        st.write("Categorical Features")
        selected_categorical_features: list[str] = st.multiselect(
            "Select categorical features to visualize", categorical_cols, index=1
        )

    # Display plots for selected numeric features
    for feature in selected_numeric_features:
        st.subheader(f"Distribution of {feature} (Numeric)")
        fig: Figure = px.histogram(
            df,
            x=feature,
            nbins=30,
            title=f"Distribution of {feature}",
            marginal="rug",
            color_discrete_sequence=["blue"],
        )
        fig.update_layout(bargap=0.2)
        st.plotly_chart(fig)

    # Display plots for selected categorical features
    for feature in selected_categorical_features:
        st.subheader(f"Distribution of {feature} (Categorical)")
        fig = px.histogram(
            df,
            x=feature,
            title=f"Distribution of {feature}",
            color_discrete_sequence=["green"],
        )
        fig.update_layout(bargap=0.2)
        st.plotly_chart(fig)
