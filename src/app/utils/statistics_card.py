# -*- coding: utf-8 -*-
# """
# feature_analysis.py
# Created on Nov 07, 2024
# @ Author: Mazhar
# """

from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st


# Define function to display a statistic card with a small plot
def display_stat_card(
    column_name: str,
    data: pd.DataFrame,
    figsize: tuple[int, int] = (2, 1),
    background_color: str = "gray",
    color: str = "white",
    exclude_cols: list[str] = None,
) -> None:

    # Select only columns with numerical data with exclude_cols
    data: pd.DataFrame = data.drop(columns=exclude_cols)
    numerical_data: pd.DataFrame = data.select_dtypes(include=["number"])

    # Calculate key statistics
    max_val: float = data[column_name].max()
    mean_val: float = data[column_name].mean()
    min_val: float = data[column_name].min()

    # Calculate the correlation matrix once for efficiency
    correlation_matrix: pd.DataFrame = numerical_data.corr()

    # Get the highest absolute correlation for the target column, excluding self-correlation
    correlations: pd.Series = correlation_matrix[column_name].drop(column_name)
    max_corr_val: float = correlations.abs().max()
    most_correlated_column: str = correlations.abs().idxmax()

    stat_col, plot_col = st.columns([1, 1.5])

    with stat_col:
        st.markdown(
            f"<b style='font-size: 16px;'>{column_name}</b>", unsafe_allow_html=True
        )
        # st.markdown(
        #     f"""
        #     <div style="line-height: 0.8; font-size: 0.1px;">
        #         <p style="margin: 0; padding: 0;">Min: {min_val:.2f}, Max:  {max_val:.2f}</p>
        #         <p style="margin: 0; padding: 0;">Mean: {mean_val:.2f}</p>
        #         <p style="margin: 0; padding: 0;">Corr: {max_corr_val:.2f} with '{most_correlated_column}'</p>
        #     </div>
        #     """,
        #     unsafe_allow_html=True,
        # )
        st.markdown(
            f"""
            <div style="line-height: 0.8; font-size: 0.1px;">
                <p style="margin: 0; padding: 0;">Mean: {mean_val:.2f}</p>
                <p style="margin: 0; padding: 0;">Corr: {max_corr_val:.2f} with '{most_correlated_column}'</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with plot_col:
        # Plot distribution
        fig, ax = plt.subplots(figsize=figsize)  # Use figsize as a keyword argument
        sns.histplot(data[column_name], ax=ax, kde=True, color=color)

        # Set background color
        ax.set_facecolor(background_color)

        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticks([])
        ax.set_yticks([])
        st.pyplot(fig)
