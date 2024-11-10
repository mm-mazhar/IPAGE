# -*- coding: utf-8 -*-
# """
# feature_analysis.py
# Created on Nov 07, 2024
# @ Author: Mazhar
# """

import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st


# Define function to display a statistic card with a small plot
def display_stat_card(column_name, data, figsize=(2, 1), background_color="gray",color="white") -> None:

    # Select only columns with numerical data
    numerical_data = data.select_dtypes(include=["number"])

    # Calculate key statistics
    max_val = data[column_name].max()
    mean_val = data[column_name].mean()
    min_val = data[column_name].min()

    # Calculate the correlation matrix once for efficiency
    correlation_matrix = numerical_data.corr()

    # Get the highest absolute correlation for the target column, excluding self-correlation
    correlations = correlation_matrix[column_name].drop(column_name)
    max_corr_val = correlations.abs().max()
    most_correlated_column = correlations.abs().idxmax()

    stat_col, plot_col = st.columns([1, 1.5])

    with stat_col:
        st.markdown(
            f"<b style='font-size: 16px;'>{column_name}</b>", unsafe_allow_html=True
        )
        st.markdown(
            f"""
            <div style="line-height: 0.8; font-size: 0.1px;">
                <p style="margin: 0; padding: 0;">Min: {min_val:.2f}, Max:  {max_val:.2f}</p>
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
