# -*- coding: utf-8 -*-
# """
# elbow_curve_plot.py
# Created on Nov 07, 2024
# @ Author: Mazhar
# """


import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def plot_elbow_curve(
    data: pd.DataFrame,
    max_clusters: int = 10,
    background_color: str = "white",
    text_color: str = "black",
) -> None:
    """
    Plots the elbow curve to determine the optimal number of clusters.

    Parameters:
    - data: pd.DataFrame - The input data for clustering.
    - max_clusters: int - The maximum number of clusters to test.
    """
    # Selecting only the numeric columns for clustering and standardizing the data
    numeric_data: pd.DataFrame = data.select_dtypes(include=["float64"]).dropna()
    scaler = StandardScaler()
    scaled_data: np.ndarray = scaler.fit_transform(numeric_data)

    # Calculate inertia for a range of cluster numbers
    inertia: list = []
    K = range(1, max_clusters + 1)  # Test a range of cluster numbers

    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42, verbose=0)
        kmeans.fit(scaled_data)
        inertia.append(kmeans.inertia_)

    # Create the elbow curve using Plotly Express
    fig = px.line(
        x=list(K),
        y=inertia,
        markers=True,
        labels={"x": "Number of Clusters", "y": "Inertia"},
        title="Elbow Method for Optimal Number of Clusters",
    )
    # Customize layout
    fig.update_layout(
        plot_bgcolor=background_color,  # Background color
        paper_bgcolor=background_color,  # Paper background color
        font=dict(color=text_color),  # General text color
        title_font=dict(color=text_color),
        xaxis=dict(
            title=dict(
                text="Number of Clusters",
                font=dict(color=text_color),  # X-axis title color
            ),
            tickfont=dict(color=text_color),  # X-axis tick label color
            # showline=True,  # Show x-axis line
            # showgrid=True,  # Show x-axis grid
            zeroline=True,  # Show x-axis zero line
        ),
        yaxis=dict(
            title=dict(
                text="Inertia", font=dict(color=text_color)  # Y-axis title color
            ),
            tickfont=dict(color=text_color),  # Y-axis tick label color
            # showline=True,  # Show y-axis line
            # showgrid=True,  # Show y-axis grid
            zeroline=True,  # Show y-axis zero line
        ),
        legend=dict(
            title="Legend",
            font=dict(color=text_color),  # Legend text color
            bgcolor=background_color,  # Legend background color
        ),
    )
    st.plotly_chart(fig)


# # Usage
# data: pd.DataFrame = load_data(cfg.PATHS.DATASET)
# plot_elbow_curve(data)
