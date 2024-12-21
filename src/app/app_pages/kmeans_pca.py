# -*- coding: utf-8 -*-
# """
# kmeans_pca.py
# Created on Dec 17, 2024
# @ Author: Mazhar
# """

import pandas as pd
import streamlit as st
from utils.elbow_curve_plot import plot_elbow_curve
from utils.kmeans_and_pca import *


def display_kmeans_pca_info() -> None:
    """
    Function to display K-Means and PCA explanation along with cluster insights using Streamlit expanders.
    """
    with st.expander(
        "K-Means and Principal Component Analysis (Un-Supervised Learning)"
    ):
        # st.write(
        #     """
        #     - Performing **clustering** to uncover any underlying patterns within these soil profiles.

        #     - **K-means** clustering will be a suitable choice for this unsupervised analysis.

        #     - The **elbow** point in the inertia plot suggests an optimal number of clusters around 3 or 4.

        #     - We will proceed with **K-means clustering** using **3 clusters** and then use **PCA** for visualization of
        #     these clusters to see how different soil profiles group together.
        #     """
        # )
        st.write(
            """
            - To be written.
            """
        )

    with st.expander("Insights"):
        # st.markdown(
        #     """
        #     ### Cluster Insights

        #     - **Cluster 0**:
        #       - Concentrated around the negative side of the PCA Component 1 axis, with points mostly on the left.
        #       - Likely represents soils with **moderate nutrient levels**.

        #     - **Cluster 1**:
        #       - Spans a range in both PCA Component 1 and PCA Component 2, extending towards the right side of the plot.
        #       - This grouping may indicate **nutrient-rich soils**, as it diverges from the other clusters.

        #     - **Cluster 2**:
        #       - Contains only a few points.
        #       - Suggests soils with **distinct characteristics**, possibly with lower fertility or unique nutrient imbalances.

        #     **Summary**:
        #     - The cluster on the left is concentrated around moderate nutrient levels.
        #     - The cluster on the right may include nutrient-rich soils.
        #     - The isolated points in Cluster 2 may represent outliers or soils with unique properties.
        #     """
        # )
        st.markdown(
            """
            ### Cluster Insights

            - To be written.
            """
        )


def kmeans_pca_and_elbow_curve(data, n_clusters, background_color, text_color):
    """
    Function to display K-Means clustering with PCA results and the Elbow curve in two columns.

    Args:
        data (pd.DataFrame): The input dataset for clustering.
        n_clusters (int): The number of clusters for K-Means.
        background_color (str): Background color for visualizations.
        text_color (str): Text color for visualizations.

    Returns:
        pd.DataFrame: The clustered data after performing K-Means and PCA.
    """
    col1, col2 = st.columns([2, 1])

    with col1:
        # Perform K-Means and PCA
        clustered_data: None = perform_kmeans_and_pca(
            data,
            n_clusters=n_clusters,
            background_color=background_color,
            text_color=text_color,
        )

    with col2:
        # Plot Elbow Curve
        plot_elbow_curve(
            data,
            background_color=background_color,
            text_color=text_color,
        )

    return clustered_data


def plot_cluster_visualizations(
    clustered_data, numeric_feature, background_color, text_color, sub_title_color
) -> None:
    """
    Function to display box plots and histograms by clusters in two columns.

    Args:
        clustered_data (pd.DataFrame): Data containing cluster information.
        numeric_feature (str): Numeric feature for visualizing clusters.
        background_color (str): Background color for the plots.
        text_color (str): Text color for the plots.
        sub_title_color (str): Color for the subtitles.
    """
    col1, col2 = st.columns(2)

    with col1:
        plot_cluster_boxplots(
            clustered_data,
            numeric_feature,
            background_color=background_color,
            text_color=text_color,
        )

    with col2:
        plot_cluster_histograms(
            clustered_data,
            numeric_feature,
            background_color=background_color,
            text_color=text_color,
        )
