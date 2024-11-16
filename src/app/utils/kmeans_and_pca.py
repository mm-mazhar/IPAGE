# -*- coding: utf-8 -*-
# """
# kmeans_and_pca.py
# Created on Nov 10, 2024
# @ Author: Mazhar
# """


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from utils.utils import display_centered_title


def perform_kmeans_and_pca(
    data: pd.DataFrame,
    n_clusters: int = 3,
    background_color: str = "white",
    text_color: str = "black",
) -> None:
    """
    Performs K-means clustering and PCA, then plots the results using Plotly Express.

    Parameters:
    - data: pd.DataFrame - The input data for clustering.
    - n_clusters: int - The number of clusters for K-means.
    """
    df: pd.DataFrame = data.copy()
    # Selecting only the numeric columns for clustering and standardizing the data
    numeric_data: pd.DataFrame = df.select_dtypes(include=["float64"]).dropna()
    scaler = StandardScaler()
    scaled_data: np.ndarray = scaler.fit_transform(numeric_data)

    # Aligning clustering output with the original data by resetting the index on numeric data before clustering
    numeric_data.reset_index(drop=True, inplace=True)

    # Applying K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters: np.ndarray = kmeans.fit_predict(scaled_data)

    # Adding the cluster labels to the filtered numeric data
    numeric_data["Cluster"] = clusters

    # Re-running PCA for visualization with cluster labels
    pca = PCA(n_components=2)
    pca_data: np.ndarray = pca.fit_transform(scaled_data)
    numeric_data["PCA1"] = pca_data[:, 0]
    numeric_data["PCA2"] = pca_data[:, 1]

    # Plotting PCA components with cluster labels using Plotly Express
    fig = px.scatter(
        numeric_data,
        x="PCA1",
        y="PCA2",
        color="Cluster",
        title="PCA of Soil Data with K-means Clusters",
        labels={"PCA1": "PCA Component 1", "PCA2": "PCA Component 2"},
        color_continuous_scale=px.colors.qualitative.Set1,
    )
    # Customize layout
    fig.update_layout(
        plot_bgcolor=background_color,  # Background color
        paper_bgcolor=background_color,  # Paper background color
        font=dict(color=text_color),  # General text color
        title_font=dict(color=text_color),
        xaxis=dict(
            title=dict(
                text="PCA Component 1",
                font=dict(color=text_color),  # X-axis title color
            ),
            tickfont=dict(color=text_color),  # X-axis tick label color
            zeroline=True,  # Show x-axis zero line
            range=[-2, 4],
        ),
        yaxis=dict(
            title=dict(
                text="PCA Component 2",
                font=dict(color=text_color),  # Y-axis title color
            ),
            tickfont=dict(color=text_color),  # Y-axis tick label color
            zeroline=False,  # Show y-axis zero line
        ),
        legend=dict(
            title="Legend",
            font=dict(color=text_color),  # Legend text color
            bgcolor=background_color,  # Legend background color
        ),
    )
    st.plotly_chart(fig)

    # Return the original DataFrame with the cluster labels
    df["Cluster"] = clusters
    return df


def cluster_statistics(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate descriptive statistics for each cluster.

    Parameters:
    - data: pd.DataFrame - The input data with cluster labels.

    Returns:
    - pd.DataFrame - Summary statistics for each cluster.
    """
    return data.groupby("Cluster").describe()


def plot_cluster_histograms(
    data: pd.DataFrame,
    feature: str,
    background_color: str = "white",
    text_color: str = "black",
) -> None:
    """
    Plot histograms of a feature for each cluster using Plotly Express.

    Parameters:
    - data: pd.DataFrame - The input data with cluster labels.
    - feature: str - The feature to plot.
    - background_color: str - Background color for the plot.
    - text_color: str - Text color for the plot.
    """
    # display_centered_title("Cluster Histograms", color="red")
    # st.markdown("""---""")

    # Create the histogram using Plotly Express
    fig = px.histogram(
        data,
        x=feature,
        color="Cluster",
        barmode="stack",
        title=f"Distribution of {feature} by Cluster",
        color_discrete_sequence=px.colors.qualitative.Set1,
    )

    # Customize layout
    fig.update_layout(
        plot_bgcolor=background_color,  # Background color
        paper_bgcolor=background_color,  # Paper background color
        font=dict(color=text_color),  # General text color
        title_font=dict(color=text_color),
        xaxis=dict(
            title=dict(text=feature, font=dict(color=text_color)),  # X-axis title color
            tickfont=dict(color=text_color),  # X-axis tick label color
        ),
        yaxis=dict(
            title=dict(text="Count", font=dict(color=text_color)),  # Y-axis title color
            tickfont=dict(color=text_color),  # Y-axis tick label color
        ),
        legend=dict(
            title="Legend",
            font=dict(color=text_color),  # Legend text color
            bgcolor=background_color,  # Legend background color
        ),
    )

    st.plotly_chart(fig)


def plot_cluster_comparisons(
    data: pd.DataFrame, background_color: str = "white", text_color: str = "black"
) -> None:
    """
    Plot comparisons of clusters with both categorical features using Plotly Express.

    Parameters:
    - data: pd.DataFrame - The input data with cluster labels.
    - background_color: str - Background color for the plot.
    - text_color: str - Text color for the plot.
    """
    col1, col2 = st.columns([1, 8])
    with col1:
        # Add vertical space to center the content
        for _ in range(12):
            st.write("")  # Add empty strings to create space

        # Select a single categorical feature for comparison
        categorical_feature = st.selectbox(
            "Select Categorical Feature",
            options=data.select_dtypes(include=["object", "category"]).columns,
            index=0,
            key="plot_cluster_comparisons_categorical",
        )
    with col2:
        # Plot categorical feature comparison
        fig = px.histogram(
            data,
            x=categorical_feature,
            color="Cluster",
            barmode="group",
            title=f"Bar Plot of {categorical_feature} by Cluster",
            color_discrete_sequence=px.colors.qualitative.Set1,
        )
        fig.update_layout(
            plot_bgcolor=background_color,
            paper_bgcolor=background_color,
            font=dict(color=text_color),
            title_font=dict(color=text_color),
            xaxis=dict(
                title=dict(text=categorical_feature, font=dict(color=text_color)),
                tickfont=dict(color=text_color),
            ),
            yaxis=dict(
                title=dict(text="Count", font=dict(color=text_color)),
                tickfont=dict(color=text_color),
            ),
            legend=dict(
                title="Legend",
                font=dict(color=text_color),
                bgcolor=background_color,
            ),
        )
        st.plotly_chart(fig)


def plot_cluster_boxplots(
    data: pd.DataFrame,
    feature: str,
    background_color: str = "white",
    text_color: str = "black",
) -> None:
    """
    Plot box plots of a feature for each cluster using Plotly Express.

    Parameters:
    - data: pd.DataFrame - The input data with cluster labels.
    - feature: str - The feature to plot.
    - background_color: str - Background color for the plot.
    - text_color: str - Text color for the plot.
    """

    # Create the box plot using Plotly Express
    fig = px.box(
        data,
        x="Cluster",
        y=feature,
        color="Cluster",
        title=f"Box Plot of {feature} by Cluster",
        color_discrete_sequence=px.colors.qualitative.Set1,
    )

    # Customize layout
    fig.update_layout(
        plot_bgcolor=background_color,  # Background color
        paper_bgcolor=background_color,  # Paper background color
        font=dict(color=text_color),  # General text color
        title_font=dict(color=text_color),
        xaxis=dict(
            title=dict(
                text="Cluster", font=dict(color=text_color)  # X-axis title color
            ),
            tickfont=dict(color=text_color),  # X-axis tick label color
        ),
        yaxis=dict(
            title=dict(text=feature, font=dict(color=text_color)),  # Y-axis title color
            tickfont=dict(color=text_color),  # Y-axis tick label color
        ),
        legend=dict(
            title="Legend",
            font=dict(color=text_color),  # Legend text color
            bgcolor=background_color,  # Legend background color
        ),
    )

    st.plotly_chart(fig)


def plot_cluster_pairplot(
    data: pd.DataFrame, background_color: str = "white", text_color: str = "black"
) -> None:
    """
    Plot pair plots for a list of features colored by cluster using Plotly Express.

    Parameters:
    - data: pd.DataFrame - The input data with cluster labels.
    - background_color: str - Background color for the plot.
    - text_color: str - Text color for the plot.
    """
    # Select features for pair plot
    features = st.multiselect(
        "Select Features for Pair Plot",
        options=data.select_dtypes(include=["float64"]).columns,
        default=data.select_dtypes(include=["float64"]).columns[
            :3
        ],  # Default to first three features
        key="plot_cluster_pairplot",
    )

    if not features:
        st.warning("Please select at least one feature to plot.")
        return

    # Create the pair plot using Plotly Express
    fig = px.scatter_matrix(
        data,
        dimensions=features,
        color="Cluster",
        title="Pair Plot of Features by Cluster",
        color_discrete_sequence=px.colors.qualitative.Set1,
    )

    # Customize layout
    fig.update_layout(
        plot_bgcolor=background_color,  # Background color
        paper_bgcolor=background_color,  # Paper background color
        font=dict(color=text_color),  # General text color
        title_font=dict(color=text_color),
        legend=dict(
            title="Legend",
            font=dict(color=text_color),  # Legend text color
            bgcolor=background_color,  # Legend background color
        ),
    )

    st.plotly_chart(fig)


# # Usage
# # Assuming `clustered_data` is the DataFrame returned from `perform_kmeans_and_pca`
# clustered_data = perform_kmeans_and_pca(data)

# # Calculate statistics
# stats = cluster_statistics(clustered_data)
# print(stats)

# # Plot histograms for a specific feature
# plot_cluster_histograms(clustered_data, "FeatureName")

# # Plot box plots for a specific feature
# plot_cluster_boxplots(clustered_data, "FeatureName")

# # Plot pair plots for a list of features
# plot_cluster_pairplot(clustered_data, ["Feature1", "Feature2", "Feature3"])
