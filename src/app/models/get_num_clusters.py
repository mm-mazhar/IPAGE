# -*- coding: utf-8 -*-
# """
# get_num_clusters.py
# Created on Nov 07, 2024
# @ Author: Mazhar
# """

import os
import sys

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from logging import Logger

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import seaborn as sns
import streamlit as st
from configs.common_configs import get_config, get_logger
from easydict import EasyDict
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Get the configuration
cfg: EasyDict = get_config()

# Initialize logger
logger: Logger = get_logger()


# Load dataset
@st.cache_data
def load_data(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path)


data: pd.DataFrame = load_data(cfg.PATHS.DATASET)

# Selecting only the numeric columns for clustering and standardizing the data
numeric_data: pd.DataFrame = data.select_dtypes(include=["float64"]).dropna()
scaler = StandardScaler()
scaled_data: np.ndarray = scaler.fit_transform(numeric_data)

# Performing K-means clustering with an optimal number of clusters (testing range 2-6)
inertia: list = []
for k in range(2, 7):
    kmeans = KMeans(n_clusters=k, random_state=42, verbose=0)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)

# print(f"Inertia: {inertia}")

# # Plot inertia to determine the "elbow" point
# plt.figure(figsize=(10, 6))
# plt.plot(range(2, 7), inertia, marker="o")
# plt.title("Elbow Method for Optimal Number of Clusters")
# plt.xlabel("Number of Clusters")
# plt.ylabel("Inertia")
# plt.show()

# # Calculate inertia for a range of cluster numbers
# inertia = []
# K = range(1, 11)  # Test a range of cluster numbers

# for k in K:
#     kmeans = KMeans(n_clusters=k, random_state=42, verbose=0)
#     kmeans.fit(scaled_data)
#     inertia.append(kmeans.inertia_)

# # Plot the elbow curve using Plotly
# fig = go.Figure()
# fig.add_trace(go.Scatter(x=list(K), y=inertia, mode='lines+markers', marker=dict(color='blue')))
# fig.update_layout(
#     title="Elbow Method for Optimal Number of Clusters",
#     xaxis_title="Number of Clusters",
#     yaxis_title="Inertia",
#     template="plotly_white"
# )
# fig.show()
