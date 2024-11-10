# -*- coding: utf-8 -*-
# """
# k_means_and_pca.py
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

############################################################################
############## Applying K-means clustering with 3 clusters #################
############################################################################

# Aligning clustering output with the original data by resetting the index on numeric data before clustering
numeric_data.reset_index(drop=True, inplace=True)

# Re-running K-means clustering on the aligned data
kmeans = KMeans(n_clusters=3, random_state=42)
clusters: np.ndarray = kmeans.fit_predict(scaled_data)

# Adding the cluster labels to the filtered numeric data
numeric_data["Cluster"] = clusters

# Merging cluster information back to the main dataset by aligning indexes
data["Cluster"] = numeric_data["Cluster"]

# Re-running PCA for visualization with cluster labels
pca = PCA(n_components=2)
pca_data: np.ndarray = pca.fit_transform(scaled_data)
numeric_data["PCA1"] = pca_data[:, 0]
numeric_data["PCA2"] = pca_data[:, 1]

# Plotting PCA components with cluster labels
plt.figure(figsize=(10, 8))
sns.scatterplot(x="PCA1", y="PCA2", hue="Cluster", data=numeric_data, palette="Set1")
plt.title("PCA of Soil Data with K-means Clusters")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend(title="Cluster")
plt.show()
