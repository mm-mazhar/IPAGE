# -*- coding: utf-8 -*-
# """
# app.py
# Created on Dec 17, 2024
# @ Author: Mazhar
# """

import os
import time
import warnings
from logging import Logger
from typing import Any

import folium
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import sweetviz as sv
from app_pages import config_dashboard_sidebar  # prediction_page,
from app_pages import (
    config_kmeans_pca_sidebar,
    display_kmeans_pca_info,
    display_stat_cards,
    display_sweetviz_report_page,
    kmeans_pca_and_elbow_curve,
    plot_cluster_visualizations,
    plot_violin_strip_scatter,
    plots_bar_dist_pie,
)
from app_pages.about import display_about_text
from app_pages.prediction_page import display_prediction_page
from configs.common_configs import get_config, get_logger
from easydict import EasyDict
from pandas.io.formats.style import Styler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from streamlit_folium import st_folium
from utils.correlation import *
from utils.data_table import display_styled_data_table
from utils.feature_analysis import plot_stacked_bar_chart
from utils.folium_map import create_folium_map
from utils.kmeans_and_pca import cluster_statistics, plot_cluster_comparisons
from utils.utils import highlight_ideal_values  # load_data,
from utils.utils import (
    display_centered_title,
    get_categorical_columns,
    get_numeric_columns,
    title_h3,
)

# from tpot import TPOTRegressor

# Suppress only UserWarning
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=SyntaxWarning)
# Suppress specific warnings from a module
warnings.filterwarnings(
    "ignore", category=UserWarning, message=".*optional dependency `torch`.*"
)


# Get the configuration
cfg: EasyDict = get_config()

# Initialize logger
logger: Logger = get_logger()

# print(f"ROOT_DIR: {cfg.ROOT_DIR}")
# print(f"DATASET PATH: {cfg.PATHS.DATASET}")

st.set_page_config(
    page_title=cfg.STAPP.CONFIGS.PAGE_TITLE,
    page_icon=cfg.STAPP.CONFIGS.PAGE_ICON,
    layout=cfg.STAPP.CONFIGS.LAYOUT,
)

# Exclude 'longitude' and 'latitude' from numeric columns
exclude_cols: list[str] = [cfg.STAPP.EXCLUDE_COLS.COL_1, cfg.STAPP.EXCLUDE_COLS.COL_2]


# Load dataset
@st.cache_data
def load_data(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path)


def main() -> None:

    df: pd.DataFrame = load_data(cfg.PATHS.DATASET)
    data: pd.DataFrame = df.copy()

    # Sidebar Configuration
    with st.sidebar:
        # Sidebar for navigation
        st.sidebar.title(cfg.STAPP.CONFIGS.SIDEBAR_TITLE)
        page: Any = st.sidebar.selectbox(
            "Go to", list(cfg.STAPP["PAGES"].values()), index=1
        )
        # print(f"PAGE: {page}")

    ######################
    # PAGE | Data Overview
    ######################
    if page == cfg.STAPP["PAGES"]["DATA_OVERVIEW"]:
        # Display centered title
        display_centered_title(
            cfg.STAPP["PAGES"]["DATA_OVERVIEW"],
            color=cfg.STAPP["STYLES"]["TITLE_COLOR"],
        )
        # Call the function
        display_sweetviz_report_page(cfg, data, exclude_cols)

    ####################
    # PAGE | Dashboard
    ####################

    elif page == cfg.STAPP["PAGES"]["DASHBOARD"]:
        # Config Sidebar of Dashboard
        cfg_dashboard_sidebar: dict[str, Any] = config_dashboard_sidebar(
            data,
            exclude_cols,
            get_numeric_columns=get_numeric_columns,  # Pass function references
            get_categorical_columns=get_categorical_columns,
        )

        # Access the returned values
        selected_method: str = cfg_dashboard_sidebar["selected_method"]
        selected_numeric_feature: str = cfg_dashboard_sidebar[
            "selected_numeric_feature"
        ]
        selected_numeric_feature_x: str = cfg_dashboard_sidebar[
            "selected_numeric_feature_x"
        ]
        selected_categorical_feature: str = cfg_dashboard_sidebar[
            "selected_categorical_feature"
        ]
        selected_features: list[str] = cfg_dashboard_sidebar["selected_features"]
        selected_columns: list[str] = cfg_dashboard_sidebar["selected_columns"]

        # Display Centered Title
        display_centered_title(
            cfg.STAPP["PAGES"]["DASHBOARD"], color=cfg.STAPP["STYLES"]["TITLE_COLOR"]
        )

        fig_size: tuple[int, int] = tuple(cfg.STAPP.STATISTICS_CARD.FIG_SIZE)
        background_color: str = cfg.STAPP.STATISTICS_CARD.DIST_PLOT_BACKGROUND_COLOR
        dist_plot_color: str = cfg.STAPP.STATISTICS_CARD.DIST_PLOT_COLOR

        # print(f"Fig Size: {fig_size}")
        print(f"Dist Plot Color: {dist_plot_color}")

        #############################################
        # PAGE | Dashboard| ROW 1 | Statistics Cards
        #############################################
        # Separator
        st.markdown("""---""")
        display_stat_cards(
            cfg=cfg,
            data=data,
            fig_size=fig_size,
            background_color=background_color,
            dist_plot_color=dist_plot_color,
            exclude_cols=exclude_cols,
        )

        #######################################################
        # PAGE | Dashboard | ROW 2 | Map Display
        #######################################################
        st.markdown("""---""")

        title_h3(
            "Geospatial Visualization",
            color=cfg.STAPP["STYLES"]["SUB_TITLE_COLOR"],
        )
        map_data: pd.DataFrame = data[
            ["latitude", "longitude", "Area"]
            + (
                [selected_numeric_feature, selected_categorical_feature]
                # if selected_numeric_feature
                # else (
                #     [selected_categorical_feature]
                #     if selected_categorical_feature
                #     else []
                # )
            )
        ].dropna()

        if not map_data.empty:
            folium_map: folium.Map = create_folium_map(
                map_data,
                selected_numeric_feature=selected_numeric_feature,
                selected_categorical_feature=selected_categorical_feature,
            )
            st_folium(
                folium_map,
                width=cfg.STAPP["STYLES"]["FOLIUM"]["WIDTH"],
                height=cfg.STAPP["STYLES"]["FOLIUM"]["HEIGHT"],
                zoom=cfg.STAPP["STYLES"]["FOLIUM"]["ZOOM_START"],
            )
        else:
            st.warning("No data to display on the map.")

        ##############################################
        # PAGE | Dashboard | ROW 3 | Feature Analysis
        ##############################################
        # Separator
        st.markdown("""---""")
        # Centered Title H3
        title_h3(
            "Feature Analysis",
            color=cfg.STAPP["STYLES"]["SUB_TITLE_COLOR"],
        )

        background_color: str = cfg.STAPP["STYLES"]["BACKGROUND"]
        text_color: str = cfg.STAPP["STYLES"]["TEXT_COLOR"]
        width: str = cfg.STAPP["STYLES"]["WIDTH"]
        height: str = cfg.STAPP["STYLES"]["HEIGHT"]

        print(f"Background Color: {background_color}")
        print(f"Text Color: {text_color}")
        print(f"Width: {width}")
        print(f"Height: {height}")

        plots_bar_dist_pie(
            data=data,
            selected_numeric_feature=selected_numeric_feature,
            selected_method=selected_method,
            selected_categorical_feature=selected_categorical_feature,
            background_color=background_color,
            text_color=text_color,
        )

        ##############################################
        # PAGE | Dashboard | ROW 4 | Feature Analysis
        ##############################################
        # Separator
        # st.markdown("""---""")

        plot_violin_strip_scatter(
            data=data,
            selected_numeric_feature_x=selected_numeric_feature_x,
            selected_numeric_feature=selected_numeric_feature,
            selected_categorical_feature=selected_categorical_feature,
            background_color=background_color,
            text_color=text_color,
        )

        ##############################################
        # PAGE | Dashboard | ROW 5 | Feature Analysis
        ##############################################
        # # Separator
        # st.markdown("""---""")
        # Called from utils/feature_analysis.py
        # Visualize Stacked Bar Chart
        plot_stacked_bar_chart(
            data,
            selected_features,
            background_color=background_color,
            text_color=text_color,
            # width=width,
            # height=height,
        )

        ################################################
        # PAGE | Dashboard | ROW 6 | Data Table Display
        ################################################
        # Separator
        st.markdown("""---""")
        # Centered Title H3
        title_h3("Data Table", color="red")

        # Display Data Table
        # Called from utils/data_table.py
        display_styled_data_table(
            data=data,
            selected_columns=selected_columns,
            cfg=cfg,
            highlight_ideal_values=highlight_ideal_values,
        )

    #########################
    # PAGE | PREDICTION
    #########################
    elif page == cfg.STAPP["PAGES"]["PREDICTION"]:
        display_prediction_page(cfg)

    #########################
    # PAGE | ABOUT
    #########################
    elif page == cfg.STAPP["PAGES"]["ABOUT"]:
        # Centered title
        display_centered_title(
            cfg.STAPP["PAGES"]["ABOUT"], color=cfg.STAPP["STYLES"]["TITLE_COLOR"]
        )
        # Display K-Means and PCA Info (Expanders)
        display_about_text()


if __name__ == "__main__":
    main()
