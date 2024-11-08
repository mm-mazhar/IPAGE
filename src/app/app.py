# -*- coding: utf-8 -*-
# """
# app.py
# Created on Nov 07, 2024
# @ Author: Mazhar
# """

import os
import time
import warnings
from calendar import c
from logging import Logger

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import sweetviz as sv
from configs.common_configs import get_config, get_logger
from easydict import EasyDict
from httpx import get
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from utils.correlation import *
from utils.feature_analysis import *
from utils.sweetviz import generate_sweetviz_report, st_display_sweetviz
from utils.utils import (
    display_centered_title,
    get_categorical_columns,
    get_numeric_columns,
    load_data,
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
    page_title="Streamlit App",
    page_icon=":bar_chart:",
    layout="wide",
)


def main() -> None:
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", list(cfg.STAPP["PAGES"].values()))
    # print(f"PAGE: {page}")

    # Data Overview Page
    if page == cfg.STAPP["PAGES"]["DATA_OVERVIEW"]:
        # Centered title
        display_centered_title(cfg.STAPP["PAGES"]["DATA_OVERVIEW"])

        # Load data
        data: pd.DataFrame = load_data(cfg.PATHS.DATASET)
        report_path: str = cfg.PATHS.REPORT

        # Generate dfSummary and convert to HTML
        # summary_html = dfSummary(data).to_html()

        # Display the summary in Streamlit
        # st.components.v1.html(summary_html, width=1000, height=1100, scrolling=True)

        if st.button("Generate Sweetviz Report"):
            generate_sweetviz_report(data, report_path)
            if os.path.exists(report_path):
                st_display_sweetviz(report_path)

        # Display the report
        elif os.path.exists(report_path):
            st_display_sweetviz(report_path)
        # Display the report in an iframe within Streamlit
        # st.components.v1.html(report_html, width=1000, height=1000, scrolling=True)

        else:
            st.write("No report available. Please press the button to generate it.")

    elif page == cfg.STAPP["PAGES"]["CORR"]:
        # Centered title
        display_centered_title(cfg.STAPP["PAGES"]["CORR"])
        # Load data
        data: pd.DataFrame = load_data(cfg.PATHS.DATASET)
        # Get Numeric Columns
        numerical_features: list[str] = get_numeric_columns(data)
        # Plot Correlation Heatmap
        # plot_correlation_heatmap(data, numerical_features)

    elif page == cfg.STAPP["PAGES"]["FEATURE_ANALYSIS"]:
        # Centered title
        display_centered_title(cfg.STAPP["PAGES"]["FEATURE_ANALYSIS"])
        # Load data
        data: pd.DataFrame = load_data(cfg.PATHS.DATASET)
        # Get Numeric Columns
        numerical_features: list[str] = get_numeric_columns(data)
        categorical_features: list[str] = get_categorical_columns(data)

        background_color = cfg.STAPP["STYLES"]["BACKGROUND"]
        text_color = cfg.STAPP["STYLES"]["TEXT"]
        width = cfg.STAPP["STYLES"]["WIDTH"]
        height = cfg.STAPP["STYLES"]["HEIGHT"]

        # print(f"Background Color: {background_color}")
        # print(f"Text Color: {text_color}")
        # print(f"Width: {width}")
        # print(f"Height: {height}")

        # Visualize Bar Plot
        visualize_bar_plot(
            data,
            numerical_features,
            categorical_features,
            flag="mean",
            background_color=background_color,
            text_color=text_color,
            # width=width,
            # height=height,
        )

        # # Visualize Comparison Box
        # visualize_comparison_box(
        #     data,
        #     numerical_features,
        #     categorical_features,
        #     background_color=background_color,
        #     text_color=text_color,
        #     # width=width,
        #     # height=height,
        # )

        # # Visualize Comparison Violin
        # visualize_comparison_violin(
        #     data,
        #     numerical_features,
        #     categorical_features,
        #     background_color=background_color,
        #     text_color=text_color,
        #     # width=width,
        #     # height=height,
        # )

        # # Visualize Comparison Strip
        # visualize_comparison_strip(
        #     data,
        #     numerical_features,
        #     categorical_features,
        #     background_color=background_color,
        #     text_color=text_color,
        #     # width=width,
        #     # height=height,
        # )

        # Separator
        st.markdown("""---""")

        # Visualize Stacked Bar Chart
        plot_stacked_bar_chart(
            data,
            categorical_features,
            background_color=background_color,
            text_color=text_color,
            # width=width,
            # height=height,
        )

    elif page == cfg.STAPP["PAGES"]["RANDOM_FOREST"]:
        # Centered title
        display_centered_title(cfg.STAPP["PAGES"]["RANDOM_FOREST"])

    elif page == cfg.STAPP["PAGES"]["ABOUT"]:
        # Centered title
        display_centered_title(cfg.STAPP["PAGES"]["ABOUT"])


if __name__ == "__main__":
    main()


# # EDA Page
# elif page == cfg.STAPP["PAGES"]["EDA"]:
#     pass
#     # st.title("Exploratory Data Analysis")
#     # iPAGE_data = load_data()
#     # iPAGE_data_encoded = pd.get_dummies(iPAGE_data, columns=['Soil Series (Area Wise)', 'Land Type', 'Soil Type'], drop_first=True)

#     # # Histograms
#     # st.subheader("Feature Distribution Histograms")
#     # fig, ax = plt.subplots(figsize=(15, 10))
#     # iPAGE_data_encoded.hist(ax=ax, bins=20, edgecolor='black', color='lime')
#     # st.pyplot(fig)

#     # # Correlation Matrix
#     # st.subheader("Correlation Matrix")
#     # fig, ax = plt.subplots(figsize=(14, 10))
#     # correlation = iPAGE_data_encoded.corr()
#     # sns.heatmap(correlation, annot=True, cmap='YlGn', linewidths=0.5, ax=ax)
#     # st.pyplot(fig)

# # Model Inference Page
# elif page == cfg.STAPP["PAGES"]["RANDOM_FOREST"]:
#     pass
#     # st.title("Model Inference")
#     # iPAGE_data = load_data()
#     # iPAGE_data_encoded = pd.get_dummies(iPAGE_data, columns=['Soil Series (Area Wise)', 'Land Type', 'Soil Type'], drop_first=True)

#     # # Define Features and Target Variables
#     # X = iPAGE_data_encoded.drop(columns=['B (ug/g)'])
#     # y_b = iPAGE_data_encoded['B (ug/g)']
#     # y_soc = iPAGE_data_encoded['SOC (%)']

#     # # Train-Test Split and Scaling
#     # X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(X, y_b, test_size=0.3, random_state=42)
#     # X_train_soc, X_test_soc, y_train_soc, y_test_soc = train_test_split(X, y_soc, test_size=0.3, random_state=42)

#     # scaler = StandardScaler()
#     # X_train_b_scaled = scaler.fit_transform(X_train_b)
#     # X_test_b_scaled = scaler.transform(X_test_b)
#     # X_train_soc_scaled = scaler.fit_transform(X_train_soc)
#     # X_test_soc_scaled = scaler.transform(X_test_soc)

#     # # AutoML with TPOT for Boron (B)
#     # tpot_b = TPOTRegressor(verbosity=2, generations=5, population_size=50, random_state=42)
#     # tpot_b.fit(X_train_b_scaled, y_train_b)

#     # # AutoML with TPOT for SOC (TOC)
#     # tpot_soc = TPOTRegressor(verbosity=2, generations=5, population_size=50, random_state=42)
#     # tpot_soc.fit(X_train_soc_scaled, y_train_soc)

#     # # Evaluate Performance for B(Boron)
#     # y_pred_b = tpot_b.predict(X_test_b_scaled)
#     # r2_b = r2_score(y_test_b, y_pred_b)
#     # mae_b = mean_absolute_error(y_test_b, y_pred_b)
#     # mse_b = mean_squared_error(y_test_b, y_pred_b)
#     # rmse_b = np.sqrt(mse_b)

#     # st.subheader("Boron (B) Prediction Metrics")
#     # st.write(f"R²: {r2_b}")
#     # st.write(f"MAE: {mae_b}")
#     # st.write(f"MSE: {mse_b}")
#     # st.write(f"RMSE: {rmse_b}")

#     # # Evaluate Performance for SOC (%)
#     # y_pred_soc = tpot_soc.predict(X_test_soc_scaled)
#     # r2_soc = r2_score(y_test_soc, y_pred_soc)
#     # mae_soc = mean_absolute_error(y_test_soc, y_pred_soc)
#     # mse_soc = mean_squared_error(y_test_soc, y_pred_soc)
#     # rmse_soc = np.sqrt(mse_soc)

#     # st.subheader("SOC (%) Prediction Metrics")
#     # st.write(f"R²: {r2_soc}")
#     # st.write(f"MAE: {mae_soc}")
#     # st.write(f"MSE: {mse_soc}")
#     # st.write(f"RMSE: {rmse_soc}")
