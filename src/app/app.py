# -*- coding: utf-8 -*-
# """
# app.py
# Created on Nov 10, 2024
# @ Author: Mazhar
# """

import os
import time
import warnings
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
from pandas.io.formats.style import Styler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from utils.correlation import *
from utils.dist_plots import *
from utils.elbow_curve_plot import plot_elbow_curve
from utils.feature_analysis import *
from utils.kmeans_and_pca import *
from utils.pie_plot import *
from utils.scatter_plot import *
from utils.statistics_card import display_stat_card
from utils.sweetviz import generate_sweetviz_report, st_display_sweetviz
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
    page_title="Streamlit App",
    page_icon=":bar_chart:",
    layout="wide",
)


# Load dataset
@st.cache_data
def load_data(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path)


def main() -> None:
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", list(cfg.STAPP["PAGES"].values()))
    # print(f"PAGE: {page}")

    data: pd.DataFrame = load_data(cfg.PATHS.DATASET)

    ####################
    # Data Overview Page
    ####################
    if page == cfg.STAPP["PAGES"]["DATA_OVERVIEW"]:
        # Centered title
        display_centered_title(
            cfg.STAPP["PAGES"]["DATA_OVERVIEW"],
            color=cfg.STAPP["STYLES"]["TITLE_COLOR"],
        )

        # # Load data
        # data: pd.DataFrame = load_data(cfg.PATHS.DATASET)
        report_path: str = cfg.PATHS.REPORT
        width: int = cfg.STAPP.SWEETVIZ.WIDTH
        height: int = cfg.STAPP.SWEETVIZ.HEIGHT

        # print(f"Width: {width}")
        # print(f"Height: {height}")

        # Generate dfSummary and convert to HTML
        # summary_html = dfSummary(data).to_html()

        # Display the summary in Streamlit
        # st.components.v1.html(summary_html, width=1000, height=1100, scrolling=True)

        if st.button("Generate Sweetviz Report"):
            generate_sweetviz_report(data, report_path)
            if os.path.exists(report_path):
                st_display_sweetviz(report_path, width, height)

        # Display the report
        elif os.path.exists(report_path):
            st_display_sweetviz(report_path, width, height)
        # Display the report in an iframe within Streamlit
        # st.components.v1.html(report_html, width=1000, height=1000, scrolling=True)

        else:
            st.write("No report available. Please press the button to generate it.")

    ###########################
    # Exploratory Data Analysis
    ###########################

    elif page == cfg.STAPP["PAGES"]["EDA"]:
        # Centered title
        display_centered_title(
            cfg.STAPP["PAGES"]["EDA"], color=cfg.STAPP["STYLES"]["TITLE_COLOR"]
        )

        fig_size = tuple(cfg.STAPP.STATISTICS_CARD.FIG_SIZE)
        background_color = cfg.STAPP.STATISTICS_CARD.DIST_PLOT_BACKGROUND_COLOR
        dist_plot_color = cfg.STAPP.STATISTICS_CARD.DIST_PLOT_COLOR

        # print(f"Fig Size: {fig_size}")
        print(f"Dist Plot Color: {dist_plot_color}")

        # # Load data
        # data: pd.DataFrame = load_data(cfg.PATHS.DATASET)

        ########################
        # Statistics Cards Row 1
        ########################
        # Separator
        st.markdown("""---""")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            display_stat_card(
                "pH",
                data,
                figsize=(fig_size[0], fig_size[1]),
                background_color=background_color,
                color=dist_plot_color,
            )
        with col2:
            display_stat_card(
                "SOC (%)",
                data,
                figsize=(fig_size[0], fig_size[1]),
                background_color=background_color,
                color=dist_plot_color,
            )
        with col3:
            display_stat_card(
                "Boron B (ug/g)",
                data,
                figsize=(fig_size[0], fig_size[1]),
                background_color=background_color,
                color=dist_plot_color,
            )
        with col4:
            display_stat_card(
                "Zinc Zn (ug/g)",
                data,
                figsize=(fig_size[0], fig_size[1]),
                background_color=background_color,
                color=dist_plot_color,
            )

        ########################
        # Feature Analysis Row 2
        ########################
        # Separator
        st.markdown("""---""")
        # Centered Title H3
        title_h3(
            "Feature Analysis",
            color=cfg.STAPP["STYLES"]["SUB_TITLE_COLOR"],
        )

        # Get Numeric Columns
        numerical_features: list[str] = get_numeric_columns(data)
        categorical_features: list[str] = get_categorical_columns(data)

        background_color: str = cfg.STAPP["STYLES"]["BACKGROUND"]
        text_color: str = cfg.STAPP["STYLES"]["TEXT_COLOR"]
        width = cfg.STAPP["STYLES"]["WIDTH"]
        height = cfg.STAPP["STYLES"]["HEIGHT"]

        print(f"Background Color: {background_color}")
        print(f"Text Color: {text_color}")
        print(f"Width: {width}")
        print(f"Height: {height}")

        col1, col2 = st.columns(2)

        with col1:
            # Visualize Bar Plot
            visualize_bar_plot(
                data,
                numerical_features,
                categorical_features,
                background_color=background_color,
                text_color=text_color,
                # width=width,
                # height=height,
            )

        # Separator
        # st.markdown("""---""")
        with col2:
            # Visualize Stacked Bar Chart
            plot_stacked_bar_chart(
                data,
                categorical_features,
                background_color=background_color,
                text_color=text_color,
                # width=width,
                # height=height,
            )

        ########################
        # Feature Analysis Row 3
        ########################
        # Separator
        # st.markdown("""---""")

        col1, col2 = st.columns(2)

        with col1:
            # Visualize Comparison Box
            visualize_comparison_box(
                data,
                numerical_features,
                categorical_features,
                background_color=background_color,
                text_color=text_color,
                # width=width,
                # height=height,
            )

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

        with col2:
            # plot_dist_chart(
            #     data,
            #     categorical_features,
            #     background_color=background_color,
            #     text_color=text_color,
            #     # width=width,
            #     # height=height,
            # )
            plot_pie_chart(
                data,
                categorical_features,
                background_color=background_color,
                text_color=text_color,
            )

        ########################
        # Feature Analysis Row 4
        ########################
        # # Separator
        # st.markdown("""---""")
        scatter_plot(
            data,
            background_color=background_color,
            text_color=text_color,
        )

        ########################
        # Data Table Row 5
        ########################
        # Separator
        st.markdown("""---""")
        # Centered Title H3
        title_h3("Data Table", color="red")

        # Define ideal ranges from config
        ideal_ranges: dict = {
            "pH": tuple(cfg.STAPP["IDEAL_RANGES"]["pH"]),
            "SOC (%)": tuple(cfg.STAPP["IDEAL_RANGES"]["SOC_PERCENT"]),
            "Boron B (ug/g)": tuple(cfg.STAPP["IDEAL_RANGES"]["BORON_B_UG_G"]),
            "Zinc Zn (ug/g)": tuple(cfg.STAPP["IDEAL_RANGES"]["ZINC_ZN_UG_G"]),
        }

        # st.markdown("**Note on Ideal Ranges for Farming:**")
        with st.expander("View Ideal Ranges"):
            st.write(
                f"""
                - **pH**: {ideal_ranges["pH"][0]} to {ideal_ranges["pH"][1]}, 
                - **SOC (%)**: {ideal_ranges["SOC (%)"][0]} to {ideal_ranges["SOC (%)"][1]}, 
                - **Boron (B) (ug/g)**: {ideal_ranges["Boron B (ug/g)"][0]} to {ideal_ranges["Boron B (ug/g)"][1]}, 
                - **Zinc (Zn) (ug/g)**: {ideal_ranges["Zinc Zn (ug/g)"][0]} to {ideal_ranges["Zinc Zn (ug/g)"][1]}
                """
            )

        color: str = cfg.STAPP["STYLES"]["COL_BACKGROUND_COLOR"]

        # Column selection
        all_columns: list[str] = data.columns.tolist()
        selected_columns: list[str] = st.multiselect(
            "Select columns to display", all_columns, default=all_columns
        )

        # Apply the styling using Styler.map
        styled_data: Styler = data[selected_columns].style
        styled_data = styled_data.apply(
            lambda col: [
                highlight_ideal_values(val, *ideal_ranges["pH"], color=color)
                for val in col
            ],
            subset=["pH"] if "pH" in selected_columns else [],
        )
        styled_data = styled_data.apply(
            lambda col: [
                highlight_ideal_values(val, *ideal_ranges["SOC (%)"], color=color)
                for val in col
            ],
            subset=["SOC (%)"] if "SOC (%)" in selected_columns else [],
        )
        styled_data = styled_data.apply(
            lambda col: [
                highlight_ideal_values(
                    val, *ideal_ranges["Boron B (ug/g)"], color=color
                )
                for val in col
            ],
            subset=["Boron B (ug/g)"] if "Boron B (ug/g)" in selected_columns else [],
        )
        styled_data = styled_data.apply(
            lambda col: [
                highlight_ideal_values(
                    val, *ideal_ranges["Zinc Zn (ug/g)"], color=color
                )
                for val in col
            ],
            subset=["Zinc Zn (ug/g)"] if "Zinc Zn (ug/g)" in selected_columns else [],
        )

        # Display the data
        st.write(styled_data)

    elif page == cfg.STAPP["PAGES"]["K_MEANS_PCA"]:
        # Centered title
        display_centered_title(
            cfg.STAPP["PAGES"]["K_MEANS_PCA"],
            color=cfg.STAPP["STYLES"]["TITLE_COLOR"],
        )
        ########################
        # K-Means and PCA
        ########################
        # Separator
        # st.markdown("""---""")

        background_color: str = cfg.STAPP["STYLES"]["BACKGROUND"]
        text_color: str = cfg.STAPP["STYLES"]["TEXT_COLOR"]
        sub_title_color = cfg.STAPP["STYLES"]["TITLE_COLOR"]

        # print(f"Backgound Color: {background_color}")
        # print(f"Text Color: {text_color}")

        with st.expander(
            "K-Means and Principal Component Analysis (Un-Supervised Learning)"
        ):
            st.write(
                f"""
                - Performing **clustering** to uncover any underlying patterns within these soil profiles. 
                
                - **K-means** clustering will be a suitable choice for this unsupervised analysis.
               
                - The **elbow** point in the inertia plot suggests an optimal number of clusters around 3 or 4. 
                
                - We will proceed with **K-means clustering** using **3 clusters** and then use **PCA** for visualization of 
                these clusters to see how different soil profiles group together.
                """
            )
        with st.expander("Insights"):
            st.markdown(
                """
                ### Cluster Insights

                - **Cluster 0**: 
                - Concentrated around the negative side of the PCA Component 1 axis, with points mostly on the left.
                - Likely represents soils with **moderate nutrient levels**.

                - **Cluster 1**: 
                - Spans a range in both PCA Component 1 and PCA Component 2, extending towards the right side of the plot.
                - This grouping may indicate **nutrient-rich soils**, as it diverges from the other clusters.

                - **Cluster 2**: 
                - Contains only few points.
                - Suggests soils with **distinct characteristics**, possibly with lower fertility or unique nutrient imbalances.

                **Summary**:
                - The cluster on the left is concentrated around moderate nutrient levels.
                - The cluster on the right may include nutrient-rich soils.
                - The isolated points in Cluster 2 may represent outliers or soils with unique properties.
                """
            )
        ####################################
        ### PCA ### Plots ( Row 1)
        ####################################
        col1, col2 = st.columns([2, 1])

        with col1:
            clustered_data: pd.DataFrame = perform_kmeans_and_pca(
                data,
                n_clusters=3,
                background_color=background_color,
                text_color=text_color,
            )

        with col2:
            plot_elbow_curve(
                data, background_color=background_color, text_color=text_color
            )

        ############################################
        ### Further Analysis on Clusters (Row 2) ###
        ############################################
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 4, 4])
        with col1:
            # Add vertical space to center the content
            for _ in range(14):
                st.write("")  # Add empty strings to create space

            # Select feature for histogram
            feature: str = st.selectbox(
                "Select Feature",
                clustered_data.select_dtypes(include=["float64"]).columns,
                key="plot_clusters",
                index=1,
            )
        with col2:
            title_h3("Box Plots by Clusters", color=sub_title_color)
            plot_cluster_boxplots(
                clustered_data,
                feature,
                background_color=background_color,
                text_color=text_color,
            )
        with col3:
            title_h3("Histograms by Clusters", color=sub_title_color)
            plot_cluster_histograms(
                clustered_data,
                feature,
                background_color=background_color,
                text_color=text_color,
            )
        ############################################
        ### Further Analysis on Clusters (Row 3) ###
        ############################################
        st.markdown("---")
        title_h3(
            "Comparisons of clusters with categorical features",
            color=sub_title_color,
        )
        plot_cluster_comparisons(
            clustered_data, background_color=background_color, text_color=text_color
        )
        ############################################
        ### Further Analysis on Clusters (Row 4) ###
        ############################################
        st.markdown("---")
        # Calculate statistics
        stats = cluster_statistics(clustered_data)
        # Display statistics in Streamlit
        title_h3("Descriptive Statistics for Each Cluster", color=sub_title_color)
        # st.subheader("Descriptive Statistics for Each Cluster")
        st.dataframe(stats)  # Use st.table(stats) for a static table

    elif page == cfg.STAPP["PAGES"]["RANDOM_FOREST"]:
        # Centered title
        display_centered_title(
            cfg.STAPP["PAGES"]["RANDOM_FOREST"],
            color=cfg.STAPP["STYLES"]["TITLE_COLOR"],
        )

    elif page == cfg.STAPP["PAGES"]["ABOUT"]:
        # Centered title
        display_centered_title(
            cfg.STAPP["PAGES"]["ABOUT"], color=cfg.STAPP["STYLES"]["TITLE_COLOR"]
        )


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
