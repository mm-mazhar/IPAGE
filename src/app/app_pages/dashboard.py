# -*- coding: utf-8 -*-
# """
# dashboard.py
# Created on Dec 17, 2024
# @ Author: Mazhar
# """

import streamlit as st
from utils.feature_analysis import *
from utils.statistics_card import display_stat_card


def display_stat_cards(
    cfg, data, fig_size, background_color, dist_plot_color, exclude_cols
) -> None:
    """
    Function to display multiple stat cards in columns.

    Args:
        cfg: Configuration object containing STATISTICS_CARD attributes.
        data: The dataset to be displayed/analyzed.
        fig_size (tuple): Tuple specifying figure size (width, height).
        background_color (str): Background color for the stat card.
        dist_plot_color (str): Color for the distribution plot.
        exclude_cols (list): Columns to be excluded.
    """
    col1, col2, col3, col4 = st.columns(4)
    stat_cards = [
        cfg.STAPP.STATISTICS_CARD.COL_1,
        cfg.STAPP.STATISTICS_CARD.COL_2,
        cfg.STAPP.STATISTICS_CARD.COL_3,
        cfg.STAPP.STATISTICS_CARD.COL_4,
    ]

    for col, card in zip([col1, col2, col3, col4], stat_cards):
        with col:
            display_stat_card(
                card,
                data,
                figsize=(fig_size[0], fig_size[1]),
                background_color=background_color,
                color=dist_plot_color,
                exclude_cols=exclude_cols,
            )


def plots_bar_dist_pie(
    data,
    selected_numeric_feature,
    selected_method,
    selected_categorical_feature,
    background_color,
    text_color,
) -> None:
    """
    Function to display a bar plot and pie chart side by side.

    Args:
        data: The dataset used for visualization.
        selected_numeric_feature (str): Selected numeric feature for the bar plot.
        selected_method (str): Aggregation method for the bar plot.
        selected_categorical_feature (str): Selected categorical feature for the pie chart.
        background_color (str): Background color for the visualizations.
        text_color (str): Text color for the visualizations.
    """
    col1, col2 = st.columns(2)

    with col1:
        # Visualize Bar Plot
        visualize_bar_plot(
            data,
            selected_numeric_feature,
            selected_method,
            selected_categorical_feature,
            background_color=background_color,
            text_color=text_color,
            # width=width,
            # height=height,
        )

    with col2:
        # # Plot Distribution Chart
        # plot_dist_chart(
        #     data,
        #     selected_categorical_feature,
        #     background_color=background_color,
        #     text_color=text_color,
        #     # width=width,
        #     # height=height,
        # )

        # Plot Pie Chart
        plot_pie_chart(
            data,
            selected_categorical_feature,
            background_color=background_color,
            text_color=text_color,
        )


def plot_violin_strip_scatter(
    data,
    selected_numeric_feature_x,
    selected_numeric_feature,
    selected_categorical_feature,
    background_color,
    text_color,
) -> None:
    """
    Function to display comparison visualizations and a stacked bar chart side by side.

    Args:
        data: The dataset used for visualization.
        selected_numeric_feature_x (str): Selected numeric feature for the scatter plot.
        selected_numeric_feature (str): Selected numeric feature for comparison visuals.
        selected_categorical_feature (str): Selected categorical feature for comparison visuals.
        selected_features (list): Features to be used for the stacked bar chart.
        background_color (str): Background color for the visualizations.
        text_color (str): Text color for the visualizations.
    """
    col1, col2 = st.columns(2)

    with col1:
        # # Visualize Comparison Box
        # visualize_comparison_box(
        #     data,
        #     selected_numeric_feature,
        #     selected_categorical_feature,
        #     background_color=background_color,
        #     text_color=text_color,
        #     # width=width,
        #     # height=height,
        # )

        # Visualize Comparison Violin
        visualize_comparison_violin(
            data,
            selected_numeric_feature,
            selected_categorical_feature,
            background_color=background_color,
            text_color=text_color,
        )

        # # Visualize Comparison Strip
        # visualize_comparison_strip(
        #     data,
        #     selected_numeric_feature,
        #     selected_categorical_feature,
        #     background_color=background_color,
        #     text_color=text_color,
        #     # width=width,
        #     # height=height,
        # )

    with col2:
        # Visualize Scatter Plot
        scatter_plot(
            data,
            selected_numeric_feature_x,
            selected_numeric_feature,
            selected_categorical_feature,
            background_color=background_color,
            text_color=text_color,
        )
