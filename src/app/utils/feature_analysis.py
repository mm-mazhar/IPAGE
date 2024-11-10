# -*- coding: utf-8 -*-
# """
# feature_analysis.py
# Created on Nov 07, 2024
# @ Author: Mazhar
# """

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.graph_objs._figure import Figure
from utils.utils import display_centered_title


def visualize_bar_plot(
    df: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
    key="visualize_bar_plot",
    background_color: str = "white",
    text_color: str = "black",
    # width: int = 1000,
    # height: int = 600,
) -> None:
    """Visualize the aggregation of a numeric feature across different categories using a bar plot."""
    # display_centered_title("Bar Plot", color="red")
    st.markdown("""---""")
    # # Ensure width and height are integers
    # width = int(width)
    # height = int(height)

    methods: list[str] = ["mean", "sum", "count"]

    # Create two columns for feature selection
    col1, col2, col3 = st.columns(3)

    # Use the first column for numeric feature selection
    with col1:
        # st.write("Select Numeric Feature")
        selected_numeric_feature: str = st.selectbox(
            "Select Numeric Feature", numeric_cols, index=1, key=f"{key}_numeric"
        )

    with col2:
        # st.write("Select Method")
        selected_method: str = st.selectbox(
            "Select Method", methods, index=0, key=f"{key}_method"
        )

    # Use the second column for categorical feature selection
    with col3:
        # st.write("Select Categorical Feature")
        selected_categorical_feature: str = st.selectbox(
            "Select Categorical Feature",
            categorical_cols,
            index=1,
            key=f"{key}_categorical",
        )

    # Determine the aggregation based on the selected method
    if selected_numeric_feature and selected_categorical_feature:
        if selected_method == "mean":
            # st.subheader(
            #     f"Mean of {selected_numeric_feature} by {selected_categorical_feature}"
            # )
            aggregated_values: pd.DataFrame = (
                df.groupby(selected_categorical_feature)[selected_numeric_feature]
                .mean()
                .reset_index()
            )
        elif selected_method == "sum":
            # st.subheader(
            #     f"Sum of {selected_numeric_feature} by {selected_categorical_feature}"
            # )
            aggregated_values = (
                df.groupby(selected_categorical_feature)[selected_numeric_feature]
                .sum()
                .reset_index()
            )
        elif selected_method == "count":
            # st.subheader(
            #     f"Count of {selected_numeric_feature} by {selected_categorical_feature}"
            # )
            aggregated_values = (
                df.groupby(selected_categorical_feature)[selected_numeric_feature]
                .count()
                .reset_index()
            )
        else:
            st.error("Invalid method. Please use 'mean', 'sum', or 'count'.")
            return

        # Plot the bar plot
        fig: Figure = px.bar(
            aggregated_values,
            x=selected_categorical_feature,
            y=selected_numeric_feature,
            title=f"Bar Plot of {selected_method.capitalize()} {selected_numeric_feature} by {selected_categorical_feature}",
            color=selected_categorical_feature,
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        # Optional background color and dimensions
        fig.update_layout(
            plot_bgcolor=background_color,
            paper_bgcolor=background_color,
            title_font=dict(color=text_color),
            xaxis=dict(
                title_font=dict(color=text_color), tickfont=dict(color=text_color)
            ),
            yaxis=dict(
                title_font=dict(color=text_color), tickfont=dict(color=text_color)
            ),
            # width=width,  # Ensure width is set
            # height=height,  # Ensure height is set
        )
        st.plotly_chart(fig)


def visualize_comparison_box(
    df: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
    key="visualize_comparison_box",
    background_color: str = "white",
    text_color: str = "black",
    # width: int = 1000,
    # height: int = 600,
) -> None:
    """Visualize the comparison between selected numeric and categorical columns using Plotly."""
    # display_centered_title("Box Plot", color="red")
    st.markdown("""---""")
    # # Ensure width and height are integers
    # width = int(width)
    # height = int(height)

    # Create two columns for feature selection
    col1, col2 = st.columns(2)

    # Use the first column for numeric feature selection
    with col1:
        # st.write("Select Numeric Feature")
        selected_numeric_feature: str = st.selectbox(
            "Select Numeric Feature", numeric_cols, index=1, key=f"{key}_numeric"
        )

    # Use the second column for categorical feature selection
    with col2:
        # st.write("Select Categorical Feature")
        selected_categorical_feature: str = st.selectbox(
            "Select Categorical Feature",
            categorical_cols,
            index=1,
            key=f"{key}_categorical",
        )

    # Plot the comparison
    if selected_numeric_feature and selected_categorical_feature:
        # st.subheader(
        #     f"Comparison of {selected_numeric_feature} by {selected_categorical_feature}"
        # )
        fig: Figure = px.box(
            df,
            x=selected_categorical_feature,
            y=selected_numeric_feature,
            title=f"Box Plot of {selected_numeric_feature} by {selected_categorical_feature}",
            color=selected_categorical_feature,
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        # Optional background color
        fig.update_layout(
            plot_bgcolor=background_color,
            paper_bgcolor=background_color,
        )
        fig.update_layout(
            plot_bgcolor=background_color,
            paper_bgcolor=background_color,
            title_font=dict(color=text_color),
            xaxis=dict(
                title_font=dict(color=text_color), tickfont=dict(color=text_color)
            ),
            yaxis=dict(
                title_font=dict(color=text_color), tickfont=dict(color=text_color)
            ),
            # width=width,
            # height=height,
        )
        st.plotly_chart(fig)


def visualize_comparison_violin(
    df: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
    key="visualize_comparison_violin",
    background_color: str = "white",
    text_color: str = "black",
    # width: int = 1000,
    # height: int = 600,
) -> None:
    """Visualize the comparison between selected numeric and categorical columns using Plotly."""
    # display_centered_title("Violin Plot", color="red")
    st.markdown("""---""")
    # # Ensure width and height are integers
    # width = int(width)
    # height = int(height)

    # Create two columns for feature selection
    col1, col2 = st.columns(2)

    # Use the first column for numeric feature selection
    with col1:
        # st.write("Select Numeric Feature")
        selected_numeric_feature: str = st.selectbox(
            "Select Numeric Feature", numeric_cols, index=1, key=f"{key}_numeric"
        )

    # Use the second column for categorical feature selection
    with col2:
        # st.write("Select Categorical Feature")
        selected_categorical_feature: str = st.selectbox(
            "Select Categorical Feature",
            categorical_cols,
            index=1,
            key=f"{key}_categorical",
        )

    # Plot the comparison
    if selected_numeric_feature and selected_categorical_feature:
        # st.subheader(
        #     f"Comparison of {selected_numeric_feature} by {selected_categorical_feature}"
        # )
        fig: Figure = px.violin(
            df,
            x=selected_categorical_feature,
            y=selected_numeric_feature,
            title=f"Violin Plot of {selected_numeric_feature} by {selected_categorical_feature}",
            color=selected_categorical_feature,
            box=True,  # Adds a box plot inside the violin plot
            points="all",  # Shows all points
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_layout(
            plot_bgcolor=background_color,
            paper_bgcolor=background_color,
            title_font=dict(color=text_color),
            xaxis=dict(
                title_font=dict(color=text_color), tickfont=dict(color=text_color)
            ),
            yaxis=dict(
                title_font=dict(color=text_color), tickfont=dict(color=text_color)
            ),
            # width=width,
            # height=height,
        )
        st.plotly_chart(fig)


def visualize_comparison_strip(
    df: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
    key="visualize_comparison_strip",
    background_color: str = "white",
    text_color: str = "black",
    # width: int = 1000,
    # height: int = 600,
) -> None:
    """Visualize the comparison between selected numeric and categorical columns using Plotly."""
    # display_centered_title("Strip Plot", color="red")
    st.markdown("""---""")
    # # Ensure width and height are integers
    # width = int(width)
    # height = int(height)

    # Create two columns for feature selection
    col1, col2 = st.columns(2)

    # Use the first column for numeric feature selection
    with col1:
        # st.write("Select Numeric Feature")
        selected_numeric_feature: str = st.selectbox(
            "Select Numeric Feature", numeric_cols, index=1, key=f"{key}_numeric"
        )

    # Use the second column for categorical feature selection
    with col2:
        # st.write("Select Categorical Feature")
        selected_categorical_feature: str = st.selectbox(
            "Select Categorical Feature",
            categorical_cols,
            index=1,
            key=f"{key}_categorical",
        )

    # Plot the comparison
    if selected_numeric_feature and selected_categorical_feature:
        # st.subheader(
        #     f"Comparison of {selected_numeric_feature} by {selected_categorical_feature}"
        # )
        fig: Figure = px.strip(
            df,
            x=selected_categorical_feature,
            y=selected_numeric_feature,
            title=f"Strip Plot of {selected_numeric_feature} by {selected_categorical_feature}",
            color=selected_categorical_feature,
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        # Optional background and text color
        fig.update_layout(
            plot_bgcolor=background_color,
            paper_bgcolor=background_color,
            title_font=dict(color=text_color),
            xaxis=dict(
                title_font=dict(color=text_color), tickfont=dict(color=text_color)
            ),
            yaxis=dict(
                title_font=dict(color=text_color), tickfont=dict(color=text_color)
            ),
            # width=width,
            # height=height,
        )
        st.plotly_chart(fig)


def plot_stacked_bar_chart(
    df: pd.DataFrame,
    categorical_cols: list[str],
    key="plot_stacked_bar_chart",
    background_color: str = "white",
    text_color: str = "black",
    # width: int = 1000,
    # height: int = 600,
) -> None:
    """Plot a stacked bar chart for selected categorical features with customizable colors."""
    # display_centered_title("Stacked Bar Chart", color="red")
    st.markdown("""---""")
    # # Ensure width and height are integers
    # width = int(width)
    # height = int(height)

    # Allow users to select which categorical features to include
    selected_features: list[str] = st.multiselect(
        "Select Categorical Features",
        categorical_cols,
        default=categorical_cols[:2],
        key=f"{key}_categorical",
    )

    if len(selected_features) < 2:
        st.warning("Please select at least two categorical features.")
        return

    # Create a list to hold the traces
    traces: list[go.Bar] = []

    # Iterate over pairs of selected features
    for i in range(len(selected_features) - 1):
        feature1: str = selected_features[i]
        feature2: str = selected_features[i + 1]

        # Create a crosstab for the current pair of features
        crosstab: pd.DataFrame = pd.crosstab(df[feature1], df[feature2])

        # Add a trace for each category in feature2
        for category in crosstab.columns:
            traces.append(
                go.Bar(
                    x=crosstab.index,
                    y=crosstab[category],
                    name=f"{feature2}: {category}",
                )
            )

    # Create the figure
    fig = go.Figure(data=traces)

    # Update layout for stacked bars with custom colors
    fig.update_layout(
        barmode="stack",
        title="Stacked Bar Chart of Selected Categorical Features",
        xaxis_title="Categories",
        yaxis_title="Count",
        legend_title="Categories",
        plot_bgcolor=background_color,
        paper_bgcolor=background_color,
        font=dict(color=text_color),
        title_font=dict(color=text_color),
        # width=width,
        # height=height,
    )

    # Display the chart
    st.plotly_chart(fig)


# # Usage
# categorical_features = ['Area', 'soil group', 'Land class', 'knit (surface)']
# plot_stacked_bar_chart(data, categorical_features)
