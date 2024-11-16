# -*- coding: utf-8 -*-
# """
# scatter_plot.py
# Created on Nov 07, 2024
# @ Author: Mazhar
# """

import pandas as pd
import plotly.express as px
import streamlit as st


def scatter_plot(
    df: pd.DataFrame,
    key="scatter_plot",
    background_color: str = "white",
    text_color: str = "black",
) -> None:
    """Visualize data with respect to 'Data Collection Year' and other features."""
    st.markdown("""---""")

    # Create two columns with different widths
    col1, col2 = st.columns(
        [3, 1]
    )  # 3:1 ratio for larger left and smaller right column

    with col2:
        # Add vertical space to center the content
        for _ in range(6):
            st.write("")  # Add empty strings to create space

        # Exclude 'Data Collection Year' from numerical features
        numerical_features = [
            col
            for col in df.select_dtypes(include=["number"]).columns
            if col != "Data Collection Year"
        ]
        categorical_features = df.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

        # Ensure at least one categorical column is selected
        if not categorical_features:
            st.error("No categorical columns available.")
            return

        # Select time filter using a slider
        min_year = int(df["Data Collection Year"].min())
        max_year = int(df["Data Collection Year"].max())
        selected_year = st.slider(
            "Select Data Collection Year",
            min_value=min_year,
            max_value=max_year,
            value=min_year,  # Default to the minimum year
            step=1,
            key=f"{key}_year",
        )

        # Select categorical and numerical features
        selected_categorical_feature = st.selectbox(
            "Select Categorical Feature",
            categorical_features,
            key=f"{key}_categorical",
        )

        selected_numerical_feature = st.selectbox(
            "Select Numerical Feature",
            numerical_features,
            key=f"{key}_numerical",
        )

    with col1:
        # Filter data by selected year
        filtered_df = df[df["Data Collection Year"] == selected_year]

        # Plot with respect to 'Data Collection Year'
        if selected_categorical_feature and selected_numerical_feature:
            fig = px.scatter(
                filtered_df,
                x="Data Collection Year",
                y=selected_numerical_feature,
                color=selected_categorical_feature,
                title=f"{selected_numerical_feature} over Data Collection Year by {selected_categorical_feature}",
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
                # legend=dict(
                #     title=dict(
                #         text=selected_categorical_feature, font=dict(color=text_color)
                #     ),
                #     font=dict(color=text_color),
                # ),
            )
            st.plotly_chart(fig)


# Example usage
# visualize_data(data)
