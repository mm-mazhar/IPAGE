# -*- coding: utf-8 -*-
# """
# dist_plots.py
# Created on Nov 07, 2024
# @ Author: Mazhar
# """

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.graph_objs._figure import Figure
from utils.utils import display_centered_title

def plot_dist_chart(
    df: pd.DataFrame,
    categorical_cols: list[str],
    key="dist_plot",
    background_color: str = "white",
    text_color: str = "black",
    # width: int = 1000,
    # height: int = 600,
) -> None:
    """Plot a distribution bar chart for selected categorical features with customizable colors."""
    st.markdown("""---""")

    # Allow users to select which categorical features to include
    selected_feature: str = st.selectbox(
        "Select Categorical Feature",
        categorical_cols,
        index=0,
        key=f"{key}_categorical",
    )

    if selected_feature and selected_feature in df.columns:
        # Plot distribution plot for the selected feature
        fig = px.histogram(
            df,
            x=selected_feature,
            color=selected_feature,  # Use the selected feature for coloring
            title=f"Distribution of {selected_feature}",
            color_discrete_sequence=px.colors.qualitative.Set2,  # Use a color sequence
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
            #     title=dict(text=selected_feature, font=dict(color=text_color)),
            #     font=dict(color=text_color),
            # ),
        )
        st.plotly_chart(fig)
    else:
        st.error("Selected feature is not a valid column in the DataFrame.")
        
