# -*- coding: utf-8 -*-
# """
# pie_plot.py
# Created on Nov 07, 2024
# @ Author: Mazhar
# """

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.graph_objs._figure import Figure
from utils.utils import display_centered_title

def plot_pie_chart(
    df: pd.DataFrame,
    categorical_cols: list[str],
    key="pie_chart",
    background_color: str = "white",
    text_color: str = "black",
) -> None:
    """Plot a pie chart for selected categorical features with customizable colors."""
    st.markdown("""---""")

    # Allow users to select which categorical features to include
    selected_feature: str = st.selectbox(
        "Select Categorical Feature for Pie Chart",
        categorical_cols,
        index=0,
        key=f"{key}_categorical",
    )

    if selected_feature and selected_feature in df.columns:
        # Plot pie chart for the selected feature
        fig = px.pie(
            df,
            names=selected_feature,
            title=f"Pie Chart of {selected_feature}",
            color_discrete_sequence=px.colors.qualitative.Set2,  # Use a color sequence
        )
        fig.update_layout(
            plot_bgcolor=background_color,
            paper_bgcolor=background_color,
            title_font=dict(color=text_color),
            legend=dict(
                title=dict(text=selected_feature, font=dict(color=text_color)),
                font=dict(color=text_color),
            ),
        )
        st.plotly_chart(fig)
    else:
        st.error("Selected feature is not a valid column in the DataFrame.")