# -*- coding: utf-8 -*-
# """
# folium_map.py
# Created on Dec 19, 2024
# @ Author: Mazhar
# """

from typing import Any

import folium
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm, colors
from streamlit_folium import st_folium


def create_folium_map(
    data, selected_numeric_feature=None, selected_categorical_feature=None
) -> folium.Map:
    """
    Create a Folium map with data points and include selected numerical and categorical features in the tooltip.

    Args:
        data (pd.DataFrame): Data containing latitude, longitude, and features to visualize.
        selected_numeric_feature (str, optional): The numerical feature to visualize in the tooltip.
        selected_categorical_feature (str, optional): The categorical feature to visualize in the tooltip.

    Returns:
        folium.Map: A Folium map object with visualizations.
    """
    # Create base map centered on the data
    m = folium.Map(
        location=[data["latitude"].mean(), data["longitude"].mean()],
        left="10%",
        top="0%",
        position="relative",
        tiles="OpenStreetMap",
        zoom_control=True,
        # zoom_start=2,
        # width="100%",  # or specify a fixed width like '800px'
        # height="300px",  # specify the height to make it more rectangular
    )

    # Iterate through each data point
    for _, row in data.iterrows():
        # Create a tooltip that includes area, numerical feature, and categorical feature
        tooltip_text: str = f"Area: {row['Area']}<br>"
        if selected_numeric_feature:
            tooltip_text += (
                f"{selected_numeric_feature}: {row[selected_numeric_feature]:.2f}<br>"
            )
        if selected_categorical_feature:
            tooltip_text += f"{selected_categorical_feature}: {row[selected_categorical_feature]}<br>"

        # Add a marker or circle marker depending on the numerical feature
        if selected_numeric_feature and isinstance(
            row[selected_numeric_feature], (int, float)
        ):
            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]],
                radius=6,
                fill=True,
                fill_color=plt.cm.viridis(
                    row[selected_numeric_feature] / data[selected_numeric_feature].max()
                ),
                color="red",
                fill_opacity=0.5,
                tooltip=tooltip_text,
            ).add_to(m)
        else:
            folium.Marker(
                location=[row["latitude"], row["longitude"]],
                tooltip=tooltip_text,
            ).add_to(m)

    return m
