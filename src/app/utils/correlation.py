# -*- coding: utf-8 -*-
# """
# correlation.py
# Created on Nov 07, 2024
# @ Author: Mazhar
# """

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sns
from plotly.graph_objs._figure import Figure


def plot_correlation_heatmap(
    df: pd.DataFrame, numerical_features: list[str]
) -> None:
    # Calculate the correlation matrix
    correlation_matrix = df[numerical_features].corr()

    # Set up the matplotlib figure
    plt.figure(figsize=(12, 8))

    # Create a heatmap using Seaborn
    sns.heatmap(
        correlation_matrix,
        annot=True,  # Annotate cells with correlation values
        cmap="coolwarm",  # Use the 'coolwarm' colormap
        linewidths=0.5,  # Add lines between cells
        cbar_kws={"shrink": 0.8},  # Shrink the color bar
    )

    # Add a title
    plt.title("Correlation Heatmap of Numerical Features")

    # Show the plot
    plt.show()


def generate_correlation_story(correlation_matrix: pd.DataFrame) -> str:
    story = []
    for col in correlation_matrix.columns:
        for row in correlation_matrix.index:
            if col != row:
                corr_value = correlation_matrix.loc[row, col]
                if corr_value > 0.7:
                    story.append(
                        f"The features '{row}' and '{col}' have a strong positive correlation of {corr_value:.2f}."
                    )
                elif corr_value < -0.7:
                    story.append(
                        f"The features '{row}' and '{col}' have a strong negative correlation of {corr_value:.2f}."
                    )
    return "\n".join(story)


# Example usage
# correlation_matrix = data[numerical_features].corr()
# story = generate_correlation_story(correlation_matrix)
# print(story)

# Usage
# plot_correlation_heatmap(data, numerical_features)
