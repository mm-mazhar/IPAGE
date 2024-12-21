# -*- coding: utf-8 -*-
# """
# data_overview.py
# Created on Dec 17, 2024
# @ Author: Mazhar
# """

import os

import pandas as pd
import streamlit as st
from utils.sweetviz import generate_sweetviz_report, st_display_sweetviz


def display_sweetviz_report_page(
    cfg: dict,
    data: pd.DataFrame,
    exclude_cols: list[str],
) -> None:
    """
    Function to display the Sweetviz report page in Streamlit.

    Args:
        cfg (dict): Configuration object containing paths and styles.
        data (pd.DataFrame): Input dataset.
        exclude_cols (list[str]): Columns to exclude (e.g., longitude, latitude).
    """
    
    # Paths and configurations
    report_path: str = cfg.PATHS.REPORT
    width: int = cfg.STAPP.SWEETVIZ.WIDTH
    height: int = cfg.STAPP.SWEETVIZ.HEIGHT

    # print(f"Width: {width}")
    # print(f"Height: {height}")

    # Generate dfSummary and convert to HTML
    # summary_html = dfSummary(data).to_html()

    # Display the summary in Streamlit
    # st.components.v1.html(summary_html, width=1000, height=1100, scrolling=True)

    # Generate or display Sweetviz report
    if st.button("Generate Sweetviz Report"):
        # Drop excluded columns
        data_sweetviz: pd.DataFrame = data.drop(columns=exclude_cols)
        generate_sweetviz_report(data_sweetviz, report_path)
        if os.path.exists(report_path):
            st_display_sweetviz(report_path, width, height)
    elif os.path.exists(report_path):
        # Display existing report
        st_display_sweetviz(report_path, width, height)
    else:
        st.write("No report available. Please press the button to generate it.")
