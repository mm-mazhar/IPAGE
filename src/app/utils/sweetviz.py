# -*- coding: utf-8 -*-
# """
# sweetviz.py
# Created on Nov 07, 2024
# @ Author: Mazhar
# """

import time

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import sweetviz as sv
from streamlit.delta_generator import DeltaGenerator


def st_display_sweetviz(report_path, width=1325, height=600) -> None:
    # Read HTML file content
    with open(report_path, "r", encoding="utf-8") as file:
        report_html = file.read()

    # Adjust outer div to align with the sidebar
    left_aligned_report_html = f"""
    <div style="display: flex; justify-content: flex-start; padding: 0; margin: 0;">
        <div style="width: 100%; max-width: {width}px; height: {height}px; overflow: auto; border: 1px solid #ddd; padding: 0px; text-align: left;">
            {report_html}
        </div>
    </div>
    """

    # Display the left-aligned report in Streamlit
    components.html(
        left_aligned_report_html, width=width, height=height, scrolling=True
    )

    # components.html(report_html, width=width, height=height, scrolling=True)


def generate_sweetviz_report(data: pd.DataFrame, report_path: str) -> None:
    # Display progress bar
    progress_text: DeltaGenerator = st.empty()
    progress_bar: DeltaGenerator = st.progress(0)
    progress_text.text("Initializing Sweetviz report generation...")

    # Simulate initial setup progress
    for percent_complete in range(0, 30, 10):
        time.sleep(0.5)  # Simulate initial setup time
        progress_bar.progress(percent_complete)

    # Generate Sweetviz report
    progress_text.text("Analyzing data with Sweetviz...")
    report: sv.DataframeReport = sv.analyze(
        data,
    )

    # Simulate report generation progress
    for percent_complete in range(30, 70, 10):
        time.sleep(0.5)  # Simulate analysis time
        progress_bar.progress(percent_complete)

    # Save the report as HTML
    progress_text.text("Saving Sweetviz report...")
    report.show_html(
        filepath=report_path, open_browser=False, layout="widescreen", scale=0.7
    )

    # Simulate finalization progress
    for percent_complete in range(70, 101, 10):
        time.sleep(0.5)  # Simulate finalization time
        progress_bar.progress(percent_complete)

    progress_text.text("Sweetviz report generated!")
