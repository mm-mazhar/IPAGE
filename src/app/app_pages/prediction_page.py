# -*- coding: utf-8 -*-
# """
# prediction_page.py
# Created on Jan 22, 2025
# @ Author: Mazhar/Taylor Will
# """

import os
import tempfile
from typing import Any
from urllib.parse import urlencode

import pandas as pd
import requests
import streamlit as st
from utils.utils import (  # Assuming you have this utility
    display_centered_title,
    title_h4_left,
)


def single_prediction(cfg: Any) -> None:
    """
    Handle single data point prediction.

    Args:
        cfg: Configuration object containing API_URL and possibly API_KEY.
    """
    st.header("🔮 Single Point Prediction")

    with st.form("single_prediction_form"):
        # ✅ Read selected targets from session state
        targets = st.session_state.get("selected_targets", [])

        st.markdown("### Enter the features for prediction:")

        # Feature Inputs
        area = st.text_input("Area", value="Mithpukur")

        col1, col2 = st.columns(2)
        with col1:
            pH = st.number_input("pH", value=5.3)
        with col2:
            nitrogen = st.number_input("Nitrogen", value=0.08)

        col3, col4 = st.columns(2)
        with col3:
            phosphorus = st.number_input("Phosphorus", value=12.0)
        with col4:
            potassium = st.number_input("Potassium", value=0.17)

        col5, col6 = st.columns(2)
        with col5:
            sulfur = st.number_input("Sulfur", value=26.4)
        with col6:
            sand = st.number_input("Sand", value=33)

        col7, col8 = st.columns(2)
        with col7:
            silt = st.number_input("Silt", value=33)
        with col8:
            clay = st.number_input("Clay", value=33)

        submit = st.form_submit_button("Predict")

    # st.write(f"area: {area}")
    # st.write(f"pH: {pH}")
    # st.write(f"Nitrogen: {nitrogen}")
    # st.write(f"Phosphorus: {phosphorus}")
    # st.write(f"Potassium: {potassium}")
    # st.write(f"Sulfur: {sulfur}")
    # st.write(f"Sand: {sand}")
    # st.write(f"Silt: {silt}")
    # st.write(f"Clay: {clay}")

    if submit:
        input_data = {
            "Area": area,
            "pH": pH,
            "Nitrogen": nitrogen,
            "Phosphorus": phosphorus,
            "Potassium": potassium,
            "Sulfur": sulfur,
            "Sand": sand,
            "Silt": silt,
            "Clay": clay,
        }

        title_h4_left(
            "Sending data to the model...",
            color=cfg.STAPP["STYLES"]["SUB_TITLE_COLOR"],
        )

        try:
            headers = {}
            if hasattr(cfg, "API_KEY"):
                headers["Authorization"] = f"Bearer {cfg.API_KEY}"

            query_params = urlencode([("targets", t) for t in targets])

            url = f"{cfg.API.URL}{cfg.API.VER_STRING}{cfg.API.SINGLE_INFERENCE_ENDPOINT}?{query_params}"

            response = requests.post(
                url,
                json=input_data,
                headers=headers,
                timeout=cfg.API.TIME_OUT,
            )
            response.raise_for_status()

            prediction = response.json().get("prediction", None)
            if prediction is not None:
                prediction_text = "\n".join(
                    [
                        f"**{target}:** {prediction.get(target, 'Not Found')}"
                        for target in targets
                    ]
                )
                st.success("Prediction completed successfully!")
                st.success(prediction_text)
            else:
                st.error("Prediction not found in the response.")
        except requests.exceptions.RequestException as err:
            st.error(f"An error occurred: {err}")


def batch_prediction(cfg: Any) -> None:
    """
    Handle batch data point predictions.

    Args:
        cfg: Configuration object containing API_URL and possibly API_KEY.
    """
    st.header("📂 Batch Prediction")

    # ✅ Read selected targets from session state
    targets = st.session_state.get("selected_targets", [])

    st.markdown("### Upload a CSV file for batch prediction:")
    st.info(
        "The csv file must contain: Area, pH, Nitrogen, Phosphorus, Potassium, Sulfur, Sand, Silt, and Clay."
    )

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            upload_dir = cfg.PATHS.STAPP_UPLOADS_DIR
            os.makedirs(upload_dir, exist_ok=True)

            filepath = os.path.join(upload_dir, uploaded_file.name)

            with open(filepath, "wb") as buffer:
                buffer.write(uploaded_file.read())

            st.success(f"File '{uploaded_file.name}' saved successfully.")

            batch_data = pd.read_csv(filepath)
            st.markdown("#### Preview of Uploaded Data:")
            st.dataframe(batch_data.head())

            if st.button("Predict on Batch"):
                title_h4_left(
                    "Sending data to the model...",
                    color=cfg.STAPP["STYLES"]["SUB_TITLE_COLOR"],
                )

                try:
                    headers = (
                        {"Authorization": f"Bearer {cfg.API_KEY}"}
                        if hasattr(cfg, "API_KEY")
                        else {}
                    )

                    query_params = urlencode([("targets", t) for t in targets])

                    url = f"{cfg.API.URL}{cfg.API.VER_STRING}{cfg.API.BATCH_INFERENCE_ENDPOINT}?{query_params}"
                    st.write("API Endpoint: ", url)

                    with open(filepath, "rb") as file_obj:
                        files = {"file": ("file", file_obj, "text/csv")}
                        response = requests.post(
                            url, headers=headers, files=files, timeout=cfg.API.TIME_OUT
                        )
                        response.raise_for_status()

                    response_json = response.json()
                    predictions_df = pd.DataFrame(response_json)
                    predictions_df.rename(
                        columns={col: f"Pred_{col}" for col in predictions_df.columns},
                        inplace=True,
                    )

                    predictions_df.reset_index(drop=True, inplace=True)
                    batch_data.reset_index(drop=True, inplace=True)

                    batch_data = pd.concat([batch_data, predictions_df], axis=1)

                    st.success("Batch prediction completed successfully!")
                    st.dataframe(batch_data.head())

                    csv = batch_data.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="📥 Download Predictions as CSV",
                        data=csv,
                        file_name="batch_predictions.csv",
                        mime="text/csv",
                    )

                except requests.exceptions.RequestException as err:
                    st.error(f"An error occurred: {err}")

        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")


def display_prediction_page(cfg: Any) -> None:
    """
    Display the Prediction Page with single and batch prediction sections.

    Args:
        cfg: Configuration object containing API_URL and possibly API_KEY.
    """
    display_centered_title("🔮 Prediction", color=cfg.STAPP.STYLES.TITLE_COLOR)

    # Move target selection to sidebar
    with st.sidebar:
        st.markdown("### 🎯 Select Targets for Prediction:")
        all_targets = [
            cfg.STAPP.TARGETS.SOC,
            cfg.STAPP.TARGETS.Boron,
            cfg.STAPP.TARGETS.Zinc,
        ]

        # Store selected targets in session state
        st.session_state["selected_targets"] = st.multiselect(
            "Choose one or more targets",
            options=all_targets,  # Available options
            default=all_targets,  # Pre-select all by default
            key="global_target_selection",
        )

    # Single Point Prediction Section
    single_prediction(cfg)

    st.markdown("---")  # Separator

    # Batch Prediction Section
    batch_prediction(cfg)
