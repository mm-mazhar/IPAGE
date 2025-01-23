# -*- coding: utf-8 -*-
# """
# prediction_page.py
# Created on Jan 22, 2025
# @ Author: Mazhar/Taylor Will
# """

import os
import tempfile
from typing import Any

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
    # st.write(cfg.API.URL)
    # st.write("Type:", type(cfg.API.TIME_OUT))
    # st.write(f"Value: {cfg.API.TIME_OUT}")

    with st.form("single_prediction_form"):

        st.markdown("### Select Target to Predict:")
        targets = st.selectbox(
            "Select Target", ["SOC", "Boron", "Zinc"], key="single_inference"
        )

        st.markdown("### Enter the features for prediction:")

        # Row 1: Area
        area = st.text_input("Area", value="Mithpukur")

        # Row 2: pH and Nitrogen
        col1, col2 = st.columns(2)
        with col1:
            pH = st.number_input("pH", value=5.3)
        with col2:
            nitrogen = st.number_input("Nitrogen", value=0.08)

        # Row 3: Phosphorus and Potassium
        col3, col4 = st.columns(2)
        with col3:
            phosphorus = st.number_input("Phosphorus", value=12.0)
        with col4:
            potassium = st.number_input("Potassium", value=0.17)

        # Row 4: Sulfur and Sand
        col5, col6 = st.columns(2)
        with col5:
            sulfur = st.number_input("Sulfur", value=26.4)
        with col6:
            sand = st.number_input("Sand", value=33)

        # Row 5: Silt and Clay
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
        st.write(input_data)
        try:
            headers = {}
            if hasattr(cfg, "API_KEY"):
                headers["Authorization"] = f"Bearer {cfg.API_KEY}"

            # Include the targets parameter in the URL
            url = f"{cfg.API.URL}{cfg.API.VER_STRING}{cfg.API.SINGLE_INFERENCE_ENDPOINT}?targets={targets}"
            st.write("API Endpoint:", url)

            response = requests.post(
                url,
                json=input_data,
                headers=headers,
                timeout=cfg.API.TIME_OUT,  # Optional: set a timeout for the request
            )
            response.raise_for_status()

            prediction = response.json().get("prediction", None)
            if prediction is not None:
                # st.success(f"**Prediction:** {prediction}")
                st.success(f"**{targets}:** {prediction.get(targets, 'Not Found')}")
            else:
                st.error("Prediction not found in the response.")
        except requests.exceptions.Timeout:
            st.error("The request timed out. Please try again later.")
        except requests.exceptions.HTTPError as http_err:
            st.error(f"HTTP error occurred: {http_err}")
        except requests.exceptions.RequestException as err:
            st.error(f"An error occurred: {err}")
        except ValueError:
            st.error("Invalid response format received from the API.")


def batch_prediction(cfg: Any) -> None:
    """
    Handle batch data point predictions.

    Args:
        cfg: Configuration object containing API_URL and possibly API_KEY.
    """
    st.header("📂 Batch Prediction")

    st.markdown("### Select Target to Predict:")
    targets = st.selectbox(
        "Select Target", ["SOC", "Boron", "Zinc"], key="batch_inference"
    )

    st.markdown("### Upload a CSV file for batch prediction:")

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is None:
        st.write("No file uploaded yet.")

    if uploaded_file is not None:
        try:
            try:
                # Define the folder where you want to save the file
                upload_dir = cfg.PATHS.STAPP_UPLOADS_DIR  # Specify your folder name
                os.makedirs(
                    upload_dir, exist_ok=True
                )  # Create directory if it does not exist

                # Create the path of the file to save
                filepath = os.path.join(upload_dir, uploaded_file.name)

                # Read the file in bytes
                file_bytes = uploaded_file.read()

                # Write bytes to file
                try:
                    with open(filepath, "wb") as buffer:
                        buffer.write(file_bytes)
                    st.success(
                        f"File '{uploaded_file.name}' saved successfully to {filepath}"
                    )
                    # st.success(f"File Name: {os.path.basename(filepath)}")
                except Exception as e:
                    st.error(f"An error occurred while saving the file: {e}")
                    return
            except Exception as e:
                st.error(f"An error occurred while Uploading File: {e}")

            # Read the uploaded CSV
            try:
                with open(
                    filepath, "rb"
                ) as f:  # Using with to ensure correct file handle closing
                    batch_data = pd.read_csv(f)
                    st.markdown("#### Preview of Uploaded Data:")
                    st.dataframe(batch_data.head())
                    st.info(f"Length of Uploaded Data: {len(batch_data)}")
            except FileNotFoundError:
                st.error(f"File could not be found: {filepath}")
                return
            except Exception as e:
                st.error(f"An error occured while reading CSV: {e}")
                return

            # Define required features
            required_features = [
                "Area",
                "pH",
                "Nitrogen",
                "Phosphorus",
                "Potassium",
                "Sulfur",
                "Sand",
                "Silt",
                "Clay",
            ]  # Replace with actual features

            missing_features = [
                feat for feat in required_features if feat not in batch_data.columns
            ]
            if missing_features:
                st.error(
                    f"The following required features are missing from the uploaded file: {', '.join(missing_features)}"
                )
                os.remove(filepath)  # Cleanup temp file
                return

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
                    # headers["Content-Type"] = "multipart/form-data"

                    # Make the URL
                    url = f"{cfg.API.URL}{cfg.API.VER_STRING}{cfg.API.BATCH_INFERENCE_ENDPOINT}?targets={targets}"
                    st.write("API Endpoint: ", url)

                    # Convert the uploaded file to a file-like object for requests
                    # files = {"file": (uploaded_file.name, uploaded_file, "text/csv")}
                    files = {
                        "file": (
                            "file",
                            # os.path.basename(filepath),
                            open(filepath, "rb"),
                            "text/csv",
                        )
                    }
                    with open(filepath, "rb") as file_obj:
                        files = {
                            "file": (
                                "file",
                                file_obj,  # pass on the file handle
                                "text/csv",
                            )
                        }
                        response = requests.post(
                            url,
                            headers=headers,
                            files=files,  # Send the file as multipart/form-data
                            timeout=cfg.API.TIME_OUT,
                        )
                        response.raise_for_status()

                        # st.write(f"response.text: {response.text}")
                        # st.write(f"Response Type: {type(response.json())}")
                        # st.write(f"Response Length: {len(response.json()[targets])}")
                        # st.write(f"Response: {response.json()}")

                        predictions = pd.DataFrame(response.json())
                        # Rename the column to match the target
                        predictions.rename(
                            columns={f"{targets}": f"Pred_{targets}"}, inplace=True
                        )
                        # st.dataframe(predictions.head())

                        # Convert DataFrame Column to List
                        predictions = predictions[f"Pred_{targets}"].tolist()

                        # st.info(f"Length of Predictions: {len(predictions)}")
                        # st.write(f"Predictions Type: {type(predictions)}")
                        # st.write(f"Predictions: {predictions}")

                        if predictions and isinstance(predictions, list):
                            if len(predictions) != len(batch_data):
                                st.warning(
                                    "Number of predictions does not match number of input records."
                                )
                            batch_data[f"Pred_{targets}"] = predictions
                            st.success("Batch prediction completed successfully!")
                            st.dataframe(batch_data)

                            # Optionally, allow users to download the results
                            csv = batch_data.to_csv(index=False).encode("utf-8")
                            st.download_button(
                                label="📥 Download Predictions as CSV",
                                data=csv,
                                file_name="batch_predictions.csv",
                                mime="text/csv",
                            )
                        else:
                            st.error(
                                "Predictions not found or invalid format in the response."
                            )
                except requests.exceptions.Timeout:
                    st.error("The request timed out. Please try again later.")
                except requests.exceptions.HTTPError as http_err:
                    st.error(f"HTTP error occurred: {http_err}")
                except requests.exceptions.RequestException as err:
                    st.error(f"An error occurred: {err}")
                except ValueError:
                    st.error("Invalid response format received from the API.")
                except Exception as e:
                    st.error(f"An error occurred while making the request {e}")
        except pd.errors.EmptyDataError:
            st.error("The uploaded CSV file is empty.")
        except pd.errors.ParserError:
            st.error("Error parsing the uploaded CSV file.")
        except Exception as e:
            st.error(f"An unexpected error occurred while running the code: {e}")
        finally:
            try:
                if filepath and os.path.exists(filepath):
                    os.remove(filepath)
            except Exception as e:
                st.error(f"Failed to remove the file {filepath} error: {e}")


def display_prediction_page(cfg: Any) -> None:
    """
    Display the Prediction Page with single and batch prediction sections.

    Args:
        cfg: Configuration object containing API_URL and possibly API_KEY.
    """
    display_centered_title("🔮 Prediction", color=cfg.STAPP.STYLES.TITLE_COLOR)

    # Single Point Prediction Section
    single_prediction(cfg)

    st.markdown("---")  # Separator

    # Batch Prediction Section
    batch_prediction(cfg)
