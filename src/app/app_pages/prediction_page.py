# app_pages/prediction_page.py

import streamlit as st
import pandas as pd
import requests
from typing import Any
from utils.utils import display_centered_title  # Assuming you have this utility

def single_prediction(cfg: Any) -> None:
    """
    Handle single data point prediction.
    
    Args:
        cfg: Configuration object containing API_URL and possibly API_KEY.
    """
    st.header("🔮 Single Point Prediction")
    
    with st.form("single_prediction_form"):
        st.markdown("### Enter the features for prediction:")
        
        # Replace these with actual feature names and types
        feature1 = st.number_input("Feature 1", value=0.0)
        feature2 = st.number_input("Feature 2", value=0.0)
        feature3 = st.number_input("Feature 3", value=0.0)
        # Add more features as needed
        
        submit = st.form_submit_button("Predict")

    if submit:
        input_data = {
            "feature1": feature1,
            "feature2": feature2,
            "feature3": feature3,
            # Add more features here
        }

        st.markdown("### Sending data to the model...")
        try:
            headers = {}
            if hasattr(cfg, 'API_KEY'):
                headers['Authorization'] = f"Bearer {cfg.API_KEY}"
            
            response = requests.post(
                f"{cfg.API_URL}/api/v1/inference/point",
                json=input_data,
                headers=headers,
                timeout=10  # Optional: set a timeout for the request
            )
            response.raise_for_status()

            prediction = response.json().get("prediction", None)
            if prediction is not None:
                st.success(f"**Prediction:** {prediction}")
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
    
    st.markdown("### Upload a CSV file for batch prediction:")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    
    if uploaded_file is not None:
        try:
            # Read the uploaded CSV
            batch_data = pd.read_csv(uploaded_file)
            st.markdown("#### Preview of Uploaded Data:")
            st.dataframe(batch_data.head())

            # Define required features
            required_features = ["feature1", "feature2", "feature3"]  # Replace with actual features
            missing_features = [feat for feat in required_features if feat not in batch_data.columns]
            if missing_features:
                st.error(f"The following required features are missing from the uploaded file: {', '.join(missing_features)}")
                return

            if st.button("Predict on Batch"):
                st.markdown("### Sending data to the model...")
                try:
                    headers = {}
                    if hasattr(cfg, 'API_KEY'):
                        headers['Authorization'] = f"Bearer {cfg.API_KEY}"
                    
                    response = requests.post(
                        f"{cfg.API_URL}/api/v1/inference/batch",
                        json=batch_data[required_features].to_dict(orient="records"),
                        headers=headers,
                        timeout=30  # Optional: set a longer timeout for batch requests
                    )
                    response.raise_for_status()

                    predictions = response.json().get("predictions", None)
                    if predictions and isinstance(predictions, list):
                        if len(predictions) != len(batch_data):
                            st.warning("Number of predictions does not match number of input records.")
                        batch_data["Prediction"] = predictions
                        st.success("Batch prediction completed successfully!")
                        st.dataframe(batch_data)

                        # Optionally, allow users to download the results
                        csv = batch_data.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Download Predictions as CSV",
                            data=csv,
                            file_name='batch_predictions.csv',
                            mime='text/csv',
                        )
                    else:
                        st.error("Predictions not found or invalid format in the response.")
                except requests.exceptions.Timeout:
                    st.error("The request timed out. Please try again later.")
                except requests.exceptions.HTTPError as http_err:
                    st.error(f"HTTP error occurred: {http_err}")
                except requests.exceptions.RequestException as err:
                    st.error(f"An error occurred: {err}")
                except ValueError:
                    st.error("Invalid response format received from the API.")
        except pd.errors.EmptyDataError:
            st.error("The uploaded CSV file is empty.")
        except pd.errors.ParserError:
            st.error("Error parsing the uploaded CSV file.")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")

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
