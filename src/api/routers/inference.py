import os
from typing import List

import joblib
import numpy as np
import pandas as pd
from fastapi import APIRouter, File, HTTPException, Query, UploadFile, status
from fastapi.responses import FileResponse

from src.api.schema import PredictionInput, PredictionResponse, TargetSelect
from src.model.config import REQUIRED_COLUMNS
from src.model.model import MODEL_FILE_PATH

preFix_user = "/inference"
inference_router = APIRouter(prefix=preFix_user, tags=["Inference"])


@inference_router.post("/point")
def predict(
    data: PredictionInput,
    targets: List[TargetSelect] = Query(
        default=[
            target.value for target in TargetSelect
        ],  # Default: Select all targets
        description="Select one or more target variables",
    ),
):
    df = pd.DataFrame(data.dict(), index=[0])
    pred_dict = {}
    for target in targets:
        model_path = MODEL_FILE_PATH.joinpath(f"{target.name}_Ridge")
        model = joblib.load(model_path)
        # print("Test")
        # print(model.predict(df))
        pred_dict[target.name] = np.round(model.predict(df)[0], 3)
    prediction = PredictionResponse(**pred_dict)
    return {"prediction": prediction}


@inference_router.post("/batch")
async def upload_prediction_data(
    targets: List[TargetSelect] = Query(
        default=[
            target.value for target in TargetSelect
        ],  # Default: Select all targets
        description="Select one or more target variables",
    ),
    file: UploadFile = File(...),
):
    """
    Train a model with new data. The csv file must contain the columns: Area, pH, Nitrogen, Potassium, Phosphorus,
    Sulfur, Sand, Silt, and Clay.

    - `targets`: A list of target variables to predict (e.g., SOC, Boron, Zinc).
    - `file`: The uploaded CSV file containing required features.

    """
    # Define the file path where the uploaded file will be saved
    filepath = os.path.join(os.getcwd(), "data", file.filename)

    # Ensure the directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Write the uploaded file to the defined filepath
    try:
        with open(filepath, "wb") as buffer:
            buffer.write(await file.read())
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"File upload failed: {str(e)}",
        )

    # Read the saved file into a DataFrame
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File read failed: {str(e)}",
        )

    # Check for required columns
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Missing required columns: {', '.join(missing_columns)}",
        )

    pred_df = pd.DataFrame(columns=[target.name for target in targets])

    try:
        for target in targets:
            model_path = MODEL_FILE_PATH.joinpath(f"{target.name}_Ridge")
            model = joblib.load(model_path)
            features = df[
                REQUIRED_COLUMNS
            ]  # Ensure you're using only the required columns
            pred_df[target.name] = model.predict(features)[:, 0]

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Model prediction error: {str(e)}",
        )
    except Exception as e:  # Catch any other potential errors
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred during prediction: {str(e)}",
        )

    # print(pred_df)

    # Return as JSON response
    return pred_df.to_dict()
