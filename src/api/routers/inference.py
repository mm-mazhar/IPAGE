import os
from typing import List

import joblib
import numpy as np
import pandas as pd
from fastapi import APIRouter, File, HTTPException, Query, UploadFile, status
from fastapi.responses import FileResponse

from src.api.schema import PredictionInput, PredictionResponse, TargetSelect
from src.model.model import MODEL_FILE_PATH

preFix_user = "/inference"
inference_router = APIRouter(prefix=preFix_user, tags=["Inference"])


@inference_router.post("/point")
def predict(data: PredictionInput, targets: List[TargetSelect] = Query(...)):
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
    targets: List[TargetSelect] = Query(...), file: UploadFile = File(...)
):
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
    pred_df = pd.DataFrame(columns=[target.name for target in targets])

    for target in targets:
        model_path = MODEL_FILE_PATH.joinpath(f"{target.name}_Ridge")
        model = joblib.load(model_path)
        pred_df[target.name] = model.predict(df)[:, 0]

    # print(pred_df)

    # Return as JSON response
    return pred_df.to_dict()


# @inference_router.post("/batch", response_class=FileResponse)
# async def upload_prediction_data(
#     targets: List[TargetSelect] = Query(...), file: UploadFile = File(...)
# ):
#     # Define the file path where the uploaded file will be saved
#     filepath = os.path.join(os.getcwd(), "data", file.filename)

#     # Ensure the directory exists
#     os.makedirs(os.path.dirname(filepath), exist_ok=True)

#     # Write the uploaded file to the defined filepath
#     with open(filepath, "wb") as buffer:
#         buffer.write(await file.read())
#     # Read the saved file into a DataFrame
#     df = pd.read_csv(filepath)
#     pred_df = pd.DataFrame(columns=[target.name for target in targets])
#     for target in targets:
#         model_path = MODEL_FILE_PATH.joinpath(f"{target.name}_Ridge")
#         model = joblib.load(model_path)
#         pred_df[target.name] = model.predict(df)[:, 0]
#     print(pred_df)
#     return pred_df.to_dict()
