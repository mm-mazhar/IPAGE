import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import HTMLResponse
from typing import Literal, Union,List
from src.api.db import DB
from src.api.models import SoilData
from src.api.schema import RawData, PredictionInput, PredictionResponse,TargetSelect
from sqlalchemy.orm import class_mapper
from fastapi.responses import FileResponse
from src.model.model import DataPreprocessor, BaseModel, MODEL_FILE_PATH
from sklearn.linear_model import Ridge
# from catboost import CatBoostRegressor
import joblib
import pandas as pd
import numpy as np
from loguru import logger


# from app.config import settings, setup_app_logging
from .configs import (
    API_PROJECT_NAME,
    API_VER_STR,
    __version__,
    settings,
    # setup_app_logging,
)

# from api.db import models
# from api.db.database import engine

# setup logging as early as possible
# setup_app_logging(config=settings)

# Prefix for all API endpoints
preFix: str = API_VER_STR
api_project_name: str = API_PROJECT_NAME

# app = FastAPI()
app = FastAPI(
    title=f"{api_project_name}",
    openapi_url=f"{preFix}/openapi.json",
    swagger_ui_parameters={"syntaxHighlight.theme": "obsidian"},
    servers=settings.SERVERS,
)


db = DB()


def object_to_dict(obj):
    # Extract columns from the SQLAlchemy model
    columns = [c.key for c in class_mapper(obj.__class__).columns]
    return {column: getattr(obj, column) for column in columns}

@app.get("/")
def read_root():
    """Basic HTML response."""
    body: str = (
        "<html>"
        "<body style='display: flex; justify-content: center; align-items: center; height: 100vh; margin: 0; background-color: #7d8492;'>"
        "<div style='text-align: center; background-color: white; padding: 20px; border-radius: 20px;'>"
        f"<h1 style='font-weight: bold; font-family: Arial;'>{api_project_name}</h1>"
        f"<h3 style='font-weight: bold; font-family: Arial;'>Version: {__version__}</h3>"
        "<div>"
        f"<h4>Check the docs: <a href='/docs'>here</a><h4>"
        "</div>"
        "</div>"
        "</body>"
        "</html>"
    )

    return HTMLResponse(content=body)

@app.get("/data")
def get_data(limit: int = None):
    if limit:
        data = db.retrieve_data(limit=limit)
    else: 
        data = db.retrieve_data()
    result = [object_to_dict(obj) for obj in data]
    return {"status": "Success", "data": result}

@app.post("/data")
def create_data(data: RawData):
    duplicated = False
    status = "Failed"
    data = db.create_soil_data(data.dict())

    if isinstance(data, SoilData):
        status = "Success"
        data = object_to_dict(data)
    elif data is None:
        duplicated = True
    db.end_session()
    return {"status": status, "duplicated": duplicated, "data": data}

@app.post("/train/")
def train_model(target:TargetSelect):
    '''
    Training the model on combined isda and ipage data
    '''
    data = DataPreprocessor("merged_v3.csv", "data").preprocess()
    regression_model = Ridge
    model = BaseModel([target],regression_model)
    model.train(data)
    result = model.evaluate()
    model_name = f'{target}_{type(regression_model()).__name__}'
    print(model_name)
    model.save_model(model_name)
    return {"status": "Success", "metrics": result}

@app.post("/retrain/upload", response_class=FileResponse)
async def train_model_with_uploaded_data(file: UploadFile = File(...)):
    '''
    Train a model with new data. The csv file must contain the columns: Area, pH, Nitrogen, Phosphorus,
    Sulfur, Sand, Silt, and Clay
    '''
    # Define the file path where the uploaded file will be saved
    filepath = os.path.join(os.getcwd(), "data", file.filename)

    # Ensure the directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Write the uploaded file to the defined filepath
    with open(filepath, "wb") as buffer:
        buffer.write(await file.read())
    
    # Retraining pipeline
    data = DataPreprocessor(file.filename, "data").preprocess()
    regression_model = Ridge
    targets = ['SOC','Boron','Zinc']
    model = BaseModel(targets,regression_model)
    model.train(data)
    result = model.evaluate()
    for target in targets:
        model_name = f'retrain_{target}_{type(regression_model()).__name__}'
        print(model_name)
        model.save_model(model_name)
    return {"status": "Success", "metrics": result}


@app.post("/inference/point/")
def predict(data:PredictionInput,targets:List[TargetSelect] = Query(...)):
    df = pd.DataFrame(data.dict(),index=[0])
    pred_dict = {}
    for target in targets:
        model_path = MODEL_FILE_PATH.joinpath(f'{target.name}_Ridge')
        model = joblib.load(model_path)
        # print("Test")
        # print(model.predict(df))
        pred_dict[target.name] = np.round(model.predict(df)[0],3)
    prediction = PredictionResponse(**pred_dict)
    return {"prediction": prediction}

@app.post("/inference/batch/", response_class=FileResponse)
async def upload_prediction_data(targets:List[TargetSelect] = Query(...),file: UploadFile = File(...)):
    # Define the file path where the uploaded file will be saved
    filepath = os.path.join(os.getcwd(), "data", file.filename)

    # Ensure the directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Write the uploaded file to the defined filepath
    with open(filepath, "wb") as buffer:
        buffer.write(await file.read())
    # Read the saved file into a DataFrame
    df = pd.read_csv(filepath)
    pred_df = pd.DataFrame(columns=[target.name for target in targets])
    for target in targets:
        model_path = MODEL_FILE_PATH.joinpath(f'{target.name}_Ridge')
        model = joblib.load(model_path)
        pred_df[target.name] = model.predict(df)[:,0]
    print(pred_df)
    return pred_df.to_dict()


if __name__ == "__main__":
    # Use this for debugging purposes only
    logger.warning("Running in development mode. Do not run like this in production.")
    import uvicorn

    uvicorn.run(
        "src.api.app:app",
        host="localhost",
        port=settings.FASTAPI_PORT,
        log_level="debug",
        reload=True,
    )
