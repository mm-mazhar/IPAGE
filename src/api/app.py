from fastapi import FastAPI, File, UploadFile,Query
from typing import Literal, Union,List
from src.api.db import DB
from src.api.models import SoilData
from src.api.schema import RawData, PredictionInput, PredictionResponse,TargetSelect
from sqlalchemy.orm import class_mapper
from fastapi.responses import FileResponse
from src.model.model import DataPreprocessor, BaseModel, MODEL_FILE_PATH
from sklearn.linear_model import Ridge
# from catboost import CatBoostRegressor
import os
import joblib
import pandas as pd
import numpy as np


app = FastAPI()
db = DB()


def object_to_dict(obj):
    # Extract columns from the SQLAlchemy model
    columns = [c.key for c in class_mapper(obj.__class__).columns]
    return {column: getattr(obj, column) for column in columns}

@app.get("/")
def read_root():
    return {"message": "Welcome to IPAGE API"}

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
def train_model_with_uploaded_data(file: UploadFile = File(...)):
    filepath = os.path.join(os.getcwd(), "data", file.filename)
    
    return filepath


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
async def upload_prediction_data(file: UploadFile = File(...)):
    # Define the file path where the uploaded file will be saved
    filepath = os.path.join(os.getcwd(), "data", file.filename)
    print(f"Filepath: {type(filepath)=}")
    print(filepath)

    # Ensure the directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Write the uploaded file to the defined filepath
    with open(filepath, "wb") as buffer:
        buffer.write(await file.read())
    
    # Read the saved file into a DataFrame
    df = pd.read_csv(filepath)
    print(df.head())
    return filepath
