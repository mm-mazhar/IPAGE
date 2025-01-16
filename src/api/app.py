from fastapi import FastAPI, File, UploadFile
from typing import Literal, Union
from src.api.db import DB
from src.api.models import SoilData
from src.api.schema import RawData, PredictionInput, PredictionResponse
from sqlalchemy.orm import class_mapper
from fastapi.responses import FileResponse
from src.model.model import DataPreprocessor, BaseModel
import os


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
def retrain_model():
    data = DataPreprocessor("merged_v3.csv", "data").preprocess()
    model = BaseModel(["SOC"])

    model.train(data)
    result = model.evaluate()
    return {"status": "Success", "metrics": result}

@app.post("/train/upload", response_class=FileResponse)
def train_model_with_uploaded_data(file: UploadFile = File(...)):
    filepath = os.path.join(os.getcwd(), "data", file.filename)
    print(filepath)
    return filepath


@app.post("/predict/", response_model=PredictionResponse)
def predict(data: RawData):
    return {"status": "Success", "message": data}

@app.post("/predict/upload/", response_class=FileResponse)
def upload_prediction_data(file: UploadFile = File(...)):
    filepath = os.path.join(os.getcwd(), "data", file.filename)
    print(filepath)
    return filepath
