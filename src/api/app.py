from fastapi import FastAPI
from pydantic import BaseModel
from typing import Literal, Union
import datetime
from src.api.db import DB
from src.api.models import SoilData
from sqlalchemy.orm import class_mapper


app = FastAPI()
db = DB()

class RawData(BaseModel):

    longitude: float
    latitude: float
    area: float
    soil_group: Union[None, str]
    land_class: Union[
        None, 
        Literal[
            'isda', 'Medium low land', 'Medium high land',
            'high ground', 'Medium low land', 'Shallow to medium high land',
            'Deep medium high land']
        ]
    soil_type: Union[
        None,
        Literal[
            'isda', 'sandy loam', 'loam',
            'clay loam', 'unknown', 'Clay loam',
            'loam clay', 'brick', 'in the sand'
        ]]
    pH: float
    SOC: float
    Nitrogen: float
    Potassium: float
    Phosphorus: float
    Sulfur: float
    Boron: float
    Zinc: float
    Sand: float
    Silt: float
    Clay: float

class PredictionResponse(BaseModel):
    SOC: float
    Boron: float
    Zinc: float

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



# @app.post("/predict/", response_model=RawData)
@app.post("/predict/")
def predict(data: RawData):
    return {"status": "Success", "message": data}
