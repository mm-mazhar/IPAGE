from fastapi import FastAPI
from pydantic import BaseModel
from typing import Literal, Union

app = FastAPI()

class InferenceData(BaseModel):

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


@app.get("/")
def read_root():
    return {"message": "Welcome to IPAGE API"}

@app.get("/predict/")
def predict(data: InferenceData):
    return {"message": "Prediction endpoint"}