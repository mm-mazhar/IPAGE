from pydantic import BaseModel
from typing import Union, Literal

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


class PredictionInput(BaseModel):
    Area: str
    pH: float
    SOC: float
    Nitrogen: float
    Phosporus: float
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
