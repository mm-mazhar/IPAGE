from pydantic import BaseModel
from typing import Union, Literal
from enum import Enum


class RawData(BaseModel):

    longitude: float
    latitude: float
    area: float
    soil_group: Union[None, str]
    land_class: Union[
        None,
        Literal[
            "isda",
            "Medium low land",
            "Medium high land",
            "high ground",
            "Medium low land",
            "Shallow to medium high land",
            "Deep medium high land",
        ],
    ]
    soil_type: Union[
        None,
        Literal[
            "isda",
            "sandy loam",
            "loam",
            "clay loam",
            "unknown",
            "Clay loam",
            "loam clay",
            "brick",
            "in the sand",
        ],
    ]
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
    Area: str = "Mithpukur"
    pH: float = 5.3
    Nitrogen: float = 0.08
    Phosphorus: float = 12
    Potassium: float = 0.17
    Sulfur: float = 26.4
    Sand: float = 33
    Silt: float = 33
    Clay: float = 33


class PredictionResponse(BaseModel):
    SOC: float = None
    Boron: float = None
    Zinc: float = None


class TargetSelect(str, Enum):
    SOC = "SOC"
    Boron = "Boron"
    Zinc = "Zinc"


class Desc(BaseModel):
    name: str
    api_version: str
    package_version: str
