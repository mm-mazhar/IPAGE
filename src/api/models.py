"""Model for the soil data table structure"""

from sqlalchemy import Column, Integer, String, DateTime, Float
from sqlalchemy.orm import declarative_base

base = declarative_base()

class SoilData(base):
    __tablename__ = 'soil_data'

    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(DateTime)
    longitude = Column(Float)
    latitude = Column(Float)
    area = Column(Float)
    soil_group = Column(String)
    land_class = Column(String)
    soil_type = Column(String)
    pH = Column(Float)
    SOC = Column(Float)
    Nitrogen = Column(Float)
    Potassium = Column(Float)
    Phosphorus = Column(Float)
    Sulfur = Column(Float)
    Boron = Column(Float)
    Zinc = Column(Float)
    Sand = Column(Float)
    Silt = Column(Float)
    Clay = Column(Float)
