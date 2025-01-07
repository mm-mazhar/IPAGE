from .models import Base, SoilData
from sqlalchemy import create_engine, URL
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm.session import Session
from datetime import datetime
from typing import Union, List
from dotenv import load_dotenv
import os

load_dotenv()
DATABASE = os.getenv('DATABASE', 'ipage.db')
DRIVERNAME = os.getenv('DRIVERNAME', 'sqlite')


class DB:
    def __init__(self, db_url: URL = URL.create(DRIVERNAME, database=DATABASE)) -> None:
        print(db_url)
        self._engine = create_engine(db_url)
        self.__session = None
        self._initilize_db()
    
    def _initilize_db(self):
        Base.metadata.create_all(self._engine)

    @property    
    def session(self) -> Session:
        if self.__session is None:
            self.__session = sessionmaker(bind=self._engine)
        return self.__session()
    
    @session.setter
    def session(self, session: Session):
        self.__session = session
    
    def close(self):
        if self.__session is not None:
            self.__session.close()
            self.__session = None
    
    def create_soil_data(self, data: dict) -> SoilData:
        date = datetime.now()
        data['date'] = date
        soil_data = SoilData(**data)
        self.__session.add(soil_data)
        self.__session.commit()
        return soil_data
    
    def retrieve_data(self, limit: Union[int, None] = None) -> Union[List[SoilData], SoilData]:
        if limit is not None:
            return self.__session.query(SoilData).limit(limit).all()
        return self.__session.query(SoilData).all()
    
    def retrieve_data_by_date(self, model, date_from, date_to = None) -> Union[List[SoilData], SoilData]:
        if date_to is None:
            date_to = datetime.now()
        return self.__session.query(model).filter(model.date >= date_from, model.date <= date_to).all()
