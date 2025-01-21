import os
from datetime import datetime, timezone
from typing import List, Union

from dotenv import load_dotenv
from sqlalchemy import URL, create_engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm.session import Session

from .models import Base, SoilData

load_dotenv()
DATABASE = os.getenv("DATABASE", "./src/api/ipage.db")
DRIVERNAME = os.getenv("DRIVERNAME", "sqlite")


class DB:
    def __init__(self, db_url: URL = URL.create(DRIVERNAME, database=DATABASE)) -> None:
        print(db_url)
        self._engine = create_engine(db_url)
        self.__session = None
        self._initilize_db()

    def _initilize_db(self):
        Base.metadata.create_all(self._engine)

    @property
    def _session(self) -> Session:
        if self.__session is None:
            DBSession = sessionmaker(bind=self._engine)
            self.__session = DBSession()
        return self.__session

    @_session.setter
    def _session(self, session: Session):
        self.__session = session

    def end_session(self):
        if self.__session is not None:
            self.__session.close()
            self.__session = None

    def create_soil_data(self, data: dict) -> Union[SoilData, None]:
        if not self.data_exists(data):
            soil_data = SoilData(**data)
            self._session.add(soil_data)
            self._session.commit()
            return soil_data
        return

    def retrieve_data(
        self, limit: Union[int, None] = None
    ) -> Union[List[SoilData], SoilData]:
        if limit is not None:
            return self._session.query(SoilData).limit(limit).all()
        return self._session.query(SoilData).all()

    def data_exists(self, filter: dict) -> bool:
        if "created_at" in filter:
            filter.pop("created_at")
        data_exists = self._session.query(SoilData).filter_by(**filter).count()
        return True if data_exists else False

    def retrieve_data_by_date(
        self, model, date_from, date_to=None
    ) -> Union[List[SoilData], SoilData]:
        if date_to is None:
            date_to = datetime.now(timezone.utc)
        return (
            self._session.query(model)
            .filter(model.date >= date_from, model.date <= date_to)
            .all()
        )

    def clear_data(self):
        """
        Deletes all rows from the SoilData table.
        """
        try:
            self._session.query(SoilData).delete()
            self._session.commit()
            return True
        except SQLAlchemyError as e:
            print(f"Error clearing data: {e}")
            self._session.rollback()
            return False
        finally:
            self.end_session()
