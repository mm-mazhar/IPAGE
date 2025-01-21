from fastapi import APIRouter, HTTPException, Response, status
from fastapi.responses import JSONResponse

from src.api import utils
from src.api.db import DB
from src.api.models import SoilData
from src.api.schema import RawData

db = DB()

preFix_user = "/data"
data_router = APIRouter(prefix=preFix_user, tags=["Database"])


@data_router.get("/get-data")
def get_data(limit: int = None):
    """
    Retrieve data from the database
    EXAMPLE:
        GET /data/get-data
        GET /data/get-data?limit=10
    Args:
        limit (int): The number of records to retrieve from the database
    Response:
        JSONResponse:
            A JSON response containing the retrieved data or an error message
        HTTPException:
            An exception raised when an error occurs
    """
    try:
        if limit:
            data = db.retrieve_data(limit=limit)
        else:
            data = db.retrieve_data()
    except Exception as e:
        return HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )

    db.end_session()
    if not data:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"status": "Failed", "data": "No data found"},
        )

    result = [utils.object_to_dict(obj) for obj in data]
    return {"status": "Success", "data": result}


@data_router.post("/create-data")
def create_data(data: RawData, response: Response):
    """
    Create a new record in the database
    EXAMPLE:
        POST /data/create-data
        {
            "Area": "Area",
            "pH": "pH",
            "Nitrogen": "Nitrogen",
            "Phosphorus": "Phosphorus",
            "Potassium": "Potassium",
            "Sulfur": "Sulfur",
            "Zinc": "Zinc",
            ...
        }
    Args:
        data (RawData): The data to be stored in the database
    Response:
        fastapi.Response: The response object with status code 409 if the data already exists
        JSONResponse:
            A JSON response containing the status of the operation and the data stored
        HTTPException:
            An exception raised when an error occurs
    """
    duplicated = False
    status = "Failed"

    try:
        data = db.create_soil_data(data.model_dump())
    except Exception as e:
        return HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )

    if isinstance(data, SoilData):
        status = "Success"
        data = utils.object_to_dict(data)
    elif data is None:
        response.status_code = status.HTTP_409_CONFLICT
        duplicated = True
        status = "Data already exists in the database"

    db.end_session()
    return {"status": status, "duplicated": duplicated, "data": data}


@data_router.delete("/clear-data")
def clear_data():
    """
    Clear all the data from the database.
    """
    status = "Failed"
    if db.clear_data():
        status = "Success"
    return {"status": status}
