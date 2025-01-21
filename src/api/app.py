from fastapi import FastAPI, File, UploadFile, Query, status, Response, HTTPException
from typing import Literal, Union,List
from src.api.db import DB
from src.api.models import SoilData
from src.api.schema import RawData, PredictionInput, PredictionResponse,TargetSelect
from sqlalchemy.orm import class_mapper
from fastapi.responses import FileResponse, JSONResponse
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
    """Root endpoint"""
    return {"message": "Welcome to IPAGE API"}

@app.get("/data")
def get_data(limit: int = None):
    """
    Retrieve data from the database

    EXAMPLE:
        GET /data
        GET /data?limit=10
    
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
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

    db.end_session()
    if not data:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"status": "Failed", "data": "No data found"}
        )

    result = [object_to_dict(obj) for obj in data]
    return {"status": "Success", "data": result}

@app.post("/data", status_code=status.HTTP_201_CREATED)
def create_data(data: RawData, response: Response):
    """
    Create a new record in the database

    EXAMPLE:
        POST /data
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
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

    if isinstance(data, SoilData):
        status = "Success"
        data = object_to_dict(data)
    elif data is None:
        response.status_code = status.HTTP_409_CONFLICT
        duplicated = True
        status = "Data already exists in the database"

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

@app.post("/retrain/upload")
async def train_model_with_uploaded_data(file: UploadFile = File(...)):
    '''
    Train a model with new data. The csv file must contain the columns: Area, pH, Nitrogen, Phosphorus,
    Sulfur, Sand, Silt, and Clay
    '''
    # Define the file path where the uploaded file will be saved
    filepath = os.path.join(os.getcwd(), "data", file.filename)

    # Ensure the directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Write the uploaded file to the defined filepath
    with open(filepath, "wb") as buffer:
        buffer.write(await file.read())
    
    # Retraining pipeline
    data = DataPreprocessor(file.filename, "data").preprocess()
    regression_model = Ridge
    targets = ['SOC','Boron','Zinc']
    model = BaseModel(targets,regression_model)
    model.train(data)
    result = model.evaluate()
    for target in targets:
        model_name = f'retrain_{target}_{type(regression_model()).__name__}'
        print(model_name)
        model.save_model(model_name)
    return {"status": "Success", "metrics": result}

    # Ensure the directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Write the uploaded file to the defined filepath
    with open(filepath, "wb") as buffer:
        buffer.write(await file.read())
    
    # Retraining pipeline
    data = DataPreprocessor(file.filename, "data").preprocess()
    regression_model = Ridge
    targets = ['SOC','Boron','Zinc']
    model = BaseModel(targets,regression_model)
    model.train(data)
    result = model.evaluate()
    for target in targets:
        model_name = f'retrain_{target}_{type(regression_model()).__name__}'
        print(model_name)
        model.save_model(model_name)
    return {"status": "Success", "metrics": result}


@app.post("/inference/point/")
def predict(data:PredictionInput,targets:List[TargetSelect] = Query(...)):
    df = pd.DataFrame(data.dict(),index=[0])
    pred_dict = {}
    for target in targets:
        model_path = MODEL_FILE_PATH.joinpath(f'{target.name}_Ridge')
        model = joblib.load(model_path)
        pred_dict[target.name] = np.round(model.predict(df)[0],3)
        
    prediction = PredictionResponse(**pred_dict)
    return {"prediction": prediction}

@app.post("/inference/batch/", response_class=FileResponse)
async def upload_prediction_data(targets:List[TargetSelect] = Query(...),file: UploadFile = File(...)):
    # Define the file path where the uploaded file will be saved
    filepath = os.path.join(os.getcwd(), "data", file.filename)

    # Ensure the directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Write the uploaded file to the defined filepath
    with open(filepath, "wb") as buffer:
        buffer.write(await file.read())
    # Read the saved file into a DataFrame
    features = pd.read_csv(filepath)
    pred_df = pd.DataFrame(columns=[target.name for target in targets])
    for target in targets:
        model_path = MODEL_FILE_PATH.joinpath(f'{target.name}_Ridge')
        model = BaseModel.load_model(model_path)
    
        pred_df[target.name] = model.predict(features)[:,0]

    prediction_path = MODEL_FILE_PATH.parent.joinpath('predictions.csv')
    pred_df.to_csv(prediction_path,index=False)
    
    return prediction_path
