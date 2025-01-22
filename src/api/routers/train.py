import os

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from fastapi.responses import FileResponse
from sklearn.linear_model import Ridge

from src.api.schema import TargetSelect
from src.model.model import BaseModel, DataPreprocessor

preFix_user = "/train"
train_router = APIRouter(prefix=preFix_user, tags=["Model Training"])


@train_router.post("/model-train")
def train_model(target: TargetSelect):
    """
    Training the model on combined isda and ipage data
    """
    data = DataPreprocessor("merged_v3.csv", "data").preprocess()
    regression_model = Ridge
    model = BaseModel([target], regression_model)
    model.train(data)
    result = model.evaluate()
    model_name = f"{target}_{type(regression_model()).__name__}"
    print(model_name)
    model.save_model(model_name)
    return {"status": "Success", "metrics": result}


@train_router.post("/modeL-retrain/upload")
async def train_model_with_uploaded_data(file: UploadFile = File(...)):
    """
    Train a model with new data. The csv file must contain the columns: Area, pH, Nitrogen, Phosphorus,
    Sulfur, Sand, Silt, and Clay
    """
    # Define the file path where the uploaded file will be saved
    filepath = os.path.join(os.getcwd(), "data", file.filename)
    print(f"FilePath For Uploaded File: {filepath}")

    # Ensure the directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Write the uploaded file to the defined filepath
    try:
        with open(filepath, "wb") as buffer:
            buffer.write(await file.read())
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"File upload failed: {str(e)}",
        )

    # Retraining pipeline
    data = DataPreprocessor(file.filename, "data").preprocess()
    regression_model = Ridge
    targets = ["SOC", "Boron", "Zinc"]
    model = BaseModel(targets, regression_model)
    model.train(data)
    result = model.evaluate()
    for target in targets:
        model_name = f"retrain_{target}_{type(regression_model()).__name__}"
        print(model_name)
        model.save_model(model_name)

    # Return JSON response
    return {"status": "Success", "metrics": result}


# @train_router.post("/modeL-retrain/upload", response_class=FileResponse)
# async def train_model_with_uploaded_data(file: UploadFile = File(...)):
#     """
#     Train a model with new data. The csv file must contain the columns: Area, pH, Nitrogen, Phosphorus,
#     Sulfur, Sand, Silt, and Clay
#     """
#     # Define the file path where the uploaded file will be saved
#     filepath = os.path.join(os.getcwd(), "data", file.filename)
#     print(f"FilePath For Uploaded File: {filepath}")

#     # Ensure the directory exists
#     os.makedirs(os.path.dirname(filepath), exist_ok=True)

#     # Write the uploaded file to the defined filepath
#     with open(filepath, "wb") as buffer:
#         buffer.write(await file.read())

#     # Retraining pipeline
#     data = DataPreprocessor(file.filename, "data").preprocess()
#     regression_model = Ridge
#     targets = ["SOC", "Boron", "Zinc"]
#     model = BaseModel(targets, regression_model)
#     model.train(data)
#     result = model.evaluate()
#     for target in targets:
#         model_name = f"retrain_{target}_{type(regression_model()).__name__}"
#         print(model_name)
#         model.save_model(model_name)
#     return {"status": "Success", "metrics": result}
