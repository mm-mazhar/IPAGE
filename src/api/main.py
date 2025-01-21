import os
import sys

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from typing import Any

from fastapi import APIRouter, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
from sqlalchemy.orm import class_mapper

from . import routers

# from app.config import settings, setup_app_logging
from .configs import API_PROJECT_NAME, API_VER_STR, __version__, settings

# from api.db import models
# from api.db.database import engine

# setup logging as early as possible
# setup_app_logging(config=settings)

# Prefix for all API endpoints
preFix: str = API_VER_STR
api_project_name: str = API_PROJECT_NAME

# print(f"API Project Name: {api_project_name}")
# print(f"API Version: {__version__}")
# print(f"API Prefix: {preFix}")

app = FastAPI(
    title=f"{api_project_name}",
    openapi_url=f"{preFix}/openapi.json",
    swagger_ui_parameters={"syntaxHighlight.theme": "obsidian"},
    servers=settings.SERVERS,
)
app.mount("/static", StaticFiles(directory="./src/api/static"), name="static")

# models.Base.metadata.create_all(bind=engine)

root_router = APIRouter(tags=["Root"])


@app.get("/", tags=["Root"])
def index(request: Request) -> Any:
    """Basic HTML response."""
    body: str = (
        "<html>"
        "<head>"
        "<link rel='icon' href='/static/favicon.ico' type='image/x-icon' >"
        "</head>"
        "<body style='display: flex; justify-content: center; align-items: center; height: 100vh; margin: 0; background-color: #7d8492;'>"
        "<div style='text-align: center; background-color: white; padding: 20px; border-radius: 20px;'>"
        f"<h1 style='font-weight: bold; font-family: Arial;'>{api_project_name}</h1>"
        f"<h3 style='font-weight: bold; font-family: Arial;'>Version: {__version__}</h3>"
        "<div>"
        f"<h4>Check the docs: <a href='/docs'>here</a><h4>"
        "</div>"
        "</div>"
        "</body>"
        "</html>"
    )
    return HTMLResponse(content=body)


app.include_router(routers.desc_router, prefix=preFix)
app.include_router(routers.data_router, prefix=preFix)
app.include_router(routers.train_router, prefix=preFix)
app.include_router(routers.inference_router, prefix=preFix)

# Set all CORS enabled origins
if settings.BACKEND_CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

if __name__ == "__main__":
    # Use this for debugging purposes only
    logger.warning("Running in development mode. Do not run like this in production.")
    import uvicorn

    uvicorn.run(
        "src.api.main:app",
        host="localhost",
        port=settings.FASTAPI_PORT,
        log_level="debug",
        reload=True,
    )
