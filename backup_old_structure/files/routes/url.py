from fastapi import FastAPI, APIRouter, Depends
from helpers.config import get_settings, Settings

url_router = APIRouter(
    prefix="/api/v1",
    tags=["url"]
)

@url_router.post("/url")
async def upload_url(settings: Settings = Depends(get_settings)):
    return {"youtube_url": settings.youtube_url}
