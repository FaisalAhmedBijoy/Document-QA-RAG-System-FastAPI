
from fastapi import APIRouter, HTTPException
from app.config.configurations import Config

config = Config()

router = APIRouter()

@router.get("/")
async def get_index():  
    return {"message": "Welcome to the Bangla Speech Recognition API!"}

