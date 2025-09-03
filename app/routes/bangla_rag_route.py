import io
import jwt  
import json
import torch
import ffmpeg
import base64


from app.config.configurations import Config
from fastapi import APIRouter, HTTPException

config = Config()

router = APIRouter()

@router.get("/")
async def get_index():  
    return {"message": "Welcome to the Bangla Speech Recognition API!"}

