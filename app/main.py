import uvicorn
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes.bangla_rag_route import router as rag_router

# Initialize FastAPI app
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Include the router
app.include_router(rag_router, prefix="/rag/en")

logger.info("Routers included successfully")

@app.get("/")
async def get_index():
    logger.info("Hello route triggered")
    return {"message": "Hello route triggered!"}


# Entry point for running the application
def run():
    uvicorn.run("app.main:app", host="0.0.0.0", port=6038, reload=True, log_level="info")


if __name__ == "__main__":
    run()
