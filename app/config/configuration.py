import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    def __init__(self):
        self.CORS_ORIGINS = self.get_required_env("CORS_ORIGINS")
        self.CHUNK_SIZE = self.get_required_env("CHUNK_SIZE")
        self.CHUNK_OVERLAP = self.get_required_env("CHUNK_OVERLAP")
        self.GROQ_API_KEY = self.get_required_env("GROQ_API_KEY")    
        
    def get_required_env(self, env_variable):
        value = os.getenv(env_variable)
        if value is None:
            error_message = f"Invalid or missing '{env_variable}' in the environment variables"
            return error_message
        return value