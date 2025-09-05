import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    def __init__(self):
        self.CORS_ORIGINS = self.get_required_env("CORS_ORIGINS")
        self.HUGGINGFACE_EMBEDDING_MODEL = self.get_required_env("HUGGINGFACE_EMBEDDING_MODEL")
        self.GROQ_API_KEY = self.get_required_env("GROQ_API_KEY")
        self.LLM_MODEL = self.get_required_env("LLM_MODEL")
        self.VECTOR_STORE_PATH = self.get_required_env("VECTOR_STORE_PATH")    
        self.CHUNK_OVERLAP = int(self.get_required_env("CHUNK_OVERLAP"))
        self.CHUNK_SIZE = int(self.get_required_env("CHUNK_SIZE"))
        
    def get_required_env(self, env_variable):
        value = os.getenv(env_variable)
        if value is None:
            error_message = f"Invalid or missing '{env_variable}' in the environment variables"
            return error_message
        return value