# Pydantic model for request body
from pydantic import BaseModel

class QueryRequest(BaseModel):
    query: str
    expected_answer: str | None = None