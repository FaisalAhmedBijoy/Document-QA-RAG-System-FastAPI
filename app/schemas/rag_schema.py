from pydantic import BaseModel

# Schema without expected answer
class QueryOnlySchema(BaseModel):
    query: str

# Schema with expected answer (optional field)
class QueryWithReferenceSchema(BaseModel):
    query: str
    expected_answer: str | None = None
