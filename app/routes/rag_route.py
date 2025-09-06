
from fastapi import APIRouter

from app.schemas.rag_schema import QueryOnlySchema, QueryWithReferenceSchema
from app.services.rag_service import query_rag_with_reference, query_rag_without_reference

router = APIRouter()

@router.get("/")
async def get_index():  
    return {"message": "Welcome to the Document Question Answering API!"}

@router.post("/query")
async def query_rag(request: QueryOnlySchema):
    return await query_rag_without_reference(request)

@router.post("/query-with-reference")
async def query_rag_reference(request: QueryWithReferenceSchema):
    return await query_rag_with_reference(request)

    

