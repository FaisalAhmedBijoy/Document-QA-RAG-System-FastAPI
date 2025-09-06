
from fastapi import HTTPException
from fastapi.responses import JSONResponse

from app.config.configuration import Config
from app.core.logger import configure_logging
from app.processing.generate_rag_chain import create_rag_chain
from app.processing.generate_vector_db import load_vector_store
from app.processing.single_query_inference import run_inference
from app.processing.evaluate_rag import evaluate_rag_with_reference
from app.processing.generate_embeddings import get_embeddings
from app.schemas.rag_schema import QueryOnlySchema, QueryWithReferenceSchema


config=Config()
logger = configure_logging("RAG_SERVICE")
embeddings = get_embeddings()
vector_store = load_vector_store(config.VECTOR_STORE_PATH)
rag_chain = create_rag_chain(vector_store)

async def query_rag_without_reference(request: QueryOnlySchema):
    try:
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")

        logger.info(f"Processing query: {request.query}")

        answer = run_inference(rag_chain=rag_chain, query=request.query)
        response = {
            "query": request.query,
            "answer": answer,
        }
        return JSONResponse(content=response)
    except Exception as e:
        logger.error(f"Query processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def query_rag_with_reference(request: QueryWithReferenceSchema):
    try:
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")

        logger.info(f"Processing query: {request.query}")
        
        if request.expected_answer:
            result=evaluate_rag_with_reference(query=request.query, 
                                               expected_answer=request.expected_answer, 
                                               rag_chain=rag_chain, 
                                               embeddings=embeddings)
            response = {
                "query": result.get("query"),
                "expected_answer": result.get("expected_answer"),
                "actual": result.get("actual"),
                "cosine_similarity": result.get("cosine_similarity"),
                "context": result.get("context"),
            }
            return JSONResponse(content=response)
        else:
            answer = run_inference(rag_chain, request.query)
            response = {
                "query": request.query,
                "expected_answer": "N/A",
                "actual": answer,
                "cosine_similarity": "N/A",
                "context": ["N/A"],
            }
            return JSONResponse(content=response)
    except Exception as e:
        logger.error(f"Query processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))