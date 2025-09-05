
from fastapi import APIRouter, HTTPException
from app.schemas.schemas import QueryRequest
import os
import re
import logging
import numpy as np
from app.processing.generate_rag_chain import create_rag_chain
from app.processing.generate_vector_db import initialize_embeddings, load_vector_store
from app.config.configuration import Config
config=Config()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

router = APIRouter()

vector_store = load_vector_store(config.VECTOR_STORE_PATH)


@router.get("/")
async def get_index():  
    return {"message": "Welcome to the Document Question Answering API!"}


@router.post("/query")
async def query_rag(request: QueryRequest):
    try:
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")

        logger.info(f"Processing query: {request.query}")

        rag_chain = create_rag_chain(vector_store)
        retrieved_docs = rag_chain.retriever.invoke(request.query)
        context = [f"[Doc {i+1}]: {doc.page_content[:500]}" for i, doc in enumerate(retrieved_docs)]

        result = rag_chain.invoke({"query": request.query})
        answer = result.get("result", "").strip()

        response = {
            "query": request.query,
            "answer": answer,
            "context": context
        }

        if request.expected_answer:
            embeddings = initialize_embeddings()
            answer_embedding = embeddings.embed_query(answer if answer else "Information not found in the document")
            expected_embedding = embeddings.embed_query(request.expected_answer)
            sim = np.dot(answer_embedding, expected_embedding) / (
                np.linalg.norm(answer_embedding) * np.linalg.norm(expected_embedding)
            )
            response["expected_answer"] = request.expected_answer
            response["cosine_similarity"] = float(sim)

        return response
    except Exception as e:
        logger.error(f"Query processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

