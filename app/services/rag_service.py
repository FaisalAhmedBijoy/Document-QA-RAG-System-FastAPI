import os
import time
import shutil
from fastapi import HTTPException, UploadFile, File
from fastapi.responses import JSONResponse

from app.config.configuration import Config
from app.core.logger import configure_logging
from app.processing.generate_rag_chain import create_rag_chain
from app.processing.generate_vector_db import load_vector_store, create_vector_store
from app.processing.single_query_inference import run_inference
from app.processing.evaluate_rag import evaluate_rag_with_reference
from app.processing.generate_embeddings import get_embeddings
from app.processing.generate_text_chunks import generate_text_chunks_from_pdf
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

async def generate_vector_store_for_pdf(pdf_file: UploadFile,
                                        document_id: str,
                                        saved_pdf_path: str,
                                        output_text_file_path: str,
                                        saved_vector_store_path: str):
    try:
        
        # Save the uploaded PDF
        with open(saved_pdf_path, "wb") as f:
            shutil.copyfileobj(pdf_file.file, f)

        # Process the PDF into chunks
        chunks = generate_text_chunks_from_pdf(saved_pdf_path, output_text_file_path)

        # Create the FAISS vector store and save it with the document ID
        vector_store = create_vector_store(chunks, saved_vector_store_path)

        logger.info(f"PDF uploaded and processed successfully with ID: {document_id}")
        
        # Return the document ID for querying later
        return JSONResponse(content={"document_id": document_id, 
                                     "message": "PDF uploaded and vector store created."})

    except Exception as e:
        logger.error(f"Error in PDF upload and vector processing: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing the PDF: {str(e)}")
    

async def query_rag_by_document(request: QueryOnlySchema, document_id: str):
    try:
        # Load the correct vector store based on document ID
        saved_vector_store_path = f"app/data/vectorstores/faiss_index_{document_id}"
        
        if not os.path.exists(saved_vector_store_path):
            raise HTTPException(status_code=404, detail="Document ID not found.")

        vector_store = load_vector_store(saved_vector_store_path)
        rag_chain = create_rag_chain(vector_store)
        
        # Query the vector store
        answer = run_inference(rag_chain=rag_chain, query=request.query)
        response = {
            "query": request.query,
            "answer": answer,
        }
        return JSONResponse(content=response)

    except Exception as e:
        logger.error(f"Error querying document with ID {document_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing the query: {str(e)}")
    
async def get_all_vectors_list():
    try:
        if not os.path.exists(config.VECTOR_STORE_DIR):
            return JSONResponse(content={"vector_stores": [], "message": "No vector stores found."})

        # Get all folders that start with "faiss_index"
        vector_stores = [
            name for name in os.listdir(config.VECTOR_STORE_DIR)
            if os.path.isdir(os.path.join(config.VECTOR_STORE_DIR, name)) and name.startswith("faiss_index")
        ]

        # Extract document IDs from folder names
        document_ids = [name.replace("faiss_index_", "") for name in vector_stores]

        return JSONResponse(content={"vector_stores": vector_stores, "document_ids": document_ids})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)