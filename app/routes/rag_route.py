
import time
from fastapi import APIRouter
from fastapi import APIRouter, UploadFile, File, HTTPException

from app.schemas.rag_schema import QueryOnlySchema, QueryWithReferenceSchema
from app.services.rag_service import query_rag_with_reference, query_rag_without_reference, generate_vector_store_for_pdf, query_rag_by_document, get_all_vectors_list

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

@router.post("/upload-document-pdf")
async def upload_pdf(pdf_file: UploadFile = File(...)):
    
    # Generate a unique ID based on current time
    document_id = str(int(time.time() * 1000))  # Timestamp in milliseconds
    saved_pdf_path = f"app/data/pdfs/{document_id}.pdf"
    output_text_file_path = f"app/data/texts/{document_id}.txt"
    saved_vector_store_path = f"app/data/vectorstores/faiss_index_{document_id}"

    return await generate_vector_store_for_pdf(pdf_file=pdf_file,
                                               document_id=document_id, 
                                               saved_pdf_path=saved_pdf_path, 
                                               output_text_file_path=output_text_file_path, 
                                               saved_vector_store_path=saved_vector_store_path)

@router.post("/query-by-document")
async def query_by_document(request: QueryOnlySchema, document_id: str):
    return await query_rag_by_document(request, document_id)
        
@router.get("/list-vector-stores")
async def list_vector_stores():
    """
    List all saved FAISS vector store directories/files.
    Returns the document IDs extracted from the folder names.
    """
    return await get_all_vectors_list()
   