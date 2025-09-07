import os
import time
from langchain_community.vectorstores import FAISS

from app.config.configuration import Config
from app.core.logger import configure_logging
from app.processing.generate_embeddings import get_embeddings
from app.processing.generate_text_chunks import generate_text_chunks_from_pdf

config=Config()
logger = configure_logging("GENERATE_VECTOR_DB")

# Create FAISS vector store
def create_vector_store(docs, saved_vector_store_path):
    try:
        logger.info("Creating FAISS vector store")
        embeddings = get_embeddings()
        vector_store = FAISS.from_documents(docs, embeddings)
        vector_store.save_local(saved_vector_store_path)
        logger.info("FAISS vector store created and saved")
        return vector_store
    except Exception as e:
        logger.error(f"Vector store error: {e}")
        raise

# Load FAISS vector store
def load_vector_store(saved_vector_store_path):
    if not os.path.exists(saved_vector_store_path):
        raise FileNotFoundError(f"FAISS index not found at: {saved_vector_store_path}")
    embeddings = get_embeddings()
    return FAISS.load_local(saved_vector_store_path, embeddings, allow_dangerous_deserialization=True)

if __name__=='__main__':
  
    pdf_path = "app/data/pdfs/CV.pdf"
    output_text_file_path = "app/data/texts/CV.txt"
    # saved_vector_store_path = "app/data/vectorstores/faiss_index"
    # Generate a unique ID based on current time
    document_id = str(int(time.time() * 1000))  # Timestamp in milliseconds
    saved_vector_store_path = f"app/data/vectorstores/faiss_index_{document_id}"

    chunks=generate_text_chunks_from_pdf(pdf_path,output_text_file_path)
    vector_store=create_vector_store(chunks, saved_vector_store_path)
    logger.info(f"Total documents loaded and preprocessed: {len(chunks)}")
    logger.info(f"Vector store info: {vector_store}")

    # To load the vector store later
    vector_store = load_vector_store(saved_vector_store_path)
    logger.info(f"Loaded vector store info: {vector_store}")