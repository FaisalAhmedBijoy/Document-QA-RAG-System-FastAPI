import os
import logging
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from app.config.configuration import Config
from app.processing.generate_text_chunks import generate_text_chunks_from_pdf

config=Config()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# HuggingFace embeddings for Bengali
def initialize_embeddings():
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=config.HUGGINGFACE_EMBEDDING_MODEL,
            encode_kwargs={'normalize_embeddings': True}
        )
        _ = embeddings.embed_query("test")
        logger.info("HuggingFace embeddings initialized successfully")
        return embeddings
    except Exception as e:
        logger.error(f"Embedding init error: {e}")
        raise

# Create FAISS vector store
def create_vector_store(docs, saved_vector_store_path):
    try:
        logger.info("Creating FAISS vector store")
        embeddings = initialize_embeddings()
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
    embeddings = initialize_embeddings()
    return FAISS.load_local(saved_vector_store_path, embeddings, allow_dangerous_deserialization=True)

if __name__=='__main__':
  
    pdf_path = "app/data/pdfs/CV.pdf"
    output_text_file_path = "app/data/texts/CV.txt"
    saved_vector_store_path = "app/data/vectorstores/faiss_index"

    # chunks=generate_text_chunks_from_pdf(pdf_path,output_text_file_path)
    # vector_store=create_vector_store(chunks, saved_vector_store_path)
    # logger.info(f"Total documents loaded and preprocessed: {len(chunks)}")
    # logger.info(f"Vector store info: {vector_store}")

    # To load the vector store later
    vector_store = load_vector_store(saved_vector_store_path)
    logger.info(f"Loaded vector store info: {vector_store}")