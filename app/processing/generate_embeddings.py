
from langchain_huggingface import HuggingFaceEmbeddings

from app.config.configuration import Config
from app.core.logger import configure_logging

config=Config()
logger = configure_logging("GENERATE_EMBEDDINGS")

# Initialize HuggingFace embeddings for Bengali
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=config.HUGGINGFACE_EMBEDDING_MODEL,
        encode_kwargs={'normalize_embeddings': True}
    )

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

if __name__=='__main__':
  
    embeddings = initialize_embeddings()
    embeddings = get_embeddings()
    logger.info(f"Sample embedding: {embeddings.embed_query('This is a test sentence.')}")
    logger.info(f"Sample embedding: {embeddings.embed_query('আজকে আমার মন ভালো নেই।')}")
