import os
import re
import logging
import numpy as np
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from app.config.configuration import Config
from app.processing.pdf_to_text import pdf2text_pdfplumber

config=Config()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Load and preprocess PDF with pdf2text, with caching
def generate_text_chunks_from_pdf(pdf_path: str, cache_path: str):
    try:
        # Check if text file exists
        if os.path.exists(cache_path):
            logger.info(f"Loading text from: {cache_path}")
            with open(cache_path, 'r', encoding='utf-8') as f:
                text = f.read()
        else:
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"PDF not found: {pdf_path}")

            logger.info(f"Extracting text from PDF: {pdf_path}")
            # Extract text using pdf2text
            text = pdf2text_pdfplumber(pdf_path,cache_path)
            if not text or not text.strip():
                raise ValueError("No text extracted from PDF with pdf2text")
            logger.info(f"Extracted text sample: {text[:500]}")
            # Save extracted text to txt file
            with open(cache_path, 'w', encoding='utf-8') as f:
                f.write(text)
            logger.info(f"Saved extracted text to: {cache_path}")

        # Clean the extracted text
        cleaned_text = re.sub(r'\s+', ' ', text).strip()
        cleaned_text = cleaned_text.encode('utf-8', errors='replace').decode('utf-8')
        # Create a single Document object
        data = [Document(page_content=cleaned_text, metadata={"source": pdf_path})]

        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", "।", " ", ""],
        )
        chunks = text_splitter.split_documents(data)
        logger.info(f"Created {len(chunks)} chunks")
        for i, chunk in enumerate(chunks):
            logger.info(f"Chunk {i+1}: {chunk.page_content[:200]}")
        return chunks
    except Exception as e:
        logger.error(f"Error loading PDF or cache: {e}")
        raise

if __name__=='__main__':
 
    pdf_path = "app/data/pdfs/CV.pdf"
    output_text_file_path = "app/data/texts/CV.txt"
    chunks=generate_text_chunks_from_pdf(pdf_path,output_text_file_path)
    logger.info(f"Total documents loaded and preprocessed: {len(chunks)}")
    
