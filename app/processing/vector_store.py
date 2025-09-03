import os
import logging
from app.processing.pdf_to_text import pdf2text
import os
import re
import logging
import numpy as np
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from bangla_pdf_ocr import process_pdf
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from app.config.configuration import Config

config=Config()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def load_and_preprocess_pdf(pdf_dirpath: str, cache_path: str):
    try:
        # Check if text file exists
        if os.path.exists(cache_path):
            logger.info(f"Loading text from: {cache_path}")
            with open(cache_path, 'r', encoding='utf-8') as f:
                text = f.read()
        else:
            if not os.path.exists(pdf_dirpath):
                raise FileNotFoundError(f"PDF not found: {pdf_dirpath}")

            logger.info(f"Extracting text from PDF: {pdf_dirpath}")
            # Extract text using bangla_pdf_ocr
            text = pdf2text(pdf_dirpath)
            if not text or not text.strip():
                raise ValueError("No text extracted from PDF with bangla_pdf_ocr")
            logger.info(f"Extracted text sample: {text[:500]}")
            # Save extracted text to txt file
            with open(cache_path, 'w', encoding='utf-8') as f:
                f.write(text)
            logger.info(f"Saved extracted text to: {cache_path}")

        # Clean the extracted text
        cleaned_text = re.sub(r'\s+', ' ', text).strip()
        cleaned_text = cleaned_text.encode('utf-8', errors='replace').decode('utf-8')
        # Create a single Document object
        data = [Document(page_content=cleaned_text, metadata={"source": pdf_dirpath})]

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
 
    pdf_dirpath = "data/pdfs"
    output_text_file_path = "data/texts/prospectus.txt"
    docs=load_and_preprocess_pdf(pdf_dirpath,output_text_file_path)
    logger.info(f"Total documents loaded and preprocessed: {len(docs)}")
    
