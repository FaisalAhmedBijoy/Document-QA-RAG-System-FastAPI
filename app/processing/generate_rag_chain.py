from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from tenacity import retry, stop_after_attempt, wait_fixed

from app.config.configuration import Config
from app.core.logger import configure_logging
from app.processing.generate_vector_db import load_vector_store

config=Config()
logger = configure_logging("GENERATE_RAG_CHAIN")

# Initialize Groq LLM with rate limit handling
@retry(stop=stop_after_attempt(5), wait=wait_fixed(3))
def initialize_llm():
    try:
        if not config.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        logger.info("Initializing Groq LLM")
        return ChatGroq(
            groq_api_key=config.GROQ_API_KEY,
            model_name=config.LLM_MODEL,
            temperature=0.7,
            max_tokens=512,
        )
    except Exception as e:
        logger.error(f"LLM init error: {e}")
        raise

# Create RAG chain
def create_rag_chain(vector_store):
    try:
        logger.info("Creating RAG chain")
        llm = initialize_llm()
        prompt_template = """
        You are an assistant that answers questions strictly based on the provided document text.

        Rules:
        - Only use the information from the given Context.
        - Do not use outside knowledge.
        - If the answer is not found in the Context, reply exactly: "Information not found in the document."
        - Provide only the answer, without repeating the question or the context.

        Context: {context}
        Question: {question}
        Answer:
        """
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5}),
            chain_type_kwargs={"prompt": PROMPT}
        )
        logger.info("RAG chain created successfully")
        return chain
    except Exception as e:
        logger.error(f"RAG chain error: {e}")
        raise

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
    rag_chain = create_rag_chain(vector_store)
    logger.info("RAG chain is ready for inference")