import logging
import numpy as np
from app.config.configuration import Config
from app.processing.generate_vector_db import initialize_embeddings, load_vector_store
from app.processing.generate_rag_chain import create_rag_chain

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Evaluate RAG system
def evaluate_rag(query: str, expected: str, rag_chain, embeddings):
    try:
        logger.info(f"Evaluating query: {query}")
        # retrieved_docs = rag_chain.retriever.invoke(query)
        result = rag_chain.invoke({"query": query})
        answer = result.get("result", "").strip()

        answer_embedding = embeddings.embed_query(answer if answer else "Information not found in the document.")
        expected_embedding = embeddings.embed_query(expected)
        sim = np.dot(answer_embedding, expected_embedding) / (
            np.linalg.norm(answer_embedding) * np.linalg.norm(expected_embedding)
        )
        return {
            "query": query,
            "expected": expected,
            "actual": answer,
            "cosine_similarity": float(sim)
        }
    except Exception as e:
        logger.error(f"Evaluation error: {e}")
        return {"error": str(e)}

if __name__=='__main__':
  
    saved_vector_store_path = "app/data/vectorstores/faiss_index"

    # To load the vector store later
    vector_store = load_vector_store(saved_vector_store_path)
    rag_chain = create_rag_chain(vector_store)
    embeddings = initialize_embeddings()
    logger.info("RAG chain is ready for inference")

    sample_queries = [
            {"query": "Give me the email address", "expected": "faisal.cse16.kuet@gmail.com"},
            {"query": "What is the CGPA of the candidate?", "expected": "3.41"},
            {"query": "Where did he complete his undergraduate studies?", "expected": "Khulna University of Engineering & Technology"},
            {"query": "What is the current salary in business automation limited", "expected": "25,000 Taka"},
            {"query": "কোন কোম্পানিতে এখন চাকরি করে?", "expected": "বিজনেস অটোমেশন লিমিটেড"},
        ]

    for sample_query in sample_queries:
        result = evaluate_rag(sample_query["query"], sample_query["expected"], rag_chain, embeddings)
        print("--------------------------------------------------")
        print('Query:', sample_query['query'])
        print('Expected:', sample_query['expected'])
        print('Actual:', result.get('actual', 'N/A'))
        print('Cosine Similarity:', result.get('cosine_similarity', 'N/A'))
    
    logger.info("Evaluation completed")