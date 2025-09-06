import numpy as np

from app.core.logger import configure_logging
from app.processing.generate_vector_db import load_vector_store
from app.processing.generate_rag_chain import create_rag_chain
from app.processing.generate_embeddings import get_embeddings

logger = configure_logging("EVALUATE_RAG")

# Evaluate RAG system
def evaluate_rag_with_reference(query: str, expected_answer: str, rag_chain, embeddings):
    try:
        logger.info(f"Evaluating query: {query}")
        # retrieved_docs = rag_chain.retriever.invoke(query)
        result = rag_chain.invoke({"query": query})
        answer = result.get("result", "").strip()

        retrieved_docs = rag_chain.retriever.invoke(query)
        context = [f"[Doc {i+1}]: {doc.page_content[:500]}" for i, doc in enumerate(retrieved_docs)]

        answer_embedding = embeddings.embed_query(answer if answer else "Information not found in the document.")
        expected_embedding = embeddings.embed_query(expected_answer)
        sim = np.dot(answer_embedding, expected_embedding) / (
            np.linalg.norm(answer_embedding) * np.linalg.norm(expected_embedding)
        )
        return {
            "query": query,
            "expected_answer": expected_answer,
            "actual": answer,
            "cosine_similarity": float(sim),
            "context": context
        }
    except Exception as e:
        logger.error(f"Evaluation error: {e}")
        return {"error": str(e)}

if __name__=='__main__':
  
    saved_vector_store_path = "app/data/vectorstores/faiss_index"

    # To load the vector store later
    vector_store = load_vector_store(saved_vector_store_path)
    rag_chain = create_rag_chain(vector_store)
    embeddings = get_embeddings()
    logger.info("RAG chain is ready for inference")

    sample_queries = [
            {"query": "Give me the email address", "expected_answer": "faisal.cse16.kuet@gmail.com"},
            {"query": "What is the CGPA of the candidate?", "expected_answer": "3.41"},
            # {"query": "Where did he complete his undergraduate studies?", "expected_answer": "Khulna University of Engineering & Technology"},
            # {"query": "What is the current salary in business automation limited", "expected_answer": "25,000 Taka"},
            # {"query": "কোন কোম্পানিতে এখন চাকরি করে?", "expected_answer": "বিজনেস অটোমেশন লিমিটেড"},
        ]

    for sample_query in sample_queries:
        result = evaluate_rag_with_reference(sample_query["query"], sample_query["expected_answer"], rag_chain, embeddings)
        print("--------------------------------------------------")
        print('Query:', sample_query['query'])
        print('Expected Answer:', sample_query['expected_answer'])
        print('Actual:', result.get('actual', 'N/A'))
        print('Cosine Similarity:', result.get('cosine_similarity', 'N/A'))
        print('Context:', '\n'.join(result.get('context', [])))
    
    logger.info("Evaluation completed")