from app.core.logger import configure_logging
from app.processing.generate_vector_db import load_vector_store
from app.processing.generate_rag_chain import create_rag_chain

logger = configure_logging("SINGLE_QUERY_INFERENCE")

# Run inference
def run_inference(rag_chain, query: str):
    try:
        result = rag_chain.invoke({"query": query})
        return result.get("result", "").strip()
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    saved_vector_store_path = "app/data/vectorstores/faiss_index"
    vector_store = load_vector_store(saved_vector_store_path)
    rag_chain = create_rag_chain(vector_store)

    print("\nRAG Inference System Ready!")

    # query = "কোন কোম্পানিতে এখন চাকরি করে?"
    # query = "What is the current salary in business automation limited?"
    query = "What is the CGPA of the candidate?"

    answer = run_inference(rag_chain, query)
    print(f"Query: {query}\nAnswer: {answer}")
