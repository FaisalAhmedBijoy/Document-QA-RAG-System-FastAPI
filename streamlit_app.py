import streamlit_app as st

from app.config.configuration import Config
from app.core.logger import configure_logging
from app.processing.generate_vector_db import load_vector_store
from app.processing.generate_rag_chain import create_rag_chain
from app.processing.single_query_inference import run_inference as rag_run_inference

config = Config()
logger = configure_logging("STREAMLIT_APP")

def main():
    st.set_page_config(page_title="Document Q&A System", page_icon="📚", layout="wide")

    # Sidebar: chat history
    with st.sidebar:
        st.header("Chat History")
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        if st.session_state.chat_history:
            for i, entry in enumerate(reversed(st.session_state.chat_history)):
                with st.expander(f"Query {len(st.session_state.chat_history) - i}: {entry['query'][:50]}..."):
                    st.write(f"**Query**: {entry['query']}")
                    st.write(f"**Answer**: {entry['answer']}")
        else:
            st.write("No queries yet.")

    # Main content
    st.title("📚 Document Q&A System")
    st.markdown("Ask questions in Bengali or English about the document, and the system will provide answers based on the context.")

    # Initialize RAG chain once
    if "rag_chain" not in st.session_state:
        try:
            vector_store = load_vector_store(config.VECTOR_STORE_PATH)
            st.session_state.rag_chain = create_rag_chain(vector_store)
            st.success("RAG system initialized successfully!")
        except Exception as e:
            st.error(f"Failed to initialize RAG system: {str(e)}")
            logger.error(f"Failed to initialize RAG system: {str(e)}")
            return

    # Query input
    query = st.text_input(
        "Enter your question in Bengali or English:",
        placeholder="Example: What is the email address of the candiate?"
    )

    if st.button("Submit Query"):
        if not query.strip():
            st.warning("Please enter a query.")
            return

        with st.spinner("Processing your query..."):
            try:
                # ✅ Correct order & using the imported function
                answer_text = rag_run_inference(
                    rag_chain=st.session_state.rag_chain,
                    query=query
                )

                st.subheader("Result")
                st.write(f"**Query**: {query}")
                st.write(f"**Answer**: {answer_text}")

                # Save to history
                st.session_state.chat_history.append({
                    "query": query,
                    "answer": answer_text
                })

            except Exception as e:
                st.error(f"Error processing query: {str(e)}")
                logger.error(f"Error processing query: {str(e)}")

if __name__ == "__main__":
    main()
