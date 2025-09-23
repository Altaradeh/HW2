import streamlit as st
import os
import shelve
from dotenv import load_dotenv
from langchain.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
import chromadb

# --- 1. Environment and Configuration Setup ---
# Load environment variables from .env file
load_dotenv()

# Set up the OpenAI API key
if "OPENAI_API_KEY" not in os.environ or not os.environ["OPENAI_API_KEY"]:
    st.error("OpenAI API key is not set. Please add it to your .env file.")
    st.stop()

# Define paths for the PDF and the Chroma DB persistence
PDF_PATH = "data/document.pdf"
CHROMA_DB_PATH = "chroma_db"
CACHE_DB_PATH = "cache_db"

# Ensure the data directory exists
if not os.path.exists("data"):
    os.makedirs("data")

# --- 2. Data Loading and Processing ---
def load_and_process_pdf():
    """
    Loads the PDF, splits it into chunks, and creates a vector store using ChromaDB.
    This process is performed only once and persisted to disk.
    """
    st.info("Reading and processing the PDF file. This may take a few moments...")
    
    # Use PyPDFLoader to load the document
    loader = PyPDFLoader(PDF_PATH)
    pages = loader.load()

    # Split the document into chunks for better retrieval
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
        length_function=len
    )
    docs = text_splitter.split_documents(pages)

    # Use OpenAI's embedding model
    embedding_model = OpenAIEmbeddings()

    # Create the vector store and persist it
    vector_store = Chroma.from_documents(
        docs,
        embedding_model,
        persist_directory=CHROMA_DB_PATH
    )
    vector_store.persist()
    st.success("PDF processed and vector store created successfully!")

    return vector_store

# --- 3. Streamlit UI and Logic ---
def main():
    """
    The main Streamlit application logic.
    """
    st.set_page_config(page_title="PDF RAG POC", layout="wide")
    st.title("📄 PDF-Based Retrieval-Augmented Generation (RAG)")
    st.write("Ask a question about the document and get an answer from the PDF only.")
    
    # Check if the Chroma DB is already persisted. If not, create it.
    if not os.path.exists(CHROMA_DB_PATH):
        if not os.path.exists(PDF_PATH):
            st.error(f"PDF file not found at: {PDF_PATH}")
            st.stop()
        
        # This will run only on the first execution or if the DB folder is deleted
        with st.spinner("Preparing the knowledge base from the PDF..."):
            vector_store = load_and_process_pdf()
    else:
        # Load the existing vector store
        st.info("Loading existing knowledge base from disk...")
        embedding_model = OpenAIEmbeddings()
        vector_store = Chroma(
            persist_directory=CHROMA_DB_PATH,
            embedding_function=embedding_model
        )
        st.success("Knowledge base loaded!")

    # Setup the retrieval chain
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    # Custom prompt to enforce the "PDF only" rule
    prompt_template = """
    You are an AI assistant tasked with answering questions based **ONLY** on the context provided below.
    If the context does not contain the answer, you **MUST** respond with "I cannot find the answer to your request in the document."
    Do not use any external knowledge.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    qa_prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    # Use RetrievalQA to create the chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs={"prompt": qa_prompt}
    )

    # --- 4. File-based Caching with `shelve` ---
    def get_from_cache(question):
        """Retrieves a response from the file-based cache."""
        with shelve.open(CACHE_DB_PATH) as cache:
            return cache.get(question)

    def add_to_cache(question, answer):
        """Adds a new question-answer pair to the file-based cache."""
        with shelve.open(CACHE_DB_PATH) as cache:
            cache[question] = answer

    # --- 5. User Interaction ---
    user_question = st.text_input("Enter your question:", "")

    if st.button("Get Answer"):
        if not user_question:
            st.warning("Please enter a question.")
            return

        # Check cache before querying the LLM
        cached_answer = get_from_cache(user_question)
        if cached_answer:
            st.info("Answer retrieved from cache.")
            st.markdown(f"**Answer:**\n\n{cached_answer}")
            return

        # If not in cache, query the RAG chain
        with st.spinner("Finding the answer..."):
            try:
                response = qa_chain({"query": user_question})
                answer = response["result"]

                # Cache the new answer
                add_to_cache(user_question, answer)

                st.markdown(f"**Answer:**\n\n{answer}")

                # Optional: display the source document snippets
                # st.subheader("Source Documents")
                # for i, doc in enumerate(response["source_documents"]):
                #     st.write(f"**Snippet from Page {doc.metadata.get('page', 'N/A')}:**")
                #     st.code(doc.page_content[:300] + "...") # Show a truncated snippet

            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.info("Please check your OpenAI API key and try again.")
    
    st.markdown("---")
    st.markdown("This application uses a file-based cache. Your questions and the corresponding answers are stored in the `cache_db` file to avoid re-running the LLM for the same query.")
    st.markdown("The PDF embeddings are stored in the `chroma_db` directory for persistence.")

if __name__ == "__main__":
    main()
