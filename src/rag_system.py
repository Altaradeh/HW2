import openai
import faiss
import numpy as np
import streamlit as st
import os
import pickle
import tempfile
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from typing import List, Dict, Any
import tiktoken

# --- Load environment variables ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

# --- Constants ---
FAISS_INDEX_PATH = "faiss_db/index.faiss"
FAISS_META_PATH = "faiss_db/meta.pkl"
CHUNK_SIZE = 512  # tokens
CHUNK_OVERLAP = 64  # tokens

# --- Utility Functions ---

def ensure_dirs():
    os.makedirs("faiss_db", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

def pdf_to_text_chunks(pdf_file, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    reader = PdfReader(pdf_file)
    all_chunks = []
    for page_num, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        tokens = tiktoken.encoding_for_model("text-embedding-ada-002").encode(text)
        start = 0
        while start < len(tokens):
            end = min(start + chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = tiktoken.encoding_for_model("text-embedding-ada-002").decode(chunk_tokens)
            all_chunks.append({
                "text": chunk_text,
                "page": page_num + 1,
                "chunk_id": f"{page_num+1}-{start}-{end}"
            })
            start += chunk_size - overlap
    return all_chunks

def embed_texts(texts, model="text-embedding-3-small"):
    response = openai.embeddings.create(input=texts, model=model)
    return np.array([r.embedding for r in response.data]).astype("float32")

class FAISSDB:
    def __init__(self, dim, db_path="faiss_db"):
        self.index_path = os.path.join(db_path, "faiss.index")
        self.meta_path = os.path.join(db_path, "meta.pkl")
        os.makedirs(db_path, exist_ok=True)
        self.dim = dim
        self._load()

    def _load(self):
        if os.path.exists(self.index_path) and os.path.exists(self.meta_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.meta_path, "rb") as f:
                self.meta = pickle.load(f)
        else:
            self.index = faiss.IndexFlatL2(self.dim)
            self.meta = []

    def add(self, texts, metadatas, model="text-embedding-3-small"):
        embeds = embed_texts(texts, model)
        self.index.add(embeds)
        self.meta.extend(metadatas)
        self.save()

    def search(self, query, k=5, model="text-embedding-3-small"):
        q_emb = embed_texts([query], model)
        D, I = self.index.search(q_emb, k)
        return [self.meta[i] for i in I[0] if i < len(self.meta)]

    def save(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "wb") as f:
            pickle.dump(self.meta, f)

# --- OpenAI Agent/Assistant setup ---
def get_or_create_agent():
    agents = openai.beta.assistants.list()
    for agent in agents.data:
        if agent.name == "RAG Agent":
            return agent
    return openai.beta.assistants.create(
        name="RAG Agent",
        instructions="You are a helpful assistant that answers questions using a search_documents function.",
        model="gpt-4o",
        tools=[{
            "type": "function",
            "function": {
                "name": "search_documents",
                "description": "Search the vector database for relevant context.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"}
                    },
                    "required": ["query"]
                }
            }
        }]
    )

def run_agent(agent, user_query, search_fn):
    thread = openai.beta.threads.create()
    openai.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=user_query
    )
    run = openai.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=agent.id
    )
    while run.status in ["queued", "in_progress", "requires_action"]:
        if run.status == "requires_action":
            for tool_call in run.required_action.submit_tool_outputs.tool_calls:
                if tool_call.function.name == "search_documents":
                    args = eval(tool_call.function.arguments)
                    results = search_fn(args["query"])
                    openai.beta.threads.runs.submit_tool_outputs(
                        thread_id=thread.id,
                        run_id=run.id,
                        tool_outputs=[{
                            "tool_call_id": tool_call.id,
                            "output": str(results)
                        }]
                    )
        run = openai.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
    messages = openai.beta.threads.messages.list(thread_id=thread.id, order="desc", limit=1)
    return messages.data[0].content[0].text.value

# --- Streamlit UI ---

ensure_dirs()
st.set_page_config(page_title="RAG System with OpenAI Agents & FAISS", layout="wide")

# --- Example ingestion function to ensure 'text' key is present ---
def add_documents_to_faiss(db, texts, extra_metadata=None):
    # extra_metadata: list of dicts or None
    if extra_metadata is None:
        metadatas = [{"text": t} for t in texts]
    else:
        metadatas = []
        for t, meta in zip(texts, extra_metadata):
            m = dict(meta) if meta else {}
            m["text"] = t
            metadatas.append(m)
    db.add(texts, metadatas)

def add_documents_to_faiss(chunks, source):
    # Assuming this function adds documents to the FAISS index
    db = FAISSDB(dim=1536)
    texts = [chunk["text"] for chunk in chunks]
    metadatas = [{"source": source, "chunk_id": chunk["chunk_id"]} for chunk in chunks]
    db.add(texts, metadatas)

st.sidebar.title("📚 Document Management")
uploaded_files = st.sidebar.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True)
if st.sidebar.button("Index Manual(s)") and uploaded_files:
    for file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.read())
            tmp.flush()
            st.sidebar.write(f"Processing: {file.name}")
            chunks = pdf_to_text_chunks(tmp.name)
            add_documents_to_faiss(chunks, file.name)
    st.sidebar.success("Documents indexed!")

st.sidebar.markdown("---")
st.sidebar.title("🔎 Database Stats")

def load_faiss_index():
    db = FAISSDB(dim=1536)
    return db.index, db.meta

index, meta = load_faiss_index()
def load_faiss_index():
    db = FAISSDB(dim=1536)
    return db.index, db.meta

st.sidebar.write(f"**Documents:** {len(set(m['source'] for m in meta)) if meta else 0}")
st.sidebar.write(f"**Chunks:** {len(meta) if meta else 0}")

st.title("🤖 RAG System with OpenAI Agents & FAISS")
st.markdown("Ask questions about your uploaded manuals. The agent will search, analyze, and cite relevant information.")

if "history" not in st.session_state:
    st.session_state["history"] = []

query = st.text_input("Ask a question about your documents:", key="query")
search_sensitivity = st.slider("Search Sensitivity (Top K)", 1, 10, 5)

if st.button("Ask") and query:
    db = FAISSDB(dim=1536)
    agent = get_or_create_agent()
    def search_fn(q):
        results = db.search(q, k=5)
        return "\n".join([r["text"] for r in results])
    answer = run_agent(agent, query, search_fn)
    st.write(answer)

if st.button("Reset Conversation"):
    st.session_state["history"] = []

st.markdown("### 📝 Conversation History")
for msg in st.session_state["history"]:
    if msg["role"] == "user":
        st.write(f"**User:** {msg['content']}")
    else:
        st.write(f"**Agent:** {msg['content']}")