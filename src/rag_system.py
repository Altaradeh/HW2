from dotenv import load_dotenv
import os
load_dotenv()
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")
import faiss
import numpy as np
import streamlit as st
import os
import pickle
from PyPDF2 import PdfReader
from io import BytesIO

# --- Embedding and FAISS setup ---
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

    def reset(self):
        self.index = faiss.IndexFlatL2(self.dim)
        self.meta = []
        self.save()

# --- PDF Chunking ---
def pdf_to_chunks(pdf_file, chunk_size=600, overlap=100):
    reader = PdfReader(pdf_file)
    all_chunks = []
    for page_num, page in enumerate(reader.pages, 1):
        text = page.extract_text() or ""
        words = text.split()
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i+chunk_size]
            chunk_text = " ".join(chunk_words)
            if chunk_text.strip():
                all_chunks.append({
                    "text": chunk_text,
                    "page": page_num
                })
    return all_chunks

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
    import json
    while run.status in ["queued", "in_progress", "requires_action"]:
        if run.status == "requires_action":
            tool_outputs = []
            for tool_call in run.required_action.submit_tool_outputs.tool_calls:
                if tool_call.function.name == "search_documents":
                    args = eval(tool_call.function.arguments)
                    results = search_fn(args["query"])
                    tool_outputs.append({
                        "tool_call_id": tool_call.id,
                        "output": json.dumps(results)
                    })
            if tool_outputs:
                openai.beta.threads.runs.submit_tool_outputs(
                    thread_id=thread.id,
                    run_id=run.id,
                    tool_outputs=tool_outputs
                )
        run = openai.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
    messages = openai.beta.threads.messages.list(thread_id=thread.id, order="desc", limit=1)
    answer = messages.data[0].content[0].text.value
    # Fetch the tool output from the run steps
    tool_output = None
    try:
        steps = openai.beta.threads.runs.steps.list(thread_id=thread.id, run_id=run.id)
        # Debug: print all step details
        import streamlit as st
        #st.write("[DEBUG] All run steps:", steps.data)
        for step in steps.data:
            # Check for tool_calls step type
            if hasattr(step, "type") and step.type == "tool_calls":
                if hasattr(step, "step_details") and hasattr(step.step_details, "tool_calls"):
                    for tool_call in step.step_details.tool_calls:
                        if hasattr(tool_call, "output") and tool_call.output:
                            tool_output = tool_call.output
    except Exception as e:
        st.write(f"[DEBUG] Exception while extracting tool output: {e}")
        tool_output = None
    return answer, tool_output

# --- Streamlit UI ---
st.set_page_config(page_title="RAG System with OpenAI Agents & FAISS", layout="wide")
st.title("📚 RAG System with OpenAI Agents & FAISS")

# Sidebar: PDF upload and DB controls
with st.sidebar:
    st.header("📄 Upload PDF(s)")
    uploaded_files = st.file_uploader("Upload PDF manuals", type=["pdf"], accept_multiple_files=True)
    st.markdown("---")
    st.header("🗄️ Database Controls")
    reset_pw = st.text_input("Enter password to reset DB", type="password", key="reset_db_pw")
    if st.button("Reset Database"):
        if reset_pw == "letmein":  # Change 'letmein' to your desired password
            db = FAISSDB(dim=1536)
            db.reset()
            st.success("Database reset!")
        else:
            st.error("Incorrect password. Database not reset.")
    st.markdown("---")
    st.header("📊 Stats")
    db = FAISSDB(dim=1536)
    st.write(f"**Chunks:** {len(db.meta)}")

# Main: Indexing and Query
if uploaded_files:
    st.subheader("Indexing PDFs...")
    db = FAISSDB(dim=1536)
    for file in uploaded_files:
        with st.spinner(f"Processing {file.name}..."):
            chunks = pdf_to_chunks(BytesIO(file.read()))
            texts = []
            metadatas = []
            for c in chunks:
                # Ensure all fields are present and correct
                text = c.get("text", "")
                page = c.get("page", "N/A")
                source = file.name
                texts.append(text)
                metadatas.append({"text": text, "page": page, "source": source})
            db.add(texts, metadatas)
    st.success("All PDFs indexed!")
    st.info("If you previously indexed PDFs with missing metadata, please reset the database and re-upload your PDFs for correct chunk display.")

st.header("💬 Ask a Question")
def on_query_enter():
    if query.strip():
        st.session_state["loading"] = True
        st.session_state["trigger_answer"] = True
        st.rerun()

if "trigger_answer" not in st.session_state:
    st.session_state["trigger_answer"] = False

query = st.text_input(
    "Enter your question:",
    key="main_query_input",
    on_change=on_query_enter
)
if "loading" not in st.session_state:
    st.session_state["loading"] = False
if "answer" not in st.session_state:
    st.session_state["answer"] = None
if "tool_output" not in st.session_state:
    st.session_state["tool_output"] = None

get_answer_btn = st.button("🔎 Get Answer", disabled=st.session_state["loading"])
if (get_answer_btn or st.session_state["trigger_answer"]) and query:
    st.session_state["loading"] = True
    st.session_state["trigger_answer"] = False
    st.rerun()

if st.session_state["loading"]:
    import json
    db = FAISSDB(dim=1536)
    agent = get_or_create_agent()
    with st.spinner("Waiting for agent response..."):
        def search_fn(q):
            # Return full chunk metadata for JSON serialization (including text and page)
            results = db.search(q, k=5)
            # Ensure only serializable fields are included
            return [
                {
                    "text": r.get("text", ""),
                    "page": r.get("page", "N/A"),
                    "source": r.get("source", "N/A")
                }
                for r in results
            ]
        answer, tool_output = run_agent(agent, query, search_fn)
    st.session_state["answer"] = answer
    st.session_state["tool_output"] = tool_output
    st.session_state["loading"] = False
    st.rerun()

# Display answer and chunks if available
if st.session_state["answer"] is not None:
    st.subheader("🤖 Agent Answer")
    st.write(st.session_state["answer"])
    st.markdown("---")
    st.subheader("🔍 Retrieved Chunks")
    chunks = []
    tool_output = st.session_state["tool_output"]
    if tool_output:
        import json
        try:
            if isinstance(tool_output, list):
                chunks = tool_output
            elif isinstance(tool_output, str):
                chunks = json.loads(tool_output)
        except Exception as e:
            st.write(f"[DEBUG] Failed to parse tool output: {e}")
            chunks = []
    if not chunks:
        db = FAISSDB(dim=1536)
        chunks = db.search(query, k=5)
    for r in chunks:
        st.write(f"**Page:** {r.get('page', 'N/A')} | **Source:** {r.get('source','N/A')}")
        #st.write(r.get('text',''))
        st.markdown("---")
