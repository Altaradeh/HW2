# RAG System with Streamlit and OpenAI Assistants
# Installation: pip install streamlit pypdf2 openai chromadb tiktoken python-dotenv
# Environment: OPENAI_API_KEY required

import streamlit as st
import os
import hashlib
import json
import logging
from typing import List, Dict, Any, Tuple
from pathlib import Path
import tiktoken
from dotenv import load_dotenv

# PDF processing
import PyPDF2
from io import BytesIO

# Embeddings and vector database
import openai
import chromadb
from chromadb.config import Settings

# OpenAI API (for both embeddings and chat completion)
from openai import OpenAI

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize OpenAI client
try:
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
except Exception as e:
    logger.error(f"Failed to initialize OpenAI client: {e}")
    st.error("Please ensure OPENAI_API_KEY is set in your environment")

# Constants
CHUNK_SIZE = 600  # Smaller chunks for better granularity
CHUNK_OVERLAP = 150  # More overlap for better context preservation
EMBEDDING_MODEL = "text-embedding-3-small"
ASSISTANT_MODEL = "gpt-4o-mini"  # Model for the assistant
DEFAULT_K = 6  # More chunks to retrieve
SIMILARITY_THRESHOLD = 0.3  # Lower threshold = more permissive search

# Assistant configuration
ASSISTANT_INSTRUCTIONS = """You are a RAG assistant specialized in answering questions from user manuals. 

Your capabilities:
- Search through indexed document chunks using the search_documents function
- Provide comprehensive answers with proper citations
- Handle follow-up questions with conversation context
- Synthesize information from multiple sources

Instructions:
- Always use the search_documents function to find relevant information
- Cite chunk IDs and page numbers in your responses
- If search returns no results, try rephrasing the search query
- Provide helpful, detailed answers based on the retrieved context
- Maintain conversation context for follow-up questions"""


class RAGSystem:
    """Main RAG System class handling all components with OpenAI Assistant integration"""
    
    def __init__(self, persist_directory: str = "./chroma_db"):
        """Initialize the RAG system with persistent vector database and OpenAI assistant"""
        self.persist_directory = persist_directory
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # Initialize ChromaDB with persistence
        try:
            # Create directory if it doesn't exist
            os.makedirs(persist_directory, exist_ok=True)

            self.chroma_client = chromadb.PersistentClient()
            self.collection_name = "manual_chunks"
            
            # Get or create collection
            try:
                self.collection = self.chroma_client.get_collection(name=self.collection_name)
                logger.info("Loaded existing collection")
            except Exception:
                self.collection = self.chroma_client.create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
                logger.info("Created new collection")
                
        except Exception as e:
            logger.error(f"Error initializing ChromaDB: {e}")
            # Fallback: recreate the database
            try:
                if os.path.exists(persist_directory):
                    import shutil
                    shutil.rmtree(persist_directory)
                os.makedirs(persist_directory, exist_ok=True)

                self.chroma_client = chromadb.PersistentClient(persist_directory=persist_directory)
                self.collection = self.chroma_client.create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
                logger.info("Recreated ChromaDB after error")
            except Exception as e2:
                logger.error(f"Failed to recreate ChromaDB: {e2}")
                st.error(f"Database initialization error: {e2}")
        
        # Initialize OpenAI Assistant
        self.assistant = None
        self.thread = None
        self._initialize_assistant()

    def _initialize_assistant(self):
        """Initialize OpenAI Assistant with RAG capabilities"""
        try:
            # Create or get existing assistant
            assistants = openai_client.beta.assistants.list()
            existing_assistant = None
            
            for assistant in assistants.data:
                if assistant.name == "RAG Document Assistant":
                    existing_assistant = assistant
                    break
            
            if existing_assistant:
                self.assistant = existing_assistant
                logger.info("Using existing RAG assistant")
            else:
                # Create new assistant with function calling
                self.assistant = openai_client.beta.assistants.create(
                    name="RAG Document Assistant",
                    instructions=ASSISTANT_INSTRUCTIONS,
                    model=ASSISTANT_MODEL,
                    tools=[
                        {
                            "type": "function",
                            "function": {
                                "name": "search_documents",
                                "description": "Search through indexed document chunks for relevant information",
                                "parameters": {
                                    "type": "object",
                                    "properties": {
                                        "query": {
                                            "type": "string",
                                            "description": "The search query to find relevant document chunks"
                                        },
                                        "num_results": {
                                            "type": "integer",
                                            "description": "Number of chunks to retrieve (default: 6)",
                                            "default": 6
                                        }
                                    },
                                    "required": ["query"]
                                }
                            }
                        }
                    ]
                )
                logger.info("Created new RAG assistant")
            
            # Create a new conversation thread
            self.thread = openai_client.beta.threads.create()
            
        except Exception as e:
            logger.error(f"Error initializing assistant: {e}")
            st.error(f"Error setting up AI assistant: {e}")

    def search_documents_function(self, query: str, num_results: int = 6) -> str:
        """
        Function for the assistant to search documents
        This will be called by the OpenAI assistant via function calling
        """
        try:
            chunks = self.retrieve_relevant_chunks(query, k=num_results)
            
            if not chunks:
                return "No relevant documents found for this query."
            
            # Format results for the assistant
            results = []
            for chunk in chunks:
                similarity_pct = int(chunk.get("similarity", 0) * 100)
                result = f"""
Chunk ID: {chunk['metadata']['chunk_id']}
Page: {chunk['metadata']['page_number']}
Source: {chunk['metadata']['source_file']}
Similarity: {similarity_pct}%
Content: {chunk['text'][:800]}{'...' if len(chunk['text']) > 800 else ''}
---"""
                results.append(result)
            
            return "\n".join(results)
            
        except Exception as e:
            logger.error(f"Error in search function: {e}")
            return f"Error searching documents: {e}"

    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken"""
        return len(self.tokenizer.encode(text))

    def load_and_chunk_pdfs(self, uploaded_files) -> List[Dict[str, Any]]:
        """
        Extract text from uploaded PDFs and split into chunks with metadata
        
        Args:
            uploaded_files: List of uploaded PDF files from Streamlit
            
        Returns:
            List of chunk dictionaries with text, metadata, and ids
        """
        all_chunks = []
        
        for uploaded_file in uploaded_files:
            try:
                # Read PDF content
                pdf_reader = PyPDF2.PdfReader(BytesIO(uploaded_file.read()))
                
                # Extract text from all pages
                full_text = ""
                page_texts = {}
                
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    page_text = page.extract_text()
                    page_texts[page_num] = page_text
                    full_text += f"\n--- Page {page_num} ---\n" + page_text
                
                # Create file hash for caching
                uploaded_file.seek(0)
                file_hash = hashlib.md5(uploaded_file.read()).hexdigest()
                
                # Check if already processed
                if self._is_file_processed(file_hash):
                    logger.info(f"File {uploaded_file.name} already processed, skipping")
                    continue
                
                # Split into chunks
                chunks = self._split_text_into_chunks(full_text, uploaded_file.name, page_texts)
                
                # Add file hash to metadata
                for chunk in chunks:
                    chunk["metadata"]["file_hash"] = file_hash
                
                all_chunks.extend(chunks)
                logger.info(f"Processed {uploaded_file.name}: {len(chunks)} chunks created")
                
            except Exception as e:
                logger.error(f"Error processing {uploaded_file.name}: {e}")
                st.error(f"Error processing {uploaded_file.name}: {e}")
                continue
        
        return all_chunks

    def _is_file_processed(self, file_hash: str) -> bool:
        """Check if file with given hash has already been processed"""
        try:
            if not hasattr(self, 'collection') or self.collection is None:
                return False
                
            results = self.collection.get(
                where={"file_hash": file_hash},
                limit=1
            )
            return len(results["ids"]) > 0
        except Exception as e:
            logger.warning(f"Error checking file processing status: {e}")
            return False

    def _split_text_into_chunks(self, text: str, filename: str, page_texts: Dict[int, str]) -> List[Dict[str, Any]]:
        """
        Split text into chunks with overlap, preserving page information
        Uses both sentence-aware and sliding window approaches for better retrieval
        
        Args:
            text: Full document text
            filename: Source filename
            page_texts: Dictionary mapping page numbers to page text
            
        Returns:
            List of chunk dictionaries
        """
        chunks = []
        
        # Clean and preprocess text
        text = self._clean_text(text)
        
        # Split into sentences first for better boundaries
        sentences = self._split_into_sentences(text)
        
        current_chunk = []
        current_tokens = 0
        chunk_id = 0
        
        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence)
            
            # Check if adding this sentence would exceed chunk size
            if current_tokens + sentence_tokens > CHUNK_SIZE and current_chunk:
                # Create chunk from current sentences
                chunk_text = " ".join(current_chunk)
                page_num = self._find_page_number(chunk_text, page_texts)
                
                chunk = {
                    "id": f"{filename}_{chunk_id}",
                    "text": chunk_text,
                    "metadata": {
                        "chunk_id": chunk_id,
                        "source_file": filename,
                        "page_number": page_num,
                        "token_count": current_tokens,
                        "sentence_count": len(current_chunk)
                    }
                }
                chunks.append(chunk)
                
                # Start new chunk with overlap (keep last few sentences)
                overlap_sentences = max(1, min(3, len(current_chunk) // 3))
                overlap_text = current_chunk[-overlap_sentences:] if len(current_chunk) > overlap_sentences else []
                
                current_chunk = overlap_text + [sentence]
                current_tokens = sum(self.count_tokens(s) for s in current_chunk)
                chunk_id += 1
            else:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens
        
        # Handle remaining sentences
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            page_num = self._find_page_number(chunk_text, page_texts)
            
            chunk = {
                "id": f"{filename}_{chunk_id}",
                "text": chunk_text,
                "metadata": {
                    "chunk_id": chunk_id,
                    "source_file": filename,
                    "page_number": page_num,
                    "token_count": current_tokens,
                    "sentence_count": len(current_chunk)
                }
            }
            chunks.append(chunk)
        
        return chunks

    def _clean_text(self, text: str) -> str:
        """Clean and preprocess text for better chunking"""
        import re
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page markers but keep page info
        text = re.sub(r'\n--- Page \d+ ---\n', ' [PAGE_BREAK] ', text)
        
        # Fix common PDF extraction issues
        text = re.sub(r'([a-z])([A-Z])', r'\1. \2', text)  # Add periods between sentences
        text = re.sub(r'([.!?])\s*([a-z])', lambda m: m.group(1) + ' ' + m.group(2).upper(), text)
        
        return text.strip()

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using simple rules"""
        import re
        
        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        
        # Clean and filter sentences
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10:  # Filter very short fragments
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences

    def _find_page_number(self, chunk_text: str, page_texts: Dict[int, str]) -> int:
        """Find the most likely page number for a chunk"""
        # Look for page markers first
        for page_num in page_texts.keys():
            if f"--- Page {page_num} ---" in chunk_text:
                return page_num
        
        # Fallback: find page with most overlap
        best_page = 1
        max_overlap = 0
        
        for page_num, page_text in page_texts.items():
            # Count common words
            chunk_words = set(chunk_text.lower().split())
            page_words = set(page_text.lower().split())
            overlap = len(chunk_words.intersection(page_words))
            
            if overlap > max_overlap:
                max_overlap = overlap
                best_page = page_num
        
        return best_page

    def embed_and_store_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Generate embeddings for chunks and store in vector database
        
        Args:
            chunks: List of chunk dictionaries to embed and store
        """
        if not chunks:
            return
        
        try:
            # Process in batches to avoid token limits
            BATCH_SIZE = 100  # Process 100 chunks at a time
            MAX_TOKENS_PER_BATCH = 250000  # Safety margin below 300k limit
            
            total_processed = 0
            
            for i in range(0, len(chunks), BATCH_SIZE):
                batch_chunks = chunks[i:i + BATCH_SIZE]
                
                # Check token count for this batch
                batch_texts = [chunk["text"] for chunk in batch_chunks]
                batch_token_count = sum(self.count_tokens(text) for text in batch_texts)
                
                # If batch is too large, process smaller sub-batches
                if batch_token_count > MAX_TOKENS_PER_BATCH:
                    # Process one chunk at a time for this batch
                    for single_chunk in batch_chunks:
                        self._process_single_chunk_batch([single_chunk])
                        total_processed += 1
                        
                        # Update progress
                        if total_processed % 10 == 0:
                            st.write(f"Processed {total_processed}/{len(chunks)} chunks...")
                else:
                    # Process the whole batch
                    self._process_single_chunk_batch(batch_chunks)
                    total_processed += len(batch_chunks)
                    
                    # Update progress
                    st.write(f"Processed {total_processed}/{len(chunks)} chunks...")
            
            logger.info(f"Successfully stored {len(chunks)} chunks in vector database")
            
        except Exception as e:
            logger.error(f"Error embedding and storing chunks: {e}")
            st.error(f"Error processing embeddings: {e}")
            raise

    def _process_single_chunk_batch(self, batch_chunks: List[Dict[str, Any]]) -> None:
        """
        Process a single batch of chunks for embedding and storage
        
        Args:
            batch_chunks: List of chunks to process in this batch
        """
        texts = [chunk["text"] for chunk in batch_chunks]
        ids = [chunk["id"] for chunk in batch_chunks]
        metadatas = [chunk["metadata"] for chunk in batch_chunks]
        
        # Get embeddings from OpenAI
        response = openai_client.embeddings.create(
            input=texts,
            model=EMBEDDING_MODEL
        )
        
        embeddings = [item.embedding for item in response.data]
        
        # Store in ChromaDB
        self.collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )

    def retrieve_relevant_chunks(self, query: str, k: int = DEFAULT_K) -> List[Dict[str, Any]]:
        """
        Retrieve most relevant chunks for a query using multiple strategies
        
        Args:
            query: User question
            k: Number of chunks to retrieve
            
        Returns:
            List of relevant chunks with metadata and similarity scores
        """
        try:
            # Expand query with synonyms and variations
            expanded_queries = self._expand_query(query)
            
            all_results = []
            
            # Search with each query variation
            for search_query in expanded_queries:
                # Generate query embedding
                query_response = openai_client.embeddings.create(
                    input=[search_query],
                    model=EMBEDDING_MODEL
                )
                query_embedding = query_response.data[0].embedding
                
                # Search vector database with more results
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=min(k * 2, 20)  # Get more candidates
                )
                
                # Add to results
                if results["ids"] and len(results["ids"]) > 0:
                    for i in range(len(results["ids"][0])):
                        similarity = 1 - results["distances"][0][i] if "distances" in results else 1
                        
                        # Apply similarity threshold
                        if similarity >= SIMILARITY_THRESHOLD:
                            chunk = {
                                "id": results["ids"][0][i],
                                "text": results["documents"][0][i],
                                "metadata": results["metadatas"][0][i],
                                "similarity": similarity,
                                "query_variant": search_query
                            }
                            all_results.append(chunk)
            
            # Remove duplicates and rank by similarity
            unique_results = self._deduplicate_and_rank(all_results)
            
            # Return top k results
            final_results = unique_results[:k]
            
            logger.info(f"Retrieved {len(final_results)} chunks for query (from {len(all_results)} candidates)")
            
            # If no good results, try fuzzy matching
            if not final_results:
                logger.info("No semantic matches found, trying keyword search...")
                final_results = self._keyword_fallback_search(query, k)
            
            return final_results
            
        except Exception as e:
            logger.error(f"Error retrieving chunks: {e}")
            st.error(f"Error during retrieval: {e}")
            return []

    def _expand_query(self, query: str) -> List[str]:
        """Expand query with variations to improve search recall"""
        queries = [query]
        
        # Add question variations
        query_lower = query.lower().strip()
        
        # Remove question words for keyword-style search
        question_words = ['what', 'how', 'when', 'where', 'why', 'which', 'who', 'is', 'are', 'can', 'should', 'do', 'does']
        keywords = ' '.join([word for word in query_lower.split() if word not in question_words])
        if keywords and keywords != query_lower:
            queries.append(keywords)
        
        # Add variations with different phrasing
        if 'how to' in query_lower:
            queries.append(query_lower.replace('how to', 'procedure for'))
            queries.append(query_lower.replace('how to', 'steps to'))
        
        if '?' in query:
            queries.append(query.replace('?', ''))
        
        return list(set(queries))  # Remove duplicates

    def _deduplicate_and_rank(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate chunks and rank by best similarity"""
        seen_ids = set()
        unique_results = []
        
        # Sort by similarity score (highest first)
        sorted_results = sorted(results, key=lambda x: x["similarity"], reverse=True)
        
        for result in sorted_results:
            if result["id"] not in seen_ids:
                seen_ids.add(result["id"])
                unique_results.append(result)
        
        return unique_results

    def _keyword_fallback_search(self, query: str, k: int) -> List[Dict[str, Any]]:
        """Fallback keyword-based search when semantic search fails"""
        try:
            # Get all documents
            all_docs = self.collection.get()
            
            if not all_docs["documents"]:
                return []
            
            # Simple keyword matching
            query_words = set(query.lower().split())
            scored_chunks = []
            
            for i, doc in enumerate(all_docs["documents"]):
                doc_words = set(doc.lower().split())
                
                # Calculate overlap score
                overlap = len(query_words.intersection(doc_words))
                if overlap > 0:
                    score = overlap / len(query_words)  # Normalize by query length
                    
                    chunk = {
                        "id": all_docs["ids"][i],
                        "text": doc,
                        "metadata": all_docs["metadatas"][i],
                        "similarity": score,
                        "search_type": "keyword_fallback"
                    }
                    scored_chunks.append(chunk)
            
            # Sort by score and return top k
            scored_chunks.sort(key=lambda x: x["similarity"], reverse=True)
            return scored_chunks[:k]
            
        except Exception as e:
            logger.error(f"Keyword fallback search failed: {e}")
            return []

    def answer_with_assistant(self, query: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Generate answer using OpenAI Assistant with function calling
        
        Args:
            query: User question
            
        Returns:
            Tuple of (assistant's answer, retrieved chunks for display)
        """
        try:
            # Add user message to thread
            openai_client.beta.threads.messages.create(
                thread_id=self.thread.id,
                role="user",
                content=query
            )
            
                      # ...existing code...
            
            # Wait for any previous run to complete before adding a new message
            if hasattr(self, 'thread') and self.thread is not None:
                try:
                    runs = openai_client.beta.threads.runs.list(thread_id=self.thread.id, order="desc", limit=1)
                    if runs.data:
                        last_run = runs.data[0]
                        while last_run.status in ['queued', 'in_progress', 'requires_action']:
                            import time
                            time.sleep(1)
                            last_run = openai_client.beta.threads.runs.retrieve(thread_id=self.thread.id, run_id=last_run.id)
                except Exception as e:
                    logger.warning(f"Error checking previous run status: {e}")
            else:
                logger.warning("No active thread found.")
            
            # Run the assistant
            run = openai_client.beta.threads.runs.create(
                thread_id=self.thread.id,
                assistant_id=self.assistant.id
            )
            # ...existing code...
            
            # Wait for completion and handle function calls
            retrieved_chunks = []
            
            while run.status in ['queued', 'in_progress', 'requires_action']:
                if run.status == 'requires_action':
                    # Handle function calls
                    tool_calls = run.required_action.submit_tool_outputs.tool_calls
                    tool_outputs = []
                    
                    for tool_call in tool_calls:
                        if tool_call.function.name == "search_documents":
                            # Parse function arguments
                            import json
                            args = json.loads(tool_call.function.arguments)
                            search_query = args.get("query", "")
                            num_results = args.get("num_results", 6)
                            
                            # Perform search
                            chunks = self.retrieve_relevant_chunks(search_query, k=num_results)
                            retrieved_chunks.extend(chunks)  # Store for UI display
                            
                            # Get search results for assistant
                            search_results = self.search_documents_function(search_query, num_results)
                            
                            tool_outputs.append({
                                "tool_call_id": tool_call.id,
                                "output": search_results
                            })
                    
                    # Submit tool outputs
                    run = openai_client.beta.threads.runs.submit_tool_outputs(
                        thread_id=self.thread.id,
                        run_id=run.id,
                        tool_outputs=tool_outputs
                    )
                
                # Wait and check status
                import time
                time.sleep(1)
                run = openai_client.beta.threads.runs.retrieve(
                    thread_id=self.thread.id,
                    run_id=run.id
                )
            
            if run.status == 'completed':
                # Get the assistant's response
                messages = openai_client.beta.threads.messages.list(
                    thread_id=self.thread.id,
                    order="desc",
                    limit=1
                )
                
                if messages.data:
                    answer = messages.data[0].content[0].text.value
                    
                    # Log the interaction
                    self._log_interaction(query, answer, retrieved_chunks)
                    
                    return answer, retrieved_chunks
                else:
                    return "No response received from assistant.", []
            else:
                logger.error(f"Assistant run failed with status: {run.status}")
                return f"Assistant encountered an error (status: {run.status})", []
                
        except Exception as e:
            logger.error(f"Error with assistant: {e}")
            st.error(f"Error generating answer: {e}")
            return "Sorry, I encountered an error while generating the answer.", []

    def reset_conversation(self):
        """Reset the conversation thread for a fresh start"""
        try:
            self.thread = openai_client.beta.threads.create()
            logger.info("Conversation reset")
        except Exception as e:
            logger.error(f"Error resetting conversation: {e}")

    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get the current conversation history"""
        try:
            messages = openai_client.beta.threads.messages.list(
                thread_id=self.thread.id,
                order="asc"
            )
            
            history = []
            for message in messages.data:
                history.append({
                    "role": message.role,
                    "content": message.content[0].text.value,
                    "timestamp": message.created_at
                })
            
            return history
        except Exception as e:
            logger.error(f"Error getting conversation history: {e}")
            return []

    def _log_interaction(self, query: str, answer: str, chunks: List[Dict[str, Any]]):
        """Log questions and answers for evaluation"""
        try:
            log_entry = {
                "query": query,
                "answer": answer,
                "chunks_used": [
                    {
                        "chunk_id": chunk["metadata"]["chunk_id"],
                        "page": chunk["metadata"]["page_number"],
                        "source": chunk["metadata"]["source_file"]
                    }
                    for chunk in chunks
                ]
            }
            
            # Create logs directory if it doesn't exist
            os.makedirs("logs", exist_ok=True)
            
            # Append to log file
            with open("logs/rag_interactions.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
                
        except Exception as e:
            logger.warning(f"Failed to log interaction: {e}")


def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="RAG System with Claude",
        page_icon="📚",
        layout="wide"
    )
    
    st.title("📚 RAG System with OpenAI")
    st.markdown("Upload PDF manuals and ask questions")
    
    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        st.error("⚠️ Please set OPENAI_API_KEY environment variable")
        st.stop()
    
    # Initialize RAG system
    if "rag_system" not in st.session_state:
        st.session_state.rag_system = RAGSystem()
    
    rag_system = st.session_state.rag_system
    
    # Sidebar for file upload and indexing
    with st.sidebar:
        st.header("📄 Document Management")
        
        uploaded_files = st.file_uploader(
            "Upload PDF manuals",
            type=["pdf"],
            accept_multiple_files=True,
            help="Upload one or more PDF files to create your knowledge base"
        )
        
        if uploaded_files:
            st.write(f"Uploaded {len(uploaded_files)} file(s)")
            for file in uploaded_files:
                st.write(f"- {file.name}")
        
        if st.button("🔄 Index Manual(s)", disabled=not uploaded_files):
            try:
                with st.spinner("Processing PDFs..."):
                    # Load and chunk PDFs
                    chunks = rag_system.load_and_chunk_pdfs(uploaded_files)
                    
                    if chunks:
                        # Embed and store chunks
                        rag_system.embed_and_store_chunks(chunks)
                        st.success(f"✅ Successfully indexed {len(chunks)} chunks!")
                    else:
                        st.info("ℹ️ No new content to index (files may already be processed)")
                        
            except Exception as e:
                st.error(f"❌ Error during indexing: {e}")
        
        # Show database stats
        try:
            if hasattr(rag_system, 'collection') and rag_system.collection is not None:
                collection_count = rag_system.collection.count()
                st.metric("📊 Total Chunks", collection_count)
            else:
                st.metric("📊 Total Chunks", 0)
        except Exception as e:
            logger.warning(f"Error getting collection count: {e}")
            st.metric("📊 Total Chunks", "Error")
    
    # Main query interface
    st.header("💬 Ask Questions")
    
    # Query input
    query = st.text_input(
        "Enter your question:",
        placeholder="What is the procedure for...",
        help="Ask questions about the uploaded manuals"
    )
    
    # Settings
    col1, col2, col3 = st.columns([2, 1, 1])
    with col2:
        k_chunks = st.selectbox("Chunks to retrieve", [4, 6, 8, 10], index=1)
    with col3:
        similarity_threshold = st.slider("Search sensitivity", 0.1, 0.8, 0.3, 0.1,
                                       help="Lower = more permissive search")
    
    # Process query
    if query and st.button("🔍 Get Answer"):
        try:
            # Check if database has content
            if rag_system.collection.count() == 0:
                st.warning("⚠️ Please upload and index some PDF files first!")
                st.stop()
            
            with st.spinner("Searching for relevant information..."):
                # Update similarity threshold
                rag_system.similarity_threshold = similarity_threshold
                
                # Retrieve relevant chunks
                relevant_chunks = rag_system.retrieve_relevant_chunks(query, k=k_chunks)
            
            if not relevant_chunks:
                st.warning("No relevant information found. Try rephrasing your question.")
            else:
                with st.spinner("Generating answer with OpenAI..."):
                    # Generate answer
                    answer, retrieved_chunks = rag_system.answer_with_assistant(query)
                
                # Display answer
                st.header("🤖 Answer")
                st.markdown(answer)
                
                # Display retrieved chunks
                with st.expander("📋 Retrieved Context Chunks"):
                    for i, chunk in enumerate(relevant_chunks, 1):
                        st.subheader(f"Chunk {i}")
                        
                        # Metadata with similarity score
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Chunk ID", chunk["metadata"]["chunk_id"])
                        with col2:
                            st.metric("Page", chunk["metadata"]["page_number"])
                        with col3:
                            similarity_pct = int(chunk.get("similarity", 0) * 100)
                            st.metric("Similarity", f"{similarity_pct}%")
                        with col4:
                            search_type = chunk.get("search_type", "semantic")
                            st.metric("Search Type", search_type)
                        
                        # Content
                        st.text_area(
                            f"Content {i}",
                            chunk["text"][:500] + "..." if len(chunk["text"]) > 500 else chunk["text"],
                            height=100,
                            key=f"chunk_{i}"
                        )
                        
                        if i < len(relevant_chunks):
                            st.divider()
                        
        except Exception as e:
            st.error(f"❌ Error processing query: {e}")
    
    # Footer
    st.markdown("---")
    st.markdown("**Tip**: Ask specific questions about your uploaded manuals.")


if __name__ == "__main__":
    main()