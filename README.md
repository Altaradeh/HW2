# RAG System with OpenAI Agents & FAISS

A sophisticated Retrieval-Augmented Generation (RAG) system that combines OpenAI's experimental Agents SDK with FAISS vector database for intelligent document querying and analysis.

## 🌟 Features

### 🤖 **OpenAI Agents Integration**
- **Advanced Agent Capabilities**: Uses OpenAI's cutting-edge Agents SDK for intelligent reasoning
- **Multi-Tool Support**: Agents can search documents, get statistics, and suggest search terms
- **Context-Aware Conversations**: Maintains conversation history and context
- **Self-Improving**: Agents can try different search strategies automatically

### 🔍 **Enhanced Search & Retrieval**
- **FAISS Vector Database**: Fast, efficient similarity search optimized for cloud deployment
- **Multiple Search Strategies**: Semantic, keyword, and hybrid search modes
- **Smart Query Expansion**: Automatically tries different phrasings and synonyms
- **Relevance Scoring**: Shows similarity percentages for retrieved chunks

### 📚 **Document Processing**
- **PDF Upload & Processing**: Handles large PDF manuals with intelligent chunking
- **Sentence-Aware Splitting**: Preserves document structure and context
- **Metadata Preservation**: Tracks source files, page numbers, and chunk IDs
- **Incremental Updates**: Add new documents without reprocessing existing ones

### ☁️ **Streamlit Cloud Ready**
- **No System Dependencies**: Uses FAISS instead of ChromaDB (no SQLite issues)
- **Persistent Storage**: Data survives app restarts
- **Poetry Package Management**: Clean dependency management for deployment
- **Environment Variable Configuration**: Secure API key management

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- OpenAI API key
- Poetry (recommended) or pip

### Installation with Poetry

1. **Clone and Setup**:
```bash
git clone <repository-url>
cd rag-system
```

2. **Install Dependencies**:
```bash
poetry install
```

3. **Set Environment Variables**:
```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

4. **Run the Application**:
```bash
poetry run streamlit run rag_system.py
```

### Alternative Installation with Pip

```bash
pip install streamlit openai openai-agents pypdf2 faiss-cpu numpy tiktoken python-dotenv scikit-learn pandas
```

## 📁 Project Structure

```
rag-system/
├── rag_system.py          # Main application file
├── pyproject.toml         # Poetry configuration
├── README.md              # This file
├── .env.example           # Environment variables template
├── .streamlit/
│   └── config.toml        # Streamlit configuration
├── faiss_db/              # FAISS database storage (created automatically)
└── logs/                  # Interaction logs (created automatically)
```

## ⚙️ Configuration

### Environment Variables

Create a `.env` file with:

```bash
OPENAI_API_KEY=your_openai_api_key_here
```

### Streamlit Cloud Deployment

1. **Set up repository** on GitHub
2. **Deploy to Streamlit Cloud**:
   - Connect your GitHub repository
   - Set `OPENAI_API_KEY` in Streamlit Cloud secrets
   - Deploy!

## 🎯 Usage

### 1. **Upload Documents**
- Upload one or more PDF files using the sidebar
- Click "Index Manual(s)" to process and store embeddings

### 2. **Chat with Your Documents**
- Ask questions in natural language
- The agent will automatically search for relevant information
- Get comprehensive answers with source citations

### 3. **Advanced Features**
- **Adjust search sensitivity** with the slider
- **Reset conversation** for a fresh start
- **View retrieved chunks** to see source material
- **Monitor database statistics** in the sidebar

## 🧠 How It Works

### Agent Workflow
1. **User Query** → Processed by OpenAI Agent
2. **Intelligent Analysis** → Agent decides search strategy
3. **Multi-Tool Execution** → Searches documents, gets stats, suggests alternatives
4. **Context Synthesis** → Combines information from multiple sources
5. **Comprehensive Response** → Provides well-cited, detailed answers

### Search Strategies
- **Semantic Search**: Vector similarity using embeddings
- **Keyword Search**: Traditional text matching (fallback)
- **Hybrid Search**: Combines both approaches for maximum recall

### Vector Database
- **FAISS IndexFlatIP**: Optimized for cosine similarity
- **Normalized Embeddings**: Ensures accurate similarity scoring
- **Persistent Storage**: Index and metadata saved to disk
- **Error Recovery**: Automatic database health checks and recovery

## 🔧 Development

### Running Tests
```bash
poetry run pytest
```

### Code Formatting
```bash
poetry run black rag_system.py
poetry run flake8 rag_system.py
```

### Type Checking
```bash
poetry run mypy rag_system.py
```

## 📊 Performance & Scaling

### Optimization Features
- **Batch Processing**: Handles large documents without API limits
- **Smart Chunking**: Sentence-aware splitting preserves context
- **Caching**: Avoids reprocessing identical files
- **Progress Tracking**: Real-time feedback during indexing

### Resource Usage
- **Memory Efficient**: FAISS optimized for cloud environments
- **Storage Efficient**: Compressed embeddings and metadata
- **API Efficient**: Batched requests and smart retry logic

## 🆘 Troubleshooting

### Common Issues

**"FAISS database connection lost"**
- Click "Recover FAISS Database" in sidebar
- If persists, restart the application

**"No relevant information found"**
- Try different search terms or broader queries
- Use the "hybrid" search type for better coverage
- Check if documents were properly indexed

**API Errors**
- Verify OPENAI_API_KEY is set correctly
- Check API quota and billing status
- Try reducing batch size for large documents

### Debug Mode
Enable "Show Development Info" to see:
- Database file status
- Agent availability
- Conversation history length
- Document statistics

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and formatting
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- **OpenAI** for the Agents SDK and GPT models
- **Meta AI** for FAISS vector database
- **Streamlit** for the amazing web framework
- **Python Poetry** for dependency management