# RAG Document Q&A Assistant

A Retrieval-Augmented Generation (RAG) system that lets you upload PDF documents and ask natural language questions about their content. The app retrieves the most relevant sections from your documents and uses an LLM to generate accurate, grounded answers — instead of relying on the model's general knowledge alone.

## Features

- Upload any PDF document via API
- Ask natural language questions about the uploaded content
- Answers are grounded in your actual documents (reduces hallucination)
- Built as a REST API using FastAPI, with interactive Swagger docs

## Tech Stack

- **Language:** Python
- **Backend Framework:** FastAPI
- **Embeddings:** `sentence-transformers` (all-MiniLM-L6-v2)
- **Vector Store:** FAISS
- **LLM:** Groq API (openai/gpt-oss-20b)
- **PDF Processing:** pypdf

## How It Works

1. **Document Ingestion** — PDF text is extracted and split into overlapping chunks (500 characters, 50 character overlap) to preserve context across boundaries.
2. **Embedding** — Each chunk is converted into a 384-dimensional vector using a sentence-transformer model.
3. **Storage** — Vectors are stored in a FAISS index for fast similarity search.
4. **Retrieval** — When a question is asked, it's embedded the same way, and the top 3 most similar chunks are retrieved using L2 distance.
5. **Generation** — Retrieved chunks are passed as context to an LLM (via Groq), which generates a natural language answer grounded in that context.

## Project Structure
rag-qa-assistant/
├── main.py # FastAPI app with /upload and /ask endpoints
├── load_pdf.py # PDF text extraction and chunking
├── embed_and_store.py # Embedding generation and FAISS storage
├── retrieve.py # Standalone retrieval testing script
├── rag_query.py # Standalone full RAG pipeline (CLI version)
├── requirements.txt # Python dependencies
└── .gitignore

## Setup & Installation

1. Clone the repository:
```bash
git clone https://github.com/SWADESHNANDA77/rag-document-qa-assistant.git
cd rag-document-qa-assistant
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the root directory:
GROQ_API_KEY=your_api_key_here
Get a free API key at [console.groq.com](https://console.groq.com)

5. Run the API:
```bash
uvicorn main:app --reload
```

6. Open your browser to `http://127.0.0.1:8000/docs` to test the endpoints interactively.

## API Endpoints

### `POST /upload`
Upload a PDF document to be processed and added to the knowledge base.

### `POST /ask`
Ask a question. Returns an answer grounded in the uploaded documents.

**Example response:**
```json
{
  "question": "What is DBMS?",
  "answer": "DBMS (Database Management System) is a collection of programs that manages a database...",
  "sources_used": 3
}
```

## Design Decisions

- **Chunk size (500 chars) with 50-char overlap**: balances context preservation with retrieval precision — small enough for focused retrieval, with overlap to avoid losing meaning at chunk boundaries.
- **FAISS over other vector DBs**: lightweight, no external service dependency, fast for small-to-medium document collections.
- **Groq for inference**: free tier, extremely fast inference speed, good for rapid development and demos.

## Future Improvements

- [ ] Deploy on AWS (EC2/Lambda)
- [ ] Add a simple frontend (Streamlit)
- [ ] Support multiple file formats (docx, txt)
- [ ] Add chat history / conversational memory
- [ ] Improve chunking strategy (semantic chunking instead of fixed-size)

## Author

Swadesh Nanda
