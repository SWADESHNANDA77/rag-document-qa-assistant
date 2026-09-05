from fastapi import FastAPI, UploadFile, File
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import os
import numpy as np
from dotenv import load_dotenv
from groq import Groq

from load_pdf import load_pdf_text, chunk_text

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

app = FastAPI(title="RAG Document Q&A Assistant")

# Load model once at startup (not on every request — that would be slow)
print("Loading embedding model...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

VECTOR_STORE_PATH = "vector_store.index"
CHUNKS_PATH = "chunks.pkl"

def load_or_create_store():
    if os.path.exists(VECTOR_STORE_PATH) and os.path.exists(CHUNKS_PATH):
        index = faiss.read_index(VECTOR_STORE_PATH)
        with open(CHUNKS_PATH, "rb") as f:
            chunks = pickle.load(f)
        return index, chunks
    else:
        # Empty store to start
        dimension = 384  # matches all-MiniLM-L6-v2 output size
        index = faiss.IndexFlatL2(dimension)
        return index, []

index, chunks = load_or_create_store()

def save_store():
    faiss.write_index(index, VECTOR_STORE_PATH)
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    global index, chunks

    # Save uploaded file temporarily
    temp_path = f"docs/{file.filename}"
    os.makedirs("docs", exist_ok=True)
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    # Process it: extract text, chunk, embed
    text = load_pdf_text(temp_path)
    new_chunks = chunk_text(text)

    new_embeddings = embed_model.encode(new_chunks).astype("float32")
    index.add(new_embeddings)
    chunks.extend(new_chunks)

    save_store()

    return {
        "filename": file.filename,
        "chunks_added": len(new_chunks),
        "total_chunks": len(chunks)
    }

@app.post("/ask")
async def ask_question(question: str):
    if len(chunks) == 0:
        return {"answer": "No documents uploaded yet. Please upload a PDF first."}

    query_embedding = embed_model.encode([question]).astype("float32")
    distances, indices = index.search(query_embedding, min(3, len(chunks)))
    retrieved_chunks = [chunks[idx] for idx in indices[0]]

    context = "\n\n".join(retrieved_chunks)
    prompt = f"""Answer the question based only on the context below.
If the answer isn't in the context, say "I don't have enough information to answer that."

Context:
{context}

Question: {question}

Answer:"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    return {
        "question": question,
        "answer": response.choices[0].message.content,
        "sources_used": len(retrieved_chunks)
    }

@app.get("/")
async def root():
    return {"message": "RAG Q&A API is running. Visit /docs to test endpoints."}