from load_pdf import load_pdf_text, chunk_text
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import pickle

def get_chunks():
    pdf_folder = "docs"
    pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith(".pdf")]
    pdf_path = os.path.join(pdf_folder, pdf_files[0])
    text = load_pdf_text(pdf_path)
    chunks = chunk_text(text)
    return chunks

def build_vector_store(chunks):
    print("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print(f"Generating embeddings for {len(chunks)} chunks...")
    embeddings = model.encode(chunks, show_progress_bar=True)

    embeddings = np.array(embeddings).astype("float32")
    dimension = embeddings.shape[1]

    print(f"Embedding dimension: {dimension}")

    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    return index, chunks

if __name__ == "__main__":
    chunks = get_chunks()
    index, chunks = build_vector_store(chunks)

    # Save the index and chunks so we don't have to rebuild every time
    faiss.write_index(index, "vector_store.index")
    with open("chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

    print(f"\nDone! Stored {index.ntotal} vectors.")
    print("Saved as: vector_store.index and chunks.pkl")