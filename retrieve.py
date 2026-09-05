from sentence_transformers import SentenceTransformer
import faiss
import pickle
import numpy as np

def load_vector_store():
    index = faiss.read_index("vector_store.index")
    with open("chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    return index, chunks

def retrieve(query, index, chunks, model, top_k=3):
    query_embedding = model.encode([query]).astype("float32")
    distances, indices = index.search(query_embedding, top_k)

    results = []
    for i, idx in enumerate(indices[0]):
        results.append({
            "chunk": chunks[idx],
            "distance": distances[0][i]
        })
    return results

if __name__ == "__main__":
    print("Loading model and vector store...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    index, chunks = load_vector_store()

    query = input("\nAsk a question about your document: ")

    results = retrieve(query, index, chunks, model, top_k=3)

    print(f"\n--- Top {len(results)} relevant chunks ---\n")
    for i, r in enumerate(results):
        print(f"[Result {i+1}] (distance: {r['distance']:.4f})")
        print(r["chunk"])
        print("-" * 50)