from sentence_transformers import SentenceTransformer
import faiss
import pickle
import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()  # loads variables from .env

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def load_vector_store():
    index = faiss.read_index("vector_store.index")
    with open("chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    return index, chunks

def retrieve(query, index, chunks, model, top_k=3):
    query_embedding = model.encode([query]).astype("float32")
    distances, indices = index.search(query_embedding, top_k)
    return [chunks[idx] for idx in indices[0]]

def generate_answer(query, retrieved_chunks):
    context = "\n\n".join(retrieved_chunks)

    prompt = f"""Answer the question based only on the context below. 
If the answer isn't in the context, say "I don't have enough information to answer that."

Context:
{context}

Question: {query}

Answer:"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    print("Loading model and vector store...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    index, chunks = load_vector_store()

    query = input("\nAsk a question about your document: ")

    retrieved_chunks = retrieve(query, index, chunks, model, top_k=3)
    answer = generate_answer(query, retrieved_chunks)

    print("\n--- Answer ---")
    print(answer)