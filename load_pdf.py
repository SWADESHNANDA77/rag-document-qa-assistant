from pypdf import PdfReader
import os

def load_pdf_text(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap  # overlap helps preserve context across chunks
    return chunks

if __name__ == "__main__":
    pdf_folder = "docs"
    pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith(".pdf")]

    if not pdf_files:
        print("No PDF found in docs/ folder. Add one and rerun.")
    else:
        pdf_path = os.path.join(pdf_folder, pdf_files[0])
        print(f"Loading: {pdf_path}")

        text = load_pdf_text(pdf_path)
        print(f"Total characters extracted: {len(text)}")

        chunks = chunk_text(text)
        print(f"Number of chunks: {len(chunks)}")
        print("\n--- First chunk preview ---")
        print(chunks[0])