import faiss
import numpy as np
import pickle
import PyPDF2
from groq import Groq

# Set your Groq API key
GROQ_API_KEY = "YOUR_KEY"
client = Groq(api_key=GROQ_API_KEY)


def get_embeddings(text_chunks):
    """Get embeddings using sentence-transformers (local model)
    Note: Groq doesn't provide embedding models, so we use a local alternative"""
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(text_chunks, show_progress_bar=True)
        return embeddings
    except ImportError:
        print("❌ Please install sentence-transformers: pip install sentence-transformers")
        return None


def pdf_to_vectors(pdf_path):
    # Read PDF
    print(f"📄 Reading PDF: {pdf_path}")
    with open(pdf_path, 'rb') as f:
        pdf_reader = PyPDF2.PdfReader(f)
        total_pages = len(pdf_reader.pages)

        # Extract text from each page separately
        page_texts = []
        for page_num, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            page_texts.append({
                'text': page_text,
                'page_number': page_num + 1
            })

        # Combine all text for chunking
        text = ''.join([p['text'] for p in page_texts])

    print(f"📊 Total pages: {total_pages}")
    print(f"📊 Total text length: {len(text):,} characters")
    print(f"📊 Average characters per page: {len(text) // total_pages:,}")

    # Create chunks with page tracking
    chunks = []
    chunk_metadata = []

    for i in range(0, len(text), 400):
        chunk_text = text[i:i + 500]
        chunks.append(chunk_text)

        # Estimate which page this chunk belongs to
        estimated_page = min((i // (len(text) // total_pages)) + 1, total_pages)
        chunk_metadata.append({
            'start_pos': i,
            'estimated_page': estimated_page
        })

    print(f"✂️ Created {len(chunks)} chunks")

    # Get embeddings using local model
    print("🔄 Getting embeddings using local model...")
    embeddings = get_embeddings(chunks)

    if embeddings is None:
        print("❌ Failed to generate embeddings")
        return None, None

    # Create FAISS index
    print("🗂️ Creating FAISS index...")
    embeddings = np.array(embeddings)
    embedding_dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(embedding_dim)

    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(embeddings.astype('float32'))
    index.add(embeddings.astype('float32'))

    # Save to files
    print("💾 Saving to files...")
    faiss.write_index(index, "vectors.index")
    with open("chunks.pkl", "wb") as f:
        pickle.dump({
            'chunks': chunks,
            'metadata': chunk_metadata,
            'total_pages': total_pages,
            'embedding_dim': embedding_dim
        }, f)

    print("✅ Vector database created successfully!")
    print(f"📁 Files saved: vectors.index, chunks.pkl")
    print(f"📊 Vector shape: {embeddings.shape}")
    print(f"📢 Embedding dimensions: {embedding_dim}")

    return embeddings, chunks


# Usage
if __name__ == "__main__":
    # Convert PDF to vectors (run this once)
    pdf_file = "social distancing detection report.pdf"
    embeddings, chunks = pdf_to_vectors(pdf_file)

    if embeddings is not None:
        print("\n🎉 Setup complete! Now you can run 'question-vector.py' to chat with your PDF!")
    else:
        print("\n❌ Setup failed. Please install required packages:")
        print("   pip install sentence-transformers")