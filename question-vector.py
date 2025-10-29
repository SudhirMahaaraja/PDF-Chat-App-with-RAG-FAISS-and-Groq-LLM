import faiss
import numpy as np
import pickle
import os
from groq import Groq

# Set your Groq API key
GROQ_API_KEY = "YOUR_KEY"
client = Groq(api_key=GROQ_API_KEY)


def get_query_embedding(query):
    """Get embedding for the query using the same local model"""
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embedding = model.encode([query])
        return embedding
    except ImportError:
        print("❌ Please install sentence-transformers: pip install sentence-transformers")
        return None


def ask_question(question):
    # Check if vector files exist
    if not os.path.exists("vectors.index") or not os.path.exists("chunks.pkl"):
        print("❌ Error: Vector database not found!")
        print("🔧 Please run 'pdf-vector-groq.py' first to create the database.")
        return None

    try:
        # Load saved data
        index = faiss.read_index("vectors.index")
        with open("chunks.pkl", "rb") as f:
            data = pickle.load(f)

        chunks = data['chunks']
        metadata = data['metadata']
        total_pages = data['total_pages']

        # Get question embedding
        query_vector = get_query_embedding(question)
        if query_vector is None:
            return None

        # Normalize for cosine similarity
        query_vector = query_vector.astype('float32')
        faiss.normalize_L2(query_vector)

        # Search similar chunks (retrieve top 3)
        scores, indices = index.search(query_vector, 3)

        # Show similarity scores and page info for debugging
        print(f"🔍 Found {len(indices[0])} relevant chunks:")
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            page_num = metadata[idx]['estimated_page']
            print(f"   Chunk {i + 1}: Score {score:.3f} (≈Page {page_num})")

        # Build context with page information
        context_parts = []
        for idx in indices[0]:
            chunk_text = chunks[idx]
            page_num = metadata[idx]['estimated_page']
            context_parts.append(f"[Page {page_num}]: {chunk_text}")

        context = '\n\n'.join(context_parts)

        # Get answer from Groq
        print("🤖 Generating answer using Groq...")
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": f"You are answering questions about a {total_pages}-page Social Distancing detection report. When providing answers, mention page numbers when relevant. Be specific and cite information from the provided context."
                },
                {
                    "role": "user",
                    "content": f"Context from the document:\n{context}\n\nQuestion: {question}\n\nPlease answer based on the context provided above:"
                }
            ],
            model="llama-3.1-8b-instant",  # You can also use: "mixtral-8x7b-32768", "llama-3.1-70b-versatile"
            temperature=0.3,
            max_tokens=1024,
        )

        return chat_completion.choices[0].message.content

    except Exception as e:
        print(f"❌ Error processing question: {str(e)}")
        return None


def main():
    # Check if database exists
    if not os.path.exists("vectors.index") or not os.path.exists("chunks.pkl"):
        print("❌ Vector database not found!")
        print("🔧 Please run 'pdf-vector.py' first to create the database.")
        print("📋 Steps:")
        print("   1. Run: python pdf-vector.py")
        print("   2. Then run: python question-vector.py")
        return

    # Load database info
    try:
        index = faiss.read_index("vectors.index")
        with open("chunks.pkl", "rb") as f:
            data = pickle.load(f)

        chunks = data['chunks']
        total_pages = data['total_pages']

        print(f"✅ Database loaded: {len(chunks)} chunks from {total_pages} pages")
        print(f"📄 Document: Tourist Behaviour Analysis Report")
    except Exception as e:
        print(f"❌ Error loading database: {str(e)}")
        return

    # Interactive question loop
    print("\n" + "=" * 60)
    print("🤖 RAG System Ready! Ask questions about the Tourist Behaviour Analysis Report")
    print("💡 Type 'bye', 'quit', 'exit', or 'q' to exit")
    print("📊 Type 'info' to see database statistics")
    print("💭 Type 'examples' to see sample questions")
    print("=" * 60)

    while True:
        question = input("\n❓ Your question: ").strip()

        # Check for exit commands
        if question.lower() in ['bye', 'quit', 'exit', 'q']:
            print("👋 Goodbye! Thanks for using the RAG system!")
            break

        # Show database info
        if question.lower() == 'info':
            print(f"📊 Database Info:")
            print(f"   • Document: Tourist Behaviour Analysis Report")
            print(f"   • Total pages: {total_pages}")
            print(f"   • Total chunks: {len(chunks)}")
            print(f"   • Embedding model: all-MiniLM-L6-v2")
            print(f"   • LLM: Groq (llama-3.1-8b-instant)")
            continue

        # Show example questions
        if question.lower() == 'examples':
            print("\n💭 Example Questions:")
            print("   • What states were analyzed in this project?")
            print("   • What clustering algorithm was used?")
            print("   • What were the main findings about Goa tourism?")
            print("   • How was the data extracted from Instagram?")
            print("   • What is the future scope of this project?")
            print("   • Who were the team members?")
            print("   • What were the challenges faced during data extraction?")
            continue

        # Skip empty questions
        if not question:
            print("⚠️ Please enter a question!")
            continue

        print("🔍 Searching and generating answer...")
        answer = ask_question(question)

        if answer:
            print(f"\n🤖 Answer:\n{answer}")
        else:
            print("❌ Sorry, I couldn't generate an answer. Please try a different question.")


if __name__ == "__main__":
    main()