import streamlit as st
import faiss
import numpy as np
import pickle
import os
import PyPDF2
from groq import Groq
from sentence_transformers import SentenceTransformer
from datetime import datetime
import tempfile

# Page configuration
st.set_page_config(
    page_title="PDF Chat System",
    page_icon="🧙‍♂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        padding: 1rem 0;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: black;
        border-left: 4px solid #1E88E5;
    }
    .assistant-message {
        background-color: black;
        border-left: 4px solid #43A047;
    }
    .source-info {
        background-color: dark-teal;
        padding: 0.5rem;
        border-radius: 0.3rem;
        font-size: 0.85rem;
        margin-top: 0.5rem;
    }
    .stButton button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'current_pdf' not in st.session_state:
    st.session_state.current_pdf = None
if 'embedding_model' not in st.session_state:
    st.session_state.embedding_model = None
if 'groq_client' not in st.session_state:
    st.session_state.groq_client = None
if 'vector_index' not in st.session_state:
    st.session_state.vector_index = None
if 'chunks_data' not in st.session_state:
    st.session_state.chunks_data = None


# Functions
@st.cache_resource
def load_embedding_model():
    """Load the sentence transformer model"""
    return SentenceTransformer('all-MiniLM-L6-v2')


def initialize_groq_client(api_key):
    """Initialize Groq client"""
    return Groq(api_key=api_key)


def extract_text_from_pdf(pdf_file):
    """Extract text from uploaded PDF"""
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    total_pages = len(pdf_reader.pages)

    page_texts = []
    for page_num, page in enumerate(pdf_reader.pages):
        page_text = page.extract_text()
        page_texts.append({
            'text': page_text,
            'page_number': page_num + 1
        })

    text = ''.join([p['text'] for p in page_texts])
    return text, total_pages, page_texts


def create_chunks(text, total_pages, chunk_size=500, overlap=100):
    """Create text chunks with metadata"""
    chunks = []
    chunk_metadata = []

    for i in range(0, len(text), chunk_size - overlap):
        chunk_text = text[i:i + chunk_size]
        chunks.append(chunk_text)

        estimated_page = min((i // (len(text) // total_pages)) + 1, total_pages)
        chunk_metadata.append({
            'start_pos': i,
            'estimated_page': estimated_page
        })

    return chunks, chunk_metadata


def create_vector_index(chunks, embedding_model):
    """Create FAISS index from chunks"""
    embeddings = embedding_model.encode(chunks, show_progress_bar=True)
    embeddings = np.array(embeddings).astype('float32')

    embedding_dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(embedding_dim)

    faiss.normalize_L2(embeddings)
    index.add(embeddings)

    return index, embeddings


def get_query_embedding(query, embedding_model):
    """Get embedding for query"""
    embedding = embedding_model.encode([query])
    embedding = embedding.astype('float32')
    faiss.normalize_L2(embedding)
    return embedding


def search_similar_chunks(query, index, chunks, metadata, embedding_model, top_k=3):
    """Search for similar chunks"""
    query_vector = get_query_embedding(query, embedding_model)
    scores, indices = index.search(query_vector, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        results.append({
            'chunk': chunks[idx],
            'page': metadata[idx]['estimated_page'],
            'score': float(score)
        })

    return results


def generate_answer(question, context_results, groq_client, total_pages, pdf_name):
    """Generate answer using Groq"""
    context_parts = []
    for result in context_results:
        context_parts.append(f"[Page {result['page']}]: {result['chunk']}")

    context = '\n\n'.join(context_parts)

    chat_completion = groq_client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": f"You are answering questions about a {total_pages}-page document titled '{pdf_name}'. When providing answers, mention page numbers when relevant. Be specific and cite information from the provided context."
            },
            {
                "role": "user",
                "content": f"Context from the document:\n{context}\n\nQuestion: {question}\n\nPlease answer based on the context provided above:"
            }
        ],
        model="llama-3.1-8b-instant",
        temperature=0.5,
        max_tokens=1024,
    )

    return chat_completion.choices[0].message.content, context_results


# Main App
st.markdown('<div class="main-header">🧙‍♂️PDF Chat System</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")

    # API Key Input
    groq_api_key = st.text_input("Groq API Key", type="password", help="Enter your Groq API key from console.groq.com")

    if groq_api_key:
        if st.session_state.groq_client is None:
            st.session_state.groq_client = initialize_groq_client(groq_api_key)
            st.success("Groq API connected!")

    st.markdown("---")

    # PDF Upload/Selection
    st.header("📄 PDF Document")

    pdf_option = st.radio("Choose PDF source:", ["Upload New PDF", "Use Default PDF"])

    if pdf_option == "Upload New PDF":
        uploaded_file = st.file_uploader("Upload PDF", type=['pdf'])

        if uploaded_file and st.button("Process PDF"):
            with st.spinner("Processing PDF..."):
                try:
                    # Load embedding model
                    if st.session_state.embedding_model is None:
                        st.session_state.embedding_model = load_embedding_model()

                    # Extract text
                    text, total_pages, page_texts = extract_text_from_pdf(uploaded_file)

                    # Create chunks
                    chunks, metadata = create_chunks(text, total_pages)

                    # Create vector index
                    index, embeddings = create_vector_index(chunks, st.session_state.embedding_model)

                    # Store in session state
                    st.session_state.vector_index = index
                    st.session_state.chunks_data = {
                        'chunks': chunks,
                        'metadata': metadata,
                        'total_pages': total_pages
                    }
                    st.session_state.current_pdf = uploaded_file.name

                    st.success(f"PDF processed! {len(chunks)} chunks created from {total_pages} pages.")

                except Exception as e:
                    st.error(f"Error processing PDF: {str(e)}")

    else:
        default_pdf_path = "social distancing detection report.pdf"

        if os.path.exists(default_pdf_path):
            st.info(f"Using: {default_pdf_path}")

            if st.button("Load Default PDF"):
                with st.spinner("Loading PDF..."):
                    try:
                        # Load embedding model
                        if st.session_state.embedding_model is None:
                            st.session_state.embedding_model = load_embedding_model()

                        # Extract text
                        with open(default_pdf_path, 'rb') as f:
                            text, total_pages, page_texts = extract_text_from_pdf(f)

                        # Create chunks
                        chunks, metadata = create_chunks(text, total_pages)

                        # Create vector index
                        index, embeddings = create_vector_index(chunks, st.session_state.embedding_model)

                        # Store in session state
                        st.session_state.vector_index = index
                        st.session_state.chunks_data = {
                            'chunks': chunks,
                            'metadata': metadata,
                            'total_pages': total_pages
                        }
                        st.session_state.current_pdf = default_pdf_path

                        st.success(f"PDF loaded! {len(chunks)} chunks from {total_pages} pages.")

                    except Exception as e:
                        st.error(f"Error loading PDF: {str(e)}")
        else:
            st.warning(f"Default PDF not found: {default_pdf_path}")

    st.markdown("---")

    # Current Document Info
    if st.session_state.current_pdf:
        st.header("📊 Current Document")
        st.info(f"**File:** {st.session_state.current_pdf}")
        if st.session_state.chunks_data:
            st.metric("Total Pages", st.session_state.chunks_data['total_pages'])
            st.metric("Total Chunks", len(st.session_state.chunks_data['chunks']))

    st.markdown("---")

    # Clear History
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

# Main Content Area
col1, col2 = st.columns([2, 1])

with col1:
    st.header("💬 Chat Interface")

    # Display chat history
    chat_container = st.container()
    with chat_container:
        for chat in st.session_state.chat_history:
            # User message
            st.markdown(f"""
            <div class="chat-message user-message">
                <strong>You:</strong> {chat['question']}
                <div style="font-size: 0.75rem; color: #666; margin-top: 0.3rem;">
                    {chat['timestamp']}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Assistant message
            st.markdown(f"""
            <div class="chat-message assistant-message">
                <strong>Assistant:</strong> {chat['answer']}
            </div>
            """, unsafe_allow_html=True)

            # Source information
            if 'sources' in chat and chat['sources']:
                source_text = " | ".join([f"Page {s['page']} (Score: {s['score']:.2f})" for s in chat['sources']])
                st.markdown(f"""
                <div class="source-info">
                    <strong>📍 Sources:</strong> {source_text}
                </div>
                """, unsafe_allow_html=True)

    # Question input
    st.markdown("---")
    question = st.text_input("Ask a question about the document:", key="question_input")

    col_ask, col_example = st.columns([1, 1])

    with col_ask:
        ask_button = st.button("🚀 Ask Question", type="primary")

    with col_example:
        if st.button("💡 Show Examples"):
            st.session_state.show_examples = not st.session_state.get('show_examples', False)

    if st.session_state.get('show_examples', False):
        st.info("""
        **Example Questions:**
        - What deep learning model and dataset are used for social distancing detection in the proposed system?
        - What are the two components of the project's hardware requirements? 
        - Who is the Head of the Department for Artificial Intelligence and Data Science at R.M.K. College Of Engineering and Technology?
        - According to the Activity Diagram, what action is taken if the "Distance Measured" is less than the "Minimum Distance"?
        - Name three existing methods for social distancing detection mentioned in the report. 
        """)

    # Process question
    if ask_button and question:
        if not st.session_state.groq_client:
            st.error("Please enter your Groq API key in the sidebar!")
        elif not st.session_state.vector_index:
            st.error("Please load a PDF document first!")
        else:
            with st.spinner("Searching and generating answer..."):
                try:
                    # Search similar chunks
                    results = search_similar_chunks(
                        question,
                        st.session_state.vector_index,
                        st.session_state.chunks_data['chunks'],
                        st.session_state.chunks_data['metadata'],
                        st.session_state.embedding_model
                    )

                    # Generate answer
                    answer, sources = generate_answer(
                        question,
                        results,
                        st.session_state.groq_client,
                        st.session_state.chunks_data['total_pages'],
                        st.session_state.current_pdf
                    )

                    # Add to chat history
                    st.session_state.chat_history.append({
                        'question': question,
                        'answer': answer,
                        'sources': sources,
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })

                    st.rerun()

                except Exception as e:
                    st.error(f"Error generating answer: {str(e)}")

with col2:
    st.header("📜 Chat History")

    if st.session_state.chat_history:
        st.metric("Total Questions", len(st.session_state.chat_history))

        st.markdown("---")

        for idx, chat in enumerate(reversed(st.session_state.chat_history), 1):
            with st.expander(f"Q{len(st.session_state.chat_history) - idx + 1}: {chat['question'][:50]}..."):
                st.markdown(f"**Question:** {chat['question']}")
                st.markdown(f"**Time:** {chat['timestamp']}")
                if 'sources' in chat:
                    pages = [s['page'] for s in chat['sources']]
                    st.markdown(f"**Pages Referenced:** {', '.join(map(str, pages))}")
    else:
        st.info("No chat history yet. Start asking questions!")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>Built with Streamlit, Groq, and FAISS | Powered by sentence-transformers</p>
</div>
""", unsafe_allow_html=True)