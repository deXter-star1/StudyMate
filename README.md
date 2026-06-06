# 📚 StudyMate

An AI-powered study assistant that answers questions about your own course material. Upload your PDFs and StudyMate uses retrieval-augmented generation (RAG) to answer questions grounded in those documents — with source citations.

**Live demo:** https://studymate-dex.streamlit.app

## What it does
- Upload one or more course PDFs
- Builds a searchable knowledge base from them
- Ask questions in plain English; answers are drawn only from your documents
- Each answer cites the source file and page it came from

## How it works — the RAG pipeline
1. **Chunking** — each PDF is split into overlapping text chunks
2. **Embeddings** — each chunk becomes a vector via Google's `gemini-embedding-001` model
3. **Retrieval** — the question is embedded too, and cosine similarity selects the most relevant chunks
4. **Generation** — only those chunks go to Gemini, which writes a grounded, cited answer

## Tech stack
- **Python**
- **Streamlit** — web UI
- **Google Gemini API** — embeddings + answer generation
- **pypdf** — PDF text extraction
- **NumPy** — in-memory vector store and cosine-similarity search

## Run locally
1. Clone this repo and open the folder
2. Install dependencies: `pip install -r requirements.txt`
3. Start the app: `streamlit run app.py`
4. Paste your free Gemini API key (from Google AI Studio) into the sidebar

## Note on design
The vector store is implemented in-memory with NumPy to keep the retrieval logic transparent and the app dependency-light. Swapping in a dedicated vector database such as Chroma is a small change.
