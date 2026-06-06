import streamlit as st
from pypdf import PdfReader
from google import genai
from google.genai import errors
import numpy as np
import time

st.set_page_config(page_title="StudyMate", page_icon="📚", layout="centered")

# --- Clean, minimal styling ---
st.markdown(
    """
    <style>
    [data-testid="stToolbar"], header[data-testid="stHeader"] {display: none;}
    .block-container {padding-top: 3rem; padding-bottom: 4rem; max-width: 820px;}
    .stButton button {font-weight: 700; letter-spacing: 0.02em; padding: 0.55rem 1.7rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Hero header ---
st.markdown(
    """
    <div style="margin-bottom: 2.5rem;">
      <h1 style="font-size: 3.4rem; font-weight: 900; letter-spacing: -0.03em;
                 text-transform: uppercase; line-height: 0.95; margin: 0;">StudyMate</h1>
      <p style="font-size: 1.05rem; color: #666; font-weight: 500; margin-top: 0.6rem;">
        Upload your course PDFs. Build a knowledge base. Ask anything.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

api_key = st.sidebar.text_input("Gemini API key", type="password")

# Remember the knowledge base across reruns
if "chunks" not in st.session_state:
    st.session_state.chunks = []      # each: {"text", "source", "page"}
    st.session_state.vectors = None   # numpy array of embeddings

# Split a page's text into smaller pieces
def split_text(text, size=1500, overlap=200):
    pieces = []
    start = 0
    while start < len(text):
        pieces.append(text[start:start + size])
        start += size - overlap
    return pieces

# Turn a list of texts into embedding vectors
def embed(texts):
    client = genai.Client(api_key=api_key)
    vectors = []
    for i in range(0, len(texts), 100):   # max 100 per request
        batch = texts[i:i + 100]
        result = client.models.embed_content(
            model="gemini-embedding-001",
            contents=batch,
        )
        for e in result.embeddings:
            vectors.append(e.values)
    return np.array(vectors)

# --- Phase 1: upload + index ---
uploaded_files = st.file_uploader(
    "Upload one or more PDFs", type="pdf", accept_multiple_files=True
)

if st.button("Build knowledge base", type="primary"):
    if not api_key:
        st.error("Paste your Gemini API key in the sidebar first.")
    elif not uploaded_files:
        st.error("Upload at least one PDF first.")
    else:
        chunks = []
        for f in uploaded_files:
            reader = PdfReader(f)
            for page_num, page in enumerate(reader.pages, start=1):
                page_text = page.extract_text() or ""
                for piece in split_text(page_text):
                    if piece.strip():
                        chunks.append(
                            {"text": piece, "source": f.name, "page": page_num}
                        )
        if not chunks:
            st.error("Couldn't read any text from those PDFs.")
        else:
            with st.spinner(f"Indexing {len(chunks)} chunks..."):
                try:
                    st.session_state.vectors = embed([c["text"] for c in chunks])
                    st.session_state.chunks = chunks
                    st.success(
                        f"Indexed {len(chunks)} chunks from "
                        f"{len(uploaded_files)} document(s). Ask away below."
                    )
                except errors.ServerError:
                    st.warning("Gemini's busy. Wait a few seconds and click again.")

# --- Phase 2: ask ---
question = st.text_input("Your question")

if st.button("Ask", type="primary") and question:
    if not api_key:
        st.error("Paste your Gemini API key in the sidebar first.")
    elif st.session_state.vectors is None:
        st.error("Build the knowledge base first (button above).")
    else:
        q_vector = None
        try:
            q_vector = embed([question])[0]
        except errors.ServerError:
            st.warning("Gemini's busy. Wait a few seconds and hit Ask again.")

        if q_vector is not None:
            # cosine similarity: find the chunks closest in meaning
            docs = st.session_state.vectors
            q = q_vector / (np.linalg.norm(q_vector) + 1e-10)
            d = docs / (np.linalg.norm(docs, axis=1, keepdims=True) + 1e-10)
            scores = d @ q
            top_idx = np.argsort(scores)[::-1][:4]   # 4 best matches
            picked = [st.session_state.chunks[i] for i in top_idx]

            context = ""
            for c in picked:
                context += f"[{c['source']}, page {c['page']}]\n{c['text']}\n\n"

            prompt = (
                "Answer the question using ONLY the context below. "
                "Cite the source and page in your answer. "
                "If the answer isn't in the context, say so.\n\n"
                f"CONTEXT:\n{context}\n"
                f"QUESTION: {question}"
            )

            client = genai.Client(api_key=api_key)
            answer = None
            with st.spinner("Thinking..."):
                for attempt in range(3):
                    try:
                        response = client.models.generate_content(
                            model="gemini-2.5-flash-lite",
                            contents=prompt,
                        )
                        answer = response.text
                        break
                    except errors.ServerError:
                        time.sleep(2)

            if answer:
                st.write(answer)
                with st.expander("📄 Sources used"):
                    for c in picked:
                        st.markdown(f"**{c['source']} — page {c['page']}**")
                        st.write(c["text"][:300] + "...")
            else:
                st.warning("Gemini's busy right now. Wait a few seconds and hit Ask again.")
