import streamlit as st
from rag_engine import process_pdfs, retrieve_context, get_gemini_response
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Streamlit app config
st.set_page_config(page_title="MEDRAG - Medical AI Assistant", layout="wide")
st.title("🩺 MEDRAG - Medical Document RAG Assistant")

DATA_FOLDER = "data"

# Cache the PDF processing
@st.cache_data
def process_data():
    process_pdfs(DATA_FOLDER)

# Process PDFs once
process_data()

# Input query
query = st.text_input("🔍 Ask your medical question:")

if query:
    with st.spinner("Retrieving relevant chunks and generating answer..."):
        chunks = retrieve_context(query)
        answer = get_gemini_response(query, chunks)

    st.subheader("💬 Gemini's Answer")
    st.markdown(answer)

    with st.expander("📄 Retrieved Chunks"):
        for i, chunk in enumerate(chunks):
            st.markdown(f"**Chunk {i + 1}:**")
            st.info(chunk)
