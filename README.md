# 📄 PDF Chatbot using Gemini
## 🌐 Live Demo
👉 https://pdf-chatbot-gemini-t9chehy6yrptas5mhv3ucb.streamlit.app/

A document-aware chatbot that allows users to upload PDFs and ask questions using semantic search and RAG.

## 🚀 Tech Stack
- Python
- Streamlit
- LangChain
- FAISS (Vector Database)
- Google Gemini (LLM)
- HuggingFace / Gemini Embeddings

## 🧠 How it works
1. Upload PDF files
2. Extract and chunk text
3. Generate embeddings
4. Store vectors in FAISS
5. Answer questions using Retrieval-Augmented Generation (RAG)

## ▶️ Run Locally

```bash
conda create -n genai python=3.10
conda activate genai
pip install -r requirements.txt
streamlit run app.py
