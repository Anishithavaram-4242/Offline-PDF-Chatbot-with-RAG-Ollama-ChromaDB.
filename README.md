# 🔍 Offline PDF Chatbot with RAG, Ollama & ChromaDB

This project is a local RAG (Retrieval-Augmented Generation) system that allows you to chat with any PDF file using a locally hosted LLM (via Ollama). Perfect for offline, private, and fast GenAI interaction.

---

## 🚀 Features

- ✅ Parse any PDF using UnstructuredPDFLoader
- ✅ Generate and store embeddings using FastEmbed and ChromaDB
- ✅ Query the document with natural language
- ✅ Run LLM (like Gemma 2B) locally using Ollama
- ✅ Completely offline — no API keys or cloud LLMs
- ✅ Model warm-up for faster first response



![PDF Chatbot Screenshot](screenshot.png))

---

## 📦 Tech Stack

- Python
- [LangChain](https://github.com/langchain-ai/langchain)
- [Ollama](https://ollama.com)
- [ChromaDB](https://www.trychroma.com/)
- [UnstructuredPDFLoader](https://github.com/Unstructured-IO/unstructured)
- [FastEmbed](https://github.com/langchain-ai/fastembed)

---

## 📁 How it Works

1. Upload a PDF file
2. Split into chunks and embed
3. Store embeddings in ChromaDB
4. Use similarity search to find relevant chunks
5. Ask a question — LLM answers based on retrieved context

---

## ⚙️ Setup Instructions

```bash
# 1. Clone repo
git clone https://github.com/your-username/pdf-ollama-rag

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate  # or source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run Ollama (in a separate terminal)
ollama run gemma:2b

# 5. Start app
python app.py
