# DocuLlama-AI
# 📄 DocuLlama-AI — Chat with Your PDFs Using LLaMA Locally

DocuLlama-AI lets you **upload a PDF and interact with it conversationally**, powered entirely by a **locally-running LLaMA model** via **Ollama** and **LangChain**.

> 🔐 100% local. No API keys. No cloud. Just you and your documents.

---

## 🚀 Features

- 📁 Upload any PDF (notes, books, contracts, etc.)
- 🧠 Ask any question about its content
- 💬 Instant LLM-powered replies using **LLaMA3**
- 🔎 Context-aware retrieval using **vector embeddings**
- 🔌 Works completely offline (once set up)

---

## 🧠 How It Works

1. PDF is converted into chunks of text  
2. Chunks are embedded into vectors using `mxbai-embed-large`  
3. Stored in a **Chroma** vector database  
4. A retriever finds the most relevant chunks  
5. A **locally-running LLaMA 3** model answers your query using LangChain + RAG

---

## 🛠️ Setup Instructions
pip install -r requirements.txt

### ✅ 1. Install Ollama

Install Ollama (to run LLaMA 3 locally):

ollama pull llama3
ollama pull mxbai-embed-large




