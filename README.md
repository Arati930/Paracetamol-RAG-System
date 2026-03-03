# 📄 Paracetamol RAG Question Answering System

## 📌 Project Overview
This project implements a Retrieval-Augmented Generation (RAG) system for answering questions from a Paracetamol document (PDF).  

The system uses:
- LangChain
- FAISS Vector Database
- HuggingFace Embeddings
- GPT-2 (Free LLM)
- Streamlit (Web Interface)

---

## 🚀 How It Works

1. Upload a PDF document.
2. The document is split into smaller text chunks.
3. Each chunk is converted into embeddings.
4. FAISS stores the embeddings for similarity search.
5. When a question is asked:
   - Relevant chunks are retrieved.
   - The retrieved context is passed to the LLM.
   - The model generates an answer.
6. The answer and similarity scores are displayed.

---

## 🧠 Technologies Used

- Python
- LangChain
- FAISS
- HuggingFace Transformers
- Streamlit

---

## 📂 Project Structure
Paracetamol-RAG-System/
│
├── app.py
├── rag_pipeline.py
├── requirements.txt
├── paracetamol.pdf
└── README.md


---

## ▶️ How to Run Locally

1. Install dependencies:
pip install -r requirements.txt

2. Run Streamlit:
streamlit run app.py


3. Upload the PDF and ask questions.

---

## 📊 Features

- Document chunking
- Embedding generation
- FAISS similarity search
- Retrieval with similarity score
- Context-aware answer generation
- Interactive web interface

---

## 🎯 Example Question

**Q:** What is the maximum daily dose of paracetamol?  
**A:** The recommended maximum daily dose for adults is 3–4 grams (3000–4000 mg).

---

## 👩‍💻 Author

Arati Todalabagi  
AI & Data Science Student

