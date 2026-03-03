import streamlit as st
from rag_pipeline import build_rag

st.set_page_config(page_title="Paracetamol RAG System", layout="wide")

st.title("📄 Paracetamol RAG Question Answering System")

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file is not None:

    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    retriever, llm = build_rag("temp.pdf")

    question = st.text_input("Ask a question about the document:")

    if question:
        with st.spinner("Generating answer..."):

            # 🔥 Similarity Search with Score (Bonus Feature)
            docs_with_scores = retriever.vectorstore.similarity_search_with_score(
                question, k=3
            )

            docs = [doc for doc, score in docs_with_scores]

            context = "\n\n".join([doc.page_content for doc in docs])

            prompt = f"""
Use the following context to answer clearly and accurately.

Context:
{context}

Question:
{question}

Answer:
"""

            answer = llm.invoke(prompt)

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("✅ Answer")
                st.write(answer)

            with col2:
                st.subheader("📚 Retrieved Chunks with Similarity Score")

                for doc, score in docs_with_scores:
                    st.write(f"**Similarity Score:** {score:.4f}")
                    st.write(doc.page_content)
                    st.write("--------")