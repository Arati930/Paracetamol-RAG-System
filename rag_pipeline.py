from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFacePipeline

from transformers import pipeline


def build_rag(pdf_path):

    # 1. Load PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # 2. Chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    chunks = text_splitter.split_documents(documents)

    # 3. Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    # 4. Store in FAISS
    vectorstore = FAISS.from_documents(chunks, embeddings)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # 5. FREE LLM (GPT-2 - Stable Version)
    hf_pipeline = pipeline(
        task="text-generation",
        model="gpt2",
        max_new_tokens=120,
        temperature=0.2,
        do_sample=False
    )

    llm = HuggingFacePipeline(pipeline=hf_pipeline)

    return retriever, llm