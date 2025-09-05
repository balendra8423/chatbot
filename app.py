

import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace, HuggingFaceEmbeddings

load_dotenv()

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

def process_input(input_type, input_data):
    if input_type == 'Link':
        docs = []
        for url in input_data:
            loader = WebBaseLoader(url)
            docs.extend(loader.load())
        splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
        chunks = splitter.split_documents(docs)
        texts = [d.page_content for d in chunks]
        vector_store = FAISS.from_texts(texts, embedding)
        return vector_store

    if input_type == 'Text':
        text = input_data or ""
        splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
        texts = splitter.split_text(text)
        vector_store = FAISS.from_texts(texts, embedding)
        return vector_store

    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{input_type.lower()}") as temp_file:
        temp_file.write(input_data.read())
        temp_path = temp_file.name

    if input_type == 'PDF':
        loader = PyPDFLoader(temp_path)
    elif input_type == 'DOCX':
        loader = Docx2txtLoader(temp_path)
    elif input_type == 'TXT':
        loader = TextLoader(temp_path, encoding='utf-8')
    else:
        raise ValueError("Unsupported input type")

    docs = loader.load()
    text = "".join(doc.page_content for doc in docs)
    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    texts = splitter.split_text(text)
    vector_store = FAISS.from_texts(texts, embedding)
    return vector_store

def answer_question(vectorstore, query):
    qa = RetrievalQA.from_chain_type(llm=model, retriever=vectorstore.as_retriever())
    return qa.run(query)

st.title('🧠 Personal Chatbot')

input_type = st.selectbox("Input Type", ['Link', 'PDF', 'Text', 'DOCX', 'TXT'])

input_data = None
if input_type == 'Link':
    number_input = int(st.number_input(min_value=1, max_value=20, step=1, label='Enter number of links'))
    urls = []
    for i in range(number_input):
        url = st.sidebar.text_input(f"URL {i+1}")
        if url:
            urls.append(url)
    input_data = urls
elif input_type == 'Text':
    input_data = st.text_area("Enter the text")
else:
    file_types = {'PDF': 'pdf', 'TXT': 'txt', 'DOCX': 'docx'}
    input_data = st.file_uploader(f"Upload a {input_type} file", type=[file_types.get(input_type, 'txt')])

if st.button("Proceed"):
    if (input_type == 'Link' and input_data) or (input_type == 'Text' and input_data) or (input_type in {'PDF','DOCX','TXT'} and input_data is not None):
        vectorstore = process_input(input_type, input_data)
        st.session_state["vectorstore"] = vectorstore
        st.success("✅ Vector Store created successfully!")
    else:
        st.warning("Please provide valid input.")

if "vectorstore" in st.session_state:
    query = st.text_input("Ask your question")
    if st.button("Submit"):
        answer = answer_question(st.session_state["vectorstore"], query)
        st.write("💬 Answer:")
        st.write(answer)

