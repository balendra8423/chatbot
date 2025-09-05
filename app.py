
                            
import streamlit as st
import tempfile
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_huggingface import HuggingFaceEmbeddings
import os
load_dotenv()
api_key = st.secrets["api_keys"]["Embeddings"]

embedding = HuggingFaceEmbeddings(
    api_key=api_key,
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)

def process_input(input_type, input_data):
    documents = ""
    if input_type == 'Link':
        docs = []
        for url in input_data:
            loader = WebBaseLoader(url)
            docs.extend(loader.load())
        documents = docs
    else:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{input_type.lower()}") as temp_file:
            temp_file.write(input_data.read())
            temp_path = temp_file.name
        if input_type == 'PDF':
            loader = PyPDFLoader(temp_path)
        elif input_type == 'DOCX':
            loader = Docx2txtLoader(temp_path)
        elif input_type == 'TXT':
            loader = TextLoader(temp_path, encoding='utf-8')
        elif input_type == 'Text':
            documents = input_data
        if input_type != 'Text':
            docs = loader.load()
            text = "".join(doc.page_content for doc in docs)
            documents = text
    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    if input_type == 'Link':
        texts = splitter.split_documents(documents)
        texts = [doc.page_content for doc in texts]
    else:
        texts = splitter.split_text(documents)
    vector_store = FAISS.from_texts(texts, embedding)
    return vector_store

def answer_question(vectorstore, query):
    qa = RetrievalQA.from_chain_type(llm=model, retriever=vectorstore.as_retriever())
    return qa.run(query)

st.title('🧠 Personal Chatbot')

input_type = st.selectbox("Input Type", ['Link', 'PDF', 'Text', 'DOCX', 'TXT'])

input_data = None
if input_type == 'Link':
    number_input = st.number_input(min_value=1, max_value=20, step=1, label='Enter number of links')
    input_data = []
    for i in range(number_input):
        url = st.sidebar.text_input(f"URL {i+1}")
        if url:
            input_data.append(url)
elif input_type == 'Text':
    input_data = st.text_area("Enter the text")
else:
    file_types = {'PDF': 'pdf', 'TXT': 'txt', 'DOCX': 'docx'}
    input_data = st.file_uploader(f"Upload a {input_type} file", type=[file_types.get(input_type, 'txt')])

if st.button("Proceed"):
    if input_data:
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





