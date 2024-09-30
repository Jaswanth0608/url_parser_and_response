import os
import validators
import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma, FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import time
import pickle

def is_valid_url(url):
    return validators.url(url)

def get_page_content(url):
    loader = WebBaseLoader(web_path=(url))
    text_doc = loader.load()
    return text_doc

def text_splitter_func(document_text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 200, chunk_overlap = 200)
    doc_text = text_splitter.split_documents(document_text)
    return doc_text

def build_db(model_name):
    embeddings = OllamaEmbeddings(model = model_name)
    vector_db = FAISS.from_documents(doc_text, embeddings)
    return vector_db

st.title("VAIA CLIMATE ASSIGNMENT")
st.sidebar.title("VAIA CLIMATE ASSIGNMENT")

# link = "https://medium.com/@suparnadutta05/table-extraction-drawing-insights-from-table-data-a-survey-report-96c710ebcf55"
db_path = r"E:\Jaswanth\Datasets\Assignments\vaia_climate\index.pkl"
embedding_model_name = "llama3.2"
model_name = "llama3.2"

url = st.sidebar.text_input("Enter URL")
process = st.sidebar.button("Process Text")
place_holder = st.empty()
if process and is_valid_url(url):
    place_holder.text("Extracting Text From URL...✅✅✅")
    place_holder.text("Data Loading...Started...✅✅✅")
    document_text = get_page_content(url)
    place_holder.text("Text Splitter...Started...✅✅✅")
    doc_text = text_splitter_func(document_text)
    place_holder.text("Embedding Vector Started Building...✅✅✅")
    vector_db = build_db(model_name=embedding_model_name)
    time.sleep(2)
    with open(db_path, "wb") as f:
        pickle.dump(vector_db, f)

llm = Ollama(model = model_name)
prompt = ChatPromptTemplate.from_template(
    """
    Answer the following question based on the content provided to you.
    <context>
    {context}
    <context>
    Question: {input}
    """
)

query = place_holder.text_input("Enter question: ")
if query:
    if os.path.exists(db_path):
        with open(db_path, "rb") as f:
            db = pickle.load(f)
            chain = create_stuff_documents_chain(llm, prompt)
            response_chain = create_retrieval_chain(db.as_retriever(), chain)
            response = response_chain.invoke({"input": query})
            st.write(response['answer'])