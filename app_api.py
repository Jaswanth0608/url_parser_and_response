import os
import validators
import pickle
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

app = FastAPI(title="URL Parser and Query API")

DB_PATH = r"E:\Jaswanth\Datasets\Assignments\vaia_climate\index.pkl"
EMBEDDING_MODEL_NAME = "llama3.2"
LLM_MODEL_NAME = "llama3.2"

llm = Ollama(model=LLM_MODEL_NAME)
prompt = ChatPromptTemplate.from_template(
    """
    Answer the following question based on the content provided to you.
    <context>
    {context}
    <context>
    Question: {input}
    """
)

class URLRequest(BaseModel):
    url: str

class QueryRequest(BaseModel):
    query: str

def is_valid_url(url: str) -> bool:
    return validators.url(url)

def get_page_content(url: str):
    try:
        loader = WebBaseLoader(web_path=url)
        text_doc = loader.load()
        return text_doc
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error loading URL content: {str(e)}")

def text_splitter_func(document_text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    doc_text = text_splitter.split_documents(document_text)
    return doc_text

def build_db(doc_text, model_name):
    embeddings = OllamaEmbeddings(model=model_name)
    vector_db = FAISS.from_documents(doc_text, embeddings)
    return vector_db

def load_vector_db():
    if os.path.exists(DB_PATH):
        with open(DB_PATH, "rb") as f:
            vector_db = pickle.load(f)
            return vector_db
    else:
        return None

def save_vector_db(vector_db):
    with open(DB_PATH, "wb") as f:
        pickle.dump(vector_db, f)

@app.post("/url-parser", summary="Parse URL and store embeddings")
def url_parser(request: URLRequest):
    url = request.url
    if not is_valid_url(url):
        raise HTTPException(status_code=400, detail="Invalid URL provided.")

    try:
        document_text = get_page_content(url)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    doc_text = text_splitter_func(document_text)

    try:
        new_vector_db = build_db(doc_text, EMBEDDING_MODEL_NAME)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error building embeddings: {str(e)}")

    existing_vector_db = load_vector_db()

    if existing_vector_db:
        try:
            existing_vector_db.merge_from(new_vector_db)
            vector_db = existing_vector_db
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error merging vector databases: {str(e)}")
    else:
        vector_db = new_vector_db

    try:
        save_vector_db(vector_db)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving vector database: {str(e)}")

    return {"message": "URL processed and embeddings stored successfully.", "Text": document_text}

@app.post("/query", summary="Query the stored data")
def query_data(request: QueryRequest):
    query = request.query
    if not query:
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")

    vector_db = load_vector_db()
    if not vector_db:
        raise HTTPException(status_code=404, detail="Vector database not found. Please process a URL first.")

    try:
        retriever = vector_db.as_retriever()
        chain = create_stuff_documents_chain(llm, prompt)
        response_chain = create_retrieval_chain(retriever, chain)
        response = response_chain.invoke({"input": query})
        answer = response.get('answer', "No answer found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during query processing: {str(e)}")

    return {"answer": answer}
