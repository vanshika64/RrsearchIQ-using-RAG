import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings


from dotenv import load_dotenv
load_dotenv()

groq_api_key=os.getenv("GROQ_API_KEY")
st.title("Research IQ")
llm=ChatGroq(
    model_name="llama-3.3-70b-versatile",
    api_key=groq_api_key
)

prompt=ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question
    <context>
    {context}
    </context>
    Question:{input}
    """
)
def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings=HuggingFaceEmbeddings()
        st.session_state.loader=PyPDFDirectoryLoader("../papers")
        st.session_state.docs=st.session_state.loader.load()
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
        st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.session_state.vectors=FAISS.from_documents(st.session_state.embeddings)
    
    
    
prompt1=st.text_input("Enter your question from the documents")
if st.button("Documents Embeddings"):
    vector_embedding()
    st.write("Vector Store DB is Ready")
