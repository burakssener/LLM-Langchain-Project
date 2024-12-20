import os
import pickle

import langchain
import streamlit as st
from dotenv import load_dotenv
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.document_loaders import (
    SeleniumURLLoader,
    TextLoader,
    UnstructuredURLLoader,
)
from langchain_openai import OpenAI, OpenAIEmbeddings

load_dotenv()


st.title("Gpt: It's starting...")
st.sidebar.title("Articles/Books")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)


clicked_urls = st.sidebar.button("Submit")

placeholder = st.empty()

file_path = "vector_index_data.pkl"

if clicked_urls:
    loader = SeleniumURLLoader(urls=urls)
    placeholder.text("Loading Data...")
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", "."],
        chunk_size=1000,
    )
    placeholder.text("Text Splitter works...")
    docs = text_splitter.split_documents(data)
    embeddings = OpenAIEmbeddings()
    vectorestore_openai = FAISS.from_documents(docs, embeddings)
    placeholder.text("Embedding Vector started...")

    with open(file_path, "wb") as f:
        pickle.dump(docs, f, protocol=pickle.HIGHEST_PROTOCOL)
        print("success")


query = placeholder.text_input("ASK: ")
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            docs = pickle.load(f)
        embeddings = OpenAIEmbeddings()
        vectorIndex = FAISS.from_documents(docs, embeddings)
    llm = OpenAI(
        model_name="gpt-3.5-turbo-instruct",
        temperature=0.8,
    )

    chain = RetrievalQAWithSourcesChain.from_llm(
        llm=llm, retriever=vectorIndex.as_retriever()
    )
    result = chain({"question": query}, return_only_outputs=True)
    st.header("Answer")
    st.write(result["answer"])

    sources = result.get("sources", "")
    if sources:
        st.subheader("Sources:")
        sources_list = sources.split("\n")  # Split the sources by newline
        for source in sources_list:
            st.write(source)
