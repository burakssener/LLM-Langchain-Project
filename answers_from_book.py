import os

from dotenv import load_dotenv

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY", "sk-*******************************")


from langchain_community.document_loaders import TextLoader

loader = TextLoader("./$100M_Offers.txt", encoding="UTF-8")
data = loader.load()


len(data)


from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

docs = text_splitter.split_documents(data)


# import faiss
import pickle

from langchain.chains import RetrievalQAWithSourcesChain
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

vectorindex_openai = FAISS.from_documents(docs, embeddings)

file_path = "vector_index_data.pkl"
with open(file_path, "wb") as f:
    pickle.dump(docs, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("success")


if os.path.exists(file_path):
    with open(file_path, "rb") as f:
        docs = pickle.load(f)

    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorIndex = FAISS.from_documents(docs, embeddings)


import langchain
from langchain_openai import OpenAI

llm = OpenAI(
    model_name="gpt-3.5-turbo-instruct", openai_api_key=openai_api_key, temperature=0.8
)

chain = RetrievalQAWithSourcesChain.from_llm(
    llm=llm, retriever=vectorIndex.as_retriever()
)

query = "Can you give me a list that contains best practices when creating grand slam offer. to create a grand slam offer. Make it concise and best for prompts. The Instructions you will give will be used in the prompt creation to make Chat Gpt to create best offer for the businesses."


langchain.debug = True

chain({"question": query}, return_only_outputs=True)


"""try:
    with open('vector_index.faiss', 'w') as test_file:
        test_file.write('.')
        print("Write permissions are fine.")
except IOError:
    print("Write permissions are not available.")
    


faiss.write_index(vectorIndex, 'vector_index.faiss')

vectorIndex = faiss.read_index('vector_index.faiss')"""
