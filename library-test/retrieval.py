import os
import json

with open("../key.json", "r", encoding="utf-8") as file:
    key_data = json.load(file)

os.environ["OPENAI_API_KEY"] = key_data["openai_key"]

from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

embeddings = OpenAIEmbeddings()
loader = PyPDFLoader("../sample/The_Adventures_of_Tom_Sawyer.pdf")
document = loader.load()

db = FAISS.from_documents(document, embeddings)

llm = ChatOpenAI(
    temperature=0,
    model_name="gpt-3.5-turbo",
)

retriever = db.as_retriever()
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

query = "주인공의 이름을 알려줘"
result = qa({"query": query})
print(result["result"])
