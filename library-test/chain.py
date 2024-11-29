from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_openai.chat_models import ChatOpenAI
import os
import json

with open("../key.json", "r", encoding="utf-8") as file:
    key_data = json.load(file)
os.environ["OPENAI_API_KEY"] = key_data["openai_key"]

llm = ChatOpenAI(temperature=0, model_name="gpt-4")

# prompt = PromptTemplate(
#     input_variables=["country"],
#     template="{country}의 수도는 어디야?",
# )

# chain = LLMChain(llm=llm, prompt=prompt)
# res = chain.run("미국")
# print(res)

prompt1 = PromptTemplate(
    input_variables=["sentence"],
    template="다음 문장을 한글로 번역하세요.\n\n{sentence}",
)

chain1 = LLMChain(llm=llm, prompt=prompt1, output_key="translation")

prompt2 = PromptTemplate.from_template(
    "다음 문장을 한 문장으로 요약하세요.\n\n{translation}"
)

chain2 = LLMChain(llm=llm, prompt=prompt2, output_key="summary")

from langchain.chains import SequentialChain

all_chain = SequentialChain(
    chains=[chain1, chain2],
    input_variables=["sentence"],
    output_variables=["translation", "summary"],
)

sentence = """
One limitation of LLMs is their lack of contextual information (e.g., access to some specific docuements or emails). You can combat this by giving LLMs access to the specific external data.
For this, you first need to load the external data with a document loader.
LangChain provides a variety of loaders for different types of documents ranging from PDFs and emails to websites and YouTube videos.
"""

res = all_chain(sentence)
print(res)
