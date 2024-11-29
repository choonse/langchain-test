import os
import json

with open("key.json", "r", encoding="utf-8") as file:
    key_data = json.load(file)
os.environ["OPENAI_API_KEY"] = key_data["openai_key"]

from langchain_community.chat_models import ChatOpenAI

llm = ChatOpenAI(temperature=0, model_name="gpt-4")

from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType

tools = load_tools(["wikipedia", "llm-math"], llm=llm)

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    description="계산이 필요할 때",
    verbose=True,
)

res = agent.run("Stevie Wonder의 나이와 태어난 년도")
print(res)
