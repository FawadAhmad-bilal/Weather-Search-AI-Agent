from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_classic.agents import create_react_agent,AgentExecutor
from langchain_classic import hub
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
import requests
import json
from dotenv import load_dotenv
import os
load_dotenv()
api_key=os.getenv("WEATHER_API_KEY")
search=DuckDuckGoSearchRun()

@tool
def weather_check(city_name:str)->float:
    '''you are the weather assistent and have to tell about the weather the user ask about'''
    url=f'https://api.openweathermap.org/data/2.5/weather'
    response=requests.get(url,params=({'q':city_name,'appid':api_key,'units':'metric'}),timeout=10)
    return response.json()

llm=ChatGroq(model='openai/gpt-oss-120b')

# prompt=hub.pull('hwchase17/react')
template="""Answer the following question if you can.you have the following tools access
# {tools}
use the following format:
Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}
"""
prompt=PromptTemplate.from_template(template)

agent=create_react_agent(
    llm=llm,
    prompt=prompt,
    tools=[search,weather_check]
)

agent_executor=AgentExecutor(
    agent=agent,
    tools=[search,weather_check],
    verbose=True,
    handle_parsing_errors=True
)

response=agent_executor.invoke({'input':'where is swat in pakistan and whats its weather'})
print(response)
print(response['output'])
