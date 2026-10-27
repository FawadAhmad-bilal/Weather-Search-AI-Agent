"""ReAct agent in plain Python (no LangChain).

Loop: Question -> Thought -> Action -> Observation -> ... -> Final Answer
"""
import os
import re

import requests
from ddgs import DDGS
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
weather_key = os.getenv("WEATHER_API_KEY")
client = Groq()  # reads GROQ_API_KEY from .env


def search(query: str) -> str:
    """Search the web with DuckDuckGo (no API key needed)."""
    results = DDGS().text(query, max_results=3)
    return " ".join(r["body"] for r in results)


def weather_check(city_name: str) -> dict:
    """Get the current weather of a city from OpenWeatherMap."""
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"q": city_name, "appid": weather_key, "units": "metric"}
    return requests.get(url, params=params, timeout=10).json()


TOOLS = {"search": search, "weather_check": weather_check}

SYSTEM = """Answer the question as best you can. You have these tools:
search: searches the web. Input is a search query.
weather_check: current weather of a city. Input is a city name, like 'Mingora,PK'.

Use exactly this format:
Question: the input question
Thought: think about what to do
Action: search or weather_check
Action Input: the input for the action
Observation: the result of the action
... (repeat Thought/Action/Action Input/Observation if needed)
Thought: I now know the final answer
Final Answer: the final answer to the question
"""


def run_agent(question: str, max_steps: int = 6) -> str:
    scratchpad = f"Question: {question}\nThought:"

    for _ in range(max_steps):  # limit steps so it can never loop forever
        reply = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": scratchpad},
            ],
            stop=["Observation:"],  # we run the tool, not the model
        ).choices[0].message.content

        scratchpad += reply
        if "Final Answer:" in reply:
            return reply.split("Final Answer:")[-1].strip()

        action = re.search(r"Action:\s*(.+)", reply).group(1).strip()
        action_input = re.search(r"Action Input:\s*(.+)", reply).group(1).strip()
        observation = TOOLS[action](action_input)
        scratchpad += f"\nObservation: {observation}\nThought:"

    return "Stopped: maximum steps reached."


if __name__ == "__main__":
    print(run_agent("where is swat in pakistan and what is its weather"))
