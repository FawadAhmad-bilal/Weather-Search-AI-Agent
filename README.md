# Weather & Search AI Agent

A ReAct agent that answers questions by combining live web search
(DuckDuckGo) with real-time weather data (OpenWeatherMap).

## How it works

```
Question -> Thought -> Action (tool) -> Observation -> ... -> Final Answer
```

The LLM (`gpt-oss-120b` on Groq) decides which tool to call, reads the
result, and repeats until it can answer.

## Two implementations

- `agent_from_scratch.py`: the ReAct loop in plain Python
- `agent_langchain.py`: the same agent built with LangChain

## Setup

1. Clone the repository and open the folder:
   `git clone https://github.com/FawadAhmad-bilal/weather-search-agent.git`
2. Create and activate a virtual environment:
   `python -m venv venv`
3. Install the packages:
   `pip install -r requirements.txt`
4. Copy `.env.example` to `.env` and add your own keys
   (Groq: https://console.groq.com, OpenWeatherMap: https://openweathermap.org/api)
5. Run:
   `python agent_langchain.py` or `python agent_from_scratch.py`

## Example

Question: "Where is Swat in Pakistan and what is its weather?"

The agent searched the web for the location. The weather API returned
`city not found` for "Swat, Pakistan", so the agent recovered by itself
and tried "Mingora, Pakistan", which worked.

<!-- Add a terminal screenshot or demo GIF here -->

## What I learned

- How the ReAct pattern (Thought, Action, Observation) works
- Why a tool's docstring matters: the model reads it to choose the tool
- Debugging real errors (blocked public prompts, missing prompt variables)

## Limitations

- DuckDuckGo search is unofficial and can be rate limited
- Uses `langchain_classic` (legacy agents); next step is LangGraph
- This is a learning prototype, not a production system
