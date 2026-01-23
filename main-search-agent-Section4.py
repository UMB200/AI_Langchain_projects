from dotenv import load_dotenv
from langsmith import Client
from langchain_classic import hub
from langchain_classic.agents import AgentExecutor
from langchain_classic.agents.react.agent import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

load_dotenv()

tavily_tools = [TavilySearch()]
llm = ChatOpenAI(model="gpt-4")
react_prompt = hub.pull("hwchase17/react")
agent = create_react_agent(
    llm=llm, tools=tavily_tools, prompt=react_prompt)

agent_executor = AgentExecutor(agent=agent, tools=tavily_tools, verbose=True)

def main():
    result = agent_executor.invoke(
        input={
        "input": "search for 3 job postings for a Sr. IT PM using AI in DC area on LinkedIn and list their details"
    })
    print(result)


if __name__ == "__main__":
    main()
