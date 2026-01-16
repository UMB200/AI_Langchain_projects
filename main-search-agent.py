from dotenv import load_dotenv
import os
load_dotenv()
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from tavily import TavilyClient
from langchain_tavily import TavilySearch

#tavily_api_key = os.getenv("TAVILY_API_KEY")
# tavily = TavilyClient()

# @tool
# def search(query: str) -> str:
#     """Search the web for information
    
#     Args:
#         query: The query to search for
#     Returns:
#         The search results
#     """
#     print(f"Searching for {query}")
#     return tavily.search(query)

# llm = ChatOpenAI(model="gpt-5")
# tools = [search]
# agent = create_agent(llm, tools)

llm = ChatOpenAI(model="gpt-5")
tools = [TavilySearch()]
agent = create_agent(llm, tools)



def main():
    print("Hello from langchain search-agent!")
    result = agent.invoke({"messages": HumanMessage(
        content="search for 3 job postings for an ai engineer using langchain in the bay area on linkedin and list their details?")})
    print(result)

if __name__ == "__main__":
    main()