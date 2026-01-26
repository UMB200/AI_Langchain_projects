from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain.agents import create_agent


from prompt import REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS
from schemas import AgentResponse

load_dotenv()

tavily_tools = [TavilySearch()]
llm = ChatOpenAI(model="gpt-4o")


agent = create_agent(
    model=llm, tools=tavily_tools, response_format=AgentResponse
)

def main():
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "search for 3 job postings for a IT Project Manager using AI in DC area on LinkedIn and list their details",
                }
            ]
        }
    )
    structured_result = result.get("structured_response", None)
    print(structured_result if structured_result is not None else result)

if __name__ == "__main__":
    main()
