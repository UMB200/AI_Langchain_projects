from dotenv import load_dotenv
import os

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

load_dotenv()
langsmith_api_key = os.getenv("LANGSMITH_API_KEY")

def main():
    # print(os.getenv("OPENAI_API_KEY")) the same as os.environ.get("OPENAI_API_KEY") but in the course used os.environ.get("OPENAI_API_KEY")
    #print(os.environ.get("OPENAI_API_KEY"))
    print("Hello from langchain-course!")
    information = """
    LangChain is a software framework that helps facilitate the integration of large language models (LLMs) into applications. 
    As a language model integration framework, LangChain's use-cases largely overlap with those of language models in general, 
    including document analysis and summarization, chatbots, and code analysis.
    """

    summary_template = """
    Given the following information: {information} about LangChain, I want you to create:
    1. Short summary
    2. 2 interesting facts about LangChain
    """

    summary_prompt_template = PromptTemplate(
        template=summary_template,
        input_variables=["information"]
    )

    llm = ChatOpenAI(model="gpt-5", temperature=0)
    #llm = ChatOllama(model="gemma3:270m", temperature=0)
    chain = summary_prompt_template | llm
    response = chain.invoke({"information": information})
    print(response.content)

if __name__ == "__main__":
    main()
