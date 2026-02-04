from dotenv import load_dotenv
from typing import List
from langchain_openai import ChatOpenAI
from callbacks import AgentCallbackHandler
from langchain_core.messages import HumanMessage, ToolMessage
from langchain.tools import tool, BaseTool

load_dotenv()  # Load environment variables from .env file


@tool
def get_text_length(text: str) -> int:
    """Returns the length of a text by characters"""
    print(f"get_text_length enter with {text=}")
    text = text.strip("'\n").strip(
        '"'
    )  # stripping away non alphabetic characters just in case

def find_tool_by_name(tools: List[BaseTool], tool_name: str) ->BaseTool:
    for tool in tools:
        if tool.name == tool_name:
            return tool
    raise ValueError(f"Tool with name {tool_name} not found.")

if __name__ == "__main__":
    print("This is the LangChain Tools (.bind_tools).")
    langchain_tools = [get_text_length]

    llm_chatgpt = ChatOpenAI(temperature=0, callbacks=[AgentCallbackHandler()])
    llm_with_tools = llm_chatgpt.bind_tools(langchain_tools)
    messages = [HumanMessage(content="What is the length of the text LangChain")]
    
    while True:
        ai_message = llm_with_tools.invoke(messages)

        tool_calls = getattr(ai_message, "tool_calls", None) or []
        if len(tool_calls) > 0:
            messages.append(ai_message)
            for tool_call in tool_calls:
                tool_name = tool_call.get("name")
                tool_args = tool_call.get("args", {})
                tool_call_id = tool_call.get("id")

                tool_to_call = find_tool_by_name(langchain_tools, tool_name)
                observation = tool_to_call.invoke(tool_args)
                print(f"Observation: {observation}")
                messages.append(ToolMessage(content=str(observation), tool_call_id=tool_call_id))
            continue
        print(f"Final Answer: {ai_message.content}")
        break