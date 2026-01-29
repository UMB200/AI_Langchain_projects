from dotenv import load_dotenv
from langchain_classic.agents.output_parsers import ReActSingleInputOutputParser
from langchain_core.agents import AgentAction, AgentFinish
from langchain_classic.agents.format_scratchpad import format_log_to_str
from typing import Union, List
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool, Tool, render_text_description
from langchain_openai import ChatOpenAI
from callbacks import AgentCallbackHandler

load_dotenv()  # Load environment variables from .env file


@tool
def get_text_length(text: str) -> int:
    """Returns the length of the given text."""
    print(f"get_text_length enter with {text=}")
    text = text.strip("'\n").strip('"')
    return len(text)

def find_tool_by_name(tools: List[Tool], tool_name: str) ->Tool:
    for tool in tools:
        if tool.name == tool_name:
            return tool
    raise ValueError(f"Tool with name {tool_name} not found.")

if __name__ == "__main__":
    print("This is the ReAct LangChain Section_5 module.")
    # print(get_text_length.invoke({"text": "LangChain"}))
    langchain_tools = [get_text_length]

    prompt_template = """
    Answer the following questions as best you can. You have access to the following tools:

    {tools}

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    IMPORTANT: When you have the answer, you must output "Final Answer:" followed by the result.

    Begin!

    Question: {input}
    Thought: {agent_scratchpad}
    """

    prompt_template = PromptTemplate.from_template(template=prompt_template).partial(
        tools=render_text_description(langchain_tools),
        tool_names=", ".join([tool.name for tool in langchain_tools]),
    )

    llm_chatgpt = ChatOpenAI(temperature=0, stop=["\nObservation", "Observation:"], callbacks=[AgentCallbackHandler()])
    intermediate_steps = []
    
    agent = ({
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_log_to_str(x["agent_scratchpad"]),} 
        | prompt_template 
        | llm_chatgpt 
        | ReActSingleInputOutputParser())
    
    agent_step: Union[AgentAction, AgentFinish] = agent.invoke(
        {
            "input": "What is the length of the word: LangChain?", 
            "agent_scratchpad": intermediate_steps})
    
    agent_step = ""

    while not isinstance(agent_step, AgentFinish):
        agent_step: Union[AgentAction, AgentFinish] = agent.invoke(
            {
                "input": "What is the length of the word: LangChain?", 
                "agent_scratchpad": intermediate_steps})
    
        print(f"Agent step: {agent_step}")

        if isinstance(agent_step, AgentAction):
            tool_name = agent_step.tool
            tool_found = find_tool_by_name(langchain_tools, tool_name)
            tool_input = agent_step.tool_input
            observation = tool_found.func(str(tool_input))
            print(f"Final answer of observation in AgentAction: {observation=}")
            intermediate_steps.append((agent_step, str(observation)))

    if isinstance(agent_step, AgentFinish):
        print(f"Final answer of of observation in AgentFinish: {agent_step.return_values}") 