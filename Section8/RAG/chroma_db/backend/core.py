import os
from typing import Any, Dict
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import ToolMessage
from langchain.tools import tool
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vector_store = PineconeVectorStore(index_name="langchain-docs-2025", embedding=embeddings)

model = init_chat_model("gpt-5.2", model_provider="openai")

@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve relevant documentation to help answer user queries about LangChain"""
    # Retrieve top 4 most similar documents
    retrieved_most4_docs = vector_store.as_retriever().invoke(query, k=4)

    #Serialize documents for the model
    serialized_docs = "\n\n".join(
        (f"Source: {doc.metadata.get('source', 'Unknown')}\n\nContent: {doc.page_content}")
        for doc in retrieved_most4_docs
    )
    return serialized_docs, retrieved_most4_docs

def run_agent(query: str) -> Dict[str, Any]:
    """
    Run the RAG pipeline to answer a query using retrieved documentation
    Args:
        query: The user's question
    Return:
        Dictionary containing:
            answer: The generated answer
            context: List of retrieved documents
    """

    # Create the agent with retrieval tool
    sys_prompt = (
        "You are a helpful AI assistant that answers questions about LangChain documentation. "
        "You have access to a tool that retrieves relevant documentation. "
        "Use the tool to find relevant information before answering questions. "
        "Always cite the sources you use in your answers. "
        "If you cannot find the answer in the retrieved documentation, say so."
    )

    agent = create_agent(model, tools=[retrieve_context], system_prompt=sys_prompt)

    # Buld message list
    msgs = [{"role": "user", "content": query}]

    # Invoke the agent
    response = agent.invoke({"messages": msgs})

    # Extract the ansnwer frm the last AI message
    answer = response["messages"][-1].content

    # Extract content documents from ToolMessage artificats 
    context_docs = []
    for msg in response["messages"]:
        # Check if this is a ToolMessage with artifact
        if isinstance(msg, ToolMessage) and hasattr(msg, "artifact"):
            # The artifact should contain the list of Document object
            if isinstance(msg.artifact, list):
                context_docs.extend(msg.artifact)
    return{
        "answer": answer,
        "context": context_docs
    }

if __name__ == '__main__':
    result = run_agent(query="what are deep agents")
    print(result)
