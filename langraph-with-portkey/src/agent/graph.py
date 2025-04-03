"""Define a simple chatbot agent.

This agent has responds to a user query with a built-in web search tool
"""


from typing import Annotated
from langchain_openai import ChatOpenAI
from portkey_ai import createHeaders, PORTKEY_GATEWAY_URL, Portkey
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langgraph.types import Command, interrupt
from langchain_core.tools import tool
from typing_extensions import TypedDict
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools.tavily_search import TavilySearchResults
from portkey_ai.langchain import LangchainCallbackHandler





class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

@tool
def human_assistance(query: str) -> str:
    """Request assistance from a human."""
    human_response = interrupt({"query": query})
    return human_response["data"]

portkey_handler = LangchainCallbackHandler(
    api_key="YOUR_PORTKEY_API_KEY",
    metadata={
        "session_id": "session_1",  # Use consistent metadata across your application
        "agent_id": "research_agent_1",  # Specific to the current agent
    }
)

tool = TavilySearchResults(max_results=2)
tools = [tool, human_assistance]

Portkey(api_key="YOUR_PORTKEY_API_KEY", instrumentation=True, base_url="https://api.portkey.ai/v1")
graph_builder = StateGraph(State)

llm = ChatOpenAI(
    api_key="dummy",
    base_url="https://api.portkey.ai/v1", #base url
    default_headers=createHeaders(
        api_key="YOUR_PORTKEY_API_KEY",
        virtual_key="YOUR_VRITUAL_KEY",
        trace_id="example-final" # optional
    )
)

# llm= ChatOpenAI(
#     api_key="dummy"
# )

llm_with_tools = llm.bind_tools(tools)


def chatbot(state: State):
    message = llm_with_tools.invoke(state["messages"])
    assert len(message.tool_calls) <= 1
    return {"messages": [message]}


graph_builder.add_node("chatbot", chatbot)
tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)
graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")
graph = graph_builder.compile()