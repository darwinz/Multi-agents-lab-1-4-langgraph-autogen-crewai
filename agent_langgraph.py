"""
agent_langgraph.py  –  Lab 1.4 skeleton: LangGraph implementation.

Your job
--------
Implement the two functions marked TODO so that a StateGraph agent
solves TASK_DESCRIPTION using the three tools from tools/code_tools.py.

Do NOT change:
  - The AgentState TypedDict definition
  - The function signatures of `tool_node` and `agent_node`
  - The graph wiring in `build_graph()`
  - Any imports from llm_client, tools, or task

Run:
  python agent_langgraph.py
"""

from __future__ import annotations

import json
import time
from typing import Annotated, Literal

from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from llm_client import DEFAULT_MODEL, get_client
from task import TASK_DESCRIPTION
from tools.code_tools import ALL_SCHEMAS, WORKSPACE, dispatch
from tools.token_counter import UsageTracker

# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------
from typing import TypedDict


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    step_count: int
    done: bool


# ---------------------------------------------------------------------------
# Tracker (module-level so both nodes can write to it)
# ---------------------------------------------------------------------------
tracker = UsageTracker(framework="LangGraph")

# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

MAX_STEPS = 15
# Caps each reply so OpenRouter doesn't reserve the model's full output budget
MAX_RESPONSE_TOKENS = 2048

SYSTEM_PROMPT = f"""You are a Coder Agent. Use the available tools to complete
the user's task step by step. When the task is fully done, output the single
word DONE on its own line and nothing else.

read_file and write_file take paths relative to the workspace directory
{WORKSPACE}. exec_python runs from a different directory, so code you execute
must use that absolute path, e.g. `import os, sys; os.chdir({str(WORKSPACE)!r});
sys.path.insert(0, {str(WORKSPACE)!r})`."""


def tool_node(state: AgentState) -> dict:
    """Execute all tool calls requested in the most recent AI message.

    This node is already implemented for you as a reference.
    Study it to understand how to extract and dispatch tool calls.
    """
    last_message: AIMessage = state["messages"][-1]
    tool_results = []

    for tool_call in last_message.tool_calls:
        result = dispatch(tool_call["name"], tool_call["args"])
        tool_results.append(
            ToolMessage(content=result, tool_call_id=tool_call["id"])
        )

    return {"messages": tool_results, "step_count": state["step_count"]}


def agent_node(state: AgentState) -> dict:
    """Call the LLM and return the next message.

    TODO: Implement this function.

    Steps:
    1. Build a ChatOpenAI instance using get_client() and DEFAULT_MODEL.
       Hint: ChatOpenAI accepts an `openai_client` kwarg and a `model` kwarg.
    2. Bind the tools from ALL_SCHEMAS to the model using `.bind_tools()`.
       Hint: ALL_SCHEMAS is a list of dicts; bind_tools accepts a list of
       dicts with keys "name", "description", "parameters".
    3. Add a SystemMessage at the start of the message list if one is not
       already present (check state["messages"][0].type != "system").
    4. Invoke the model with state["messages"].
    5. Record token usage via tracker.record_step().
       Hint: usage data is on response.response_metadata["token_usage"]
             keys: "prompt_tokens", "completion_tokens"
    6. Check if the AI content contains the word "DONE" – if so, set
       done=True in the returned state dict.
    7. Return {"messages": [response], "step_count": ..., "done": ...}.

    """
    from langchain_core.messages import SystemMessage

    openai_client = get_client()
    # langchain-openai 1.x has no `openai_client` kwarg; an injected client goes
    # in `root_client` (the OpenAI instance) and `client` (its chat.completions)
    llm = ChatOpenAI(
        model=DEFAULT_MODEL,
        root_client=openai_client,
        client=openai_client.chat.completions,
        max_tokens=MAX_RESPONSE_TOKENS,
    ).bind_tools(ALL_SCHEMAS)

    messages = state["messages"]
    if messages[0].type != "system":
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages

    start = time.time()
    response = llm.invoke(messages)
    usage = response.response_metadata.get("token_usage", {})
    tracker.record_step(
        prompt_tokens=usage.get("prompt_tokens", 0),
        completion_tokens=usage.get("completion_tokens", 0),
        latency_s=time.time() - start,
    )

    return {
        "messages": [response],
        "step_count": state["step_count"] + 1,
        "done": "DONE" in (response.content or ""),
    }


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def should_continue(state: AgentState) -> Literal["tools", "agent", "__end__"]:
    """Decide the next node after agent_node runs.

    This function is already implemented for you.
    """
    if state["done"] or state["step_count"] >= MAX_STEPS:
        return END
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return "agent"


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    """Wire up the StateGraph.  Do NOT modify this function."""
    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue)
    graph.add_edge("tools", "agent")
    return graph.compile()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run() -> str:
    """Run the agent and return its final message content."""
    from langchain_core.messages import HumanMessage, SystemMessage

    app = build_graph()
    initial_state: AgentState = {
        "messages": [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=TASK_DESCRIPTION),
        ],
        "step_count": 0,
        "done": False,
    }

    final_state = app.invoke(initial_state)
    tracker.print_summary()

    last = final_state["messages"][-1]
    return last.content if hasattr(last, "content") else str(last)


if __name__ == "__main__":
    result = run()
    print("\n--- Agent finished ---")
    print(result)
