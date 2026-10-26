"""
agent_autogen.py  –  Lab 1.4 skeleton: AutoGen implementation.

Your job
--------
Implement the two functions marked TODO so that an AutoGen
AssistantAgent + UserProxyAgent pair solves TASK_DESCRIPTION using
the three tools from tools/code_tools.py.

Do NOT change:
  - The function signatures of `make_assistant` and `make_proxy`
  - The `run()` function structure after the TODO comment
  - Any imports from llm_client, tools, or task

Run:
  python agent_autogen.py
"""

from __future__ import annotations

import os

import autogen
from autogen import AssistantAgent, UserProxyAgent

from llm_client import DEFAULT_MODEL
from task import TASK_DESCRIPTION
from tools.code_tools import dispatch, exec_python, read_file, write_file
from tools.code_tools import (
    EXEC_PYTHON_SCHEMA,
    READ_FILE_SCHEMA,
    WORKSPACE,
    WRITE_FILE_SCHEMA,
)
from tools.token_counter import UsageTracker
from dotenv import load_dotenv

load_dotenv(override=True)

tracker = UsageTracker(framework="AutoGen")

# ---------------------------------------------------------------------------
# LLM config – AutoGen uses its own config dict format.
# We point it at the Helicone proxy by overriding the base_url.
# ---------------------------------------------------------------------------

def _llm_config() -> dict:
    """Return an AutoGen llm_config dict that routes through Helicone.

    This function is already implemented for you. Study it to understand
    how AutoGen's llm_config dict maps onto the OpenAI client.
    """
    _HELICONE_BASE = os.getenv("HELICONE_BASE_URL")
    _OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    _HELICONE_API_KEY = os.getenv("HELICONE_API_KEY")
    return {
        "config_list": [
            {
                "model": DEFAULT_MODEL,
                "api_key": _OPENROUTER_API_KEY,
                "base_url": _HELICONE_BASE,
                "api_type": "openai",
                "default_headers": {
                    "Helicone-Auth": f"Bearer {_HELICONE_API_KEY}",
                },
            }
        ],
        "temperature": 0,
    }


# ---------------------------------------------------------------------------
# Tool functions registered with AutoGen
# ---------------------------------------------------------------------------
# AutoGen's function-calling integration requires plain Python callables.
# The three tool functions (read_file, write_file, exec_python) are imported
# directly from tools.code_tools above.

MAX_RESPONSE_TOKENS = 2048

SYSTEM_PROMPT = f"""You are a Coder Agent. Use the available tools to complete
the user's task step by step. When the task is fully done, output the single
word DONE on its own line and nothing else. Do NOT use code blocks for tool
calls – call the registered functions directly.

read_file and write_file take paths relative to the workspace directory
{WORKSPACE}. exec_python runs from a different directory, so code you execute
must use that absolute path, e.g. `import os, sys; os.chdir({str(WORKSPACE)!r});
sys.path.insert(0, {str(WORKSPACE)!r})`."""

_TOOLS = [
    (READ_FILE_SCHEMA, read_file),
    (WRITE_FILE_SCHEMA, write_file),
    (EXEC_PYTHON_SCHEMA, exec_python),
]


# ---------------------------------------------------------------------------
# Agent factory functions
# ---------------------------------------------------------------------------

def make_assistant() -> AssistantAgent:
    """Create and return an AssistantAgent with the three tools registered.

    TODO: Implement this function.

    Steps:
    1. Create an AssistantAgent with:
       - name="coder_agent"
       - system_message=SYSTEM_PROMPT
       - llm_config=_llm_config()
    2. Register each tool for LLM use with:
         agent.register_for_llm(name="...", description="...")(function)
       Register: read_file, write_file, exec_python.
       Hint: Use the descriptions from tools/code_tools.py (READ_FILE_SCHEMA,
             WRITE_FILE_SCHEMA, EXEC_PYTHON_SCHEMA).
    3. Return the agent.

    """
    llm_config = _llm_config()
    # Caps each reply so OpenRouter doesn't reserve the model's full output budget
    llm_config["max_tokens"] = MAX_RESPONSE_TOKENS
    agent = AssistantAgent(
        name="coder_agent",
        system_message=SYSTEM_PROMPT,
        llm_config=llm_config,
    )
    for schema, function in _TOOLS:
        agent.register_for_llm(name=schema["name"], description=schema["description"])(function)
    return agent


def make_proxy() -> UserProxyAgent:
    """Create and return a UserProxyAgent that executes tool calls.

    TODO: Implement this function.

    Steps:
    1. Create a UserProxyAgent with:
       - name="user_proxy"
       - human_input_mode="NEVER"
       - max_consecutive_auto_reply=15
       - is_termination_msg: a lambda that returns True when the message
         content contains "DONE"
       - code_execution_config=False  (we use our own exec_python tool)
    2. Register each tool for execution with:
         proxy.register_for_execution(name="...")(function)
       Register the same three functions as in make_assistant().
    3. Return the proxy.

    """
    proxy = UserProxyAgent(
        name="user_proxy",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=15,
        is_termination_msg=lambda msg: "DONE" in (msg.get("content") or ""),
        code_execution_config=False,
    )
    for schema, function in _TOOLS:
        proxy.register_for_execution(name=schema["name"])(function)
    return proxy


# ---------------------------------------------------------------------------
# Token tracking helper
# ---------------------------------------------------------------------------

def _record_usage(chat_result: autogen.ChatResult) -> None:
    """Extract token usage from a ChatResult and record each exchange.

    This function is already implemented for you.
    AutoGen accumulates usage in chat_result.cost – we split it into
    per-step records using the message count as a proxy for step count.
    """
    usage = getattr(chat_result, "cost", {})
    # usage dict shape: {"usage_including_cached_inference": {model: {prompt_tokens, ...}}}
    for model_data in usage.get("usage_including_cached_inference", {}).values():
        # The dict also holds a "total_cost" number alongside the per-model dicts
        if not isinstance(model_data, dict):
            continue
        prompt = model_data.get("prompt_tokens", 0)
        completion = model_data.get("completion_tokens", 0)
        if prompt or completion:
            tracker.record_step(prompt_tokens=prompt, completion_tokens=completion)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run() -> str:
    """Initiate the two-agent chat and return the final assistant message."""
    assistant = make_assistant()
    proxy = make_proxy()

    # TODO: Call proxy.initiate_chat() with:
    #   - recipient=assistant
    #   - message=TASK_DESCRIPTION
    #   - max_turns=15
    # Store the result in `result`, then call _record_usage(result).
    result = proxy.initiate_chat(recipient=assistant, message=TASK_DESCRIPTION, max_turns=15)
    _record_usage(result)

    tracker.print_summary()

    # Return the last message from the assistant
    last = result.chat_history[-1]["content"] if result.chat_history else ""
    return last


if __name__ == "__main__":
    output = run()
    print("\n--- Agent finished ---")
    print(output)
