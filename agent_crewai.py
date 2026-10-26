"""
agent_crewai.py  –  Lab 1.4 skeleton: CrewAI implementation.

Your job
--------
Implement the three functions marked TODO so that a single CrewAI Agent
and Task solves TASK_DESCRIPTION using the three tools from
tools/code_tools.py.

Do NOT change:
  - The HeliconeOpenAI class
  - The function signatures of `make_tools`, `make_agent`, `make_task`
  - The `run()` function structure after the TODO comment
  - Any imports from llm_client, tools, or task

Run:
  python agent_crewai.py
"""

from __future__ import annotations

import os
import time

from crewai import LLM, Agent, Crew, Task
from crewai.tools import tool as crewai_tool
from langchain_openai import ChatOpenAI

from llm_client import DEFAULT_MODEL
from task import TASK_DESCRIPTION
from tools.code_tools import exec_python, read_file, write_file
from tools.code_tools import WORKSPACE
from tools.token_counter import UsageTracker
from dotenv import load_dotenv

load_dotenv(override=True)

tracker = UsageTracker(framework="CrewAI")

# ---------------------------------------------------------------------------
# LLM wrapper – CrewAI accepts a LangChain-compatible ChatOpenAI object.
# ---------------------------------------------------------------------------


MAX_RESPONSE_TOKENS = 2048


def _make_llm() -> LLM:
    # CrewAI 1.x converts a LangChain ChatOpenAI into its own LLM and drops
    # default_headers, so Helicone-Auth never got sent. Build CrewAI's LLM directly.
    _HELICONE_BASE = os.getenv("HELICONE_BASE_URL")
    _OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    _HELICONE_API_KEY = os.getenv("HELICONE_API_KEY")

    return LLM(
        model=DEFAULT_MODEL,
        base_url=_HELICONE_BASE,
        api_key=_OPENROUTER_API_KEY,
        default_headers={"Helicone-Auth": f"Bearer {_HELICONE_API_KEY}"},
        temperature=0,
        # Caps each reply so OpenRouter doesn't reserve the model's full output budget
        max_tokens=MAX_RESPONSE_TOKENS,
    )


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

def make_tools() -> list:
    """Wrap the three code tools as CrewAI tool objects and return them.

    TODO: Implement this function.

    Steps:
    1. Use the @crewai_tool decorator (already imported as `crewai_tool`)
       to create three tool functions:
         - crewai_read_file(path: str) -> str
         - crewai_write_file(path: str, content: str) -> str
         - crewai_exec_python(code: str) -> str
       Each should delegate directly to the matching function imported from
       tools.code_tools.
    2. Return a list containing the three decorated tools.

    Hints:
    - @crewai_tool decorates a plain function; the docstring becomes the
      tool description shown to the LLM.
    - CrewAI infers parameter names and types from the function signature.
    - Example:
        @crewai_tool
        def my_tool(x: str) -> str:
            \"\"\"One-line description.\"\"\"
            return some_function(x)

    """

    @crewai_tool
    def crewai_read_file(path: str) -> str:
        """Read a file from the sandbox workspace, given its path relative to the workspace."""
        return read_file(path)

    @crewai_tool
    def crewai_write_file(path: str, content: str) -> str:
        """Write text content to a file in the sandbox workspace, given its path relative to the workspace."""
        return write_file(path, content)

    @crewai_tool
    def crewai_exec_python(code: str) -> str:
        """Execute a Python code snippet in a subprocess and return its stdout + stderr."""
        return exec_python(code)

    return [crewai_read_file, crewai_write_file, crewai_exec_python]


def make_agent(tools: list) -> Agent:
    """Create and return a CrewAI Agent for the coding task.

    TODO: Implement this function.

    Steps:
    1. Create an Agent with:
       - role="Coder Agent"
       - goal="Complete the assigned coding task correctly and efficiently."
       - backstory: A short sentence describing the agent as an expert Python
         developer who always writes clean, tested code.
       - tools=tools  (the list returned by make_tools())
       - llm=_make_llm()
       - verbose=True
       - max_iter=15
    2. Return the agent.

    """
    return Agent(
        role="Coder Agent",
        goal="Complete the assigned coding task correctly and efficiently.",
        backstory=(
            "You are an expert Python developer who always writes clean, tested code. "
            "Your file tools take paths relative to the workspace directory "
            f"{WORKSPACE}. Python you execute runs from a different directory, so it "
            "must use that absolute path, e.g. "
            f"`import os, sys; os.chdir({str(WORKSPACE)!r}); sys.path.insert(0, {str(WORKSPACE)!r})`."
        ),
        tools=tools,
        llm=_make_llm(),
        verbose=True,
        max_iter=15,
    )


def make_task(agent: Agent) -> Task:
    """Create and return a Task that assigns TASK_DESCRIPTION to the agent.

    TODO: Implement this function.

    Steps:
    1. Create a Task with:
       - description=TASK_DESCRIPTION
       - expected_output: A short string such as
         "The word_freq.py module is written, tested, and produces correct output."
       - agent=agent
    2. Return the task.

    """
    return Task(
        description=TASK_DESCRIPTION,
        expected_output="The word_freq.py module is written, tested, and produces correct output.",
        agent=agent,
    )


# ---------------------------------------------------------------------------
# Token tracking helper
# ---------------------------------------------------------------------------

def _record_usage(crew: Crew) -> None:
    """Extract token usage from a Crew's usage_metrics and record it.

    This function is already implemented for you.
    """
    metrics = getattr(crew, "usage_metrics", None)
    if metrics:
        prompt = getattr(metrics, "prompt_tokens", 0) or 0
        completion = getattr(metrics, "completion_tokens", 0) or 0
        tracker.record_step(prompt_tokens=prompt, completion_tokens=completion)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run() -> str:
    """Assemble and run the crew; return the task output."""
    tools = make_tools()
    agent = make_agent(tools)
    task = make_task(agent)

    # TODO: Create a Crew with:
    #   - agents=[agent]
    #   - tasks=[task]
    #   - verbose=True
    # Call crew.kickoff() and store the result in `output`.
    # Then call _record_usage(crew).
    crew = Crew(agents=[agent], tasks=[task], verbose=True)
    output = crew.kickoff()
    _record_usage(crew)

    tracker.print_summary()
    return str(output)


if __name__ == "__main__":
    result = run()
    print("\n--- Agent finished ---")
    print(result)
