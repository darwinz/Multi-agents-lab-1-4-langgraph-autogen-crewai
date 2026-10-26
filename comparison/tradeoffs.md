# Lab 1.4 – Framework Comparison: Tradeoffs

Fill in each section after running all three framework implementations
with `python compare.py`. Your answers should be specific and grounded
in what you observed, not generic descriptions from documentation.

---

## 1. Quantitative Results

Copy the comparison table printed by `compare.py` here.

```
===========================================================================
  Framework    Status          Steps     Prompt   Completion      Total
---------------------------------------------------------------------------
  LangGraph    ok                  4       3473          550       4023
  AutoGen      ok                  1       3425          559       3984
  CrewAI       ok                  1       2667          941       3608
===========================================================================
```

Model: `gpt-4.1-mini` via Helicone → OpenRouter, temperature default
(LangGraph) / 0 (AutoGen, CrewAI), `max_tokens=2048` per reply in all three.

**The Steps column is not comparable across frameworks.** LangGraph records
one step per LLM call (in `agent_node`). The provided AutoGen and CrewAI
helpers read a single aggregated usage figure after the run, so they always
record one step. Counting from the run log instead:

| Framework | LLM calls (from log) | Tool calls |
|-----------|----------------------|------------|
| LangGraph | 4 | 3 (write, write, exec) |
| AutoGen   | 4 | 3 (write, write, exec) |
| CrewAI    | ~6 | 3 (write, write, exec) |

These are single runs. A standalone LangGraph run earlier the same day took
5 calls and 5,553 tokens, so run-to-run variance is about ±30%, which is larger
than the gaps between frameworks in the table.

---

## 2. Developer Experience

### LangGraph
- How much boilerplate did you write?
- How easy was it to understand what the agent was doing at each step?
- What was the hardest part to implement?

*Answer:* About 30 lines, all in `agent_node`: build the model, bind tools,
call it, record usage, set `done`. The graph, routing and tool node were
provided, but they are plain functions I could read top to bottom, so I always
knew what ran next and why. It was the only framework where I recorded usage
per call myself, which is why its step count is the only accurate one.
Hardest part: the README hint `ChatOpenAI(openai_client=...)` doesn't exist in
langchain-openai 1.x. The constructor only warns that it moved the kwarg
into `model_kwargs`, builds its own client from `OPENAI_API_KEY` and
`OPENAI_BASE_URL`, and would then send `openai_client` as a request
parameter. The injected client has to go in `root_client` and
`client` (`root_client.chat.completions`) instead.

### AutoGen
- How much boilerplate did you write?
- How easy was it to understand what the agent was doing at each step?
- What was the hardest part to implement?

*Answer:* About 25 lines. Every tool is registered twice: `register_for_llm`
on the assistant (so the model sees the schema) and `register_for_execution`
on the proxy (so something runs it). That split is the core AutoGen idea: the
assistant only proposes calls, and the proxy executes them and replies with
the result as the next chat message. The console transcript was the easiest
of the three to follow, because every tool call and result is printed as a
message between the two agents. Hardest part: packaging, not code.
`requirements.txt` installs both `pyautogen` (now 0.10, Microsoft's shim over
the new `autogen-agentchat` API) and `autogen` (AG2 0.14, which keeps the
classic `AssistantAgent`/`register_for_llm` API this lab uses). `import autogen`
resolves to AG2. Also, the provided `_record_usage` crashed: AG2 puts a
`"total_cost": 0` number next to the per-model dicts, and the helper called
`.get()` on it.

### CrewAI
- How much boilerplate did you write?
- How easy was it to understand what the agent was doing at each step?
- What was the hardest part to implement?

*Answer:* About 45 lines, but mostly declarative: three `@tool` wrappers whose
docstrings become the tool descriptions, then role, goal and backstory strings
for the agent, and a task with an `expected_output`. I never wrote or saw the
loop. The verbose panels show tool starts and completions, but not the prompts
CrewAI builds or why it made a given call. Hardest part: CrewAI 1.x doesn't use
a LangChain `ChatOpenAI` directly. It copies the model name and a few fields
into its own `LLM` and silently drops `default_headers`, so the
`Helicone-Auth` header in the starter was never sent. I replaced `_make_llm`
with a `crewai.LLM` that sets the base URL, key and header explicitly.

---

## 3. Token Efficiency

Which framework used the fewest tokens for the same task? Why do you
think that is? (Hint: look at the system prompts each framework injects
automatically vs. what you wrote yourself.)

*Answer:* CrewAI reported the fewest total tokens (3,608), but I don't trust
that ranking. LangGraph (4,023) and AutoGen (3,984) are nearly identical
because they do the same thing: native OpenAI function calling, my system
prompt, four calls, and the full history resent each time. CrewAI had the
most completion tokens (941 vs ~550) because its final answer repeats the
whole module source to satisfy `expected_output`, while the other two end
with the single word `DONE`. CrewAI's prompt-token figure is the lowest even
though it made the most calls and injects its own role/goal/backstory
framing. It also logged "Using an approximate input token count for
gpt-4.1-mini" on every call, so its usage figure may be partly estimated.
The Helicone dashboard has the provider-reported counts and would settle it.
With ±30% run-to-run variance, these three totals are effectively a tie.

---

## 4. Control vs. Abstraction

Rank the three frameworks from most explicit control (you decide
everything) to highest abstraction (the framework decides most things).
Justify your ranking with one concrete example from your implementation.

*Answer:* **LangGraph → AutoGen → CrewAI.**

- **LangGraph:** I decide when the loop ends. `agent_node` sets `done` when
  the reply contains "DONE" and increments `step_count`, and `should_continue`
  reads both. Nothing happens that isn't in an edge I can point to.
- **AutoGen:** I supply the termination rule (`is_termination_msg` on the
  proxy) and a turn cap. The framework owns turn-taking, decides who speaks
  next, and formats tool results as chat messages.
- **CrewAI:** I supply `max_iter=15` and a description. The prompt format, the
  tool-call loop, when the agent stops, and how the final answer is phrased
  are all internal. Even the LLM object I passed in was rebuilt into a
  different one, and part of my configuration was lost in that conversion.

---

## 5. When would you use each?

Complete the table with one concrete use-case per framework.

| Framework | Best suited for |
|-----------|-----------------|
| LangGraph | A coding agent with hard guarantees (iteration caps, "only finish after tests pass", human approval before writes), where every transition has to be inspectable and testable. This is the capstone Coder agent. |
| AutoGen   | Two or more agents talking a problem through, e.g. a Coder proposing patches and a Reviewer agent critiquing them, where the conversation itself is the workflow and a readable transcript is valuable. |
| CrewAI    | Quickly prototyping a role-based pipeline (researcher → writer → editor) where the steps are coarse, the exact prompts matter less than the roles, and fine-grained control over the loop isn't needed. |

---

## 6. Surprises

What was the most surprising difference you found between the three
implementations? (One paragraph.)

*Answer:* How much of the "same task" was not actually the same. The
provided `exec_python` runs `python -c` in the current directory, while
`write_file` writes into a temp workspace. None of the three agents could
import the module they had just written until I told each one the absolute
workspace path. Each framework then needed that note in a different place:
the system prompt for LangGraph and AutoGen, the backstory for CrewAI. Each
framework also measures its own work differently. LangGraph's tracker counts
LLM calls, AutoGen's and CrewAI's count one aggregate, and CrewAI estimates
input tokens. So the comparison table looks precise but compares three
different things. The quiet failures surprised me most. CrewAI dropped my
Helicone header without a warning, and langchain-openai turned a
nonexistent `openai_client` kwarg into a one-line warning plus a separately
built client. The more a framework abstracts, the more of these I had to
find by reading its source.
