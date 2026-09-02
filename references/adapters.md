# Framework Adapters

CONTINUUM plugs into agent frameworks without becoming one. The adapter layer
(`src/continuum/adapters/`) is a thin, optional facade over the core
primitives: storage, the append-only event log, the action ledger,
checkpointing, and the recovery engine. Each adapter records progress and side
effects through the ledger and routes external effects through the two-phase
intercept/complete protocol.

The adapters do not replace a framework's own persistence (LangGraph
checkpointers, LangChain memory, OpenAI session state). They add a second,
semantic layer: exactly-once side effects, durable checkpoints, and a recovery
verdict that is safe to act on.

## Installation

Adapters are optional installs so the core stays standard-library-only.

```bash
pip install "continuum-agent[openai]"     # OpenAI Agents SDK
pip install "continuum-agent[langgraph]"  # LangGraph
pip install "continuum-agent[langchain]"  # LangChain (LCEL + create_agent)
```

The `dev` extra installs all of them so mypy can type-check the adapter modules
and the integration tests can run in CI.

## The adapter base

`GenericAgentAdapter` is the shared facade. The framework adapters extend it:

- `capture_state` / `restore_state` - durable semantic checkpoints.
- `intercept_action` - idempotent external side effects via the action ledger.
- `resume` - recovery assessment (`RESUME`, `REQUEST_HUMAN`, `ABORT`, ...).
- `start_run` - creates the run and records `RUN_STARTED` as the log's first event.

`start_run` records `RUN_STARTED` itself. Without that event, projection,
replay, and restore fail with "the log never recorded RUN_STARTED", and a
non-empty log whose first event is not `RUN_STARTED` is refused rather than
silently misordered.

## Generic Python

The in-process facade for hand-rolled loops. It writes trusted
(`Origin.DETERMINISTIC`) state.

```python
from continuum.adapters import GenericAgentAdapter

adapter = GenericAgentAdapter(store)
adapter.start_run(goal="analyze documents", run_id="r1")
```

## OpenAI Agents SDK

Wraps `function_tool` and `RunHooks`. Importing the adapter requires
`openai-agents` to be installed. State is reported with
`Origin.EXTERNAL_AGENT` provenance (see Provenance below).

## LangGraph

Wraps a `StateGraph`. Add `checkpoint_node` as a graph node and wrap tools with
`wrap_tool`.

```python
from continuum.adapters import LangGraphAgentAdapter
from langgraph.graph import StateGraph, START, END

adapter = LangGraphAgentAdapter(store)
adapter.start_run(goal="process orders", run_id="lg1")


@adapter.wrap_tool("notify.customer")
def notify(order_id: str, *, continuum_run_id: str = "") -> str:
    return send_email(order_id)


def checkpoint(state: dict) -> dict:
    return adapter.checkpoint_node(state)


builder = StateGraph(MyState)
builder.add_node("work", work)
builder.add_node("checkpoint", checkpoint)
builder.add_edge(START, "work")
builder.add_edge("work", "checkpoint")
builder.add_edge("checkpoint", END)
graph = builder.compile()
graph.invoke({"continuum_run_id": "lg1", "order_id": "O-1"})
```

`checkpoint_node` reads `continuum_run_id` from state, projects the current
semantic state from the event log (a fresh run with no events falls back to the
state dict; see issue #46), and writes a checkpoint. `wrap_tool` makes the side
effect idempotent across graph invocations. `assess_graph_recovery(run_id)`
returns a `RecoveryDecision` before re-invoking the graph after a crash.

## LangChain

Drops `checkpoint_node` into an LCEL `RunnableLambda` pipeline, and supports the
`langchain.agents.create_agent` tool-calling loop.

### LCEL pipeline

```python
from continuum.adapters import LangChainAgentAdapter
from langchain_core.runnables import RunnableLambda

adapter = LangChainAgentAdapter(store)
adapter.start_run(goal="process orders", run_id="lc1")


@adapter.wrap_tool("notify.customer")
def notify(order_id: str, *, continuum_run_id: str = "") -> str:
    return send_email(order_id)


def work(state: dict) -> dict:
    notify(order_id=state["order_id"], continuum_run_id=state["continuum_run_id"])
    return {**state, "done": True}


chain = RunnableLambda(work) | RunnableLambda(adapter.checkpoint_node)
chain.invoke({"continuum_run_id": "lc1", "order_id": "O-1"})
```

LangChain `RunnableLambda` replaces state rather than merging it (unlike a
LangGraph node), so each step must return the full state dict, including
`continuum_run_id`, for downstream nodes to find it.

### Real agent (create_agent)

Wire a wrapped LangChain `Tool` and a checkpoint callback into a real agent:

```python
from langchain.agents import create_agent
from langchain_core.tools import Tool
from langchain_core.callbacks import BaseCallbackHandler

adapter.start_run(goal="process orders", run_id="lc1")


@adapter.wrap_tool("notify.customer")
def _notify(order_id: str, *, continuum_run_id: str = "") -> str:
    return send_email(order_id)


def notify_tool(order_id: str) -> str:
    return _notify(order_id=order_id, continuum_run_id="lc1")


tool = Tool(name="notify", func=notify_tool, description="Notify a customer")


class Checkpointer(BaseCallbackHandler):
    def on_tool_end(self, output, **kwargs):
        adapter.checkpoint_node({"continuum_run_id": "lc1", "goal": "process orders"})


agent = create_agent(llm, [tool])
agent.invoke(
    {"messages": [("user", "notify the customer")]},
    config={"callbacks": [Checkpointer()]},
)
```

When the agent calls the tool more than once in a run, `wrap_tool` collapses
the calls into a single external side effect, provided the argument-hash key is
stable across calls.

### Explicit idempotency key (real-LLM safe)

Argument-hash dedup assumes identical arguments mean the same operation. An LLM
does not honour that assumption: it can stuff a generated sentence into a
parameter, rename the parameter, or otherwise render the same logical operation
with different argument text between calls. When the argument shape drifts, the
hash differs and dedup silently fails, re-firing the side effect.

Pass an explicit, Stripe-style key to identify the operation by its resource
identity instead of its argument bytes:

```python
@adapter.wrap_tool("notify.customer", key="notify:O-9")
def _notify(order_id: str, *, continuum_run_id: str = "") -> str:
    return send_email(order_id)
```

Or derive it per call with `key_fn` when the identity lives in an argument:

```python
@adapter.wrap_tool(
    "notify.customer",
    key_fn=lambda *a, **k: f"notify:{str(k.get('order_id', '')).strip()}",
)
def _notify(order_id: str, *, continuum_run_id: str = "") -> str:
    return send_email(order_id)
```

Both forward to `ActionLedger.claim(key=...)`, so two calls that share the action
type and key collapse to one external side effect regardless of argument drift.
`key` and `key_fn` are mutually exclusive; `key_fn` is called with the tool's
`(*args, **kwargs)`.

`LangGraphAgentAdapter.wrap_tool` and `OpenAIAgentAdapter.wrap_function_tool`
expose the same `key` / `key_fn` parameters, so the pattern applies uniformly
across all three framework adapters.

## Thin hook adapters

Three further frameworks are covered without a class adapter, by
`src/continuum/adapters/thin.py`. Each one hooks the framework's own tool-call
surface, so agent construction code does not change, and none of them needs a
CONTINUUM extra: install the framework itself and the hook surface is there.

| Framework | Interception surface | Entry point |
|:--|:--|:--|
| CrewAI | global before/after tool-call hooks in `crewai.hooks` | `install_crewai_hooks(storage, run_id)` |
| AutoGen core | `FunctionTool.run_json` wrapped in place | `wrap_autogen_tool(tool, storage, run_id)` |
| Pydantic AI | async Hooks capability | `Agent(capabilities=[wrap_pydantic_ai_hooks(storage, run_id)])` |

```python
from continuum.adapters.thin import (
    install_crewai_hooks,
    wrap_autogen_tool,
    wrap_pydantic_ai_hooks,
)

uninstall = install_crewai_hooks(store, "run_1")   # returns the uninstaller
wrap_autogen_tool(tool, store, "run_1")            # same instance, run_json replaced
hooks = wrap_pydantic_ai_hooks(store, "run_1")     # Agent(capabilities=[hooks])
```

All three route through one `ContinuumToolGuard` over the same `ActionLedger`:
claim before the tool runs, complete after it returns, fail (certain) when it
raises. Keys default to the MCP-style derivation (tool type, argument hash, run
scope). `key_fn` gives the same resource-identity dedup the class adapters get,
with a different signature: it is called as `key_fn(tool_name, args) -> str`,
where `args` is the tool's argument dict, not the `(*args, **kwargs)` the
`wrap_tool` decorators pass.

Framework imports stay lazy, so importing `thin.py` costs nothing when the
framework is absent. `install_crewai_hooks` raises `ImportError` with an install
hint when `crewai` is not importable; the AutoGen and Pydantic AI helpers are
duck-typed and never import their SDK at all.

State written through these hooks carries `Origin.EXTERNAL_AGENT` provenance,
because the framework asserts the facts, so such a run resolves to
`request_human` until `continuum confirm` (see Provenance and the human gate
below). Two further limits are worth knowing: `install_crewai_hooks` tracks only
tool names in `action_types`, so pass `None` to track every call, and its
uninstaller falls back to `clear_all_tool_call_hooks()` on older CrewAI
versions, which drops other hooks with it.

Tests: `tests/test_adapters_thin.py` covers the guard against a real ledger, the
CrewAI before/after registry and its `action_types` filter, the AutoGen
`run_json` claim/complete and its failure re-raise, and the Pydantic AI async
capability on both the result and the error path. No SDK is required there
either: each seam is exercised with a duck-typed stand-in, because what CONTINUUM
depends on is the shape of the seam, not the package.

## Transport seams

Two seams reach stacks no adapter does, by sitting on the wire rather than in
the process.

### Enforcing HTTP gateway

`continuum gateway` is a local proxy the application points at instead of the
real upstream, so an outbound call from any language needs a claim first:

```bash
continuum gateway --port 8765     # routes read from .continuum/gateway.json
```

Routes are data, not code, and `--config` names another file:

```json
{
  "upstreams": [
    {"host": "api.example.com", "methods": ["POST"],
     "prefix": "/v1/invoices", "action_type": "send_invoice",
     "key_template": "invoice:{id}"}
  ]
}
```

Decision semantics mirror `gate`: a matching request proceeds only when a live
STARTED claim exists for its derived key, a duplicate is refused because the
effect already happened, and an uncertain outcome demands reconciliation first.
After forwarding, the gateway settles the claim from the real status code,
COMPLETED on 2xx/3xx, FAILED-certain on 4xx because the upstream definitively
rejected it, FAILED-uncertain on 5xx and timeouts because the effect may still
have landed, and records `TOOL_COMPLETED` evidence with the status.

Unknown hosts are refused rather than forwarded, because a proxy that forwards
anywhere would be an open relay wearing CONTINUUM's name, and the gateway
refuses to start at all when the registry holds no upstream. `key_template`
substitutes top-level JSON body fields only, exactly as `gate` does. Tests:
`tests/test_gateway.py`.

### OpenTelemetry bridge

`continuum.otel.make_span_processor(storage)` turns telemetry a traced app
already emits into evidence, with no change to the application:

```python
provider.add_span_processor(continuum.otel.make_span_processor(storage))
```

A span counts as a tool call when any of `gen_ai.tool.name`, `tool.name`,
`openinference.tool.name`, `mcp.tool.name` or `function.name` names one, first
match wins. Qualifying spans land in the active run (or an explicit `run_id`) as
`TOOL_COMPLETED`, or `TOOL_FAILED` when the span status is not ok, carrying the
same payload shape `continuum observe` writes plus `via="otel"`.

Recognition is heuristic by design, because production pipelines use several
attribute conventions. Non-tool spans are ignored, and a span that arrives with
no active run is dropped rather than raising, matching how `observe` treats
telemetry from outside a run. Size and digest are never invented, because a span
does not carry the file's bytes. Install with
`pip install "continuum-agent[otel]"`; `make_span_processor` raises
`RuntimeError` with an install hint when OpenTelemetry is not importable, and the
pure core (`observation_from_span`, `record_span`) needs no SDK at all. Tests:
`tests/test_otel.py`.

## Exactly-once side effects and recovery

- External side effects are claimed in the action ledger before execution and
  completed after. A second call with the same run-scoped arguments returns the
  cached result without re-executing the underlying function.
- When the caller cannot guarantee stable argument text (an LLM-driven tool),
  pass an explicit `key` or `key_fn` to `wrap_tool` so dedup keys on resource
  identity rather than the argument hash.
- `checkpoint_node` persists a `STATE_CHECKPOINTED` event and a restorable
  semantic state.
- `assess_graph_recovery` / `assess_langchain_recovery` (or `resume`) validates
  the state and returns a `RecoveryDecision`.

### Provenance and the human gate

A run started through the LangGraph or LangChain adapter is recorded with
`Origin.DETERMINISTIC` provenance, because the adapter is the orchestrator
starting the run on CONTINUUM's behalf. A consistent such run resumes
(`RESUME`) without a human in the loop.

State reported over MCP, or through the OpenAI adapter, uses
`Origin.EXTERNAL_AGENT` provenance, which the validator marks `REQUIRES_REVIEW`.
That is intentional: an agent must not validate its own unverified work. The
consequence is that such runs resolve to `request_human` on `continuum resume`
until a human has eyeballed them.

## Integration tests

The adapter-specific end-to-end tests live in:

- `tests/test_integration_langgraph.py` - checkpoint durability, exactly-once
  side effect across duplicate runs, and a crash after the checkpoint that does
  not duplicate the side effect on resume.
- `tests/test_integration_langchain.py` - LCEL pipeline checkpoint and
  exactly-once side effect, and the same crash-after-checkpoint resume.
- `tests/test_integration_langchain_agent.py` - a real `create_agent`
  tool-calling loop (offline, driven by a scripted fake model) proving the
  side effect fires once even when the agent repeats the tool call, and stays
  once across separate agent invocations.
- `tests/test_integration_langchain.py::TestLangChainArchitecture::test_explicit_key_deduplicates_against_argument_drift`
  and `test_key_fn_derives_key_from_call_arguments` - prove an explicit
  `key` / `key_fn` collapses repeated calls even when the argument text drifts
  between invocations.

Each test imports its framework with `pytest.importorskip`, so it is collected
but skipped when the optional dependency is not installed.

### Real-LLM harness

`examples/langchain_real_llm.py` drives the LangChain adapter against a live
OpenAI-compatible model through OpenRouter (`ChatOpenAI` with
`base_url="https://openrouter.ai/api/v1"`). It runs a real `create_agent`
tool-calling loop over the same CONTINUUM run twice (a first pass plus a resume)
and asserts the wrapped external side effect fires exactly once. The model is
selected via `OPENROUTER_MODEL` (default `openai/gpt-4o-mini`); the key is read
from `OPENROUTER_API_KEY` and never written to disk. Run it with:

```bash
OPENROUTER_API_KEY=sk-or-... OPENROUTER_MODEL=openai/gpt-4o-mini \
  python examples/langchain_real_llm.py
```

It exercises the exact gap the scripted tests cannot: a live LLM produces
drifting argument text, and the stable `key` is what keeps the side effect
idempotent. See STATUS.md (Real-LLM framework adapter test, 2026-08-15) for the
recorded output and what it establishes.

`examples/openai_real_llm.py` and `examples/langgraph_real_llm.py` run the same
dual-invocation contract (first pass plus resume, exactly-once side effect) for the
OpenAI Agents SDK and LangGraph adapters respectively, using the same OpenRouter
setup. The OpenAI harness must use `OpenAIChatCompletionsModel` over the chat
completions endpoint, because OpenRouter does not fully support the Agents SDK's
Responses API. `examples/multitool_real_llm.py` is the richer live demo: a single
model prompt orchestrates three tools (lookup, notify, open ticket) through the
LangGraph adapter, each side effect wrapped with a *fixed* idempotency key, proving
exactly-once survives the model's argument drift across a soft resume.

`examples/langchain_real_llm_crash.py` drives the hard-crash contract with a live
model: the `crash` subcommand lets the agent call the wrapped tool (which performs a
real outbox write) and then hard-exits the process (`os._exit(137)`) before the
ledger records completion; the `resume` subcommand runs a fresh process that calls
`RecoveryEngine.assess`, which must report `request_human` / `safe=False` with one
uncertain action and an outbox that still holds exactly one entry. This shows a live
model's side effect is left uncertain on a mid-side-effect kill and is never
duplicated on reconciliation. `examples/openai_real_llm_crash.py` and
`examples/langgraph_real_llm_crash.py` drive the identical contract for the OpenAI
Agents SDK and LangGraph adapters (the OpenAI one uses `OpenAIChatCompletionsModel`
over the chat completions endpoint, as its soft-resume sibling does). All three
adapters' hard-crash paths are now verified against a live model. Same env vars
apply:

```bash
OPENROUTER_API_KEY=sk-or-... python examples/langchain_real_llm_crash.py crash
OPENROUTER_API_KEY=sk-or-... python examples/langchain_real_llm_crash.py resume
```


## Live LLM validation results (real model via OpenRouter)

All three framework adapters were driven against a live `gpt-4o-mini` through
OpenRouter (key from `OPENROUTER_API_KEY`, never written to disk). Each was proven
two ways: a soft resume (exactly-once side effect across a second clean invocation)
and a hard crash (`os._exit(137)` mid-side-effect, then a fresh process asserts the
run is blocked as uncertain). A richer `examples/multitool_real_llm.py` demo has one
prompt orchestrate `lookup_order` + `notify_customer` + `create_ticket` through the
LangGraph adapter.

| Adapter    | Soft resume (exactly-once)             | Hard crash (resume blocked)       |
|------------|----------------------------------------|-----------------------------------|
| LangChain  | PASS, 1 side effect, `resume` safe     | PASS, `request_human`, 1 uncertain |
| OpenAI SDK | PASS, 1 side effect, `request_human`*  | PASS, `request_human`, 1 uncertain |
| LangGraph  | PASS, 1 side effect, `resume` safe     | PASS, `request_human`, 1 uncertain |

\* The OpenAI adapter yields `request_human` even on a clean soft resume because it
records `Origin.EXTERNAL_AGENT`: an agent must not self-certify its own unverified
work. That is expected and safe. LangChain and LangGraph use `Origin.DETERMINISTIC`
and resume cleanly.

Two OpenAI-adapter bugs that only surface with a real model were found and fixed:
the tool JSON schema was emitted with no `type` key (OpenRouter rejected it), and the
context parameter was dropped from the inspectable signature, which bypassed
interception and let the side effect fire twice. The live runs also confirmed the
idempotency lesson: a stable business key (for example `ticket:O-9`) is required,
because a key derived from the model's rendered arguments does not dedupe the model's
argument drift and produced a duplicate ticket. Full run logs are in STATUS.md.
