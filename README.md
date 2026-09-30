# Managed sessions + memory: a small support copilot

A framework-independent Python sample of the **new** Databricks managed agent APIs,
using `AgentKitClient` for storage and `DatabricksOpenAI` for a tool-calling chat model.
No workspace URL, organization ID, endpoint name, or local user identity is embedded in the code.
The sample runs on a laptop or any Python environment with access to an enabled Databricks workspace.

| Managed sessions | Managed memory |
| --- | --- |
| The ordered transcript for one conversation | Selected durable facts/preferences across conversations |
| Reloaded on every turn; survives process restarts | Recalled through relevance-ranked search |
| Messages, model tool calls, and tool results | Small entries at stable topic paths |
| A new session starts with an empty transcript | A new session can recall the same caller's preferences |

**Retrieval caveat:** the current detailed docs specify **BM25 full-text search**, not vector
similarity. Although the introduction mentions semantic search, this sample does not claim
embedding-based or guaranteed synonym retrieval. Short descriptions and meaningful query terms matter.

## Quickstart

Requirements:
- Python 3.10+ and [uv](https://docs.astral.sh/uv/getting-started/installation/).
- A Databricks workspace with the managed sessions and memory preview enabled.
- An explicitly selected, authenticated Databricks CLI profile.
- For chat, permission to invoke an existing chat endpoint that supports function/tool calling.

```bash
uv sync
databricks auth login --host https://YOUR-WORKSPACE --profile YOUR_PROFILE
export DATABRICKS_CONFIG_PROFILE=YOUR_PROFILE
uv run memory-demo doctor
```

`doctor` is read-only: it resolves the authenticated identity, lists chat endpoints, and lists
both store types. It neither creates infrastructure nor invokes a model. Choose a listed,
tool-calling endpoint; model names differ by workspace.

For GPT-5.6-Sol, Chat Completions function tools require reasoning to be disabled:
set `DATABRICKS_REASONING_EFFORT=none` or pass the global `--reasoning-effort none` flag.
The setting is optional and omitted by default because other endpoints may not support it.

```bash
export DATABRICKS_MODEL=YOUR_CHAT_ENDPOINT
uv run memory-demo init --yes
uv run memory-demo --trace demo --cleanup
uv run memory-demo --trace chat
```

**Cost:** `init --yes` explicitly provisions missing stores and their billable Lakebase backing.
Model calls also incur normal inference costs. Chat and demo only open existing stores.
`--cleanup` removes this demo run's sessions/entries, **not** the backing stores; those may
continue incurring costs. Delete dedicated stores through the official SDK/CLI when finished,
after checking they contain nothing you want to retain.

Flags can replace environment variables. Global flags go **before** the subcommand:

```bash
uv run memory-demo --profile YOUR_PROFILE --model YOUR_CHAT_ENDPOINT \
  --session-store YOUR_SESSIONS --memory-store YOUR_MEMORY --trace demo
```

There is no automatic `DEFAULT` profile or hard-coded model fallback. Existing stores can be
used by setting `DEMO_SESSION_STORE` and `DEMO_MEMORY_STORE`, or passing the matching flags.
The defaults are `memory-demo-sessions` and `memory-demo-memory`.

To test the storage APIs without model costs:

```bash
uv run memory-demo --profile YOUR_PROFILE --trace demo --storage-only --cleanup
```

This still uses **real managed stores**. It is not an offline emulator.

## What the scripted demo proves

The demo seeds a clearly labelled fixture: a session-only `nightly-orders` job context and
a separately saved preference for concise PySpark examples. It uses a unique memory namespace
per run, so it never modifies your interactive chat preferences.

1. Create Session A and save an ordered transcript.
2. Save one explicit durable preference; this is not automatic transcript extraction.
3. Open fresh API resource objects and reload A's history, without an in-process transcript cache.
4. Create Session B and assert its history is empty.
5. Search for the saved preference and verify the temporary job was not saved as memory.
6. Update the same topic to SQL examples and verify there is still just one entry.
7. Delete A and verify the durable preference survives independently.
8. Forget the preference and verify the memory inventory is empty.

With a model configured, it also prints responses from resumed A, fresh B, and a fresh C after
forgetting. Storage checks are asserted; model wording and tool choices are illustrative and not
graded. A search failure is surfaced, not concealed by silently listing all memories.

The fresh API client check proves storage reloading, not an actual process restart. To demonstrate
that separately, quit chat and restart it with `--session-id` as shown below.

## Interactive walkthrough

In `chat`, enter:

```text
Remember my response preferences: concise answers with PySpark examples.
```

Approve the proposed memory write with `y`. Then:

```text
For this conversation only, we're troubleshooting nightly-orders.
Which job are we troubleshooting?
/memories
/history
/new
Use my saved response preferences to show a deduplication example.
What job are we troubleshooting in this new conversation?
```

The new thread has no old transcript; only explicitly retrieved memory can personalize its answer.
Try updating the preference to SQL and asking to forget it, approving each mutation. Start a
**fresh thread** when demonstrating updated/forgotten memory: old transcripts still contain
previously mentioned or retrieved facts. Forgetting does not redact those transcripts.

Commands: `/new`, `/resume ID`, `/sessions`, `/history`, `/memories`, `/quit`.
The current session ID is printed when chat starts. Stop the process, then resume it:

```bash
uv run memory-demo --trace chat --session-id YOUR_SESSION_ID
```

`--trace` prints activity, memory queries/results, and writes. It does not print credentials
or hidden model reasoning, but it **does print user memory contents**: do not share traces or
use sensitive demo data.

## Code map

- `src/memory_demo/backend.py`: AgentKit operations, identity binding, ownership checks,
  chronological replay, and canonical-topic updates.
- `src/memory_demo/agent.py`: a bounded chat/tool loop; only `recall`, `remember`, and `forget`.
- `src/memory_demo/demo.py`: inspectable storage acceptance checks and optional model conversations.
- `src/memory_demo/cli.py`: explicit configuration, read-only discovery, provisioning consent,
  and terminal chat controls.

## Safety and limitations

- **`actor_id` is not authorization.** Both stores authorize at the store level; anyone with access
  can access other actors through the API. Use separate stores for strict security boundaries.
- This is a single-caller CLI. A multi-user application must supply verified end-user identity and
  enforce its own authorization boundaries.
- Retrieved memories are untrusted input. All model-requested memory changes require local approval.
- The sample supports one writer per conversation and does not summarize oversized histories.
- A failed request may already have committed. Inspect the session or memory before retrying a write.

## Live validation

The sample passed live storage checks, the full GPT-5.6-Sol demo, interactive memory approvals,
and an actual chat-process restart/resume on September 30, 2026. Dedicated validation stores
were deleted afterward; existing workspace stores were untouched. See [VALIDATION.md](VALIDATION.md)
for results, model compatibility fixes, cleanup details, and reproduction steps.

## Tests

```bash
uv sync --group dev
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run pyright
```

Tests use explicit test doubles, do not authenticate, and never create paid infrastructure.
Use `doctor` and the storage-only demo for opt-in real workspace verification.

The direct SDK dependencies are pinned to the versions tested here. A machine-specific lockfile
is not shipped: `uv sync` generates one against your configured package index, without requiring
an internal Databricks package mirror. Transitive dependencies can still change between installs.

## Authoritative references

The guides were reviewed on September 30, 2026. Preview APIs and workspace availability may change.

- [Managed agent sessions](https://docs.databricks.com/aws/en/agents/agent-memory/managed-sessions)
- [Managed agent memory](https://docs.databricks.com/aws/en/agents/agent-memory/managed-memory)
- [AgentKit SDK reference](https://github.com/databricks/databricks-ai-bridge/blob/main/integrations/agentbricks/README.md#agentkit-sdk)
- [Public REST transport source](https://github.com/databricks/databricks-ai-bridge/blob/main/integrations/agentbricks/src/databricks_agentkit/_api_client.py)
- [DatabricksOpenAI client](https://api-docs.databricks.com/python/databricks-ai-bridge/latest/databricks_openai.html#databricks_openai.DatabricksOpenAI)
