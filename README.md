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

## Run the sample

Requirements: Python 3.10+, [uv](https://docs.astral.sh/uv/getting-started/installation/),
the Databricks CLI, an enabled workspace, and permission to use a tool-calling chat endpoint.

### 1. Install and authenticate

```bash
uv sync
databricks auth login --host https://YOUR-WORKSPACE --profile YOUR_PROFILE
export DATABRICKS_CONFIG_PROFILE=YOUR_PROFILE
uv run memory-demo doctor
```

`doctor` is read-only. It prints the authenticated user ID, available chat endpoints, and existing
session and memory stores. It does not create resources or invoke a model.

### 2. Configure the model and stores

Choose an endpoint printed by `doctor`:

```bash
export DATABRICKS_MODEL=YOUR_CHAT_ENDPOINT
# Required for GPT-5.6-Sol tool calls; omit for models that do not support this option.
export DATABRICKS_REASONING_EFFORT=none

# Run once. This creates only the two missing stores.
uv run memory-demo init --yes
```

**Cost:** `init --yes` can provision billable Lakebase backing. Chat and demo commands only open
existing stores. The defaults are `memory-demo-sessions` and `memory-demo-memory`.

### 3. Demonstrate cross-session memory

```bash
uv run memory-demo --verbose chat
```

Enter the following, approving the memory write with `y`:

```text
Remember that I prefer concise answers with PySpark examples.
/memories
/new
/history
Use my saved response preferences to explain deduplication.
/sessions
```

What this proves:

- `/new` creates a different session and `/history` initially returns `[]`.
- The verbose `[memory.searched]` event shows the agent calling managed memory.
- The new session still uses the preference saved in the first session.
- `/sessions` lists both transcripts while `/memories` lists the independent durable preference.

For a process-restart check, run `/quit`, start `chat` again, and ask for an answer using your saved
preferences. The memory remains because it is partitioned by authenticated user, not session ID.

### 4. Retrieve persisted data

Copy the user ID printed by `doctor`:

```bash
uv run memory-demo inspect --user-id YOUR_DATABRICKS_USER_ID
```

Copy a `session_id` from the returned `sessions` array to retrieve its complete history:

```bash
uv run memory-demo inspect \
  --user-id YOUR_DATABRICKS_USER_ID \
  --session-id YOUR_SESSION_ID
```

The output is JSON. The sample permits only the authenticated caller's user ID; it does not permit
arbitrary cross-user access.

### 5. Run the automated showcase

```bash
# Real storage APIs, no model cost
uv run memory-demo --verbose demo --storage-only --cleanup

# Real storage APIs plus model responses
uv run memory-demo --verbose demo --cleanup
```

The demo verifies session reload, a fresh empty session, cross-session recall, update, independent
session deletion, and explicit forgetting. It uses an isolated namespace. `--cleanup` removes only
that run's sessions and entries; it does not delete the stores or their backing infrastructure.

## Memory tools

The model receives three simple function tools defined in `src/memory_demo/agent.py`:

| Tool | Managed-memory operation |
| --- | --- |
| `recall(query)` | Search this user's durable entries |
| `remember(topic, content)` | Create or update one stable topic after approval |
| `forget(topic)` | Delete one stable topic after approval |

This intentionally combines lower-level create/get/list/update/delete operations into a smaller
agent interface. Memory extraction is not automatic: the model must call `remember`, and durable
facts are available in a new session only after it calls `recall`.

## Configuration

There is no automatic `DEFAULT` profile or hard-coded model. Environment variables can be replaced
with global flags placed **before** the subcommand:

```bash
uv run memory-demo --profile YOUR_PROFILE --model YOUR_CHAT_ENDPOINT \
  --session-store YOUR_SESSIONS --memory-store YOUR_MEMORY --verbose demo --cleanup
```

Commands: `doctor`, `init`, `chat`, `inspect`, and `demo`. Run `uv run memory-demo --help` for all
global options. In chat, use `/new`, `/resume ID`, `/sessions`, `/history`, `/memories`, or `/quit`.
`--verbose` can print user memory contents; do not share its output or use sensitive demo data.

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
