# Live validation

Validated on **September 30, 2026**, in the user-selected Databricks workspace.
All data was synthetic. No existing session or memory store was modified.

## Configuration

- Model endpoint: `databricks-gpt-5-6-sol`, discovered from the workspace rather than assumed.
- Transport: Chat Completions with function tools and `reasoning_effort="none"`.
- Storage client: `databricks-agentbricks==0.3.0` / `AgentKitClient`.
- Dedicated temporary session and memory stores, separate from the workspace's existing stores.
- Authenticated actor identity resolved from `current_user.me().id`.

The workspace/profile identifiers are intentionally not embedded in the sample or this report.

## Results

| Check | Result |
| --- | --- |
| Read-only authentication, model discovery, and both store APIs | Passed |
| Persist and reload chronological session history through fresh API objects | Passed |
| Fresh session starts with an empty transcript | Passed |
| Cross-session memory search returns the saved PySpark preference | Passed |
| Session-only job context is absent from durable memory | Passed |
| Update a canonical topic without creating a duplicate | Passed |
| Delete a session while retaining durable memory | Passed |
| Explicit forgetting removes the stored preference | Passed |
| Resumed model conversation recalls `nightly-orders` from its transcript | Passed |
| Fresh model conversation recalls PySpark preferences but does not know the old job | Passed |
| Fresh conversation after forgetting reports no saved preference | Passed |
| Interactive model-requested save, update, and delete each wait for approval | Passed |
| Exit the chat process, restart it, and resume the same session | Passed |

For the actual process-restart check, the resumed agent correctly identified
`validation-orders` from its saved session transcript.

Model behavior was observed in these runs; it is not a guarantee that every future model response
will use the same tools or wording. Retrieval is relevance-ranked BM25, not vector similarity.
Multi-user authorization and concurrent writers were not validated live.

## Compatibility fixes found during validation

1. **Session create response loss:** creates returned `Response ended prematurely` or `CANCELLED`
   despite committing. Sessions use client-assigned UUIDs. After a create error, bounded read-only
   lookups recover the exact session after propagation without repeating the write. The recovery
   and original-error paths have regression tests.
2. **SDK readiness enum:** the SDK returns `EndpointStateReady.READY`, not a plain string.
   Readiness checks now accept the enum value as well as a string.
3. **Sol tool-call setting:** GPT-5.6-Sol rejected Chat Completions function tools with default
   reasoning enabled. `--reasoning-effort none` resolves this. The flag/environment setting is
   optional and omitted by default for compatibility with other endpoints.

## Cleanup and local checks

Both dedicated validation stores were deleted through the managed-store APIs, after checking
remaining sessions were known synthetic test artifacts and the caller's memory inventory was empty.
Their absence was then verified using both store-list APIs. No validation store remains.
The underlying managed Lakebase instance lifecycle was not independently audited.

## Retained infrastructure observation

A later retained run created `memory-demo-sessions` and `memory-demo-memory`, then completed two
GPT-5.6-Sol conversations: one explicitly saved a response preference and a separate session
retrieved it. The stores and records were intentionally left in place for inspection.

Databricks reused the existing `databricks-internal-custom-agents` Lakebase project, its
`production` branch, and its `primary` endpoint. Store creation added two PostgreSQL databases:
`session-memory-demo-sessions` and `memory-memory-demo-memory`. The memory store API explicitly
reported the latter in `storage_backend.backend_id`; the session database matched the store name
and creation timestamp in Lakebase. No new Lakebase project or endpoint was created for this run.

The offline suite passes **38 tests**. Ruff lint, Ruff format checks, and Pyright pass.

## Reproduce in another workspace

Follow the README quickstart with your own explicit profile and store names. For this model:

```bash
export DATABRICKS_CONFIG_PROFILE=YOUR_PROFILE
export DATABRICKS_MODEL=databricks-gpt-5-6-sol
export DATABRICKS_REASONING_EFFORT=none
uv run memory-demo doctor
uv run memory-demo init --yes
uv run memory-demo --trace demo --cleanup
```

Choose another available tool-calling model if this endpoint is not listed in your workspace.
`init --yes` provisions billable backing stores. Demo cleanup removes test artifacts, not stores.
