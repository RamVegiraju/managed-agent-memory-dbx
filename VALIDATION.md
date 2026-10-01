# Validation report

Validated on **October 1, 2026** against the user-selected Databricks workspace. All live data was
synthetic and isolated under a unique `/memories/demos/<run>/preferences/` namespace.

## Live configuration

- Model: `databricks-gpt-5-6-sol` with `reasoning_effort="none"`.
- Client: `databricks-agentbricks==0.3.0` / `AgentKitClient`.
- Stores: existing `memory-demo-sessions` and `memory-demo-memory` resources.
- Identity: authenticated caller ID from `current_user.me().id`.

Workspace, profile, and user identifiers are intentionally absent from source and this report.

## Live results

| Scenario | Result |
| --- | --- |
| Managed `Session.extract_memories(dry_run=True)` | Passed |
| Structured reconciliation tool call | Passed |
| Split two user preferences into two atomic entries | Passed |
| Keep all canonical paths inside a unique custom namespace | Passed |
| Preference acknowledgement does not ask the user to save or remember it | Passed |
| Opening the Streamlit app creates no empty managed session | Passed |
| Canonical entry has no session-scoped identity | Passed |
| Cross-session managed search recalls the entry | Passed |
| Fresh-session agent answer uses recalled preferences | Passed |
| Correct PySpark preference to SQL with `UPDATE` | Passed |
| Repeated equivalent preference produces `NO_OP` | Passed |
| Unmentioned memory remains unchanged | Passed |
| Session-only instruction remains non-durable | Passed |
| Undecided contradictory preference produces `CONFLICT` | Passed |
| Explicit forget request produces `DELETE` | Passed |
| Removed memory is absent after deletion | Passed |
| Isolated live namespace is empty after cleanup | Passed |

An additional isolated API check confirmed that `extract_memories(dry_run=False)` created a new
session-associated entry rather than updating an existing canonical entry. This is why the sample
uses dry-run extraction and performs explicit, observable reconciliation before managed writes.

## Automated results

The offline suite passes **72 tests**, including:

- Managed session creation, replay, ownership checks, and fresh-session behavior.
- Actor and namespace binding for memory list and search.
- Exact AgentKit REST envelopes for memory CRUD, append, search, and dry-run extraction.
- `ADD`, `UPDATE`, `NO_OP`, `DELETE`, `CONFLICT`, and omission behavior.
- Multi-entry extraction, custom-namespace remapping, and compatible-fact preservation.
- Canonical path and evidence checks that validate the complete plan before any write.
- Canonical-only recall when legacy session-associated extraction entries are present.
- Streamlit lazy creation, end-session, fresh-session, and prominent failure behavior through AppTest.
- Model endpoint selection and forced structured reconciliation tool choice.

Ruff lint, Ruff formatting, Pyright, and `git diff --check` pass.

## Design review

The implementation was checked against the Databricks managed memory/session guides and the
Databricks memory-scaling article. It follows the key separation of episodic session history from
distilled semantic memory, identity-aware scoping, selective recall, atomic entries, consolidation,
explicit forgetting, and preservation of ambiguous or unmentioned memories.

## Reproduce

Follow the README quickstart with an explicit profile and available model:

```bash
export DATABRICKS_CONFIG_PROFILE=YOUR_PROFILE
export DATABRICKS_MODEL=databricks-gpt-5-6-sol
export DATABRICKS_REASONING_EFFORT=none

uv run memory-demo doctor
uv run memory-demo init --yes
uv run streamlit run streamlit_app.py
```

`init --yes` can provision billable backing. The app never provisions resources implicitly.
