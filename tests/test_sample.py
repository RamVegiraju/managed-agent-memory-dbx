"""Verify context boundaries, approved tool execution, and portable configuration."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest
from databricks.sdk.service.serving import EndpointStateReady
from databricks_agentbricks.errors import AgentCliError

from memory_demo.agent import (
    END_SESSION_PROMPT,
    MEMORY_PLAN_TOOL,
    TOOLS,
    Agent,
    DatabricksModel,
    end_session_memory,
    reconcile_memories,
)
from memory_demo.backend import Backend, caller_id
from memory_demo.cli import check_model, inspect, parser, validate
from memory_demo.demo import run_demo
from memory_demo.streamlit_ui import memory_output, session_output, visible_messages
from tests.fakes import FakeMemoryStore, FakeSessionStore, ScriptedModel


@pytest.fixture
def backend() -> Backend:
    return Backend(FakeSessionStore(), FakeMemoryStore(), "verified-user")


def tool_call(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call-1",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments)},
            }
        ],
    }


def memory_plan(*decisions: dict[str, str]) -> dict[str, Any]:
    return tool_call("submit_memory_plan", {"decisions": list(decisions)})


def decision(
    action: str,
    path: str,
    content: str,
    evidence: str,
    description: str = "User preference",
) -> dict[str, str]:
    return {
        "action": action,
        "path": path,
        "content": content,
        "description": description,
        "reason": "Explicit user statement",
        "evidence": evidence,
    }


def test_session_resume_and_fresh_thread(backend: Backend) -> None:
    session = backend.new_session()
    backend.append(session.session_id, [{"role": "user", "content": "nightly-orders"}])
    reopened = Backend(backend.session_store, backend.memory_store, backend.actor_id)
    assert reopened.history(session.session_id)[0]["content"] == "nightly-orders"
    assert session.last_order == "create_time asc"
    assert reopened.history(reopened.new_session().session_id) == []


def test_session_creation_recovers_lost_response_without_repeating_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reads = 0

    class LostResponseStore(FakeSessionStore):
        def add(self, actor_id: str, session_id: str | None = None) -> Any:
            super().add(actor_id, session_id)
            raise AgentCliError("Response ended prematurely")

        def get(self, session_id: str) -> Any:
            nonlocal reads
            reads += 1
            if reads < 3:
                raise AgentCliError("Not found yet")
            return super().get(session_id)

    monkeypatch.setattr("memory_demo.backend.sleep", lambda seconds: None)
    store = LostResponseStore()
    backend = Backend(store, FakeMemoryStore(), "verified-user")
    recovered = backend.new_session()
    assert recovered.actor_id == "verified-user"
    assert store.counter == 1
    assert reads == 3
    assert len(store.entries) == 1


def test_failed_session_create_preserves_original_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class FailedStore(FakeSessionStore):
        def add(self, actor_id: str, session_id: str | None = None) -> Any:
            self.counter += 1
            raise AgentCliError("Create failed")

        def get(self, session_id: str) -> Any:
            raise AgentCliError("Not found")

    monkeypatch.setattr("memory_demo.backend.sleep", lambda seconds: None)
    store = FailedStore()
    backend = Backend(store, FakeMemoryStore(), "verified-user")
    with pytest.raises(AgentCliError, match="Create failed"):
        backend.new_session()
    assert store.counter == 1
    assert not store.entries


@pytest.mark.parametrize("operation", ["history", "delete_session", "append"])
def test_foreign_session_refused_before_items(backend: Backend, operation: str) -> None:
    foreign = backend.session_store.add(actor_id="another-user")
    with pytest.raises(PermissionError):
        if operation == "append":
            backend.append(foreign.session_id, [{"role": "user", "content": "bad"}])
        else:
            getattr(backend, operation)(foreign.session_id)
    assert foreign.reads == 0
    assert not foreign.items


def test_upsert_and_independent_deletion(backend: Backend) -> None:
    session = backend.new_session()
    assert backend.remember("response-preferences", "PySpark")["status"] == "created"
    assert backend.remember("response-preferences", "SQL")["status"] == "updated"
    assert len(backend.memories()) == 1
    assert backend.memories()[0].content == "SQL"
    backend.delete_session(session.session_id)
    assert len(backend.memories()) == 1
    backend.forget("response-preferences")
    assert backend.memories() == []


def test_end_session_managed_extraction_adds_canonical_memory(backend: Backend) -> None:
    session = backend.new_session()
    statement = "I prefer concise answers"
    backend.append(session.session_id, [{"role": "user", "content": statement}])
    path = "/memories/preferences/communication/verbosity.md"
    session.extracted_memories = [
        SimpleNamespace(
            path=path,
            content="Prefers concise answers.",
            description="Response verbosity",
            session_id=session.session_id,
        )
    ]
    model = ScriptedModel(
        [memory_plan(decision("ADD", path, "Prefers concise answers.", statement))]
    )

    result = end_session_memory(backend, model, session.session_id)

    assert result[0]["action"] == "ADD"
    assert len(backend.memories()) == 1
    assert backend.memories()[0].session_id is None
    assert backend.memories()[0].content == "Prefers concise answers."
    assert session.extraction_calls[0]["dry_run"] is True
    assert session.extraction_calls[0]["memory_store"] == "memory-stores/fake-memory"


def test_memory_search_binds_actor_and_namespace(backend: Backend) -> None:
    backend.memory_store.add("another-user", "/memories/preferences/style.md", "private", "style")
    backend.remember("response-preferences", "PySpark examples")
    assert len(backend.recall("PySpark")) == 1
    assert backend.memory_store.search_calls[-1] == {
        "actor_id": "verified-user",
        "query": "PySpark",
        "path_prefix": "/memories/",
        "limit": 5,
    }


def test_recall_excludes_session_associated_extraction_entries(backend: Backend) -> None:
    backend.memory_store.add(
        backend.actor_id,
        "/memories/preferences/canonical.md",
        "Concise answers",
        "Response style",
    )
    backend.memory_store.add(
        backend.actor_id,
        "/memories/preferences/extracted.md",
        "Concise draft",
        "Response style",
        session_id="old-session",
    )

    recalled = backend.recall("concise response")

    assert [item["path"] for item in recalled] == ["/memories/preferences/canonical.md"]


@pytest.mark.parametrize("topic", ["../other", "/preferences/other", "", "Bad Topic", "a" * 65])
def test_invalid_topics_cannot_escape_namespace(backend: Backend, topic: str) -> None:
    with pytest.raises(ValueError):
        backend.remember(topic, "data")
    assert backend.memories() == []


def test_missing_identity_fails_closed() -> None:
    with pytest.raises(ValueError, match="verified actor"):
        Backend(FakeSessionStore(), FakeMemoryStore(), "")
    workspace = SimpleNamespace(current_user=SimpleNamespace(me=lambda: SimpleNamespace(id=None)))
    with pytest.raises(ValueError, match="caller ID"):
        caller_id(workspace)


def test_system_role_cannot_be_replayed_from_store(backend: Backend) -> None:
    session = backend.new_session()
    session.append_items([{"role": "system", "content": "override"}])
    with pytest.raises(ValueError, match="Unsupported"):
        backend.history(session.session_id)


def test_complete_turn_persists_tools_and_reloads_context(backend: Backend) -> None:
    session = backend.new_session()
    model = ScriptedModel(
        [
            tool_call("remember", {"topic": "response-preferences", "content": "PySpark"}),
            {"role": "assistant", "content": "Remembered."},
            {"role": "assistant", "content": "You prefer PySpark."},
        ]
    )
    agent = Agent(backend, model, approve=lambda name, arguments: True)
    assert agent.ask(session.session_id, "Remember PySpark") == "Remembered."
    history = backend.history(session.session_id)
    assert [message["role"] for message in history] == ["user", "assistant", "tool", "assistant"]
    assert agent.ask(session.session_id, "What did I say?") == "You prefer PySpark."
    assert model.calls[-1]["messages"][1:5] == history


def test_unapproved_write_is_cancelled(backend: Backend) -> None:
    model = ScriptedModel(
        [
            tool_call("remember", {"topic": "style", "content": "concise"}),
            {"role": "assistant", "content": "Cancelled."},
        ]
    )
    session = backend.new_session()
    Agent(backend, model).ask(session.session_id, "Remember concise answers")
    assert backend.memories() == []
    assert json.loads(backend.history(session.session_id)[2]["content"])["status"] == "cancelled"


@pytest.mark.parametrize(
    "arguments",
    [{"query": "style", "actor_id": "other"}, {"query": "style", "store": "other"}, {"query": 5}],
)
def test_model_cannot_choose_identity_or_store(backend: Backend, arguments: dict[str, Any]) -> None:
    model = ScriptedModel(
        [tool_call("recall", arguments), {"role": "assistant", "content": "Invalid arguments."}]
    )
    session = backend.new_session()
    Agent(backend, model).ask(session.session_id, "Recall")
    assert backend.memory_store.search_calls == []
    assert (
        json.loads(backend.history(session.session_id)[2]["content"])["status"]
        == "invalid_arguments"
    )


def test_tool_limit_does_not_execute_last_round(backend: Backend) -> None:
    model = ScriptedModel([tool_call("recall", {"query": "style"}) for _ in range(6)])
    session = backend.new_session()
    with pytest.raises(RuntimeError, match="tool-call limit"):
        Agent(backend, model).ask(session.session_id, "Recall")
    assert len(backend.memory_store.search_calls) == 5
    assert model.calls[-1]["final"] is True
    assert backend.history(session.session_id) == []


def test_model_failure_does_not_append_partial_turn(backend: Backend) -> None:
    session = backend.new_session()
    model = ScriptedModel([{"role": "assistant", "content": None}])
    with pytest.raises(ValueError, match="empty answer"):
        Agent(backend, model).ask(session.session_id, "Hello")
    assert backend.history(session.session_id) == []


def test_tool_schemas_have_no_identity_arguments() -> None:
    for tool in TOOLS:
        assert "actor_id" not in tool["function"]["parameters"]["properties"]
        assert "store" not in tool["function"]["parameters"]["properties"]
        assert tool["function"]["parameters"]["additionalProperties"] is False


def test_end_session_prompt_never_requires_manual_save_request() -> None:
    assert "does not need to say" in END_SESSION_PROMPT
    assert "Use remember only" not in END_SESSION_PROMPT


def test_demo_leaves_other_namespaces_untouched(capsys: pytest.CaptureFixture[str]) -> None:
    sessions = FakeSessionStore()
    memories = FakeMemoryStore()
    memories.add("verified-user", "/memories/preferences/real.md", "real preference", "real")

    def factory(namespace: str) -> Backend:
        return Backend(sessions, memories, "verified-user", namespace)

    run_demo(factory, cleanup=True)
    assert len(memories.entries) == 1
    assert memories.entries[0].path == "/memories/preferences/real.md"
    assert sessions.entries == {}
    assert "Storage checks passed" in capsys.readouterr().out


def test_configuration_requires_explicit_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DATABRICKS_CONFIG_PROFILE", raising=False)
    with pytest.raises(ValueError, match="--profile"):
        validate(parser().parse_args(["doctor"]))


def test_init_requires_provisioning_consent() -> None:
    with pytest.raises(ValueError, match="billable"):
        validate(parser().parse_args(["--profile", "chosen", "init"]))
    validate(parser().parse_args(["--profile", "chosen", "init", "--yes"]))


@pytest.mark.parametrize(
    ("task", "ready"),
    [("llm/v1/embeddings", "READY"), ("llm/v1/chat", "NOT_READY"), ("llm/v1/chat", None)],
)
def test_model_check_rejects_nonchat_or_unready(task: str, ready: str | None) -> None:
    endpoint = SimpleNamespace(task=task, state=SimpleNamespace(ready=ready))
    workspace = SimpleNamespace(serving_endpoints=SimpleNamespace(get=lambda name: endpoint))
    with pytest.raises(ValueError):
        check_model(workspace, "chosen-endpoint")


@pytest.mark.parametrize("ready", [EndpointStateReady.READY, "READY"])
def test_model_check_accepts_sdk_enum_and_string(ready: Any) -> None:
    endpoint = SimpleNamespace(task="llm/v1/chat", state=SimpleNamespace(ready=ready))
    workspace = SimpleNamespace(serving_endpoints=SimpleNamespace(get=lambda name: endpoint))
    check_model(workspace, "chosen-endpoint")


def test_demo_model_uses_fresh_context_after_forgetting(
    capsys: pytest.CaptureFixture[str],
) -> None:
    sessions = FakeSessionStore()
    memories = FakeMemoryStore()
    model = ScriptedModel(
        [
            {"role": "assistant", "content": "nightly-orders"},
            {"role": "assistant", "content": "Use PySpark."},
            {"role": "assistant", "content": "No saved preferences."},
        ]
    )

    def factory(namespace: str) -> Backend:
        return Backend(sessions, memories, "verified-user", namespace)

    run_demo(factory, model, cleanup=True)
    assert len(model.calls[0]["messages"]) == 4
    assert len(model.calls[1]["messages"]) == 2
    assert len(model.calls[2]["messages"]) == 2
    assert not sessions.entries
    assert not memories.entries
    assert "Fresh Session C" in capsys.readouterr().out


def test_storage_only_requires_no_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DATABRICKS_MODEL", raising=False)
    validate(parser().parse_args(["--profile", "chosen", "demo", "--storage-only"]))
    with pytest.raises(ValueError, match="--model"):
        validate(parser().parse_args(["--profile", "chosen", "chat"]))


def test_inspect_lists_user_sessions_and_memories(capsys: pytest.CaptureFixture[str]) -> None:
    sessions = FakeSessionStore()
    memories = FakeMemoryStore()
    backend = Backend(sessions, memories, "verified-user")
    session = backend.new_session()
    backend.append(session.session_id, [{"role": "user", "content": "Hello"}])
    backend.remember("response-preferences", "Concise PySpark examples")

    inspect(backend, "verified-user")

    result = json.loads(capsys.readouterr().out)
    assert result["user_id"] == "verified-user"
    assert result["sessions"][0]["session_id"] == session.session_id
    assert result["memories"][0]["content"] == "Concise PySpark examples"
    assert "session" not in result


def test_inspect_retrieves_complete_session_history(capsys: pytest.CaptureFixture[str]) -> None:
    backend = Backend(FakeSessionStore(), FakeMemoryStore(), "verified-user")
    session = backend.new_session()
    backend.append(
        session.session_id,
        [
            {"role": "user", "content": "First"},
            {"role": "assistant", "content": "Second"},
        ],
    )

    inspect(backend, "verified-user", session.session_id)

    result = json.loads(capsys.readouterr().out)
    assert [item["content"] for item in result["session"]["history"]] == ["First", "Second"]
    assert session.last_order == "create_time asc"


def test_inspect_rejects_another_user_before_store_retrieval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = Backend(FakeSessionStore(), FakeMemoryStore(), "verified-user")
    monkeypatch.setattr(backend, "sessions", lambda: pytest.fail("sessions were retrieved"))
    monkeypatch.setattr(backend, "memories", lambda: pytest.fail("memories were retrieved"))

    with pytest.raises(PermissionError, match="authenticated"):
        inspect(backend, "another-user")


def test_inspect_parser_uses_verbose_not_trace() -> None:
    arguments = parser().parse_args(
        ["--profile", "chosen", "--verbose", "inspect", "--user-id", "verified-user"]
    )
    assert arguments.verbose is True
    assert arguments.user_id == "verified-user"
    with pytest.raises(SystemExit):
        parser().parse_args(["--profile", "chosen", "--trace", "doctor"])


def test_streamlit_chat_hides_tool_protocol_messages() -> None:
    history = [
        {"role": "user", "content": "Remember concise answers"},
        {"role": "assistant", "content": None, "tool_calls": []},
        {"role": "tool", "content": '{"status": "created"}'},
        {"role": "assistant", "content": "Remembered."},
    ]
    assert visible_messages(history) == [
        {"role": "user", "content": "Remember concise answers"},
        {"role": "assistant", "content": "Remembered."},
    ]


def test_streamlit_raw_outputs_expose_user_partitions(backend: Backend) -> None:
    session = backend.new_session()
    backend.remember("style", "Concise answers")

    assert session_output(backend) == {
        "user_id": "verified-user",
        "sessions": [{"session_id": session.session_id, "create_time": None}],
    }
    assert memory_output(backend) == {
        "user_id": "verified-user",
        "memories": [
            {
                "path": "/memories/style.md",
                "description": "style",
                "content": "Concise answers",
                "session_id": None,
            }
        ],
    }


def test_reconciliation_uses_only_user_statements_and_structured_tool() -> None:
    path = "/memories/preferences/coding/examples.md"
    statement = "I prefer SQL examples"
    model = ScriptedModel([memory_plan(decision("ADD", path, "Prefers SQL examples.", statement))])
    history = [
        {"role": "user", "content": statement},
        {"role": "assistant", "content": "I suggest Python"},
        {"role": "tool", "content": "private tool output"},
    ]

    assert reconcile_memories(model, history, [], [], "/memories/") == [
        decision("ADD", path, "Prefers SQL examples.", statement)
    ]
    payload = json.loads(model.calls[0]["messages"][1]["content"])
    assert payload["user_statements"] == [statement]
    assert payload["managed_extraction_candidates"] == []
    assert payload["canonical_namespace"] == "/memories/"
    assert model.calls[0]["tools"] == [MEMORY_PLAN_TOOL]
    assert model.calls[0]["final"] is False


def test_end_session_adds_multiple_atomic_memories(backend: Backend) -> None:
    session = backend.new_session()
    statement = "Keep answers concise, and use PySpark examples"
    backend.append(session.session_id, [{"role": "user", "content": statement}])
    verbosity = "/memories/preferences/communication/verbosity"
    examples = "/memories/preferences/coding/example-framework"
    session.extracted_memories = [
        SimpleNamespace(
            path=verbosity,
            content="Prefers concise answers.",
            description="Response verbosity",
            session_id=session.session_id,
        ),
        SimpleNamespace(
            path=examples,
            content="Prefers PySpark examples.",
            description="Example framework",
            session_id=session.session_id,
        ),
    ]
    model = ScriptedModel(
        [
            memory_plan(
                decision("ADD", verbosity, "Prefers concise answers.", "Keep answers concise"),
                decision("ADD", examples, "Prefers PySpark examples.", "use PySpark examples"),
            )
        ]
    )

    results = end_session_memory(backend, model, session.session_id)

    assert [result["action"] for result in results] == ["ADD", "ADD"]
    assert {memory.path for memory in backend.memories()} == {verbosity, examples}


def test_end_session_supplies_existing_inventory_to_both_reasoning_steps(
    backend: Backend,
) -> None:
    path = "/memories/preferences/coding/language"
    backend.memory_store.add(backend.actor_id, path, "Prefers Python.", "Language")
    session = backend.new_session()
    statement = "I now prefer Scala"
    backend.append(session.session_id, [{"role": "user", "content": statement}])
    session.extracted_memories = [
        SimpleNamespace(
            path=path,
            content="Prefers Scala.",
            description="Language",
            session_id=session.session_id,
        )
    ]
    model = ScriptedModel(
        [memory_plan(decision("UPDATE", path, "Prefers Scala.", statement, "Language"))]
    )

    end_session_memory(backend, model, session.session_id)

    extraction_instructions = session.extraction_calls[0]["instructions"]
    assert path in extraction_instructions
    assert "Prefers Python." in extraction_instructions
    payload = json.loads(model.calls[0]["messages"][1]["content"])
    assert payload["existing_canonical_memories"] == [
        {"path": path, "content": "Prefers Python.", "description": "Language"}
    ]


def test_reconciliation_remaps_managed_candidate_into_custom_namespace() -> None:
    namespace = "/memories/demos/run/preferences/"
    backend = Backend(FakeSessionStore(), FakeMemoryStore(), "verified-user", namespace)
    session = backend.new_session()
    statement = "I prefer concise answers"
    backend.append(session.session_id, [{"role": "user", "content": statement}])
    session.extracted_memories = [
        SimpleNamespace(
            path="/memories/profile.md",
            content="Prefers concise answers.",
            description="Response style",
            session_id=session.session_id,
        )
    ]
    canonical = f"{namespace}communication/verbosity"
    model = ScriptedModel(
        [memory_plan(decision("ADD", canonical, "Prefers concise answers.", statement))]
    )

    end_session_memory(backend, model, session.session_id)

    assert backend.memories()[0].path == canonical
    payload = json.loads(model.calls[0]["messages"][1]["content"])
    assert payload["managed_extraction_candidates"][0]["path"] == "/memories/profile.md"
    assert payload["canonical_namespace"] == namespace


def test_end_session_refuses_foreign_session_before_extraction_or_model() -> None:
    sessions = FakeSessionStore()
    foreign = sessions.add(actor_id="another-user")
    backend = Backend(sessions, FakeMemoryStore(), "verified-user")
    model = ScriptedModel([])

    with pytest.raises(PermissionError, match="authenticated caller"):
        end_session_memory(backend, model, foreign.session_id)

    assert foreign.extraction_calls == []
    assert model.calls == []


def test_end_session_update_and_explicit_delete(backend: Backend) -> None:
    path = "/memories/preferences/coding/language.md"
    backend.memory_store.add(
        backend.actor_id, path, "Prefers Python.", "Preferred programming language"
    )
    update_session = backend.new_session()
    update_statement = "I switched from Python to Scala"
    backend.append(update_session.session_id, [{"role": "user", "content": update_statement}])
    update_model = ScriptedModel(
        [memory_plan(decision("UPDATE", path, "Prefers Scala.", update_statement))]
    )
    assert (
        end_session_memory(backend, update_model, update_session.session_id)[0]["action"]
        == "UPDATE"
    )
    assert backend.memories()[0].content == "Prefers Scala."

    delete_session = backend.new_session()
    delete_statement = "Forget my preferred programming language"
    backend.append(delete_session.session_id, [{"role": "user", "content": delete_statement}])
    delete_model = ScriptedModel([memory_plan(decision("DELETE", path, "", delete_statement, ""))])
    assert (
        end_session_memory(backend, delete_model, delete_session.session_id)[0]["action"]
        == "DELETE"
    )
    assert backend.memories() == []


def test_update_replaces_correction_and_preserves_compatible_facts(backend: Backend) -> None:
    path = "/memories/preferences/coding/style"
    backend.memory_store.add(
        backend.actor_id,
        path,
        "Prefers Python examples and concise explanations.",
        "Coding response style",
    )
    statement = "Use Scala instead of Python, but keep explanations concise"

    backend.apply_memory_decisions(
        [
            decision(
                "UPDATE",
                path,
                "Prefers Scala examples and concise explanations.",
                statement,
                "Coding response style",
            )
        ],
        [statement],
    )

    assert backend.memories()[0].content == "Prefers Scala examples and concise explanations."


def test_duplicate_path_plan_is_rejected_before_any_write(backend: Backend) -> None:
    path = "/memories/preferences/communication/verbosity"
    statement = "Keep answers concise"

    with pytest.raises(ValueError, match="duplicate path"):
        backend.apply_memory_decisions(
            [
                decision("ADD", path, "Prefers concise answers.", statement),
                decision("ADD", path, "Prefers brief answers.", statement),
            ],
            [statement],
        )

    assert backend.memories() == []


def test_invalid_later_decision_prevents_earlier_valid_write(backend: Backend) -> None:
    statement = "Keep answers concise and use SQL examples"

    with pytest.raises(ValueError, match="inside /memories/"):
        backend.apply_memory_decisions(
            [
                decision(
                    "ADD",
                    "/memories/preferences/communication/verbosity",
                    "Prefers concise answers.",
                    "Keep answers concise",
                ),
                decision(
                    "ADD",
                    "/outside/preferences/coding/examples",
                    "Prefers SQL examples.",
                    "use SQL examples",
                ),
            ],
            [statement],
        )

    assert backend.memories() == []


def test_memory_plan_rejects_delete_without_verbatim_user_evidence(backend: Backend) -> None:
    path = "/memories/preferences/coding/language.md"
    backend.memory_store.add(backend.actor_id, path, "Prefers Python.", "Language")
    with pytest.raises(ValueError, match="verbatim evidence"):
        backend.apply_memory_decisions(
            [decision("DELETE", path, "", "user asked to forget", "")],
            ["Tell me about Python"],
        )
    assert len(backend.memories()) == 1


def test_noop_conflict_and_omission_preserve_existing_memories(backend: Backend) -> None:
    verbosity = "/memories/preferences/communication/verbosity.md"
    language = "/memories/preferences/coding/language.md"
    backend.memory_store.add(backend.actor_id, verbosity, "Prefers concise answers.", "Verbosity")
    backend.memory_store.add(backend.actor_id, language, "Prefers Python.", "Language")

    results = backend.apply_memory_decisions(
        [
            decision(
                "NO_OP",
                verbosity,
                "Likes brief responses.",
                "",
                "Response verbosity",
            ),
            decision(
                "CONFLICT",
                language,
                "",
                "",
                "Preferred programming language",
            ),
        ],
        ["Maybe use Scala sometimes"],
    )

    assert [result["action"] for result in results] == ["NO_OP", "CONFLICT"]
    assert {memory.content for memory in backend.memories()} == {
        "Prefers concise answers.",
        "Prefers Python.",
    }


def test_empty_plan_never_deletes_unmentioned_memory(backend: Backend) -> None:
    path = "/memories/preferences/communication/verbosity.md"
    backend.memory_store.add(backend.actor_id, path, "Prefers concise answers.", "Verbosity")

    assert backend.apply_memory_decisions([], ["Hello there"]) == []
    assert backend.memories()[0].path == path


def test_extensionless_managed_path_is_valid(backend: Backend) -> None:
    path = "/memories/profile/identity-and-interests"
    statement = "I am an engineer in Brooklyn who enjoys tennis"

    result = backend.apply_memory_decisions(
        [decision("ADD", path, "Engineer in Brooklyn who enjoys tennis.", statement)],
        [statement],
    )

    assert result[0]["action"] == "ADD"
    assert backend.memories()[0].path == path


@pytest.mark.parametrize(
    "bad_path",
    [
        "/other/preferences/style.md",
        "/memories/preferences/../private.md",
        "/memories/preferences/Bad Topic.md",
        "/memories/preferences/trailing/",
    ],
)
def test_reconciliation_rejects_paths_outside_canonical_schema(
    backend: Backend, bad_path: str
) -> None:
    with pytest.raises(ValueError, match="path"):
        backend.apply_memory_decisions(
            [decision("ADD", bad_path, "Preference", "I prefer it")],
            ["I prefer it"],
        )
    assert backend.memories() == []


def test_model_uses_selected_endpoint_and_no_tool_calls_on_last_round() -> None:
    calls: list[dict[str, Any]] = []

    def create(**kwargs: Any) -> SimpleNamespace:
        calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Hello", tool_calls=None))]
        )

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    model = DatabricksModel(client, "selected-endpoint")
    assert model.complete([{"role": "user", "content": "Hi"}], TOOLS, True)["content"] == "Hello"
    assert calls[0]["model"] == "selected-endpoint"
    assert calls[0]["tool_choice"] == "none"
    assert "reasoning_effort" not in calls[0]


def test_reconciliation_forces_structured_plan_tool() -> None:
    calls: list[dict[str, Any]] = []

    def create(**kwargs: Any) -> SimpleNamespace:
        calls.append(kwargs)
        tool_call = SimpleNamespace(
            id="plan",
            function=SimpleNamespace(name="submit_memory_plan", arguments='{"decisions":[]}'),
        )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=None, tool_calls=[tool_call]))]
        )

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    DatabricksModel(client, "selected-endpoint").complete(
        [{"role": "user", "content": "Plan"}], [MEMORY_PLAN_TOOL], False
    )

    assert calls[0]["tool_choice"] == {
        "type": "function",
        "function": {"name": "submit_memory_plan"},
    }


def test_optional_reasoning_effort_is_forwarded() -> None:
    calls: list[dict[str, Any]] = []

    def create(**kwargs: Any) -> SimpleNamespace:
        calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Hello", tool_calls=None))]
        )

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    model = DatabricksModel(client, "selected-endpoint", reasoning_effort="none")
    model.complete([{"role": "user", "content": "Hi"}], TOOLS, False)
    assert calls[0]["reasoning_effort"] == "none"
