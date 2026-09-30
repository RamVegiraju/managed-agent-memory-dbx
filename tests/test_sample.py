"""Verify context boundaries, approved tool execution, and portable configuration."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest
from databricks.sdk.service.serving import EndpointStateReady
from databricks_agentbricks.errors import AgentCliError

from memory_demo.agent import TOOLS, Agent, DatabricksModel
from memory_demo.backend import Backend, caller_id
from memory_demo.cli import check_model, parser, validate
from memory_demo.demo import run_demo
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


def test_memory_search_binds_actor_and_namespace(backend: Backend) -> None:
    backend.memory_store.add("another-user", "/preferences/style.md", "private", "style")
    backend.remember("response-preferences", "PySpark examples")
    assert len(backend.recall("PySpark")) == 1
    assert backend.memory_store.search_calls[-1] == {
        "actor_id": "verified-user",
        "query": "PySpark",
        "path_prefix": "/preferences/",
        "limit": 5,
    }


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


def test_demo_leaves_other_namespaces_untouched(capsys: pytest.CaptureFixture[str]) -> None:
    sessions = FakeSessionStore()
    memories = FakeMemoryStore()
    memories.add("verified-user", "/preferences/real.md", "real preference", "real")

    def factory(namespace: str) -> Backend:
        return Backend(sessions, memories, "verified-user", namespace)

    run_demo(factory, cleanup=True)
    assert len(memories.entries) == 1
    assert memories.entries[0].path == "/preferences/real.md"
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
