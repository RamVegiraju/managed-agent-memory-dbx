"""Smoke-test the Streamlit entry point without Databricks network access."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from streamlit.testing.v1 import AppTest

from memory_demo.agent import DatabricksModel
from memory_demo.backend import Backend
from tests.fakes import FakeMemoryStore, FakeSessionStore


def test_streamlit_initial_screen(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DATABRICKS_CONFIG_PROFILE", raising=False)
    monkeypatch.delenv("DATABRICKS_MODEL", raising=False)
    script = Path(__file__).parents[1] / "streamlit_app.py"
    app = AppTest.from_file(script).run(timeout=10)

    assert not app.exception
    assert app.title[0].value == "Managed sessions + memory"
    assert "before starting the app" in app.warning[0].value


def test_streamlit_end_session_runs_managed_memory_flow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DATABRICKS_CONFIG_PROFILE", "test-profile")
    monkeypatch.setenv("DATABRICKS_MODEL", "test-model")
    sessions = FakeSessionStore()
    memories = FakeMemoryStore()
    backend = Backend(sessions, memories, "verified-user")
    session = backend.new_session()
    statement = "I prefer concise answers"
    backend.append(
        session.session_id,
        [
            {"role": "user", "content": statement},
            {"role": "assistant", "content": "Understood."},
        ],
    )
    path = "/memories/communication/verbosity"
    session.extracted_memories = [
        SimpleNamespace(
            path=path,
            content="Prefers concise answers.",
            description="Response verbosity",
            session_id=session.session_id,
        )
    ]
    plan = {
        "decisions": [
            {
                "action": "ADD",
                "path": path,
                "content": "Prefers concise answers.",
                "description": "Response verbosity",
                "reason": "Explicit durable preference",
                "evidence": statement,
            }
        ]
    }

    def create(**kwargs: object) -> SimpleNamespace:
        call = SimpleNamespace(
            id="plan",
            function=SimpleNamespace(
                name="submit_memory_plan",
                arguments=json.dumps(plan),
            ),
        )
        message = SimpleNamespace(content=None, tool_calls=[call])
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    model = DatabricksModel(client, "test-model")
    script = Path(__file__).parents[1] / "streamlit_app.py"
    app = AppTest.from_file(script)
    app.session_state["backend"] = backend
    app.session_state["model"] = model
    app.session_state["user_id"] = backend.actor_id
    app.session_state["session_id"] = session.session_id
    app.session_state["workspace_host"] = "https://test-workspace"
    app.session_state["activity"] = []

    app.run(timeout=10)
    assert not app.exception
    end_button = next(button for button in app.button if button.label.startswith("End session"))
    end_button.click().run(timeout=10)

    assert not app.exception
    assert len(backend.memories()) == 1
    assert backend.memories()[0].path == path
    assert app.session_state.session_id is None
    assert app.success[0].value == "Session ended. Applied 1 durable memory change(s)."
    assert app.session_state.last_memory_plan[0]["action"] == "ADD"


def test_streamlit_start_fresh_session_does_not_create_empty_record() -> None:
    sessions = FakeSessionStore()
    backend = Backend(sessions, FakeMemoryStore(), "verified-user")
    session = backend.new_session()
    model = DatabricksModel(SimpleNamespace(), "test-model")
    script = Path(__file__).parents[1] / "streamlit_app.py"
    app = AppTest.from_file(script)
    app.session_state["backend"] = backend
    app.session_state["model"] = model
    app.session_state["user_id"] = backend.actor_id
    app.session_state["session_id"] = session.session_id
    app.session_state["workspace_host"] = "https://test-workspace"
    app.session_state["activity"] = []

    app.run(timeout=10)
    fresh_button = next(button for button in app.button if button.label == "Start fresh session")
    fresh_button.click().run(timeout=10)

    assert not app.exception
    assert app.session_state.session_id is None
    assert sessions.counter == 1
    assert backend.memories() == []
    assert app.success[0].value.startswith("Ready for a fresh session")


def test_streamlit_creates_session_lazily_on_first_message() -> None:
    sessions = FakeSessionStore()
    backend = Backend(sessions, FakeMemoryStore(), "verified-user")

    def create(**kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Hello.", tool_calls=None))]
        )

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    script = Path(__file__).parents[1] / "streamlit_app.py"
    app = AppTest.from_file(script)
    app.session_state["backend"] = backend
    app.session_state["model"] = DatabricksModel(client, "test-model")
    app.session_state["user_id"] = backend.actor_id
    app.session_state["session_id"] = None
    app.session_state["workspace_host"] = "https://test-workspace"
    app.session_state["activity"] = []

    app.run(timeout=10)
    assert sessions.counter == 0
    app.chat_input[0].set_value("Hello").run(timeout=10)

    assert not app.exception
    assert sessions.counter == 1
    session_id = app.session_state.session_id
    assert isinstance(session_id, str)
    assert [item["content"] for item in backend.history(session_id)] == ["Hello", "Hello."]


def test_streamlit_end_session_failure_is_prominent_and_keeps_current_session() -> None:
    backend = Backend(FakeSessionStore(), FakeMemoryStore(), "verified-user")
    session = backend.new_session()
    statement = "I prefer concise answers"
    backend.append(session.session_id, [{"role": "user", "content": statement}])
    session.extracted_memories = []
    plan = {
        "decisions": [
            {
                "action": "ADD",
                "path": "/outside/style",
                "content": "Prefers concise answers.",
                "description": "Response style",
                "reason": "Explicit preference",
                "evidence": statement,
            }
        ]
    }

    def create(**kwargs: object) -> SimpleNamespace:
        call = SimpleNamespace(
            id="plan",
            function=SimpleNamespace(name="submit_memory_plan", arguments=json.dumps(plan)),
        )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=None, tool_calls=[call]))]
        )

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    script = Path(__file__).parents[1] / "streamlit_app.py"
    app = AppTest.from_file(script)
    app.session_state["backend"] = backend
    app.session_state["model"] = DatabricksModel(client, "test-model")
    app.session_state["user_id"] = backend.actor_id
    app.session_state["session_id"] = session.session_id
    app.session_state["workspace_host"] = "https://test-workspace"
    app.session_state["activity"] = []

    app.run(timeout=10)
    end_button = next(button for button in app.button if button.label.startswith("End session"))
    end_button.click().run(timeout=10)

    assert not app.exception
    assert app.session_state.session_id == session.session_id
    assert backend.memories() == []
    assert app.error[0].value.startswith("End session failed:")
    assert "/memories/" in app.error[0].value
