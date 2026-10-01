"""Local Streamlit UI that makes managed session and memory API boundaries visible."""

from __future__ import annotations

import os
from typing import Any

import streamlit as st
from databricks.sdk import WorkspaceClient
from databricks.sdk.core import Config
from databricks_openai import DatabricksOpenAI

from memory_demo.agent import (
    END_SESSION_PROMPT,
    RECALL_ONLY_TOOLS,
    Agent,
    DatabricksModel,
    end_session_memory,
)
from memory_demo.backend import Backend, connect
from memory_demo.cli import check_model


@st.cache_resource(show_spinner=False)
def workspace_for(profile: str) -> WorkspaceClient:
    """Create an SDK client from a CLI profile authenticated before Streamlit starts."""
    return WorkspaceClient(config=Config(profile=profile))


def visible_messages(history: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Hide function-call protocol items while preserving the exact stored history elsewhere."""
    return [
        {"role": message["role"], "content": message["content"]}
        for message in history
        if message.get("role") in {"user", "assistant"} and isinstance(message.get("content"), str)
    ]


def memory_output(backend: Backend) -> dict[str, Any]:
    """Serialize `MemoryStore.list(actor_id, path_prefix)` for the raw inspector."""
    return {
        "user_id": backend.actor_id,
        "memories": [
            {
                "path": memory.path,
                "description": memory.description,
                "content": memory.content,
                "session_id": memory.session_id,
            }
            for memory in backend.memories()
        ],
    }


def session_output(backend: Backend) -> dict[str, Any]:
    """Serialize `SessionStore.list()` after Backend filters it to the authenticated user."""
    return {
        "user_id": backend.actor_id,
        "sessions": [
            {
                "session_id": session.session_id,
                "create_time": str(session.create_time) if session.create_time else None,
            }
            for session in backend.sessions()
        ],
    }


def record_activity(event: str, details: dict[str, Any]) -> None:
    """Capture Backend observer events so users can see which managed operation occurred."""
    activity = st.session_state.setdefault("activity", [])
    activity.append({"event": event, "details": details})
    del activity[:-50]


def environment() -> dict[str, str]:
    """Read workspace-agnostic settings supplied before Streamlit starts."""
    return {
        "profile": os.getenv("DATABRICKS_CONFIG_PROFILE", ""),
        "model": os.getenv("DATABRICKS_MODEL", ""),
        "reasoning_effort": os.getenv("DATABRICKS_REASONING_EFFORT", ""),
        "session_store": os.getenv("DEMO_SESSION_STORE", "memory-demo-sessions"),
        "memory_store": os.getenv("DEMO_MEMORY_STORE", "memory-demo-memory"),
    }


def connect_from_environment(settings: dict[str, str]) -> None:
    """Open existing stores using CLI auth; defer session creation until the first message."""
    if not settings["profile"] or not settings["model"]:
        raise ValueError("DATABRICKS_CONFIG_PROFILE and DATABRICKS_MODEL must be set.")
    workspace = workspace_for(settings["profile"])
    check_model(workspace, settings["model"])
    st.session_state.activity = []
    backend = connect(
        workspace,
        settings["session_store"],
        settings["memory_store"],
        observer=record_activity,
    )
    st.session_state.update(
        backend=backend,
        model=DatabricksModel(
            DatabricksOpenAI(workspace_client=workspace),
            settings["model"],
            settings["reasoning_effort"] or None,
        ),
        user_id=backend.actor_id,
        session_id=None,
        workspace_host=workspace.config.host,
    )


def setup_instructions() -> None:
    """Explain the required CLI authentication and environment setup."""
    st.warning("Configure Databricks authentication before starting the app.")
    st.code(
        "\n".join(
            [
                "databricks auth login --host https://YOUR-WORKSPACE --profile YOUR_PROFILE",
                "export DATABRICKS_CONFIG_PROFILE=YOUR_PROFILE",
                "export DATABRICKS_MODEL=YOUR_CHAT_ENDPOINT",
                "export DATABRICKS_REASONING_EFFORT=none  # GPT-5.6-Sol only",
                "uv run streamlit run streamlit_app.py",
            ]
        ),
        language="bash",
    )


def configuration_panel(settings: dict[str, str], backend: Backend) -> None:
    """Show auto-detected identity and configuration as read-only sidebar values."""
    with st.sidebar:
        st.header("Connected as")
        st.text_input("Authenticated user ID", backend.actor_id, disabled=True)
        st.text_input("CLI profile", settings["profile"], disabled=True)
        st.text_input("Workspace", st.session_state.workspace_host, disabled=True)
        st.divider()
        st.header("Resources")
        st.text_input("Model endpoint", settings["model"], disabled=True)
        st.text_input("Session store", settings["session_store"], disabled=True)
        st.text_input("Memory store", settings["memory_store"], disabled=True)
        if st.button("Reconnect from environment", use_container_width=True):
            for key in ["backend", "model", "session_id", "user_id", "workspace_host"]:
                st.session_state.pop(key, None)
            workspace_for.clear()
            st.rerun()


def session_panel(
    backend: Backend,
    model: DatabricksModel,
    sessions: list[dict[str, Any]],
    history: list[dict[str, Any]],
) -> None:
    """End-and-capture or resume sessions through managed memory and session methods."""
    with st.sidebar:
        st.divider()
        st.header("Conversation")
        if st.button(
            "End session and capture memory",
            type="primary",
            use_container_width=True,
            disabled=not history,
        ):
            try:
                with st.spinner("Extracting durable memory…"):
                    session_id = st.session_state.session_id
                    if not isinstance(session_id, str):
                        raise ValueError("No managed session is active.")
                    decisions = end_session_memory(backend, model, session_id)
                    changed = sum(
                        decision["action"] in {"ADD", "UPDATE", "DELETE"} for decision in decisions
                    )
                    st.session_state.last_memory_plan = decisions
                    notice = (
                        f"Session ended. Applied {changed} durable memory change(s)."
                        if changed
                        else "Session ended. No durable memory changes were needed."
                    )
                    st.session_state.session_id = None
                    st.session_state.capture_notice = notice
                    st.session_state.pop("capture_error", None)
                st.rerun()
            except Exception as error:
                st.session_state.capture_error = str(error)
                st.session_state.pop("capture_notice", None)
                st.rerun()
        if st.button(
            "Start fresh session",
            use_container_width=True,
            disabled=st.session_state.session_id is None,
        ):
            st.session_state.session_id = None
            st.session_state.capture_notice = (
                "Ready for a fresh session. It will be created with your first message."
            )
            st.session_state.pop("capture_error", None)
            st.rerun()
        session_ids = [session["session_id"] for session in sessions]
        current = st.session_state.session_id
        options: list[str | None] = [None, *session_ids]
        index = options.index(current) if current in options else 0
        selected = st.selectbox(
            "Current session",
            options,
            index=index,
            format_func=lambda value: (
                "New session (created on first message)" if value is None else value
            ),
        )
        if selected != current:
            if isinstance(selected, str):
                backend.session(selected)
            st.session_state.session_id = selected
            st.rerun()
        st.caption(
            "Ending runs managed extraction, semantic reconciliation, and validated memory writes."
        )


def chat_panel(
    backend: Backend,
    model: DatabricksModel,
    history: list[dict[str, Any]],
) -> None:
    """Run Agent.ask with recall only; writes happen automatically when the session ends."""
    st.subheader("Conversation")
    st.caption("Session context: messages stored only in the selected conversation.")
    if not history:
        st.info("No transcript yet. A managed session is created when you send the first message.")
    for message in visible_messages(history):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Ask something or share a preference to carry into future sessions")
    if prompt:
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                try:
                    session_id = st.session_state.session_id
                    if not isinstance(session_id, str):
                        session_id = backend.new_session().session_id
                        st.session_state.session_id = session_id
                    agent = Agent(
                        backend,
                        model,
                        tools=RECALL_ONLY_TOOLS,
                        system_prompt=END_SESSION_PROMPT,
                    )
                    st.markdown(agent.ask(session_id, prompt))
                except Exception as error:
                    st.error(str(error))
        st.rerun()


def inspector_panel(
    memories: dict[str, Any],
    sessions: dict[str, Any],
    history: list[dict[str, Any]],
) -> None:
    """Display raw results from memory list, session list, and session item APIs."""
    st.subheader("Raw managed API output")
    st.caption("Switch views to see exactly what is partitioned by user versus session.")
    view = st.radio(
        "Inspect",
        ["User memories", "User sessions", "Selected session history", "Last memory plan"],
        horizontal=True,
        label_visibility="collapsed",
    )
    if view == "User memories":
        st.markdown("**`MemoryStore.list(actor_id=user_id)`**")
        st.json(memories, expanded=True)
    elif view == "User sessions":
        st.markdown("**`SessionStore.list()` filtered by `actor_id=user_id`**")
        st.caption("Includes every persisted session, including older or empty sessions.")
        st.json(sessions, expanded=True)
    elif view == "Selected session history":
        if st.session_state.session_id is None:
            st.markdown("**No managed session exists until the first message is sent.**")
        else:
            st.markdown('**`Session.list_items(order_by="create_time asc")`**')
        st.json(
            {"session_id": st.session_state.session_id, "history": history},
            expanded=True,
        )
    else:
        st.markdown("**Managed extraction → structured reconciliation → validated writes**")
        st.json(st.session_state.get("last_memory_plan", []), expanded=True)

    with st.expander("Recent managed API activity"):
        activity = st.session_state.get("activity", [])
        if activity:
            st.json(list(reversed(activity)), expanded=False)
        else:
            st.caption("No operations recorded yet.")


def render() -> None:
    """Auto-connect from environment settings and render chat beside raw managed data."""
    st.title("Managed sessions + memory")
    st.caption("A visual demonstration of conversation context and cross-session user memory.")
    settings = environment()
    if "backend" not in st.session_state:
        try:
            with st.spinner("Connecting with your Databricks CLI profile…"):
                connect_from_environment(settings)
            st.rerun()
        except Exception as error:
            st.error(str(error))
            setup_instructions()
            return

    backend = st.session_state.get("backend")
    model = st.session_state.get("model")
    if not isinstance(backend, Backend) or not isinstance(model, DatabricksModel):
        st.error("The UI connection state is invalid. Use Reconnect from environment.")
        return

    memories = memory_output(backend)
    sessions = session_output(backend)
    session_id = st.session_state.session_id
    history = backend.history(session_id) if isinstance(session_id, str) else []
    configuration_panel(settings, backend)
    session_panel(backend, model, sessions["sessions"], history)

    user_column, session_column, memory_column = st.columns(3)
    user_column.metric("User", backend.actor_id)
    session_column.metric("Session", session_id or "Not started")
    memory_column.metric("Durable memories", len(memories["memories"]))
    notice = st.session_state.pop("capture_notice", None)
    if notice:
        st.success(notice)
    capture_error = st.session_state.pop("capture_error", None)
    if capture_error:
        st.error(f"End session failed: {capture_error}")
    st.divider()
    chat_column, inspector_column = st.columns([3, 2], gap="large")
    with chat_column:
        chat_panel(backend, model, history)
    with inspector_column:
        inspector_panel(memories, sessions, history)
