"""Keep authenticated identity, session ownership, and memory scoping outside the model."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any
from uuid import uuid4

from databricks.sdk import WorkspaceClient
from databricks_agentbricks.errors import AgentCliError
from databricks_agentkit import AgentKitClient
from databricks_agentkit.memory_store import Memory, MemoryStore
from databricks_agentkit.session_store import Session, SessionStore

Trace = Callable[[str, dict[str, Any]], None]


def quiet_trace(event: str, details: dict[str, Any]) -> None:
    """Ignore trace events when no observer is configured."""


class Backend:
    """Bind both stores to one verified caller and one application memory namespace.

    Args:
        session_store: Existing managed session store.
        memory_store: Existing managed memory store.
        actor_id: Verified caller ID, resolved by trusted application code.
        namespace: Prefix reserved for this sample's memory entries.
        trace: Observer for API activity; never receives authentication tokens.
    """

    def __init__(
        self,
        session_store: SessionStore,
        memory_store: MemoryStore,
        actor_id: str,
        namespace: str = "/preferences/",
        trace: Trace = quiet_trace,
    ) -> None:
        if not actor_id.strip():
            raise ValueError("A verified actor ID is required; shared fallback is not allowed.")
        if not namespace.startswith("/") or not namespace.endswith("/"):
            raise ValueError("The memory namespace must start and end with '/'.")
        self.session_store = session_store
        self.memory_store = memory_store
        self.actor_id = actor_id
        self.namespace = namespace
        self.trace = trace

    def new_session(self) -> Session:
        """Create a caller-chosen ID and recover a committed create by read, never write retry."""
        session_id = str(uuid4())
        self.trace("session.creating", {"session_id": session_id})
        try:
            session = self.session_store.add(actor_id=self.actor_id, session_id=session_id)
        except AgentCliError as create_error:
            try:
                session = self.session(session_id)
            except AgentCliError:
                raise create_error
            self.trace("session.create_recovered", {"session_id": session_id})
        self.trace("session.created", {"session_id": session.session_id})
        return session

    def session(self, session_id: str) -> Session:
        """Verify ownership before exposing a conversation or any of its items."""
        session = self.session_store.get(session_id)
        if session.actor_id != self.actor_id:
            raise PermissionError("This session does not belong to the authenticated caller.")
        return session

    def sessions(self) -> list[Session]:
        """List only this caller's sessions, independent of server filter syntax."""
        return [
            session for session in self.session_store.list() if session.actor_id == self.actor_id
        ]

    def history(self, session_id: str) -> list[dict[str, Any]]:
        """Rebuild a conversation from chronological, auto-paginated session items."""
        items = self.session(session_id).list_items(order_by="create_time asc")
        history = []
        for item in items:
            if not isinstance(item.data, dict) or item.data.get("role") not in {
                "user",
                "assistant",
                "tool",
            }:
                raise ValueError("Unsupported session item; this sample stores chat messages only.")
            history.append(item.data)
        self.trace("session.loaded", {"session_id": session_id, "items": len(history)})
        return history

    def append(self, session_id: str, messages: list[dict[str, Any]]) -> None:
        """Append one completed turn, including any memory tool calls and results."""
        self.session(session_id).append_items(messages)
        self.trace("session.appended", {"session_id": session_id, "items": len(messages)})

    def delete_session(self, session_id: str) -> None:
        """Delete one owned session, without cascading to branches or touching memory."""
        self.session(session_id).delete()
        self.trace("session.deleted", {"session_id": session_id})

    def memories(self) -> list[Memory]:
        """Inspect memory through the list API, not a relevance-limited search."""
        return list(self.memory_store.list(actor_id=self.actor_id, path_prefix=self.namespace))

    def recall(self, query: str) -> list[dict[str, Any]]:
        """Search the caller's namespace with BM25 relevance-ranked retrieval."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("A non-empty memory query is required.")
        results = self.memory_store.search(
            actor_id=self.actor_id, query=query, path_prefix=self.namespace, limit=5
        )
        recalled = [
            {
                "path": result.memory.path,
                "description": result.memory.description,
                "content": result.memory.content,
                "score": result.score,
            }
            for result in results
        ]
        self.trace("memory.searched", {"query": query, "results": recalled})
        return recalled

    def remember(self, topic: str, content: str) -> dict[str, str]:
        """Create or update one canonical preference after application approval."""
        path = self._path(topic)
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Memory content must be a non-empty string.")
        existing = self._find(path)
        description = topic.replace("-", " ")
        if existing:
            existing.update(content=content, description=description)
            action = "updated"
        else:
            self.memory_store.add(
                actor_id=self.actor_id,
                path=path,
                content=content,
                description=description,
            )
            action = "created"
        self.trace(f"memory.{action}", {"path": path, "content": content})
        return {"status": action, "path": path}

    def forget(self, topic: str) -> dict[str, str]:
        """Delete one canonical preference, leaving past transcripts unchanged."""
        path = self._path(topic)
        existing = self._find(path)
        if existing:
            existing.delete()
        self.trace("memory.forgotten", {"path": path, "existed": existing is not None})
        return {"status": "deleted" if existing else "not_found", "path": path}

    def _path(self, topic: str) -> str:
        if not isinstance(topic, str) or not re.fullmatch(r"[a-z][a-z0-9-]{0,63}", topic):
            raise ValueError("Topic must be 1–64 lowercase letters, numbers, or hyphens.")
        return f"{self.namespace}{topic}.md"

    def _find(self, path: str) -> Memory | None:
        matches = [
            memory
            for memory in self.memory_store.list(actor_id=self.actor_id, path_prefix=path)
            if memory.path == path and memory.session_id is None
        ]
        if len(matches) > 1:
            raise ValueError("Multiple canonical entries exist for this topic; inspect the store.")
        return matches[0] if matches else None


def caller_id(workspace: WorkspaceClient) -> str:
    """Resolve identity from Databricks authentication, never from a command-line actor ID."""
    actor_id = workspace.current_user.me().id
    if not actor_id:
        raise ValueError("Databricks did not return a caller ID; refusing unscoped access.")
    return actor_id


def connect(
    workspace: WorkspaceClient,
    session_store: str,
    memory_store: str,
    namespace: str = "/preferences/",
    trace: Trace = quiet_trace,
) -> Backend:
    """Open existing stores; never provision infrastructure implicitly."""
    actor_id = caller_id(workspace)
    client = AgentKitClient(workspace)
    return Backend(
        client.session_stores.get(session_store),
        client.memory_stores.get(memory_store),
        actor_id,
        namespace,
        trace,
    )


def initialize(workspace: WorkspaceClient, session_store: str, memory_store: str) -> None:
    """Explicitly create only missing stores after the caller has accepted provisioning costs."""
    client = AgentKitClient(workspace)
    sessions = {store.name for store in client.session_stores.list()}
    if session_store not in sessions:
        client.session_stores.create(session_store)
    memories = {store.display_name for store in client.memory_stores.list()}
    if memory_store not in memories:
        client.memory_stores.create(memory_store)
