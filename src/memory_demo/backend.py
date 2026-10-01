"""Keep authenticated identity, session ownership, and memory scoping outside the model."""

from __future__ import annotations

import re
from collections.abc import Callable
from time import sleep
from typing import Any
from uuid import uuid4

from databricks.sdk import WorkspaceClient
from databricks_agentbricks.errors import AgentCliError
from databricks_agentkit import AgentKitClient
from databricks_agentkit.memory_store import Memory, MemoryStore
from databricks_agentkit.session_store import Session, SessionStore

ActivityObserver = Callable[[str, dict[str, Any]], None]


def quiet_observer(event: str, details: dict[str, Any]) -> None:
    """Ignore activity events when no observer is configured."""


class Backend:
    """Bind both stores to one verified caller and one application memory namespace.

    Args:
        session_store: Existing managed session store.
        memory_store: Existing managed memory store.
        actor_id: Verified caller ID, resolved by trusted application code.
        namespace: Prefix reserved for this sample's memory entries.
        observer: Callback for API activity; never receives authentication tokens.
    """

    def __init__(
        self,
        session_store: SessionStore,
        memory_store: MemoryStore,
        actor_id: str,
        namespace: str = "/memories/",
        observer: ActivityObserver = quiet_observer,
    ) -> None:
        if not actor_id.strip():
            raise ValueError("A verified actor ID is required; shared fallback is not allowed.")
        if not namespace.startswith("/") or not namespace.endswith("/"):
            raise ValueError("The memory namespace must start and end with '/'.")
        self.session_store = session_store
        self.memory_store = memory_store
        self.actor_id = actor_id
        self.namespace = namespace
        self.observer = observer

    def new_session(self) -> Session:
        """Call `SessionStore.add` and recover a committed create with bounded reads."""
        session_id = str(uuid4())
        self.observer("session.creating", {"session_id": session_id})
        try:
            session = self.session_store.add(actor_id=self.actor_id, session_id=session_id)
        except AgentCliError as create_error:
            for attempt in range(5):
                if attempt:
                    sleep(1)
                try:
                    session = self.session(session_id)
                except AgentCliError:
                    continue
                self.observer(
                    "session.create_recovered",
                    {"session_id": session_id, "read_attempts": attempt + 1},
                )
                break
            else:
                raise create_error
        self.observer("session.created", {"session_id": session.session_id})
        return session

    def session(self, session_id: str) -> Session:
        """Call `SessionStore.get`, then verify ownership before exposing the session."""
        session = self.session_store.get(session_id)
        if session.actor_id != self.actor_id:
            raise PermissionError("This session does not belong to the authenticated caller.")
        return session

    def sessions(self) -> list[Session]:
        """Call `SessionStore.list` and return only the authenticated caller's sessions."""
        return [
            session for session in self.session_store.list() if session.actor_id == self.actor_id
        ]

    def history(self, session_id: str) -> list[dict[str, Any]]:
        """Call `Session.list_items` and rebuild the auto-paginated chronological transcript."""
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
        self.observer("session.loaded", {"session_id": session_id, "items": len(history)})
        return history

    def append(self, session_id: str, messages: list[dict[str, Any]]) -> None:
        """Call `Session.append_items` with a completed turn and its memory tool protocol."""
        self.session(session_id).append_items(messages)
        self.observer("session.appended", {"session_id": session_id, "items": len(messages)})

    def delete_session(self, session_id: str) -> None:
        """Call `Session.delete` without cascading to branches or touching memory."""
        self.session(session_id).delete()
        self.observer("session.deleted", {"session_id": session_id})

    def memories(self) -> list[Memory]:
        """List canonical user memories; session transcripts retain their provenance."""
        return [
            memory
            for memory in self.memory_store.list(actor_id=self.actor_id, path_prefix=self.namespace)
            if memory.session_id is None
        ]

    def recall(self, query: str) -> list[dict[str, Any]]:
        """Call `MemoryStore.search` (`entries:search`) for relevance-ranked user recall."""
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
            if result.memory.session_id is None
        ]
        self.observer("memory.searched", {"query": query, "results": recalled})
        return recalled

    def remember(self, topic: str, content: str) -> dict[str, str]:
        """Call `MemoryStore.add` or `Memory.update` for one approved canonical topic."""
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
        self.observer(f"memory.{action}", {"path": path, "content": content})
        return {"status": action, "path": path}

    def forget(self, topic: str) -> dict[str, str]:
        """Call `Memory.delete` for one topic while leaving session transcripts unchanged."""
        path = self._path(topic)
        existing = self._find(path)
        if existing:
            existing.delete()
        self.observer("memory.forgotten", {"path": path, "existed": existing is not None})
        return {"status": "deleted" if existing else "not_found", "path": path}

    def extract_memories(self, session_id: str, instructions: str) -> list[dict[str, Any]]:
        """Call managed `Session.extract_memories` in dry-run mode for atomic candidates."""
        extracted = self.session(session_id).extract_memories(
            memory_store=self.memory_store.name,
            instructions=instructions,
            dry_run=True,
        )
        candidates = [
            {
                "path": memory.path,
                "content": memory.content,
                "description": memory.description,
                "source_session_id": memory.session_id,
            }
            for memory in extracted
        ]
        self.observer(
            "memory.extracted",
            {"session_id": session_id, "dry_run": True, "candidates": candidates},
        )
        return candidates

    def apply_memory_decisions(
        self,
        decisions: list[dict[str, str]],
        user_statements: list[str],
    ) -> list[dict[str, str]]:
        """Validate a semantic plan completely, then execute managed memory mutations."""
        existing_by_path = {memory.path: memory for memory in self.memories()}
        seen_paths: set[str] = set()
        mutating = {"ADD", "UPDATE", "DELETE"}

        for decision in decisions:
            action = decision["action"]
            path = decision["path"]
            if path in seen_paths:
                raise ValueError(f"The reconciliation plan contains duplicate path {path}.")
            seen_paths.add(path)
            self._validate_path(path)
            exists = path in existing_by_path
            if action == "ADD" and exists:
                raise ValueError(f"ADD targets existing memory {path}; use UPDATE or NO_OP.")
            if action in {"UPDATE", "DELETE", "NO_OP"} and not exists:
                raise ValueError(f"{action} targets missing memory {path}.")
            if action in {"ADD", "UPDATE"} and (
                not decision["content"].strip() or not decision["description"].strip()
            ):
                raise ValueError(f"{action} requires non-empty content and description.")
            if action in mutating:
                evidence = decision["evidence"].strip()
                if not evidence or not any(evidence in statement for statement in user_statements):
                    raise ValueError(
                        f"{action} for {path} lacks verbatim evidence from a user message."
                    )

        results: list[dict[str, str]] = []
        for decision in decisions:
            action = decision["action"]
            path = decision["path"]
            applied = action
            if action == "ADD":
                self.memory_store.add(
                    actor_id=self.actor_id,
                    path=path,
                    content=decision["content"].strip(),
                    description=decision["description"].strip(),
                )
            elif action == "UPDATE":
                existing = existing_by_path[path]
                if self._normalized(existing.content) == self._normalized(decision["content"]):
                    applied = "NO_OP"
                else:
                    existing.update(
                        content=decision["content"].strip(),
                        description=decision["description"].strip(),
                    )
            elif action == "DELETE":
                existing_by_path[path].delete()
            result = {
                "action": applied,
                "requested_action": action,
                "path": path,
                "content": decision["content"],
                "description": decision["description"],
                "reason": decision["reason"],
                "evidence": decision["evidence"],
            }
            results.append(result)
            self.observer(f"memory.reconciled_{applied.lower()}", result)
        return results

    def _path(self, topic: str) -> str:
        if not isinstance(topic, str) or not re.fullmatch(r"[a-z][a-z0-9-]{0,63}", topic):
            raise ValueError("Topic must be 1–64 lowercase letters, numbers, or hyphens.")
        return f"{self.namespace}{topic}.md"

    def _validate_path(self, path: str) -> None:
        """Require a canonical, nested Markdown-like identifier inside this app's namespace."""
        if not isinstance(path, str) or len(path) > 1024 or not path.startswith(self.namespace):
            raise ValueError(f"Memory path must be inside {self.namespace}.")
        relative = path.removeprefix(self.namespace)
        segment = r"[a-z][a-z0-9-]{0,63}"
        if not re.fullmatch(rf"(?:{segment}/)*{segment}(?:\.md)?", relative):
            raise ValueError(f"Invalid canonical memory path: {path}.")

    @staticmethod
    def _normalized(content: str | None) -> str:
        """Normalize line endings and surrounding whitespace for deterministic no-op checks."""
        return (content or "").replace("\r\n", "\n").strip()

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
    namespace: str = "/memories/",
    observer: ActivityObserver | None = None,
) -> Backend:
    """Open existing stores; never provision infrastructure implicitly."""
    actor_id = caller_id(workspace)
    client = AgentKitClient(workspace)
    return Backend(
        client.session_stores.get(session_store),
        client.memory_stores.get(memory_store),
        actor_id,
        namespace,
        observer or quiet_observer,
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
