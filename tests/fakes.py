"""Small test doubles; never used by the runnable sample."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any


@dataclass
class FakeMemory:
    actor_id: str
    path: str
    content: str
    description: str
    store: FakeMemoryStore
    session_id: str | None = None

    def update(self, content: str, description: str) -> FakeMemory:
        self.content = content
        self.description = description
        return self

    def delete(self) -> None:
        self.store.entries.remove(self)


class FakeMemoryStore:
    def __init__(self) -> None:
        self.entries: list[FakeMemory] = []
        self.search_calls: list[dict[str, Any]] = []

    def add(self, actor_id: str, path: str, content: str, description: str) -> FakeMemory:
        entry = FakeMemory(actor_id, path, content, description, self)
        self.entries.append(entry)
        return entry

    def list(self, actor_id: str, path_prefix: str) -> list[FakeMemory]:
        return [
            entry
            for entry in self.entries
            if entry.actor_id == actor_id and entry.path.startswith(path_prefix)
        ]

    def search(
        self, actor_id: str, query: str, path_prefix: str, limit: int
    ) -> list[SimpleNamespace]:
        self.search_calls.append(
            {"actor_id": actor_id, "query": query, "path_prefix": path_prefix, "limit": limit}
        )
        terms = query.lower().split()
        entries = self.list(actor_id, path_prefix)
        return [
            SimpleNamespace(memory=entry, score=1.0)
            for entry in entries
            if any(term in f"{entry.description} {entry.content}".lower() for term in terms)
        ][:limit]


class FakeSession:
    def __init__(self, store: FakeSessionStore, actor_id: str, session_id: str) -> None:
        self.store = store
        self.actor_id = actor_id
        self.session_id = session_id
        self.create_time = None
        self.items: list[dict[str, Any]] = []
        self.last_order = None
        self.reads = 0

    def append_items(self, items: list[dict[str, Any]]) -> None:
        self.items.extend(items)

    def list_items(self, order_by: str) -> list[SimpleNamespace]:
        self.reads += 1
        self.last_order = order_by
        return [SimpleNamespace(data=item) for item in self.items]

    def delete(self) -> None:
        del self.store.entries[self.session_id]


class FakeSessionStore:
    def __init__(self) -> None:
        self.entries: dict[str, FakeSession] = {}
        self.counter = 0

    def add(self, actor_id: str, session_id: str | None = None) -> FakeSession:
        self.counter += 1
        session_id = session_id or f"session-{self.counter}"
        session = FakeSession(self, actor_id, session_id)
        self.entries[session_id] = session
        return session

    def get(self, session_id: str) -> FakeSession:
        return self.entries[session_id]

    def list(self) -> list[FakeSession]:
        return list(self.entries.values())


class ScriptedModel:
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.responses = iter(responses)
        self.calls: list[dict[str, Any]] = []

    def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], final: bool
    ) -> dict[str, Any]:
        self.calls.append({"messages": list(messages), "tools": tools, "final": final})
        return next(self.responses)
