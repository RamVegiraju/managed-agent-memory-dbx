"""Exercise the actual installed AgentKit wrappers with an offline REST transport."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from databricks_agentkit import AgentKitClient


def test_real_sdk_canonical_memory_crud_uses_documented_envelopes() -> None:
    requests: list[dict[str, Any]] = []

    def respond(method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        requests.append({"method": method, "path": path, **kwargs})
        if path.endswith("/entries"):
            return {
                "name": "memory-stores/sample-memory/entries/entry-1",
                "actor_id": "verified-user",
                "path": "/memories/preferences/style",
                "content": kwargs["body"]["content"],
                "description": kwargs["body"]["description"],
            }
        if method == "PATCH":
            return {
                "name": "memory-stores/sample-memory/entries/entry-1",
                "actor_id": "verified-user",
                "path": "/memories/preferences/style",
                "content": kwargs["body"]["content"],
                "description": kwargs["body"]["description"],
            }
        if method == "DELETE":
            return {}
        return {
            "name": "memory-stores/sample-memory",
            "display_name": "sample-memory",
        }

    workspace = SimpleNamespace(api_client=SimpleNamespace(do=respond))
    store = AgentKitClient(workspace).memory_stores.get("sample-memory")
    memory = store.add(
        actor_id="verified-user",
        path="/memories/preferences/style",
        content="Concise answers.",
        description="Response style",
    )
    memory.update(content="Detailed answers.", description="Response style")
    memory.delete()

    create, update, delete = requests[-3:]
    assert create == {
        "method": "POST",
        "path": "/api/2.0/agents/memory-stores/sample-memory/entries",
        "query": None,
        "body": {
            "actor_id": "verified-user",
            "path": "/memories/preferences/style",
            "content": "Concise answers.",
            "description": "Response style",
        },
    }
    assert "session_id" not in create["body"]
    assert update == {
        "method": "PATCH",
        "path": "/api/2.0/agents/memory-stores/sample-memory/entries/entry-1",
        "query": {"update_mask": "content,description"},
        "body": {"content": "Detailed answers.", "description": "Response style"},
    }
    assert delete == {
        "method": "DELETE",
        "path": "/api/2.0/agents/memory-stores/sample-memory/entries/entry-1",
        "query": None,
        "body": None,
    }


def test_real_sdk_search_maps_wire_response_and_scopes_request() -> None:
    requests: list[dict[str, Any]] = []

    def respond(method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        requests.append({"method": method, "path": path, **kwargs})
        if path.endswith("entries:search"):
            return {
                "results": [
                    {
                        "managed_memory_entry": {
                            "name": "memory-stores/sample-memory/entries/entry-1",
                            "actor_id": "verified-user",
                            "path": "/memories/preferences/response-preferences.md",
                            "content": "Concise PySpark examples",
                        },
                        "score": 0.8,
                    }
                ]
            }
        return {
            "name": "memory-stores/sample-memory",
            "display_name": "sample-memory",
        }

    workspace = SimpleNamespace(api_client=SimpleNamespace(do=respond))
    store = AgentKitClient(workspace).memory_stores.get("sample-memory")
    result = store.search(
        actor_id="verified-user",
        query="response preferences",
        path_prefix="/memories/preferences/",
        limit=5,
    )
    assert result[0].memory.content == "Concise PySpark examples"
    assert result[0].score == 0.8
    assert requests[-1]["path"] == "/api/2.0/agents/memory-stores/sample-memory/entries:search"
    assert requests[-1]["body"] == {
        "actor_id": "verified-user",
        "query": "response preferences",
        "path_prefix": "/memories/preferences/",
        "page_size": 5,
    }


def test_real_sdk_history_auto_pages_in_requested_order() -> None:
    requests: list[dict[str, Any]] = []

    def respond(method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        requests.append({"method": method, "path": path, **kwargs})
        if path.endswith("/items"):
            if kwargs["query"].get("page_token"):
                return {
                    "session_items": [
                        {"item_id": "2", "data": {"role": "assistant", "content": "Second"}}
                    ]
                }
            return {
                "session_items": [{"item_id": "1", "data": {"role": "user", "content": "First"}}],
                "next_page_token": "next+page=",
            }
        if path.endswith("/sessions/session-a"):
            return {"session_id": "session-a", "actor_id": "verified-user"}
        return {"session_store_name": "sample-sessions"}

    workspace = SimpleNamespace(api_client=SimpleNamespace(do=respond))
    session = AgentKitClient(workspace).session_stores.get("sample-sessions").get("session-a")
    history = [item.data for item in session.list_items(order_by="create_time asc")]
    assert [message["content"] for message in history] == ["First", "Second"]
    assert requests[-1]["query"]["order_by"] == "create_time asc"
    assert requests[-1]["query"]["page_token"] == "next+page="


def test_real_sdk_appends_opaque_chat_items_in_expected_envelope() -> None:
    requests: list[dict[str, Any]] = []

    def respond(method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        requests.append({"method": method, "path": path, **kwargs})
        if path.endswith("items:append"):
            return {"session_items": []}
        if path.endswith("/sessions/session-a"):
            return {"session_id": "session-a", "actor_id": "verified-user"}
        return {"session_store_name": "sample-sessions"}

    workspace = SimpleNamespace(api_client=SimpleNamespace(do=respond))
    session = AgentKitClient(workspace).session_stores.get("sample-sessions").get("session-a")
    message = {"role": "user", "content": "Hello"}
    session.append_items([message])
    assert requests[-1]["method"] == "POST"
    assert requests[-1]["path"].endswith("/sessions/session-a/items:append")
    assert requests[-1]["body"] == {"items": [{"data": message}]}


def test_real_sdk_extracts_memories_in_dry_run_without_persisting() -> None:
    requests: list[dict[str, Any]] = []

    def respond(method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        requests.append({"method": method, "path": path, **kwargs})
        if path.endswith("/extractions"):
            return {
                "entries": [
                    {
                        "name": "memory-stores/sample-memory/entries/candidate",
                        "actor_id": "verified-user",
                        "path": "/memories/preferences/communication/verbosity.md",
                        "content": "Prefers concise answers.",
                        "session_id": "session-a",
                    }
                ]
            }
        if path.endswith("/sessions/session-a"):
            return {"session_id": "session-a", "actor_id": "verified-user"}
        return {"session_store_name": "sample-sessions"}

    workspace = SimpleNamespace(api_client=SimpleNamespace(do=respond))
    session = AgentKitClient(workspace).session_stores.get("sample-sessions").get("session-a")
    extracted = session.extract_memories(
        memory_store="sample-memory", instructions="Atomic preferences only", dry_run=True
    )

    assert extracted[0].content == "Prefers concise answers."
    assert requests[-1]["method"] == "POST"
    assert requests[-1]["path"].endswith("/sessions/session-a/extractions")
    assert requests[-1]["body"] == {
        "memory_store": "memory-stores/sample-memory",
        "instructions": "Atomic preferences only",
        "dry_run": True,
    }
