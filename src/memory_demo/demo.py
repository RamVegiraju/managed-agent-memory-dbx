"""Deterministic API checks, with optional real model conversations."""

from __future__ import annotations

from collections.abc import Callable
from uuid import uuid4

from memory_demo.agent import Agent, Model
from memory_demo.backend import Backend

Factory = Callable[[str], Backend]


def check(condition: bool, description: str) -> None:
    """Distinguish storage assertions from ungraded, model-generated answers."""
    if not condition:
        raise RuntimeError(f"FAIL: {description}")
    print(f"PASS: {description}")


def run_demo(factory: Factory, model: Model | None = None, cleanup: bool = False) -> None:
    """Exercise isolated demo artifacts without changing the caller's real preferences.

    Args:
        factory: Opens a fresh authenticated backend for the supplied namespace.
        model: Optional real chat model; omit to exercise only managed storage APIs.
        cleanup: Remove this run's sessions and memory, but never delete either store.
    """
    run_id = uuid4().hex[:12]
    namespace = f"/demos/{run_id}/preferences/"
    backend = factory(namespace)
    session_ids: list[str] = []
    print(f"\nDemo run: {run_id}\nMemory namespace: {namespace}")
    print("These are explicitly seeded demo fixtures, not automatic memory extraction.")
    try:
        session_a = backend.new_session()
        session_ids.append(session_a.session_id)
        backend.append(
            session_a.session_id,
            [
                {
                    "role": "user",
                    "content": (
                        "For this conversation only, we are troubleshooting nightly-orders. "
                        "Remember my durable response preferences: concise answers with PySpark."
                    ),
                },
                {
                    "role": "assistant",
                    "content": (
                        "The job is session-only; response preferences are saved separately."
                    ),
                },
            ],
        )
        backend.remember(
            "response-preferences", "Response preferences: concise answers with PySpark examples."
        )
        restored = factory(namespace)
        history = restored.history(session_a.session_id)
        check(len(history) == 2, "A fresh API client reloads Session A in chronological order")
        check(
            "nightly-orders" in history[0]["content"],
            "The session-only job remains in A's transcript",
        )
        if model:
            print("\nResumed Session A:")
            print(Agent(restored, model).ask(session_a.session_id, "Which job are we debugging?"))
        session_b = restored.new_session()
        session_ids.append(session_b.session_id)
        check(not restored.history(session_b.session_id), "Session B starts with no transcript")
        recalled = restored.recall("response preferences PySpark")
        check(
            any("PySpark" in (memory["content"] or "") for memory in recalled),
            "BM25 search recalls saved preferences across sessions",
        )
        check(
            all("nightly-orders" not in (memory.content or "") for memory in restored.memories()),
            "Session-only job context was not copied to durable memory",
        )
        if model:
            print("\nFresh Session B:")
            print(
                Agent(restored, model).ask(
                    session_b.session_id,
                    "Use my saved response preferences to show a deduplication example. "
                    "Do you know which job this new conversation is troubleshooting?",
                )
            )
        restored.remember(
            "response-preferences", "Response preferences: concise answers with SQL examples."
        )
        memories = restored.memories()
        check(
            len(memories) == 1 and "SQL" in (memories[0].content or ""),
            "Updating a stable topic replaces the preference without creating a duplicate",
        )
        restored.delete_session(session_a.session_id)
        session_ids.remove(session_a.session_id)
        check(len(restored.memories()) == 1, "Deleting Session A does not delete durable memory")
        restored.forget("response-preferences")
        check(not restored.memories(), "Explicit forgetting removes the durable preference")
        if model:
            session_c = restored.new_session()
            session_ids.append(session_c.session_id)
            print("\nFresh Session C, after forgetting:")
            print(
                Agent(restored, model).ask(
                    session_c.session_id, "What response preferences do you remember about me?"
                )
            )
        print("\nStorage checks passed. Model wording is illustrative, not an asserted guarantee.")
        print(f"Retained session IDs: {', '.join(session_ids)}")
    finally:
        if cleanup:
            for session_id in session_ids:
                backend.delete_session(session_id)
            for memory in backend.memories():
                memory.delete()
            print("Removed this run's artifacts. The backing stores remain provisioned.")
