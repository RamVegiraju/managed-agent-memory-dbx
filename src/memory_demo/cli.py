"""Explicit configuration and terminal controls for the portable sample."""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any

from databricks.sdk import WorkspaceClient
from databricks.sdk.errors.base import DatabricksError
from databricks_agentbricks.errors import AgentCliError
from databricks_agentkit import AgentKitClient
from openai import OpenAIError

from memory_demo.agent import Agent, DatabricksModel
from memory_demo.backend import Backend, caller_id, connect, initialize
from memory_demo.demo import run_demo


def trace(event: str, details: dict[str, Any]) -> None:
    """Print inspectable API activity without credentials or private model reasoning."""
    print(f"\n[{event}] {json.dumps(details, ensure_ascii=False, default=str)}")


def approve(name: str, arguments: dict[str, Any]) -> bool:
    """Require an explicit local user confirmation before a model-requested memory mutation."""
    print(f"\nMemory write requested: {name} {json.dumps(arguments, ensure_ascii=False)}")
    return input("Approve this durable memory change? [y/N] ").strip().lower() == "y"


def parser() -> argparse.ArgumentParser:
    """Build a CLI that never silently selects DEFAULT, a model, or a workspace."""
    result = argparse.ArgumentParser(description="Databricks managed sessions + memory sample")
    result.add_argument("--profile", default=os.getenv("DATABRICKS_CONFIG_PROFILE"))
    result.add_argument("--model", default=os.getenv("DATABRICKS_MODEL"))
    result.add_argument(
        "--reasoning-effort",
        choices=["none", "minimal", "low", "medium", "high", "xhigh", "max"],
        default=os.getenv("DATABRICKS_REASONING_EFFORT"),
        help="Optional model-specific setting; Sol chat tool calls require 'none'",
    )
    result.add_argument(
        "--session-store", default=os.getenv("DEMO_SESSION_STORE", "memory-demo-sessions")
    )
    result.add_argument(
        "--memory-store", default=os.getenv("DEMO_MEMORY_STORE", "memory-demo-memory")
    )
    result.add_argument(
        "--trace", action="store_true", help="Show memory contents and API activity"
    )
    commands = result.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="Read-only authentication, endpoint, and store discovery")
    setup = commands.add_parser("init", help="Explicitly create missing billable backing stores")
    setup.add_argument("--yes", action="store_true", help="Accept Lakebase provisioning costs")
    demo = commands.add_parser("demo", help="Run isolated session and memory acceptance checks")
    demo.add_argument("--storage-only", action="store_true", help="Do not invoke an LLM")
    demo.add_argument("--cleanup", action="store_true", help="Delete only this run's artifacts")
    chat = commands.add_parser("chat", help="Chat with approved remember/recall/forget tools")
    chat.add_argument("--session-id", help="Resume an owned session instead of creating one")
    return result


def validate(arguments: argparse.Namespace) -> None:
    """Reject ambiguous auth and invalid store configuration before any network request."""
    if not arguments.profile or not arguments.profile.strip():
        raise ValueError("Pass --profile NAME or explicitly set DATABRICKS_CONFIG_PROFILE.")
    if not re.fullmatch(r"[a-z][a-z0-9-]{1,54}[a-z0-9]", arguments.memory_store):
        raise ValueError("Memory store names must be 3–56 lowercase letters, numbers, or hyphens.")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", arguments.session_store):
        raise ValueError("Session store name must contain only letters, numbers, '_' or '-'.")
    if arguments.command == "init" and not arguments.yes:
        raise ValueError(
            "init provisions billable Lakebase backing. Review the README, "
            "then explicitly accept with init --yes."
        )
    needs_model = arguments.command == "chat" or (
        arguments.command == "demo" and not arguments.storage_only
    )
    if needs_model and (not arguments.model or not arguments.model.strip()):
        raise ValueError("Pass --model ENDPOINT or set DATABRICKS_MODEL; doctor lists endpoints.")


def check_model(workspace: WorkspaceClient, name: str) -> None:
    """Reject absent, non-chat, or unready endpoints before any model invocation."""
    endpoint = workspace.serving_endpoints.get(name)
    if endpoint.task != "llm/v1/chat":
        raise ValueError("Select a chat endpoint from doctor; this model is not llm/v1/chat.")
    ready = endpoint.state.ready if endpoint.state else None
    if getattr(ready, "value", ready) != "READY":
        raise ValueError("The selected chat endpoint is not READY; no model call was made.")


def doctor(workspace: WorkspaceClient) -> None:
    """Discover actual workspace resources without creating stores or invoking a model."""
    print(f"Workspace: {workspace.config.host}\nAuthenticated actor: {caller_id(workspace)}")
    print("\nChat endpoints (listing does not prove tool-call support):")
    for endpoint in workspace.serving_endpoints.list():
        if endpoint.task == "llm/v1/chat":
            print(f"  {endpoint.name}")
    client = AgentKitClient(workspace)
    print("\nSession stores:")
    for store in client.session_stores.list():
        print(f"  {store.name}")
    print("\nMemory stores:")
    for store in client.memory_stores.list():
        print(f"  {store.display_name}")


def chat(backend: Backend, model: DatabricksModel, session_id: str | None) -> None:
    """Expose thread controls without allowing arbitrary actor or store selection by the model."""
    session = backend.session(session_id) if session_id else backend.new_session()
    agent = Agent(backend, model, approve)
    print(f"\nSession: {session.session_id}")
    print("Commands: /new, /resume ID, /sessions, /history, /memories, /quit")
    print("Ask naturally to remember, recall, change, or forget a preference.")
    while True:
        text = input("\nYou> ").strip()
        if text == "/quit":
            return
        if not text:
            continue
        if text == "/new":
            session = backend.new_session()
            print(f"Session: {session.session_id} (fresh transcript, same caller memory)")
        elif text.startswith("/resume "):
            session = backend.session(text.removeprefix("/resume ").strip())
            print(f"Session: {session.session_id}")
        elif text == "/sessions":
            for saved in backend.sessions():
                print(f"{saved.session_id}  {saved.create_time}")
        elif text == "/history":
            print(json.dumps(backend.history(session.session_id), indent=2, ensure_ascii=False))
        elif text == "/memories":
            for memory in backend.memories():
                print(f"{memory.path}: {memory.content}")
        elif text.startswith("/"):
            print("Unknown command. Use /new, /resume ID, /sessions, /history, /memories, /quit.")
        else:
            print(f"\nAgent> {agent.ask(session.session_id, text)}")


def main() -> int:
    """Run the sample with explicit auth and actionable preview/API errors."""
    arguments = parser().parse_args()
    try:
        validate(arguments)
        workspace = WorkspaceClient(profile=arguments.profile)
        if arguments.command == "doctor":
            doctor(workspace)
            return 0
        if arguments.command == "init":
            initialize(workspace, arguments.session_store, arguments.memory_store)
            print("Stores ready. No automatic provisioning occurs in chat or demo.")
            return 0
        observer = trace if arguments.trace else None

        def factory(namespace: str) -> Backend:
            options = {"trace": observer} if observer else {}
            return connect(
                workspace,
                arguments.session_store,
                arguments.memory_store,
                namespace,
                **options,
            )

        model = None
        if arguments.model and not getattr(arguments, "storage_only", False):
            from databricks_openai import DatabricksOpenAI

            check_model(workspace, arguments.model)
            model = DatabricksModel(
                DatabricksOpenAI(workspace_client=workspace),
                arguments.model,
                arguments.reasoning_effort,
            )
        if arguments.command == "demo":
            run_demo(factory, model, arguments.cleanup)
        else:
            if model is None:
                raise ValueError("A chat model endpoint is required.")
            chat(factory("/preferences/"), model, arguments.session_id)
        return 0
    except (
        ValueError,
        RuntimeError,
        PermissionError,
        DatabricksError,
        AgentCliError,
        OpenAIError,
    ) as error:
        print(f"\nError: {error}")
        print(
            "Verify the explicit profile, preview enablement, store permissions, and a "
            "tool-calling chat endpoint. Missing stores require an explicit init --yes."
        )
        print("If a chat turn failed, memory writes may have committed; inspect before retrying.")
        return 1
    except (EOFError, KeyboardInterrupt):
        print("\nStopped. Saved sessions and memory remain available.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
