"""
claude_managed_agent.py — Anthropic Managed Agents: full lifecycle with retry.

Extends the chapter project with production-grade patterns:
  - Retry logic for the beta API (transient 503s are common in preview)
  - Environment reuse across sessions (avoids provisioning cost per task)
  - Session introspection (list running sessions, check session status)
  - Graceful cancellation (send a cancel event before the session times out)

The Managed Agents API is in beta as of Q2 2026. The beta header
`managed-agents-2026-04-01` is required on every request.

Usage:
    # Create resources once, run multiple tasks
    python claude_managed_agent.py --setup
    python claude_managed_agent.py --agent-id <id> --env-id <id> "Task 1"
    python claude_managed_agent.py --agent-id <id> --env-id <id> "Task 2"

    # List active sessions for an agent
    python claude_managed_agent.py --agent-id <id> --list-sessions

Requires: pip install requests
"""

import argparse
import json
import os
import sys
import time
from typing import Iterator

import requests

API_KEY  = os.environ["ANTHROPIC_API_KEY"]
BASE_URL = "https://api.anthropic.com/v1/managed-agents"
HEADERS  = {
    "X-API-Key":      API_KEY,
    "anthropic-beta": "managed-agents-2026-04-01",
    "Content-Type":   "application/json",
}

MAX_RETRIES = 3
RETRY_BACKOFF = [1, 3, 9]   # seconds


# ── HTTP helpers ──────────────────────────────────────────────────────

def _post(path: str, payload: dict, stream: bool = False) -> requests.Response:
    """POST with exponential retry on 503/429."""
    url = f"{BASE_URL}/{path.lstrip('/')}"
    for attempt, wait in enumerate(RETRY_BACKOFF, 1):
        resp = requests.post(url, json=payload, headers=HEADERS, stream=stream)
        if resp.status_code in (429, 503) and attempt < MAX_RETRIES:
            print(f"  [retry {attempt}/{MAX_RETRIES}] {resp.status_code} — waiting {wait}s",
                  file=sys.stderr)
            time.sleep(wait)
            continue
        resp.raise_for_status()
        return resp
    resp.raise_for_status()
    return resp  # unreachable, satisfies type checker


def _get(path: str) -> dict:
    url = f"{BASE_URL}/{path.lstrip('/')}"
    resp = requests.get(url, headers=HEADERS)
    resp.raise_for_status()
    return resp.json()


# ── Resource management ───────────────────────────────────────────────

def create_agent(name: str = "atlas-research", model: str = "claude-sonnet-4-6") -> str:
    resp = _post("agents", {
        "name": name,
        "model": model,
        "system_prompt": (
            "You are Atlas, a research assistant with access to web search, "
            "file operations, and a bash terminal. Work methodically: "
            "search before writing, verify before reporting."
        ),
        "tools": [
            {"type": "bash",            "enabled": True},
            {"type": "file_operations", "enabled": True},
            {"type": "web_search",      "enabled": True},
        ],
    })
    agent_id = resp.json()["id"]
    print(f"Created agent:       {agent_id}")
    return agent_id


def create_environment(networking: str = "limited") -> str:
    """
    'networking: limited'  — outbound via approved proxy only (recommended)
    'networking: open'     — full internet access (use carefully)
    """
    resp = _post("environments", {
        "type":                  "anthropic_cloud_sandbox",
        "networking":            networking,
        "preinstalled_packages": ["python3", "jq", "curl", "git"],
    })
    env_id = resp.json()["id"]
    print(f"Created environment: {env_id}")
    return env_id


def get_agent(agent_id: str) -> dict:
    return _get(f"agents/{agent_id}")


def list_sessions(agent_id: str) -> list[dict]:
    return _get(f"agents/{agent_id}/sessions").get("sessions", [])


# ── Session execution ─────────────────────────────────────────────────

def _parse_sse(stream: requests.Response) -> Iterator[dict]:
    """Parse Server-Sent Events from a streaming response."""
    for line in stream.iter_lines():
        if not line:
            continue
        decoded = line.decode("utf-8")
        if decoded.startswith("data:"):
            try:
                yield json.loads(decoded[5:])
            except json.JSONDecodeError:
                pass


def run_task(
    agent_id: str,
    env_id: str,
    task: str,
    metadata: dict | None = None,
) -> dict:
    """
    Create a session, execute the task, stream events to stdout.
    Returns a summary dict with output text, usage, and session ID.
    """
    session_resp = _post("sessions", {
        "agent_id":       agent_id,
        "environment_id": env_id,
        "metadata":       metadata or {},
    })
    session_id = session_resp.json()["id"]
    print(f"Session: {session_id}")
    print(f"Task:    {task[:80]}{'...' if len(task) > 80 else ''}\n")

    output_parts: list[str] = []
    usage: dict = {}

    with _post(f"sessions/{session_id}/events",
               {"type": "user_message", "text": task},
               stream=True) as stream:

        for event in _parse_sse(stream):
            etype = event.get("type")

            if etype == "tool_execution":
                tool_name  = event.get("tool_name", "?")
                tool_input = json.dumps(event.get("input", {}))
                print(f"  [tool] {tool_name}: {tool_input[:120]}")

            elif etype == "tool_result":
                # Show first 80 chars of tool output for visibility
                result = str(event.get("output", ""))[:80]
                print(f"  [result] {result}{'...' if len(result) >= 80 else ''}")

            elif etype == "text_delta":
                chunk = event.get("text", "")
                output_parts.append(chunk)
                print(chunk, end="", flush=True)

            elif etype == "session_completed":
                usage = event.get("usage", {})
                print(f"\n\n[done] input={usage.get('input_tokens')} "
                      f"output={usage.get('output_tokens')} tokens")
                break

            elif etype == "error":
                print(f"\n[error] {event.get('message')}", file=sys.stderr)
                break

    return {
        "session_id": session_id,
        "output":     "".join(output_parts),
        "usage":      usage,
    }


def cancel_session(session_id: str) -> None:
    """Send a cancel event to stop a running session gracefully."""
    _post(f"sessions/{session_id}/events", {"type": "cancel"})
    print(f"Session {session_id} cancelled.")


# ── CLI ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Claude Managed Agent — full lifecycle with retry"
    )
    parser.add_argument("task",          nargs="*", help="Task to run")
    parser.add_argument("--setup",       action="store_true", help="Create new agent + environment")
    parser.add_argument("--agent-id",    default=None)
    parser.add_argument("--env-id",      default=None)
    parser.add_argument("--list-sessions", action="store_true")
    args = parser.parse_args()

    if args.setup:
        agent_id = create_agent()
        env_id   = create_environment()
        print(f"\nSave these IDs:")
        print(f"  AGENT_ID={agent_id}")
        print(f"  ENV_ID={env_id}")
        return

    if args.list_sessions:
        if not args.agent_id:
            print("--agent-id required for --list-sessions", file=sys.stderr)
            sys.exit(1)
        sessions = list_sessions(args.agent_id)
        print(f"{len(sessions)} sessions for {args.agent_id}:")
        for s in sessions:
            print(f"  {s['id']}  status={s.get('status')}  created={s.get('created_at')}")
        return

    if not args.task:
        parser.print_help()
        sys.exit(1)

    agent_id = args.agent_id or create_agent()
    env_id   = args.env_id   or create_environment()
    task     = " ".join(args.task)

    result = run_task(agent_id, env_id, task)
    print(f"\nReuse:  --agent-id {agent_id} --env-id {env_id}")
    return result


if __name__ == "__main__":
    main()
