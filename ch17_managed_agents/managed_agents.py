"""
Atlas v0.17 — Cloud-Managed Research Agent
==========================================
Chapter 17 Project: Deploy Atlas on Anthropic's Managed Agents beta API.

The always-on daemon from ch16 (daemon loop, watchdog, heartbeat) is replaced
by a Managed Agent session. Anthropic's infrastructure handles restarts, state
management, and context compaction.

Usage:
    python managed_agents.py "Summarize recent research on RAG evaluation metrics"
    python managed_agents.py --agent-id agent_01J... "Follow-up question"

Requires: pip install requests anthropic
API note: Uses Anthropic beta header managed-agents-2026-04-01
"""

import argparse
import json
import os
import sys
import time

import requests

API_KEY  = os.environ["ANTHROPIC_API_KEY"]
BASE_URL = "https://api.anthropic.com/v1/managed-agents"
HEADERS  = {
    "X-API-Key":        API_KEY,
    "anthropic-beta":   "managed-agents-2026-04-01",
    "Content-Type":     "application/json",
}


# ── Agent lifecycle ───────────────────────────────────────────────────

def create_agent(name: str = "atlas-research") -> str:
    """
    Define the agent's identity and tool access. Create once, reuse across sessions.
    Returns the agent_id to store and pass to future sessions.
    """
    payload = {
        "name": name,
        "model": "claude-sonnet-4-6",
        "system_prompt": (
            "You are Atlas, a research assistant. You have access to web search, "
            "file operations, and a bash terminal. Work methodically: read before "
            "writing, verify before reporting. When you save findings to a file, "
            "always include the source URL and the date retrieved."
        ),
        "tools": [
            {"type": "bash",            "enabled": True},
            {"type": "file_operations", "enabled": True},
            {"type": "web_search",      "enabled": True},
        ],
    }
    resp = requests.post(f"{BASE_URL}/agents", json=payload, headers=HEADERS)
    resp.raise_for_status()
    agent_id = resp.json()["id"]
    print(f"Agent created: {agent_id}")
    return agent_id


def create_environment() -> str:
    """
    Provision a sandboxed cloud environment.
    networking=limited restricts outbound to an approved proxy — no arbitrary internet.
    """
    payload = {
        "type":                  "anthropic_cloud_sandbox",
        "networking":            "limited",
        "preinstalled_packages": ["python3", "jq", "curl", "git"],
    }
    resp = requests.post(f"{BASE_URL}/environments", json=payload, headers=HEADERS)
    resp.raise_for_status()
    env_id = resp.json()["id"]
    print(f"Environment provisioned: {env_id}")
    return env_id


# ── Session + task execution ──────────────────────────────────────────

def run_task(agent_id: str, env_id: str, task: str, verbose: bool = True) -> str:
    """
    Start a session, send the task, stream SSE events until completion.
    Returns the agent's final text output.
    """
    # Create session binding this agent to this environment
    session_resp = requests.post(
        f"{BASE_URL}/sessions",
        json={"agent_id": agent_id, "environment_id": env_id},
        headers=HEADERS,
    )
    session_resp.raise_for_status()
    session_id = session_resp.json()["id"]
    if verbose:
        print(f"Session started: {session_id}\n")

    # Stream the task as SSE
    event_payload = {"type": "user_message", "text": task}
    output_parts: list[str] = []

    with requests.post(
        f"{BASE_URL}/sessions/{session_id}/events",
        json=event_payload,
        headers=HEADERS,
        stream=True,
    ) as stream:
        stream.raise_for_status()
        for line in stream.iter_lines():
            if not line:
                continue
            decoded = line.decode("utf-8")
            if not decoded.startswith("data:"):
                continue

            try:
                event = json.loads(decoded[5:])
            except json.JSONDecodeError:
                continue

            etype = event.get("type")

            if etype == "tool_execution" and verbose:
                tool_input = json.dumps(event.get("input", {}))[:100]
                print(f"  [tool] {event['tool_name']}: {tool_input}")

            elif etype == "text_delta":
                chunk = event.get("text", "")
                output_parts.append(chunk)
                if verbose:
                    print(chunk, end="", flush=True)

            elif etype == "session_completed":
                usage = event.get("usage", {})
                if verbose:
                    print(f"\n\n[done] tokens used: {usage}")
                break

            elif etype == "error":
                print(f"\n[error] {event.get('message')}", file=sys.stderr)
                break

    return "".join(output_parts)


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Atlas v0.17 — Managed Research Agent")
    parser.add_argument("task", nargs="+", help="Research task to perform")
    parser.add_argument("--agent-id",  default=None, help="Reuse existing agent ID")
    parser.add_argument("--env-id",    default=None, help="Reuse existing environment ID")
    parser.add_argument("--quiet",     action="store_true", help="Suppress streaming output")
    args = parser.parse_args()

    task = " ".join(args.task)

    print("Atlas v0.17 — Managed Research Agent")
    print(f"Task: {task}\n")

    agent_id = args.agent_id or create_agent()
    env_id   = args.env_id   or create_environment()

    output = run_task(agent_id, env_id, task, verbose=not args.quiet)

    if args.quiet:
        print(output)

    print(f"\n\nReuse this agent:       --agent-id {agent_id}")
    print(f"Reuse this environment: --env-id {env_id}")


if __name__ == "__main__":
    main()
