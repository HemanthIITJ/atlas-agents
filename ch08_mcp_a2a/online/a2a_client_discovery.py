"""
a2a_client_discovery.py — A2A Agent Discovery and Task Delegation
==================================================================
Chapter 8 Online Example: Discover a remote agent's capabilities via its
Agent Card, then delegate a task using the A2A protocol.

The Agent Card (at /.well-known/agent.json) is the A2A equivalent of an
OpenAPI spec — it tells you what the remote agent can do, what input
formats it accepts, and where to send tasks.

Usage:
    python a2a_client_discovery.py
    pip install httpx
"""

import json
import uuid
import httpx


# ── Agent Card Discovery ─────────────────────────────────────────────

def fetch_agent_card(agent_base_url: str) -> dict:
    """
    Fetch the Agent Card from a remote A2A-compliant agent.
    The card describes capabilities, skills, and accepted input formats.
    """
    url = f"{agent_base_url.rstrip('/')}/.well-known/agent.json"
    response = httpx.get(url, timeout=10)
    response.raise_for_status()
    return response.json()


def find_matching_skill(card: dict, keyword: str) -> dict | None:
    """
    Search an agent's skills for one matching a keyword.
    Returns the first matching skill dict, or None.
    """
    for skill in card.get("skills", []):
        if keyword.lower() in skill.get("name", "").lower() or \
           keyword.lower() in skill.get("description", "").lower():
            return skill
    return None


# ── A2A Task Delegation ───────────────────────────────────────────────

def send_task(agent_base_url: str, skill_id: str, text_payload: str) -> dict:
    """
    Send a task to a remote A2A agent and wait for the result.

    A2A task format:
    - id: unique task identifier (you generate it)
    - message: the user turn, with 'parts' carrying the content
    """
    task = {
        "id": str(uuid.uuid4()),
        "skill_id": skill_id,
        "message": {
            "role": "user",
            "parts": [
                {"type": "text", "text": text_payload}
            ],
        },
    }

    endpoint = f"{agent_base_url.rstrip('/')}/tasks/send"
    response = httpx.post(endpoint, json=task, timeout=60)
    response.raise_for_status()
    return response.json()


def extract_text_result(task_result: dict) -> str:
    """
    Pull the first text artifact from an A2A task result.
    """
    artifacts = task_result.get("artifacts", [])
    if not artifacts:
        return "(no artifacts returned)"
    parts = artifacts[0].get("parts", [])
    for part in parts:
        if part.get("type") == "text":
            return part["text"]
    return "(no text part found)"


# ── Orchestrator: Discover → Match → Delegate ─────────────────────────

def delegate_to_best_agent(agents: list[str], task_keyword: str, payload: str) -> str:
    """
    Given a list of remote agent URLs, discover each one, find the
    first agent that has a skill matching task_keyword, then delegate.

    This is the core A2A pattern: dynamic discovery before delegation,
    rather than hardcoding which agent handles what.
    """
    for agent_url in agents:
        try:
            card = fetch_agent_card(agent_url)
            skill = find_matching_skill(card, task_keyword)

            if skill:
                print(f"  ✅ Found matching skill '{skill['name']}' at {agent_url}")
                result = send_task(agent_url, skill["id"], payload)
                return extract_text_result(result)
            else:
                print(f"  ⏭ {agent_url} has no '{task_keyword}' skill — skipping")

        except httpx.HTTPError as e:
            print(f"  ❌ {agent_url} unreachable: {e}")
            continue

    return "No agent available for this task."


# ── Demo ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # In a real deployment, these would be registered agent URLs
    AGENT_REGISTRY = [
        "https://security-agent.example.com",
        "https://code-review-agent.example.com",
        "https://devops-agent.example.com",
    ]

    SAMPLE_CODE = """
def transfer_funds(user_id, amount, account):
    query = f"UPDATE accounts SET balance = balance - {amount} WHERE user_id = {user_id}"
    db.execute(query)
    send_to_account(account, amount)
    """

    print("🔍 A2A Discovery: Looking for a security audit agent...\n")
    result = delegate_to_best_agent(
        agents=AGENT_REGISTRY,
        task_keyword="security",
        payload=f"Audit this code for vulnerabilities:\n{SAMPLE_CODE}",
    )
    print(f"\n📋 Result:\n{result}")