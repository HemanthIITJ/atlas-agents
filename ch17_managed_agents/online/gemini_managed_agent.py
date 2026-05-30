"""
gemini_managed_agent.py — Google Cloud Gemini Enterprise Agent Platform.

Demonstrates the Skill Registry pattern: register a governed capability once,
attach it to multiple agents. Each agent inherits the skill's IAM identity
and enforced policies automatically — no per-agent configuration needed.

The three key objects:
  1. Skill     — a governed, reusable capability (data connector, API wrapper)
  2. Agent     — a model + instruction set + set of skills
  3. Interaction — a single invocation that returns trace + output

Usage:
    # Create a skill and an agent (do once)
    python gemini_managed_agent.py --setup

    # Invoke the agent
    python gemini_managed_agent.py --agent-name projects/.../agents/atlas "Your question"

Requires: pip install google-cloud-aiplatform
Note: Uses the Vertex AI Managed Agents API (preview, Q2 2026)
"""

import argparse
import json
import os
import sys

# Vertex AI Managed Agents API — preview client
from google.cloud import aiplatform
from google.cloud.aiplatform import gapic as aip_gapic

PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "your-project-id")
LOCATION   = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")


def init_clients() -> tuple:
    """Initialize the Vertex AI clients."""
    aiplatform.init(project=PROJECT_ID, location=LOCATION)
    parent   = f"projects/{PROJECT_ID}/locations/{LOCATION}"
    skills   = aip_gapic.AgentSkillsServiceClient()
    agents   = aip_gapic.ManagedAgentsServiceClient()
    interact = aip_gapic.AgentInteractionsServiceClient()
    return parent, skills, agents, interact


# ── Skill Registry ────────────────────────────────────────────────────

def register_skill(skills_client, parent: str, name: str, dataset: str) -> str:
    """
    Register a BigQuery read-only skill in the Skill Registry.

    Skills are versioned and independently deployable. Updating a skill
    (e.g., changing encryption policy) propagates to all agents that use it
    without redeploying those agents.
    """
    skill = skills_client.create_skill(
        parent=parent,
        skill={
            "display_name": name,
            "description":  f"Read-only access to the {dataset} BigQuery dataset.",
            "connector_type": "BIGQUERY_CONNECTOR",
            "connector_config": {
                "dataset":  dataset,
                "mode":     "READ_ONLY",
                "location": LOCATION,
            },
            "enforced_policies": [
                "require_encrypted_egress",
                "log_all_queries",
                "deny_cross_region_transfer",
            ],
        },
    )
    print(f"Skill registered: {skill.name}")
    return skill.name


# ── Agent creation ────────────────────────────────────────────────────

def create_agent(agents_client, parent: str, skill_name: str) -> str:
    """
    Deploy a Managed Agent with a governed IAM service account identity.

    The agent_identity_iam is a least-privilege service account that has
    only the permissions needed to execute skills. It cannot escalate
    its own privileges or access resources outside the declared skills.
    """
    operation = agents_client.create_managed_agent(
        parent=parent,
        managed_agent={
            "display_name": "atlas-data-analyst",
            "model":        "gemini-2.5-flash",
            "instruction": (
                "You are Atlas, a data analyst assistant. Use the BigQuery skill "
                "to answer questions about product usage and research trends. "
                "Always include the SQL query you ran in your response so the "
                "user can verify it."
            ),
            "skills": [skill_name],
            "governance": {
                "agent_identity_iam":    (
                    f"atlas-agent@{PROJECT_ID}.iam.gserviceaccount.com"
                ),
                "enable_gateway_logging": True,
                "audit_log_level":        "FULL",
            },
        },
    )
    # create_managed_agent returns a long-running operation
    agent = operation.result(timeout=120)
    print(f"Agent deployed: {agent.name}")
    return agent.name


# ── Interaction ───────────────────────────────────────────────────────

def invoke_agent(
    interact_client,
    agent_name: str,
    session_id: str,
    prompt: str,
) -> dict:
    """
    Invoke the agent for one turn. Returns the full response including
    execution trace (each step the agent took to produce the answer).

    The trace is the primary observability artifact — it shows which
    skill was called, what query was run, and how the model used the result.
    """
    response = interact_client.invoke(
        agent=agent_name,
        session_id=session_id,
        input_text=prompt,
        parameters={"isolation_level": "strict"},
    )

    print(f"Status: {response.status}")
    print("\nExecution trace:")
    for step in response.trace_logs:
        print(f"  Step {step.step_number}: {step.action_description}")
        if hasattr(step, "skill_call") and step.skill_call:
            print(f"    Skill: {step.skill_call.skill_name}")
            print(f"    Query: {step.skill_call.query[:200]}")

    print(f"\nAtlas:\n{response.output_text}")
    return {
        "output":      response.output_text,
        "trace_steps": len(response.trace_logs),
        "status":      response.status,
    }


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Gemini Managed Agent — Skill Registry pattern"
    )
    parser.add_argument("prompt",       nargs="*", help="Prompt to send to the agent")
    parser.add_argument("--setup",      action="store_true", help="Register skill + create agent")
    parser.add_argument("--agent-name", default=None, help="Fully-qualified agent resource name")
    parser.add_argument("--session-id", default="atlas-session-001")
    parser.add_argument("--dataset",    default=f"{PROJECT_ID}.atlas_analytics",
                        help="BigQuery dataset for the skill")
    args = parser.parse_args()

    parent, skills_client, agents_client, interact_client = init_clients()

    if args.setup:
        skill_name = register_skill(
            skills_client, parent, "atlas-bigquery-read", args.dataset
        )
        agent_name = create_agent(agents_client, parent, skill_name)
        print(f"\nSave this agent name:")
        print(f"  AGENT_NAME={agent_name}")
        return

    if not args.prompt:
        parser.print_help()
        sys.exit(1)

    if not args.agent_name:
        print("--agent-name required. Run --setup first.", file=sys.stderr)
        sys.exit(1)

    prompt = " ".join(args.prompt)
    invoke_agent(interact_client, args.agent_name, args.session_id, prompt)


if __name__ == "__main__":
    main()
