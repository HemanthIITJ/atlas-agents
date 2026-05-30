"""
bedrock_managed_agent.py — AWS Bedrock Managed Agents (OpenAI partnership).

AWS Bedrock Managed Agents host OpenAI's frontier model natively on AWS
infrastructure. Data stays inside your AWS region; you get OpenAI's reasoning
capability alongside AWS's compliance fabric:
  - IAM execution roles with least-privilege scoping
  - Bedrock Guardrails (hallucination filtering, content moderation)
  - CloudTrail audit trail for every invocation
  - PrivateLink connectivity (data never traverses public internet)

The lifecycle has three phases:
  1. create_managed_agent  — define the agent
  2. prepare_managed_agent — compile internal execution targets (do once)
  3. invoke_managed_agent  — run the agent, stream reasoning + output

Usage:
    # Create and prepare the agent (do once per deployment)
    python bedrock_managed_agent.py --setup

    # Invoke with a prompt
    python bedrock_managed_agent.py --agent-id <id> "Analyze last quarter's S3 costs"

Requires: pip install boto3
Note: Uses AWS/OpenAI partnership model — requires Bedrock access in us-east-1
"""

import argparse
import json
import os
import sys
import time

import boto3
from botocore.exceptions import ClientError

AWS_REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")

bedrock_agent         = boto3.client("bedrock-agent",         region_name=AWS_REGION)
bedrock_agent_runtime = boto3.client("bedrock-agent-runtime", region_name=AWS_REGION)

# IAM role ARN that gives Bedrock permission to invoke the model and access data sources
EXECUTION_ROLE_ARN = os.environ.get(
    "BEDROCK_AGENT_ROLE_ARN",
    "arn:aws:iam::123456789012:role/BedrockAtlasAgentExecutionRole",
)

# Pre-configured Bedrock Guardrail — content moderation + hallucination filtering
GUARDRAIL_ID      = os.environ.get("BEDROCK_GUARDRAIL_ID",      "atlas-guardrail")
GUARDRAIL_VERSION = os.environ.get("BEDROCK_GUARDRAIL_VERSION", "1")


# ── Agent creation ────────────────────────────────────────────────────

def create_agent(name: str = "atlas-strategist") -> str:
    """
    Create a managed agent backed by OpenAI's frontier model on Bedrock.

    harnessConfig.enableMemory=True means the platform manages conversational
    state — you don't need to pass conversation history in each call.
    """
    try:
        resp = bedrock_agent.create_managed_agent(
            agentName=name,
            modelId="openai.gpt-4o-frontier-preview",
            instruction=(
                "You are Atlas, a strategic research assistant. "
                "Analyze datasets and summarize findings clearly. "
                "Cite your sources. When referencing AWS services, "
                "align recommendations with Well-Architected principles."
            ),
            harnessConfig={
                "engineType":       "OPENAI_HARNESS",
                "enableMemory":     True,
                "timeoutSeconds":   600,
                "maxTokens":        4096,
            },
            guardrailConfiguration={
                "guardrailIdentifier": GUARDRAIL_ID,
                "guardrailVersion":    GUARDRAIL_VERSION,
            },
            agentExecutionRoleArn=EXECUTION_ROLE_ARN,
            description="Atlas research agent — AWS/OpenAI partnership deployment",
        )
        agent_id = resp["agent"]["agentId"]
        print(f"Agent created: {agent_id}")
        return agent_id

    except ClientError as e:
        print(f"Failed to create agent: {e}", file=sys.stderr)
        raise


def prepare_agent(agent_id: str, wait_seconds: int = 10) -> None:
    """
    Compile the agent's internal execution targets.
    Must be called once after creation (and after any configuration changes).
    """
    bedrock_agent.prepare_managed_agent(agentId=agent_id)
    print(f"Preparing agent {agent_id}...")
    time.sleep(wait_seconds)  # Propagation across AgentCore infrastructure
    print("Agent ready.")


# ── Invocation + streaming ────────────────────────────────────────────

def invoke_agent(
    agent_id: str,
    session_id: str,
    prompt: str,
    alias_id: str = "TSTALIASID",
) -> dict:
    """
    Invoke the agent and stream its multi-step reasoning.

    The response stream contains two event types:
      'chunk' — raw output bytes (text)
      'trace' — orchestration trace (which step the model is on, tool calls)

    CloudTrail automatically records every invocation at the AWS account level
    — no additional logging code needed for compliance.
    """
    resp = bedrock_agent_runtime.invoke_managed_agent(
        agentId=agent_id,
        agentAliasId=alias_id,
        sessionId=session_id,
        inputText=prompt,
        enableTrace=True,
    )

    output_parts: list[str] = []
    trace_steps:  list[dict] = []

    print(f"Streaming response from Bedrock AgentCore:\n")

    for event in resp["completion"]:
        if "chunk" in event:
            text = event["chunk"]["bytes"].decode("utf-8")
            output_parts.append(text)
            print(text, end="", flush=True)

        elif "trace" in event:
            trace_info = event["trace"]["trace"]
            if "orchestrationTrace" in trace_info:
                step = trace_info["orchestrationTrace"].get("step", {})
                if step:
                    trace_steps.append(step)
                    # Show tool invocations inline
                    if "invocationInput" in step:
                        inv = step["invocationInput"]
                        tool_name = inv.get("actionGroupName", "tool")
                        func_name = inv.get("function", "?")
                        print(f"\n  [trace] {tool_name}.{func_name}")

    output = "".join(output_parts)
    print(f"\n\n[done] {len(trace_steps)} orchestration steps")
    print(f"[CloudTrail] All actions logged to AWS account trail automatically")

    return {
        "output":      output,
        "trace_steps": trace_steps,
        "session_id":  session_id,
    }


# ── Session management ────────────────────────────────────────────────

def end_session(agent_id: str, session_id: str, alias_id: str = "TSTALIASID") -> None:
    """
    Explicitly end a session to free managed memory.
    Without this, sessions expire after the configured timeout.
    """
    bedrock_agent_runtime.invoke_managed_agent(
        agentId=agent_id,
        agentAliasId=alias_id,
        sessionId=session_id,
        inputText="",
        endSession=True,
    )
    print(f"Session {session_id} ended.")


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="AWS Bedrock Managed Agent — OpenAI on AWS infrastructure"
    )
    parser.add_argument("prompt",       nargs="*", help="Prompt for the agent")
    parser.add_argument("--setup",      action="store_true", help="Create + prepare agent")
    parser.add_argument("--agent-id",   default=None)
    parser.add_argument("--session-id", default="atlas-bedrock-001")
    parser.add_argument("--end-session",action="store_true",
                        help="End the session after the response")
    args = parser.parse_args()

    if args.setup:
        agent_id = create_agent()
        prepare_agent(agent_id)
        print(f"\nSave: AGENT_ID={agent_id}")
        return

    if not args.prompt:
        parser.print_help()
        sys.exit(1)

    if not args.agent_id:
        print("--agent-id required. Run --setup first.", file=sys.stderr)
        sys.exit(1)

    prompt = " ".join(args.prompt)
    result = invoke_agent(args.agent_id, args.session_id, prompt)

    if args.end_session:
        end_session(args.agent_id, args.session_id)

    return result


if __name__ == "__main__":
    main()
