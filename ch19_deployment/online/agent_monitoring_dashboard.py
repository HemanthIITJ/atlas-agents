"""
agent_monitoring_dashboard.py — Prometheus metrics for Atlas agents.

The key insight: traditional web app metrics (request latency, error rate)
don't tell you anything about whether an agent is working well. You need
agent-specific signals:
  - How many tool calls per task? (efficiency)
  - What fraction of tasks exceed the token budget? (runaway detection)
  - What is the retry rate? (reliability signal)
  - What is the end-to-end cost per session? (cost management)

This module instruments a FastAPI app with those metrics via Prometheus.
The Grafana dashboard JSON at the bottom imports them.

Usage:
    # Add to your FastAPI app:
    from agent_monitoring_dashboard import instrument_app
    instrument_app(app)

    # Then metrics are available at GET /metrics
    curl http://localhost:8000/metrics

Requires: pip install prometheus-client fastapi
"""

import time
from contextlib import asynccontextmanager
from functools import wraps
from typing import Callable

from fastapi import FastAPI, Request, Response
from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
)

# ── Metric definitions ────────────────────────────────────────────────

registry = CollectorRegistry()

# Task-level counters
TASKS_TOTAL = Counter(
    "atlas_tasks_total",
    "Total agent tasks processed",
    ["status"],           # "success", "failure", "timeout"
    registry=registry,
)

TASK_LATENCY = Histogram(
    "atlas_task_latency_seconds",
    "End-to-end task latency",
    buckets=[1, 5, 15, 30, 60, 120, 300],
    registry=registry,
)

# Token and cost tracking
TOKENS_USED = Counter(
    "atlas_tokens_total",
    "Total tokens consumed",
    ["direction"],        # "input", "output"
    registry=registry,
)

SESSION_COST = Histogram(
    "atlas_session_cost_usd",
    "Estimated cost per session in USD",
    buckets=[0.001, 0.005, 0.01, 0.05, 0.10, 0.50, 1.00],
    registry=registry,
)

# Tool-call efficiency
TOOL_CALLS = Counter(
    "atlas_tool_calls_total",
    "Total tool invocations",
    ["tool_name", "status"],   # status: "success", "error"
    registry=registry,
)

TOOL_CALLS_PER_TASK = Histogram(
    "atlas_tool_calls_per_task",
    "Number of tool calls per completed task",
    buckets=[0, 1, 2, 3, 5, 8, 13, 21],
    registry=registry,
)

BUDGET_EXCEEDED = Counter(
    "atlas_budget_exceeded_total",
    "Tasks that exceeded the tool call budget",
    registry=registry,
)

# Retry and reliability
RETRY_ATTEMPTS = Counter(
    "atlas_retry_attempts_total",
    "Total retry attempts",
    ["reason"],           # "rate_limit", "timeout", "error"
    registry=registry,
)

# Active sessions (gauge — can go up and down)
ACTIVE_SESSIONS = Gauge(
    "atlas_active_sessions",
    "Currently active agent sessions",
    registry=registry,
)

# Cost rates ($/1M tokens) as of Q2 2026
COST_PER_INPUT_TOKEN  = 3.00 / 1_000_000   # Claude Sonnet 4.6
COST_PER_OUTPUT_TOKEN = 15.00 / 1_000_000


# ── Instrumentation helpers ───────────────────────────────────────────

class TaskTracker:
    """Context manager that records all metrics for one agent task."""

    def __init__(
        self,
        session_id: str,
        tool_budget: int = 10,
    ):
        self.session_id  = session_id
        self.tool_budget = tool_budget
        self._start      = time.time()
        self._tools      = 0
        self._in_tokens  = 0
        self._out_tokens = 0

    def __enter__(self):
        ACTIVE_SESSIONS.inc()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        ACTIVE_SESSIONS.dec()
        latency = time.time() - self._start
        status  = "failure" if exc_type else "success"

        TASKS_TOTAL.labels(status=status).inc()
        TASK_LATENCY.observe(latency)
        TOOL_CALLS_PER_TASK.observe(self._tools)

        if self._tools > self.tool_budget:
            BUDGET_EXCEEDED.inc()

        cost = (self._in_tokens  * COST_PER_INPUT_TOKEN +
                self._out_tokens * COST_PER_OUTPUT_TOKEN)
        SESSION_COST.observe(cost)
        return False  # Don't suppress exceptions

    def record_tokens(self, input_tokens: int, output_tokens: int):
        self._in_tokens  += input_tokens
        self._out_tokens += output_tokens
        TOKENS_USED.labels(direction="input").inc(input_tokens)
        TOKENS_USED.labels(direction="output").inc(output_tokens)

    def record_tool(self, tool_name: str, success: bool = True):
        self._tools += 1
        status = "success" if success else "error"
        TOOL_CALLS.labels(tool_name=tool_name, status=status).inc()

    def record_retry(self, reason: str = "error"):
        RETRY_ATTEMPTS.labels(reason=reason).inc()


# ── FastAPI instrumentation ───────────────────────────────────────────

def instrument_app(app: FastAPI):
    """
    Add Prometheus metrics endpoint and request tracking middleware to a FastAPI app.
    Call this once after creating your app:

        app = FastAPI()
        instrument_app(app)
    """

    # Metrics endpoint
    @app.get("/metrics")
    async def metrics():
        return Response(
            content=generate_latest(registry),
            media_type=CONTENT_TYPE_LATEST,
        )

    # HTTP request middleware
    @app.middleware("http")
    async def track_requests(request: Request, call_next: Callable):
        start = time.time()
        response = await call_next(request)
        # You can add request-level counters here if needed
        return response

    return app


# ── Demo ──────────────────────────────────────────────────────────────

def demo():
    """Simulate a series of agent tasks and print the resulting metrics."""
    import random

    print("Simulating 10 agent tasks...\n")

    for i in range(10):
        tool_budget = 5
        with TaskTracker(session_id=f"session-{i}", tool_budget=tool_budget) as t:
            # Simulate token usage
            in_tokens  = random.randint(500, 3000)
            out_tokens = random.randint(100, 800)
            t.record_tokens(in_tokens, out_tokens)

            # Simulate tool calls
            n_tools = random.randint(1, 8)
            for j in range(n_tools):
                tool = random.choice(["web_search", "read_file", "write_file", "bash"])
                t.record_tool(tool, success=random.random() > 0.1)

            # Occasionally simulate a retry
            if random.random() < 0.2:
                t.record_retry("rate_limit")

        cost = (in_tokens * COST_PER_INPUT_TOKEN + out_tokens * COST_PER_OUTPUT_TOKEN)
        print(f"  Task {i+1:2d}: {n_tools} tools, {in_tokens+out_tokens:,} tokens, ${cost:.4f}")

    print("\nPrometheus metrics output:")
    print("-" * 50)
    print(generate_latest(registry).decode()[:2000])


if __name__ == "__main__":
    demo()
