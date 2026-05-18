"""
heartbeat_monitor.py — HTTP health endpoint + external staleness checker.

Two components:

1. HeartbeatServer (runs inside the agent process):
   A tiny FastAPI server the agent POSTs to on every loop iteration.
   Exposes GET /health so load balancers and container orchestrators can
   probe liveness without reading the filesystem.

2. HeartbeatChecker (runs as a separate process):
   Polls GET /health on a schedule. If the response is stale or unreachable,
   logs a CRITICAL alert. In production, swap the log line for a PagerDuty
   call, a Slack message, or an SNS notification.

This pattern matters when you can't read the heartbeat file directly —
for example, when the agent runs in a container and the watchdog runs
outside it.

Usage (two terminals):

    Terminal 1 — start the health server (normally embedded in always_on_agent.py):
        python heartbeat_monitor.py --mode server

    Terminal 2 — start the checker:
        python heartbeat_monitor.py --mode checker --url http://localhost:8765

Requires: pip install fastapi uvicorn httpx
"""

import argparse
import logging
import threading
import time

import httpx
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Server ──────────────────────────────────────────────────────────────

app = FastAPI(title="Atlas Heartbeat")
_last_beat: float = time.time()
_task_count: int = 0
_error_count: int = 0


@app.post("/heartbeat")
def post_heartbeat(tasks_processed: int = 0, errors: int = 0):
    """Agent calls this on every loop iteration."""
    global _last_beat, _task_count, _error_count
    _last_beat = time.time()
    _task_count += tasks_processed
    _error_count += errors
    return {"ok": True}


@app.get("/health")
def get_health(stale_threshold_s: int = 60):
    """
    Returns 200 if the agent is healthy, 503 if the heartbeat is stale.
    Container orchestrators (Kubernetes, ECS) probe this endpoint.
    """
    age = time.time() - _last_beat
    healthy = age < stale_threshold_s
    return JSONResponse(
        status_code=200 if healthy else 503,
        content={
            "healthy": healthy,
            "heartbeat_age_s": round(age, 1),
            "tasks_processed": _task_count,
            "errors": _error_count,
        },
    )


class HeartbeatServer:
    """Runs the FastAPI health server in a background thread."""

    def __init__(self, port: int = 8765):
        self.port = port
        self._thread: threading.Thread | None = None

    def start(self):
        config = uvicorn.Config(app, host="0.0.0.0", port=self.port, log_level="warning")
        server = uvicorn.Server(config)
        self._thread = threading.Thread(target=server.run, daemon=True)
        self._thread.start()
        log.info(f"Health endpoint: http://localhost:{self.port}/health")

    def beat(self, tasks: int = 0, errors: int = 0):
        """Call this from your agent loop each iteration."""
        global _last_beat, _task_count, _error_count
        _last_beat = time.time()
        _task_count += tasks
        _error_count += errors


# ── Checker ─────────────────────────────────────────────────────────────

def run_checker(url: str, interval_s: int = 30, stale_threshold_s: int = 60):
    """
    Polls /health on a schedule. Logs CRITICAL if stale or unreachable.
    Replace the log line with your alerting mechanism in production.
    """
    log.info(f"Heartbeat checker started. Polling {url}/health every {interval_s}s")

    while True:
        time.sleep(interval_s)
        try:
            resp = httpx.get(
                f"{url}/health",
                params={"stale_threshold_s": stale_threshold_s},
                timeout=5,
            )
            data = resp.json()
            if resp.status_code == 200:
                log.info(
                    f"Agent healthy — age={data['heartbeat_age_s']}s, "
                    f"tasks={data['tasks_processed']}, errors={data['errors']}"
                )
            else:
                log.critical(
                    f"ALERT: Agent heartbeat stale ({data['heartbeat_age_s']}s). "
                    "Page the on-call engineer."
                )
        except httpx.ConnectError:
            log.critical(
                f"ALERT: Cannot reach agent at {url}. "
                "Process may be down. Page the on-call engineer."
            )


# ── Demo server (for --mode server) ─────────────────────────────────────

def run_demo_server(port: int):
    """Simulate an agent posting heartbeats every 5s for 60s, then stopping."""
    server = HeartbeatServer(port=port)
    server.start()

    log.info("Simulating agent heartbeats for 60s, then going silent...")
    for i in range(12):
        time.sleep(5)
        server.beat(tasks=1)
        log.info(f"Beat {i+1}/12")

    log.info("Agent 'crashed' — no more heartbeats. Checker should alert.")
    time.sleep(120)  # Stay alive so the checker can detect staleness


# ── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Heartbeat server or checker")
    parser.add_argument("--mode", choices=["server", "checker"], default="server")
    parser.add_argument("--url", default="http://localhost:8765")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--interval", type=int, default=30, help="Checker poll interval (s)")
    parser.add_argument("--stale-after", type=int, default=60, help="Stale threshold (s)")
    args = parser.parse_args()

    if args.mode == "server":
        run_demo_server(args.port)
    else:
        run_checker(args.url, args.interval, args.stale_after)


if __name__ == "__main__":
    main()
