"""
watchdog_supervisor.py — Supervise always_on_agent.py with automatic restart.

Runs the agent as a child process and monitors two things:
1. Process liveness: did it exit? (crash)
2. Heartbeat staleness: is it alive but stuck? (hang)

On crash or hang: kill the process, wait with exponential backoff, restart.
After MAX_RESTARTS consecutive failures: log a critical alert and give up.

Usage:
    python watchdog_supervisor.py [--agent always_on_agent.py] [--max-restarts 5]

Run this instead of `python always_on_agent.py` in production.

Requires: pip install anthropic  (agent dependency)
"""

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [watchdog] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

HEARTBEAT_FILE = Path("heartbeat.txt")
HEARTBEAT_TIMEOUT_S = 60   # Declare stuck if heartbeat older than this
POLL_S = 10                # How often the watchdog checks on the agent
BASE_BACKOFF_S = 5         # First restart delay; doubles each time


def heartbeat_age_s() -> float:
    """Seconds since the agent last updated its heartbeat. inf if no file."""
    if not HEARTBEAT_FILE.exists():
        return float("inf")
    try:
        last = float(HEARTBEAT_FILE.read_text().strip())
        return time.time() - last
    except ValueError:
        return float("inf")


def start_agent(cmd: list[str]) -> subprocess.Popen:
    log.info(f"Starting agent: {' '.join(cmd)}")
    return subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)


def supervise(cmd: list[str], max_restarts: int):
    restarts = 0

    while True:
        proc = start_agent(cmd)
        consecutive_stuck = 0

        # Monitor the running process
        while proc.poll() is None:
            time.sleep(POLL_S)
            age = heartbeat_age_s()

            if age > HEARTBEAT_TIMEOUT_S:
                consecutive_stuck += 1
                log.warning(
                    f"Heartbeat stale ({age:.0f}s old). "
                    f"Stuck check #{consecutive_stuck}."
                )
                if consecutive_stuck >= 2:
                    log.error("Agent appears hung. Killing.")
                    proc.kill()
                    proc.wait()
                    break
            else:
                consecutive_stuck = 0

        exit_code = proc.returncode
        if exit_code == 0:
            log.info("Agent exited cleanly (exit 0). Supervisor done.")
            return

        log.warning(f"Agent exited with code {exit_code}.")
        restarts += 1

        if restarts > max_restarts:
            log.critical(
                f"Agent has crashed {restarts} times. "
                "Giving up. Page an engineer."
            )
            sys.exit(1)

        backoff = BASE_BACKOFF_S * (2 ** (restarts - 1))
        log.warning(f"Restart #{restarts}/{max_restarts} in {backoff}s...")
        time.sleep(backoff)


def main():
    parser = argparse.ArgumentParser(description="Watchdog supervisor for Atlas daemon")
    parser.add_argument(
        "--agent",
        default="always_on_agent.py",
        help="Agent script to supervise",
    )
    parser.add_argument(
        "--max-restarts",
        type=int,
        default=5,
        help="Max consecutive restarts before giving up",
    )
    args = parser.parse_args()

    cmd = [sys.executable, args.agent]
    supervise(cmd, args.max_restarts)


if __name__ == "__main__":
    main()
