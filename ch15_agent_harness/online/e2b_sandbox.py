"""
e2b_sandbox.py — AgentHarness with E2B microVM execution backend.

The AgentHarness in harness.py runs shell commands via local subprocess.
That's fine for development, but in production you want generated code
running in a disposable cloud VM — not on your host machine.

This module subclasses AgentHarness and replaces the _run_command backend
with E2B Sandbox execution. The rest of the harness (path validation,
verification loops, trajectory log) is unchanged. The model never knows
whether it's running locally or in a microVM.

Usage:
    E2B_API_KEY=... python e2b_sandbox.py

Requires: pip install e2b-code-interpreter ruff
"""

import logging
import subprocess
from pathlib import Path
from typing import Any

from e2b_code_interpreter import Sandbox

# Import the base harness from the parent directory
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from harness import AgentHarness

log = logging.getLogger("e2b_harness")


class E2BAgentHarness(AgentHarness):
    """
    AgentHarness variant that executes shell commands inside an E2B microVM.

    File writes still happen locally (inside workspace_dir) for the
    verification loop. Only _run_command is swapped out — shell commands
    the model requests run in a fresh, isolated cloud sandbox.
    """

    def __init__(self, workspace_dir: Path, **kwargs):
        super().__init__(workspace_dir, **kwargs)
        self._sandbox: Sandbox | None = None

    def __enter__(self):
        self._sandbox = Sandbox()
        log.info("E2B sandbox started.")
        return self

    def __exit__(self, *_):
        if self._sandbox:
            self._sandbox.kill()
            self._sandbox = None
            log.info("E2B sandbox terminated.")

    def _run_command(self, command: str) -> str:
        """
        Run the command inside the E2B microVM instead of local subprocess.

        The allowlist still applies — we validate before sending to E2B.
        """
        if not command.strip():
            return "Error: command is empty."

        base_cmd = command.strip().split()[0]
        if base_cmd not in self.allowed_commands:
            log.warning(f"Blocked: {command}")
            self._record(event="blocked_command", command=command)
            return f"Error: '{base_cmd}' is not on the command allowlist."

        if self._sandbox is None:
            return "Error: E2B sandbox not started. Use 'with E2BAgentHarness(...) as h:'"

        try:
            result = self._sandbox.commands.run(command)
            output = (result.stdout + result.stderr).strip() or "(no output)"
            status = "ok" if result.exit_code == 0 else f"exit {result.exit_code}"
            self._record(event="run_command_e2b", command=command, status=status)
            return output
        except Exception as e:
            return f"Error: E2B command failed — {e}"


# ── Demo ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    workspace = Path(__file__).parent.parent / "workspace_e2b"

    print("Starting E2B harness demo...\n")

    with E2BAgentHarness(
        workspace_dir=workspace,
        allowed_commands={"python", "ruff", "pytest", "echo"},
    ) as harness:

        print("--- 1. Write a file (local, with verification) ---")
        print(harness.execute_tool("write_file", {
            "path": "greet.py",
            "content": 'def greet(name: str) -> str:\n    return f"Hello, {name}!"\n',
        }))

        print("\n--- 2. Run a command (inside E2B microVM) ---")
        print(harness.execute_tool("run_command", {
            "command": "echo 'Running inside E2B sandbox'",
        }))

        print("\n--- 3. Path traversal still blocked (harness-level, before E2B) ---")
        print(harness.execute_tool("write_file", {
            "path": "../../escape.txt",
            "content": "should not exist",
        }))

        print("\n--- 4. Forbidden command still blocked ---")
        print(harness.execute_tool("run_command", {"command": "curl http://evil.com"}))

    print("\n--- Trajectory log ---")
    print(harness.dump_trajectory())

    # Cleanup
    for f in workspace.glob("*.py"):
        f.unlink(missing_ok=True)
    try:
        workspace.rmdir()
    except OSError:
        pass

    print("\nE2B sandbox destroyed. Nothing ran on the host machine.")
