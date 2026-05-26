"""
rollback_harness.py — AgentHarness with automatic file rollback on failure.

The problem: the agent writes a file, the verification loop fails, the agent
tries to fix it but makes it worse, tries again, and after three attempts
you have a file that's more broken than when it started.

This harness extension snapshots every file before the first write in a
session. If the verification loop fails after MAX_RETRIES attempts, it
automatically restores the original file and logs the rollback. The agent
sees a clean error: "Rolled back to original — please start over."

No git required. The snapshots are in-memory for the session.

Usage:
    python rollback_harness.py

Requires: pip install ruff
"""

import logging
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))
from harness import AgentHarness

log = logging.getLogger("rollback_harness")

MAX_RETRIES = 3


class RollbackHarness(AgentHarness):
    """
    AgentHarness that snapshots files before writes and rolls back on
    repeated verification failure.

    Snapshot logic:
      - First write to a file: save original content (or "new file" marker)
      - Each subsequent write: run verification, count failures
      - After MAX_RETRIES failures: restore original, clear retry count
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._snapshots: dict[str, str | None] = {}   # rel_path → original content
        self._retry_counts: dict[str, int] = {}

    def _write_file(self, rel_path: str, content: str) -> str:
        """Write with snapshot + rollback on repeated verification failure."""
        if not rel_path:
            return "Error: path is required."
        try:
            target = self._validate_path(rel_path)
            target.parent.mkdir(parents=True, exist_ok=True)

            # Take snapshot on first write of this file this session
            if rel_path not in self._snapshots:
                if target.exists():
                    self._snapshots[rel_path] = target.read_text(encoding="utf-8")
                    log.info(f"Snapshot saved: {rel_path}")
                else:
                    self._snapshots[rel_path] = None  # New file — rollback = delete
                    log.info(f"New file tracked: {rel_path}")
                self._retry_counts[rel_path] = 0

            # Write the new content
            target.write_text(content, encoding="utf-8")
            self._record(event="write_file", path=rel_path, bytes=len(content))

            if not rel_path.endswith(".py"):
                return f"Wrote {rel_path}."

            verdict = self._verify_python(rel_path)

            if verdict.startswith("✅"):
                # Success — clear the retry counter, keep the snapshot until session ends
                self._retry_counts[rel_path] = 0
                return f"Wrote {rel_path}. {verdict}"

            # Verification failed — increment retry count
            self._retry_counts[rel_path] += 1
            attempt = self._retry_counts[rel_path]

            if attempt >= MAX_RETRIES:
                rollback_msg = self._rollback(rel_path, target)
                return (
                    f"Wrote {rel_path}. {verdict}\n"
                    f"Rolled back after {MAX_RETRIES} failed attempts. {rollback_msg}\n"
                    f"The original file has been restored. Please start this edit over."
                )

            return (
                f"Wrote {rel_path}. {verdict}\n"
                f"Attempt {attempt}/{MAX_RETRIES}. Please fix the issue and try again."
            )

        except PermissionError as e:
            self._record(event="blocked", reason=str(e))
            return str(e)
        except Exception as e:
            return f"Error: {e}"

    def _rollback(self, rel_path: str, target: Path) -> str:
        """Restore the original file content, or delete if it was new."""
        original = self._snapshots.get(rel_path)
        if original is None:
            target.unlink(missing_ok=True)
            self._record(event="rollback", path=rel_path, action="deleted")
            return "(New file deleted — it did not exist before this session.)"
        else:
            target.write_text(original, encoding="utf-8")
            self._record(event="rollback", path=rel_path, action="restored")
            return "(Restored to pre-session content.)"

    def snapshot_summary(self) -> dict[str, Any]:
        return {
            "tracked_files": list(self._snapshots),
            "retry_counts": dict(self._retry_counts),
        }


# ── Demo ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    workspace = Path(__file__).parent.parent / "workspace_rollback"
    harness = RollbackHarness(
        workspace_dir=workspace,
        allowed_commands={"python", "ruff"},
    )

    GOOD_CODE = "def add(a, b):\n    return a + b\n"
    BAD_CODE = "def add(a, b):\n    return a + b\n  broken_indent = True\n"

    print("--- Scenario: agent writes broken code three times in a row ---\n")

    print("Attempt 1 (broken):")
    print(harness.execute_tool("write_file", {"path": "utils.py", "content": BAD_CODE}))

    print("\nAttempt 2 (still broken):")
    print(harness.execute_tool("write_file", {"path": "utils.py", "content": BAD_CODE}))

    print("\nAttempt 3 (still broken — triggers rollback):")
    print(harness.execute_tool("write_file", {"path": "utils.py", "content": BAD_CODE}))

    print("\n--- After rollback: workspace state ---")
    f = workspace / "utils.py"
    if f.exists():
        print(f"utils.py exists with content:\n{f.read_text()}")
    else:
        print("utils.py does not exist (was a new file — deleted on rollback).")

    print("\n--- Scenario: agent writes good code after rollback ---\n")
    print(harness.execute_tool("write_file", {"path": "utils.py", "content": GOOD_CODE}))

    print("\n--- Snapshot summary ---")
    import json
    print(json.dumps(harness.snapshot_summary(), indent=2))

    # Cleanup
    for f in workspace.glob("*.py"):
        f.unlink(missing_ok=True)
    try:
        workspace.rmdir()
    except OSError:
        pass
